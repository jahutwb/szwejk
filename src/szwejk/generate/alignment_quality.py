"""Alignment quality and linguistic similarity scores for hybridization candidates.

Two distinct measures
---------------------
alignment_quality_score  — how reliable is the alignment?

    For TOKEN candidates the dictionary is the *primary* signal:
        dict_hit  → CS lemma confirmed by CS→PL lookup (ground truth)
        form      → char-trigram Jaccard (cognate / near-identical Slavic forms)
        upos      → part-of-speech match (structural sanity check)
        deprel    → syntactic-role match
        embedding → fallback when dict and form are both weak

    For PHRASE / SUBTREE / SENTENCE / PARAGRAPH candidates the tree-matching
    quality dominates (individual lemmas are less informative for multi-word spans):
        structural = token_support + lemma_overlap + upos + deprel
        semantic   = embedding + candidate.score

linguistic_similarity_score  — is the Czech word the right translation of the Polish word?
    Composite of dictionary validation (CS→PL lookup) and char-trigram Jaccard.
    Used in phase-3 as a tiebreaker when alignment_quality scores are tied.
"""
from __future__ import annotations

import json
from pathlib import Path

from szwejk.generate.candidate_model import ParagraphHybridCandidate
from szwejk.generate.family_tracking import _cost_family_ids, _candidate_cost_target_lemmas

# ---------------------------------------------------------------------------
# Char-trigram Jaccard (measures surface / cognate similarity)
# ---------------------------------------------------------------------------

def _char_trigrams(text: str) -> frozenset[str]:
    padded = f"  {text}  "
    return frozenset(padded[i : i + 3] for i in range(len(padded) - 2))


def char_trigram_jaccard(a: str, b: str) -> float:
    """Jaccard similarity of character trigrams — proxy for cognate closeness."""
    if not a and not b:
        return 1.0
    ta, tb = _char_trigrams(a.lower()), _char_trigrams(b.lower())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------------------
# Structural helpers (re-use signals stored in candidate metadata)
# ---------------------------------------------------------------------------

def _signals(candidate: ParagraphHybridCandidate) -> dict[str, float]:
    raw = candidate.metadata.get("signals") or {}
    return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}


def _upos(candidate: ParagraphHybridCandidate, sig: dict[str, float]) -> float:
    v = sig.get("upos", sig.get("upos_overlap", -1.0))
    if v < 0:
        # token with no explicit upos signal → infer from score
        return min(1.0, float(candidate.score) * 1.1) if candidate.granularity == "token" else 0.0
    return v


def _deprel(candidate: ParagraphHybridCandidate, sig: dict[str, float]) -> float:
    return sig.get("deprel", sig.get("dependency_overlap", 0.0))


def _lemma_signal(sig: dict[str, float]) -> float:
    return sig.get("lemma", sig.get("lemma_overlap", 0.0))


# ---------------------------------------------------------------------------
# alignment_quality_score
# ---------------------------------------------------------------------------

def _cs_lemmas_from_family_ids(candidate: ParagraphHybridCandidate) -> list[str]:
    """CS (target) lemmas extracted directly from family_ids right-side of '::'."""
    lemmas: list[str] = []
    for fid in _cost_family_ids(candidate):
        if "::" not in fid:
            continue
        cs_side = fid.split("::", 1)[1]
        for part in cs_side.split("+"):
            part = part.strip().lower()
            if part and part not in lemmas:
                lemmas.append(part)
    return lemmas


def _token_dict_hit(
    candidate: ParagraphHybridCandidate,
    wiktionary_lookup: dict[str, list[str]] | None,
) -> float:
    """Dictionary validation score for a token candidate.

    Looks up each CS lemma (right side of '::' in family_ids) → expected PL
    lemmas, checks fraction of source (PL) lemmas confirmed.
    Returns 0.0 when no dictionary entry exists — absence ≠ wrong alignment.
    """
    if not wiktionary_lookup:
        return 0.0
    pl_lemmas = _source_lemmas_of_candidate(candidate)
    cs_lemmas = _cs_lemmas_from_family_ids(candidate)
    if not pl_lemmas or not cs_lemmas:
        return 0.0
    hits: list[float] = []
    for cs in cs_lemmas:
        translations = wiktionary_lookup.get(cs)
        if translations:
            hit = len(pl_lemmas & set(translations)) / len(pl_lemmas)
            hits.append(hit)
    return sum(hits) / len(hits) if hits else 0.0


def alignment_quality_score(
    candidate: ParagraphHybridCandidate,
    wiktionary_lookup: dict[str, list[str]] | None = None,
) -> float:
    """Returns [0, 1] — how reliable is this alignment?

    TOKEN candidates — dictionary is the primary signal:
        If dict_hit > 0 (CS lemma found in dictionary and PL lemma confirmed):
            0.55 * dict_hit + 0.25 * form + 0.15 * upos + 0.05 * deprel
        If no dictionary entry (might be a valid cognate or proper noun):
            0.40 * embedding + 0.35 * form + 0.20 * upos + 0.05 * deprel

    Non-token candidates — structural tree matching + semantic confidence:
        0.50 * structural(token_support + lemma + upos + deprel) + 0.50 * embedding
    """
    sig = _signals(candidate)
    base = max(0.0, min(1.0, float(candidate.score)))
    embedding = min(1.0, max(base, sig.get("embedding", base)))
    upos = _upos(candidate, sig)
    deprel = _deprel(candidate, sig)
    lemma = _lemma_signal(sig)
    token_support = float(candidate.metadata.get("token_support_ratio",
                          1.0 if candidate.granularity == "token" else 0.0))

    if candidate.granularity == "token":
        form = char_trigram_jaccard(candidate.source_text, candidate.target_text)
        dict_hit = _token_dict_hit(candidate, wiktionary_lookup)
        if dict_hit > 0.0:
            # Dictionary confirms the translation — trust it heavily
            return min(1.0, 0.55 * dict_hit + 0.25 * form + 0.15 * upos + 0.05 * deprel)
        else:
            # No dict entry: rely on form (cognates) + embedding + structure
            return min(1.0, 0.40 * embedding + 0.35 * form + 0.20 * upos + 0.05 * deprel)

    # Non-token: structural tree match + semantic confidence
    structural = (0.40 * token_support
                  + 0.25 * lemma
                  + 0.20 * upos
                  + 0.15 * deprel)
    return min(1.0, 0.50 * structural + 0.50 * embedding)


# Minimum quality required to enter phase-2 pool (per granularity)
ALIGNMENT_QUALITY_THRESHOLD: dict[str, float] = {
    "token": 0.38,
    "phrase": 0.32,
    "subtree": 0.32,
    "span": 0.32,
    "sentence": 0.42,
    "paragraph": 0.50,
}


def alignment_quality_threshold(granularity: str) -> float:
    return ALIGNMENT_QUALITY_THRESHOLD.get(granularity, 0.38)


# ---------------------------------------------------------------------------
# linguistic_similarity_score
# ---------------------------------------------------------------------------

def _source_lemmas_of_candidate(candidate: ParagraphHybridCandidate) -> set[str]:
    """Polish lemmas on the source side (left of '::' in family_ids)."""
    lemmas: set[str] = set()
    for fid in _cost_family_ids(candidate):
        left = fid.split("::")[0] if "::" in fid else fid
        for part in left.split("+"):
            part = part.strip().lower()
            if part:
                lemmas.add(part)
    return lemmas


def linguistic_similarity_score(
    candidate: ParagraphHybridCandidate,
    wiktionary_lookup: dict[str, list[str]] | None = None,
) -> float:
    """Returns [0, 1] — is the Czech side the right translation of the Polish side?

    Primary signal: Wiktionary CS→PL lookup.
    Fallback (word not in dict, or no dict): char-trigram Jaccard.
    """
    # Form similarity between source and target text (cognate proxy)
    form_score = char_trigram_jaccard(candidate.source_text, candidate.target_text)

    if not wiktionary_lookup:
        return form_score

    pl_lemmas = _source_lemmas_of_candidate(candidate)
    cs_lemmas = [lemma.lower() for lemma in _candidate_cost_target_lemmas(candidate) if lemma]
    if not cs_lemmas or not pl_lemmas:
        return form_score

    dict_hits: list[float] = []
    for cs in cs_lemmas:
        pl_translations = wiktionary_lookup.get(cs)
        if pl_translations:
            # fraction of source PL lemmas confirmed by the dictionary
            hit = len(pl_lemmas & set(pl_translations)) / len(pl_lemmas)
            dict_hits.append(hit)
        # no entry → skip (don't penalise as 0)

    if not dict_hits:
        # No dictionary coverage for any CS lemma → fall back to form only
        return form_score

    dict_score = sum(dict_hits) / len(dict_hits)
    coverage = len(dict_hits) / len(cs_lemmas)

    # Blend: where dict has coverage, weight it 65%; form fills the rest.
    # When coverage < 1, form score compensates for missing entries.
    return coverage * (0.65 * dict_score + 0.35 * form_score) + (1.0 - coverage) * form_score


# ---------------------------------------------------------------------------
# Dictionary I/O
# ---------------------------------------------------------------------------

def build_wiktionary_lookup(path: Path | str) -> dict[str, list[str]]:
    """Load the pre-built CS→PL JSON lookup from disk."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as fh:
        return json.load(fh)

