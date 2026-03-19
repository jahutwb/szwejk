"""Alignment quality and linguistic similarity scores for hybridization candidates.

Two distinct measures
---------------------
alignment_quality_score  — how reliable is the alignment?
    Composite of structural signals (UPOS, deprel, dependency-tree match,
    token-support ratio) and semantic confidence (embedding, candidate score).
    Used in phase-2 to rank and gate candidates.

linguistic_similarity_score  — is the Czech word the right translation of the Polish word?
    Composite of dictionary validation (CS→PL Wiktionary lookup) and
    surface form similarity (char-trigram Jaccard).
    Used in phase-3 as a tiebreaker.
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

def alignment_quality_score(candidate: ParagraphHybridCandidate) -> float:
    """Returns [0, 1] — how reliable is this alignment?

    Token candidates: dominated by the enrichment score (already a composite
    of char, embedding and UPOS signals); UPOS and deprel add sanity check.

    Phrase / subtree / sentence / paragraph candidates: balance between the
    structural tree-matching quality and semantic (embedding) confidence.
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
        return min(1.0, 0.65 * embedding + 0.20 * upos + 0.15 * deprel)

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

