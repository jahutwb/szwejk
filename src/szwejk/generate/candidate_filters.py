"""Safety gates, span geometry and progress filters."""
from __future__ import annotations

from szwejk.generate.candidate_model import (
    ParagraphHybridCandidate,
    SURFACE_FUNCTION_WORDS,
    _is_span_granularity,
)
from szwejk.generate.plan_metrics import _word_count, _target_token_keys, _source_token_keys
from szwejk.generate.family_tracking import _cost_family_ids, _candidate_cost_target_lemmas
from szwejk.generate.candidate_scoring import (
    _text_similarity,
    _candidate_allowed_by_safety,
    _structural_support,
    _semantic_confidence,
)


# ---------------------------------------------------------------------------
# Extracted filter functions (verbatim from paragraph_hybridization.py)
# ---------------------------------------------------------------------------



def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return not (left[1] < right[0] or right[1] < left[0])



def _span_contains(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] >= inner[1]



def _candidate_contains(outer: ParagraphHybridCandidate, inner: ParagraphHybridCandidate) -> bool:
    outer_source_keys = _source_token_keys(outer)
    inner_source_keys = _source_token_keys(inner)
    outer_target_keys = _target_token_keys(outer)
    inner_target_keys = _target_token_keys(inner)
    if outer_source_keys and inner_source_keys and outer_target_keys and inner_target_keys:
        return inner_source_keys.issubset(outer_source_keys) and inner_target_keys.issubset(outer_target_keys)
    return _span_contains(outer.source_span, inner.source_span) and _span_contains(outer.target_span, inner.target_span)



def _conflicts(left: ParagraphHybridCandidate, right: ParagraphHybridCandidate) -> bool:
    left_source_keys = _source_token_keys(left)
    right_source_keys = _source_token_keys(right)
    left_target_keys = _target_token_keys(left)
    right_target_keys = _target_token_keys(right)
    if left_source_keys and right_source_keys and (left_source_keys & right_source_keys):
        return True
    if left_target_keys and right_target_keys and (left_target_keys & right_target_keys):
        return True
    if left.scope_id == "block" or right.scope_id == "block":
        return True
    if left.scope_id != right.scope_id:
        return False
    if left.granularity == "sentence" or right.granularity == "sentence":
        return True
    return _spans_overlap(left.source_span, right.source_span) or _spans_overlap(left.target_span, right.target_span)



def _candidate_allowed_as_surface_carrier(candidate: ParagraphHybridCandidate) -> bool:
    content_lemmas = {
        str(item).strip()
        for item in candidate.metadata.get("candidate_target_content_lemmas", [])
        if str(item).strip()
    }
    functional_lemmas = {
        str(item).strip()
        for item in candidate.metadata.get("candidate_target_functional_lemmas", [])
        if str(item).strip()
    }
    if content_lemmas:
        return True
    if not functional_lemmas:
        return True
    width = max(
        1.0,
        float(candidate.metadata.get("source_word_count", _word_count(candidate.source_text))),
        float(candidate.metadata.get("target_word_count", _word_count(candidate.target_text))),
    )
    if (_is_span_granularity(candidate.granularity) or candidate.granularity in {"sentence", "paragraph"}) and width <= 1.0:
        return False
    return width > 1.0



def _candidate_has_better_narrower_carrier(
    candidate: ParagraphHybridCandidate,
    *,
    candidates: list[ParagraphHybridCandidate],
    paragraph_target_lemmas: set[str],
) -> bool:
    if candidate.granularity != "paragraph":
        return False
    candidate_new_target = {
        lemma for lemma in _candidate_cost_target_lemmas(candidate) if lemma not in paragraph_target_lemmas
    }
    if not candidate_new_target:
        return False
    candidate_width = max(
        1.0,
        float(candidate.metadata.get("source_word_count", _word_count(candidate.source_text))),
        float(candidate.metadata.get("target_word_count", _word_count(candidate.target_text))),
    )
    for other in candidates:
        if other.candidate_id == candidate.candidate_id:
            continue
        if other.unit_index != candidate.unit_index:
            continue
        other_width = max(
            1.0,
            float(other.metadata.get("source_word_count", _word_count(other.source_text))),
            float(other.metadata.get("target_word_count", _word_count(other.target_text))),
        )
        if other_width >= candidate_width:
            continue
        other_new_target = {
            lemma for lemma in _candidate_cost_target_lemmas(other) if lemma not in paragraph_target_lemmas
        }
        if not other_new_target:
            continue
        overlap = len(other_new_target & candidate_new_target)
        overlap_ratio = overlap / max(len(other_new_target), 1)
        if other_new_target.issubset(candidate_new_target) or overlap_ratio >= 0.5:
            return True
    return False



def _token_can_stand_alone(
    pair: dict[str, object],
    *,
    blocked_standalone_upos: frozenset[str],
) -> bool:
    relation = str(pair.get("relation", ""))
    score = float(pair.get("score", 0.0))
    signals = dict(pair.get("signals", {}))
    norm = float(signals.get("norm", 0.0))
    source_upos = str(pair.get("source_upos", "")).upper()
    target_upos = str(pair.get("target_upos", "")).upper()
    source_text = str(pair.get("source_text", "")).strip().lower()
    target_text = str(pair.get("target_text", "")).strip().lower()
    lexical_or_verbish = {"PROPN", "NOUN", "ADJ", "NUM", "VERB", "ADV"}
    if source_upos in blocked_standalone_upos or target_upos in blocked_standalone_upos:
        return False
    if (
        (source_text in SURFACE_FUNCTION_WORDS or target_text in SURFACE_FUNCTION_WORDS)
        and source_upos not in lexical_or_verbish
        and target_upos not in lexical_or_verbish
    ):
        return False
    if ({source_upos, target_upos} & {"ADJ", "NUM"}) and (
        str(pair.get("prev_source_upos", "")).upper() in {"ADJ", "NUM"}
        or str(pair.get("next_source_upos", "")).upper() in {"ADJ", "NUM"}
    ):
        return False
    if relation in {"exact", "lemma_like"}:
        return True
    if relation == "cognate":
        return score >= 0.5 and norm >= 0.15
    if relation == "upos_cognate":
        return score >= 0.5 and norm >= 0.15
    return False



def _candidate_allowed_by_progress(candidate: ParagraphHybridCandidate, *, progress: float) -> bool:
    progress = max(0.0, min(1.0, progress))
    if candidate.granularity == "token":
        return _token_allowed_by_progress(candidate, progress=progress)
    if _is_span_granularity(candidate.granularity):
        return _structural_candidate_allowed(candidate, progress=progress)
    if candidate.granularity == "sentence":
        return candidate.score >= 0.62
    if candidate.granularity == "paragraph":
        return candidate.score >= 0.5
    return True



def _token_allowed_by_progress(candidate: ParagraphHybridCandidate, *, progress: float) -> bool:
    similarity = _text_similarity(candidate.source_text, candidate.target_text)
    score = float(candidate.score)
    relation = str(candidate.relation)
    source_upos = str(candidate.metadata.get("source_upos", "")).upper()
    target_upos = str(candidate.metadata.get("target_upos", "")).upper()
    source_text = candidate.source_text.strip().lower()
    target_text = candidate.target_text.strip().lower()
    lexical_upos = {"PROPN", "NOUN", "ADJ", "NUM"}
    verbish_upos = {"VERB", "ADV"}

    if (
        (source_text in SURFACE_FUNCTION_WORDS or target_text in SURFACE_FUNCTION_WORDS)
        and source_upos not in lexical_upos | verbish_upos
        and target_upos not in lexical_upos | verbish_upos
    ):
        return False

    if progress < 0.18:
        if relation == "exact":
            return True
        if source_upos in lexical_upos or target_upos in lexical_upos:
            return similarity >= 0.42 or score >= 0.68
        if source_upos in verbish_upos or target_upos in verbish_upos:
            return similarity >= 0.8
        return similarity >= 0.8 and score >= 0.85

    if progress < 0.45:
        if relation == "exact":
            return True
        if source_upos in lexical_upos or target_upos in lexical_upos:
            return similarity >= 0.34 or score >= 0.7
        if source_upos in verbish_upos or target_upos in verbish_upos:
            return similarity >= 0.68 or score >= 0.82
        return similarity >= 0.72 and score >= 0.82

    return True



def _structural_candidate_allowed(candidate: ParagraphHybridCandidate, *, progress: float) -> bool:
    score = float(candidate.score)
    source_words = _word_count(candidate.source_text)
    target_words = _word_count(candidate.target_text)
    width = max(source_words, target_words)
    support = float(candidate.metadata.get("token_support_ratio", 0.0))
    similarity = _text_similarity(candidate.source_text, candidate.target_text)

    if progress < 0.25:
        return width <= 3 and score >= 0.72 and (support >= 0.66 or similarity >= 0.5)
    if progress < 0.55:
        return width <= 6 and score >= 0.66 and (support >= 0.45 or similarity >= 0.42)
    if progress < 0.78:
        return width <= 12 and score >= 0.62
    return True



def _candidate_blocked_by_family_schedule(
    candidate: ParagraphHybridCandidate,
    *,
    paragraph_families: set[str],
) -> bool:
    new_cost_families = [family_id for family_id in _cost_family_ids(candidate) if family_id not in paragraph_families]
    if not new_cost_families:
        return False
    scheduled_intro = float(candidate.metadata.get("family_schedule_intro_unit_min", 0.0))
    covered = int(candidate.metadata.get("family_schedule_covered_count", 0) or 0)
    if covered <= 0 or scheduled_intro <= 0.0:
        return False
    return float(candidate.unit_index) < scheduled_intro



def _promotion_score_floor(candidate: ParagraphHybridCandidate) -> float:
    width = max(
        1.0,
        float(candidate.metadata.get("source_word_count", _word_count(candidate.source_text))),
        float(candidate.metadata.get("target_word_count", _word_count(candidate.target_text))),
    )
    if candidate.granularity == "sentence":
        if width <= 5:
            return 0.62
        if width <= 8:
            return 0.60
        if width <= 12:
            return 0.58
        return 0.56
    if candidate.granularity == "paragraph":
        if width <= 12:
            return 0.60
        if width <= 20:
            return 0.58
        return 0.56
    return 0.0



def _candidate_allowed_by_promotion_safety(candidate: ParagraphHybridCandidate) -> bool:
    if candidate.granularity == "sentence":
        return float(candidate.score) >= _promotion_score_floor(candidate)
    if candidate.granularity == "paragraph":
        return True
    return _candidate_allowed_by_safety(candidate)



def _candidate_allowed_by_promotion_shape(
    candidate: ParagraphHybridCandidate,
    *,
    conflicts: list[ParagraphHybridCandidate],
    base_lemmas: set[str],
) -> bool:
    _ = conflicts
    _ = base_lemmas
    if candidate.granularity != "paragraph":
        return True
    return int(candidate.metadata.get("sentence_count", 1) or 1) >= 2

