"""Scoring, penalty and bonus functions for hybridization candidates."""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher

from szwejk.generate.candidate_model import (
    ParagraphHybridCandidate,
    IDIOMATICITY_PENALTY_WEIGHT,
    TOKEN_CONFIDENCE_PENALTY,
    STRUCTURAL_CONFIDENCE_PENALTY,
    SURFACE_FUNCTION_WORDS,
    VISIBLE_FUTURE_BLEND,
    _is_span_granularity,
)
from szwejk.generate.plan_metrics import _word_count, _target_token_keys, _covered_target_tokens
from szwejk.generate.family_tracking import _cost_family_ids, _candidate_cost_target_lemmas


# ---------------------------------------------------------------------------
# Helpers re-exported here for convenience
# ---------------------------------------------------------------------------

def _text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(a=left.strip().lower(), b=right.strip().lower()).ratio()


def _candidate_visible_mass(candidate: ParagraphHybridCandidate) -> float:
    source_words = max(1, int(candidate.metadata.get("source_word_count", _word_count(candidate.source_text))))
    similarity = _text_similarity(candidate.source_text, candidate.target_text)
    score = max(0.0, min(1.0, float(candidate.score)))
    visibility = max(0.0, 1.0 - similarity)
    confidence = 0.55 + (0.45 * score)
    return source_words * visibility * confidence


def _selected_visible_mass(candidates: list[ParagraphHybridCandidate]) -> float:
    return sum(_candidate_visible_mass(c) for c in candidates)


def _candidate_idiomaticity(candidate: ParagraphHybridCandidate) -> float:
    if not (_is_span_granularity(candidate.granularity) or candidate.granularity in {"sentence", "paragraph"}):
        return 0.0
    support = float(candidate.metadata.get("token_support_ratio", 1.0))
    semantic = max(0.0, min(1.0, float(candidate.score)))
    return max(0.0, semantic - support)


def _semantic_confidence(candidate: ParagraphHybridCandidate) -> float:
    signals = dict(candidate.metadata.get("signals", {}))
    embedding = float(signals.get("embedding", candidate.score))
    return max(0.0, min(1.0, max(float(candidate.score), embedding)))


def _structural_support(candidate: ParagraphHybridCandidate) -> float:
    if candidate.granularity == "token":
        return 1.0
    support = float(candidate.metadata.get("token_support_ratio", 0.0))
    signals = dict(candidate.metadata.get("signals", {}))
    lemma = float(signals.get("lemma", signals.get("lemma_overlap", 0.0)))
    upos = float(signals.get("upos", signals.get("upos_overlap", 0.0)))
    deprel = float(signals.get("deprel", signals.get("dependency_overlap", 0.0)))
    return max(0.0, min(1.0, (0.45 * support) + (0.25 * lemma) + (0.15 * upos) + (0.15 * deprel)))


def _candidate_allowed_by_safety(candidate: ParagraphHybridCandidate) -> bool:
    if candidate.granularity == "token":
        return True
    score = float(candidate.score)
    support = _structural_support(candidate)
    semantic = _semantic_confidence(candidate)
    similarity = _text_similarity(candidate.source_text, candidate.target_text)
    width = max(_word_count(candidate.source_text), _word_count(candidate.target_text), 1)
    if _is_span_granularity(candidate.granularity):
        if semantic >= 0.72 and support < 0.28:
            return True
        if width <= 3:
            return score >= 0.58
        return score >= 0.6 and (support >= 0.2 or similarity >= 0.22)
    if candidate.granularity == "sentence":
        if semantic >= 0.74 and support < 0.24:
            return True
        if width > 8 and support < 0.28 and semantic < 0.78:
            return False
        return score >= 0.62 and (support >= 0.24 or semantic >= 0.72)
    if candidate.granularity == "paragraph":
        if width <= 3:
            return False
        if semantic >= 0.78 and support < 0.22:
            return True
        return score >= 0.68 and (support >= 0.3 or semantic >= 0.78)
    return True


# ---------------------------------------------------------------------------
# Extracted scoring functions (verbatim from paragraph_hybridization.py)
# ---------------------------------------------------------------------------



def _late_visible_bonus(candidate: ParagraphHybridCandidate, *, progress: float) -> float:
    progress = max(0.0, min(1.0, progress))
    visible_mass = _candidate_visible_mass(candidate)
    normalized_mass = min(1.0, visible_mass / 4.0)
    if candidate.granularity == "token":
        scale = 0.035
    elif _is_span_granularity(candidate.granularity):
        scale = 0.085
    elif candidate.granularity == "sentence":
        scale = 0.12
    elif candidate.granularity == "paragraph":
        scale = 0.16
    else:
        scale = 0.04
    return normalized_mass * (progress**2.4) * scale



def _effective_new_family_count(candidate: ParagraphHybridCandidate, *, new_families: list[str]) -> float:
    unique_count = float(len(set(new_families)))
    if unique_count <= 0.0:
        return 0.0
    idiomaticity = _candidate_idiomaticity(candidate)
    if idiomaticity >= 0.45 and (_is_span_granularity(candidate.granularity) or candidate.granularity == "sentence"):
        return 1.0
    return unique_count



def _new_family_penalty(
    candidate: ParagraphHybridCandidate,
    *,
    new_families: list[str],
    remaining_after: Counter[str],
    progress: float,
) -> float:
    unique_families = sorted(set(new_families))
    if not unique_families:
        return 0.0
    progress = max(0.0, min(1.0, progress))
    phase_weight = (1.0 - progress) ** 1.1
    effective_count = _effective_new_family_count(candidate, new_families=unique_families)
    raw_count = float(len(unique_families))
    family_penalty = 0.0
    for family_id in unique_families:
        remaining = max(0, int(remaining_after.get(family_id, 0)))
        deferrable = min(1.0, remaining / 10.0)
        family_penalty += 0.008 + (0.028 * deferrable * phase_weight)
    family_penalty *= effective_count / max(raw_count, 1.0)
    if (_is_span_granularity(candidate.granularity) or candidate.granularity in {"sentence", "paragraph"}) and effective_count <= 1.0:
        family_penalty *= 0.5
    return family_penalty



def _known_context_mass(candidate: ParagraphHybridCandidate, *, paragraph_target_lemmas: set[str]) -> float:
    known_count = sum(1 for lemma in _candidate_cost_target_lemmas(candidate) if lemma in paragraph_target_lemmas)
    functional_context = len([item for item in candidate.metadata.get("functional_family_ids", []) if str(item).strip()])
    target_functional_context = len(
        [item for item in candidate.metadata.get("candidate_target_functional_lemmas", []) if str(item).strip()]
    )
    width = max(
        1.0,
        float(candidate.metadata.get("source_word_count", _word_count(candidate.source_text))),
        float(candidate.metadata.get("target_word_count", _word_count(candidate.target_text))),
    )
    context_mass = known_count + max(functional_context, target_functional_context)
    if context_mass <= 0:
        return 0.0
    return context_mass + max(0.0, width - float(len(set(_candidate_cost_target_lemmas(candidate)))))



def _known_context_credit(
    candidate: ParagraphHybridCandidate,
    *,
    known_context_mass: float,
    effective_new_family_count: float,
    progress: float,
) -> float:
    if candidate.granularity == "token" or known_context_mass <= 0.0:
        return 0.0
    progress = max(0.0, min(1.0, progress))
    cheapness = 1.0 / max(1.0, effective_new_family_count)
    phase_weight = 0.65 + (0.35 * progress)
    width = max(
        1.0,
        float(candidate.metadata.get("source_word_count", _word_count(candidate.source_text))),
        float(candidate.metadata.get("target_word_count", _word_count(candidate.target_text))),
    )
    width_weight = min(1.0, width / 6.0)
    credit = known_context_mass * cheapness * phase_weight * (0.004 + (0.01 * width_weight))
    functional_context = len(
        [item for item in candidate.metadata.get("candidate_target_functional_lemmas", []) if str(item).strip()]
    )
    if functional_context > 0:
        credit += functional_context * cheapness * phase_weight * 0.016
        if effective_new_family_count <= 1.0:
            credit += functional_context * phase_weight * 0.01
    if (
        _is_span_granularity(candidate.granularity)
        and effective_new_family_count <= 1.0
        and len(_candidate_cost_target_lemmas(candidate)) <= 1
        and width <= 4.0
        and _semantic_confidence(candidate) >= 0.68
    ):
        # Small phrase expansions that carry the same lexical novelty as a token
        # should be able to beat the bare token when the added context is cheap.
        credit += 0.018 * min(known_context_mass, width)
    return min(0.14, credit)



def _early_carrier_penalty(
    candidate: ParagraphHybridCandidate,
    *,
    progress: float,
    effective_new_family_count: float,
    known_context_mass: float,
) -> float:
    return 0.0



def _local_scope_penalty(
    candidate: ParagraphHybridCandidate,
    *,
    candidates: list[ParagraphHybridCandidate],
    paragraph_target_lemmas: set[str],
) -> float:
    width = max(
        1.0,
        float(candidate.metadata.get("source_word_count", _word_count(candidate.source_text))),
        float(candidate.metadata.get("target_word_count", _word_count(candidate.target_text))),
    )
    candidate_new_target = {
        lemma for lemma in _candidate_cost_target_lemmas(candidate) if lemma not in paragraph_target_lemmas
    }

    if candidate.granularity in {"phrase", "subtree", "sentence", "paragraph"} and width > 3.0:
        for other in candidates:
            if other.candidate_id == candidate.candidate_id:
                continue
            if other.scope_id != candidate.scope_id:
                continue
            other_width = max(
                1.0,
                float(other.metadata.get("source_word_count", _word_count(other.source_text))),
                float(other.metadata.get("target_word_count", _word_count(other.target_text))),
            )
            if other_width >= width:
                continue
            other_new_target = {
                lemma for lemma in _candidate_cost_target_lemmas(other) if lemma not in paragraph_target_lemmas
            }
            if candidate_new_target and other_new_target != candidate_new_target:
                continue
            if other.source_span[0] < candidate.source_span[0] or other.source_span[1] > candidate.source_span[1]:
                continue
            return 0.045 + min(0.04, (width - other_width) * 0.01)

    if candidate.granularity in {"phrase", "subtree", "sentence", "paragraph"} and width > 2.0:
        for other in candidates:
            if other.candidate_id == candidate.candidate_id:
                continue
            if other.scope_id != candidate.scope_id:
                continue
            other_width = max(
                1.0,
                float(other.metadata.get("source_word_count", _word_count(other.source_text))),
                float(other.metadata.get("target_word_count", _word_count(other.target_text))),
            )
            if other_width >= width:
                continue
            other_new_target = {
                lemma for lemma in _candidate_cost_target_lemmas(other) if lemma not in paragraph_target_lemmas
            }
            if not other_new_target:
                continue
            if not other_new_target.issubset(candidate_new_target):
                continue
            if float(other.score) + 0.02 < float(candidate.score):
                continue
            if other.source_span[0] < candidate.source_span[0] or other.source_span[1] > candidate.source_span[1]:
                continue
            extra = len(candidate_new_target - other_new_target)
            return 0.04 + min(0.08, 0.018 * max(1, extra))

    if candidate.granularity == "paragraph":
        for other in candidates:
            if other.candidate_id == candidate.candidate_id:
                continue
            if other.scope_id != candidate.scope_id:
                continue
            other_width = max(
                1.0,
                float(other.metadata.get("source_word_count", _word_count(other.source_text))),
                float(other.metadata.get("target_word_count", _word_count(other.target_text))),
            )
            if other_width >= width:
                continue
            other_new_target = {
                lemma for lemma in _candidate_cost_target_lemmas(other) if lemma not in paragraph_target_lemmas
            }
            if not other_new_target:
                continue
            if not other_new_target.issubset(candidate_new_target):
                continue
            return 0.3 + min(0.12, 0.02 * max(1, len(candidate_new_target - other_new_target)))

    if candidate.granularity == "sentence":
        for other in candidates:
            if other.candidate_id == candidate.candidate_id or other.granularity != "paragraph":
                continue
            if other.source_text == candidate.source_text and other.target_text == candidate.target_text:
                if tuple(other.family_ids) == tuple(candidate.family_ids):
                    return 0.02
    return 0.0



def _uncovered_content_penalty(
    candidate: ParagraphHybridCandidate,
    *,
    paragraph_target_lemmas: set[str],
    progress: float,
) -> float:
    if not (_is_span_granularity(candidate.granularity) or candidate.granularity in {"sentence", "paragraph"}):
        return 0.0
    content_lemmas = {lemma for lemma in _candidate_cost_target_lemmas(candidate) if lemma}
    if not content_lemmas:
        return 0.0
    if len(content_lemmas) <= 1:
        return 0.0
    covered_lemmas = _covered_target_lemmas(candidate) | set(paragraph_target_lemmas)
    uncovered = {lemma for lemma in content_lemmas if lemma not in covered_lemmas}
    if not uncovered:
        return 0.0
    width = max(
        1.0,
        float(candidate.metadata.get("source_word_count", _word_count(candidate.source_text))),
        float(candidate.metadata.get("target_word_count", _word_count(candidate.target_text))),
    )
    ratio = len(uncovered) / max(len(content_lemmas), 1)
    progress = max(0.0, min(1.0, progress))
    phase_weight = (1.0 - progress) ** 1.15
    semantic = _semantic_confidence(candidate)
    structural = _structural_support(candidate)
    idiom_discount = 0.55 if semantic >= 0.74 and structural < 0.28 else 1.0
    width_weight = min(1.0, max(0.0, (width - 2.0) / 4.0))
    return ratio * width_weight * phase_weight * 0.11 * idiom_discount



def _lemma_urgency_bonus(candidate: ParagraphHybridCandidate, *, progress: float) -> float:
    urgency = float(candidate.metadata.get("target_lemma_urgency", 0.0))
    if urgency <= 0.0:
        return 0.0
    progress = max(0.0, min(1.0, progress))
    phase_weight = (1.0 - progress) ** 1.2
    granularity_weight = 1.0 if candidate.granularity == "token" else 0.8
    return urgency * 0.028 * phase_weight * granularity_weight



def _lemma_deferrability_penalty(candidate: ParagraphHybridCandidate, *, progress: float) -> float:
    deferrability = float(candidate.metadata.get("target_lemma_deferrability", 0.0))
    future_gain = float(candidate.metadata.get("target_lemma_future_gain_if_introduced_here", 0.0))
    global_count = float(candidate.metadata.get("target_lemma_global_count", 0.0))
    if deferrability <= 0.0 and future_gain <= 0.0:
        return 0.0
    progress = max(0.0, min(1.0, progress))
    early_phase_weight = (1.0 - progress) ** 1.45
    popularity = min(1.0, global_count / 12.0)
    opportunity = min(1.0, future_gain * 24.0)
    granularity_weight = 1.0 if candidate.granularity == "token" else 0.82
    return (0.016 * deferrability + 0.02 * popularity * opportunity) * early_phase_weight * granularity_weight



def _target_lemma_penalty(
    candidate: ParagraphHybridCandidate,
    *,
    new_target_lemmas: list[str],
    remaining_target_after: Counter[str],
    progress: float,
) -> float:
    unique_lemmas = sorted(set(new_target_lemmas))
    if not unique_lemmas:
        return 0.0
    progress = max(0.0, min(1.0, progress))
    early_phase_weight = (1.0 - progress) ** 1.25
    penalty = 0.0
    for lemma in unique_lemmas:
        remaining = max(0, int(remaining_target_after.get(lemma, 0)))
        deferrable = min(1.0, remaining / 10.0)
        penalty += 0.01 + (0.032 * deferrable * early_phase_weight)
    if _is_span_granularity(candidate.granularity) or candidate.granularity in {"sentence", "paragraph"}:
        uncovered = [lemma for lemma in unique_lemmas if lemma not in _covered_target_lemmas(candidate)]
        if uncovered:
            penalty += 0.012 * len(uncovered)
    return penalty



def _family_schedule_penalty(
    candidate: ParagraphHybridCandidate,
    *,
    progress: float,
    paragraph_families: set[str],
) -> float:
    scheduled_intro = float(candidate.metadata.get("family_schedule_intro_unit_avg", 0.0))
    if scheduled_intro <= 0.0:
        return 0.0
    covered = int(candidate.metadata.get("family_schedule_covered_count", 0) or 0)
    new_cost_families = [family_id for family_id in _cost_family_ids(candidate) if family_id not in paragraph_families]
    if not new_cost_families or covered <= 0:
        return 0.0
    current_unit = float(candidate.unit_index)
    if current_unit >= scheduled_intro:
        return 0.0
    lead_units = scheduled_intro - current_unit
    lead_ratio = lead_units / max(1.0, scheduled_intro)
    progress = max(0.0, min(1.0, progress))
    phase_weight = (1.0 - progress) ** 1.25
    granularity_weight = 1.0 if candidate.granularity == "token" else 0.88
    return min(0.2, 0.22 * lead_ratio * phase_weight * granularity_weight)



def _family_schedule_bonus(
    candidate: ParagraphHybridCandidate,
    *,
    progress: float,
    paragraph_families: set[str],
) -> float:
    scheduled_intro = float(candidate.metadata.get("family_schedule_intro_unit_avg", 0.0))
    if scheduled_intro <= 0.0:
        return 0.0
    new_cost_families = [family_id for family_id in _cost_family_ids(candidate) if family_id not in paragraph_families]
    if not new_cost_families:
        return 0.0
    current_unit = float(candidate.unit_index)
    if current_unit < scheduled_intro:
        return 0.0
    lag_units = current_unit - scheduled_intro
    lag_ratio = min(1.0, lag_units / max(1.0, scheduled_intro))
    progress = max(0.0, min(1.0, progress))
    width = max(
        1.0,
        float(candidate.metadata.get("source_word_count", _word_count(candidate.source_text))),
        float(candidate.metadata.get("target_word_count", _word_count(candidate.target_text))),
    )
    width_weight = min(1.0, width / 6.0)
    granularity_weight = 0.7 if candidate.granularity == "token" else 1.0
    return min(0.05, (0.012 + (0.02 * lag_ratio)) * progress * width_weight * granularity_weight)



def _confidence_penalty(candidate: ParagraphHybridCandidate) -> float:
    score = max(0.0, min(1.0, float(candidate.score)))
    if candidate.granularity == "token":
        penalty = (1.0 - score) * TOKEN_CONFIDENCE_PENALTY
        relation = str(candidate.relation)
        norm = float(candidate.metadata.get("signals", {}).get("norm", 0.0))
        source_upos = str(candidate.metadata.get("source_upos", "")).upper()
        target_upos = str(candidate.metadata.get("target_upos", "")).upper()
        similarity = _text_similarity(candidate.source_text, candidate.target_text)
        family_trust = float(candidate.metadata.get("family_trust", 0.0))
        if relation == "upos_cognate":
            penalty += 0.03 + (max(0.0, 0.45 - norm) * 0.10)
        elif relation == "cognate":
            penalty += 0.01 + (max(0.0, 0.35 - norm) * 0.05)
        elif relation == "lemma_like":
            penalty += 0.002
        if relation in {"exact", "lemma_like"} and ({source_upos, target_upos} & {"PROPN", "NOUN", "ADJ", "NUM"}):
            penalty *= 0.45
        if relation in {"upos_cognate", "cognate"} and similarity < 0.45:
            penalty += 0.08 + ((0.45 - similarity) * 0.12)
        if family_trust < 0.52:
            penalty += (0.52 - family_trust) * 0.18
        return penalty
    semantic = _semantic_confidence(candidate)
    structural = _structural_support(candidate)
    penalty = (1.0 - max(score, semantic)) * STRUCTURAL_CONFIDENCE_PENALTY
    target_functional_context = len(
        [item for item in candidate.metadata.get("candidate_target_functional_lemmas", []) if str(item).strip()]
    )
    target_content_lemmas = len(_candidate_cost_target_lemmas(candidate))
    if semantic >= 0.72 and structural < 0.3:
        penalty *= 0.45
    elif semantic < 0.6 and structural < 0.25:
        penalty += 0.02
    if _is_span_granularity(candidate.granularity) and target_content_lemmas <= 1 and target_functional_context > 0:
        penalty *= 0.35
    return penalty



def _idiomaticity_penalty(
    candidate: ParagraphHybridCandidate,
    *,
    progress: float,
    idiomaticity_penalty_weight: float,
) -> float:
    if idiomaticity_penalty_weight <= 0:
        return 0.0
    if not _is_span_granularity(candidate.granularity):
        return 0.0
    idiomaticity = _candidate_idiomaticity(candidate)
    if idiomaticity <= 0:
        return 0.0
    phase_weight = (1.0 - max(0.0, min(1.0, progress))) ** 1.5
    return idiomaticity * phase_weight * idiomaticity_penalty_weight



def _granularity_progress_penalty(candidate: ParagraphHybridCandidate, *, progress: float) -> float:
    progress = max(0.0, min(1.0, progress))
    phase = 1.0 - progress
    score = max(0.0, min(1.0, float(candidate.score)))
    similarity = _text_similarity(candidate.source_text, candidate.target_text)
    support = float(candidate.metadata.get("token_support_ratio", 0.0))
    width = max(_word_count(candidate.source_text), _word_count(candidate.target_text), 1)

    if candidate.granularity == "token":
        source_upos = str(candidate.metadata.get("source_upos", "")).upper()
        target_upos = str(candidate.metadata.get("target_upos", "")).upper()
        lexical_bonus = 0.1 if ({source_upos, target_upos} & {"PROPN", "NOUN", "ADJ", "NUM"}) else 0.0
        transparency = max(similarity, score) + lexical_bonus
        readability_penalty = max(0.0, 0.95 - min(1.0, transparency))
        introduction_bonus = max(0.0, min(1.0, transparency) - 0.72) * (phase**1.15) * 0.028
        return (readability_penalty * (phase**1.35) * 0.085) - introduction_bonus

    if _is_span_granularity(candidate.granularity):
        size_factor = min(1.0, max(0.0, (width - 1) / 6.0))
        structural_strength = min(1.0, (0.45 * score) + (0.35 * support) + (0.20 * similarity))
        base = 0.28 if candidate.granularity == "phrase" else 0.31
        return base * (0.45 + size_factor) * (phase**1.7) * (1.08 - structural_strength)

    if candidate.granularity == "sentence":
        return (0.24 + (0.12 * (1.0 - score))) * (phase**2.2)

    if candidate.granularity == "paragraph":
        return (0.42 + (0.14 * (1.0 - score))) * (phase**2.5)

    return 0.0



def _smooth_acceptance_slack(candidate: ParagraphHybridCandidate, *, progress: float) -> float:
    progress = max(0.0, min(1.0, progress))
    phase = 1.0 - progress
    if candidate.granularity == "token":
        similarity = _text_similarity(candidate.source_text, candidate.target_text)
        transparency = max(similarity, float(candidate.score))
        if str(candidate.metadata.get("source_upos", "")).upper() in {"PROPN", "NOUN", "ADJ", "NUM"}:
            transparency = min(1.0, transparency + 0.08)
        return 0.002 + (max(0.0, transparency - 0.68) * (phase**1.1) * 0.025)
    if _is_span_granularity(candidate.granularity):
        width = max(_word_count(candidate.source_text), _word_count(candidate.target_text), 1)
        compactness = max(0.0, 1.0 - ((width - 1) / 6.0))
        return 0.0005 + (compactness * (phase**1.45) * 0.003) + ((progress**1.8) * 0.012)
    if candidate.granularity == "sentence":
        return 0.001 + ((progress**2.0) * 0.014)
    if candidate.granularity == "paragraph":
        return 0.001 + ((progress**2.2) * 0.02)
    return 0.001

