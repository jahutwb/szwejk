"""Czechness and plan-metrics helpers — pure functions, no I/O."""
from __future__ import annotations

from collections import Counter

from szwejk.generate.candidate_model import ParagraphHybridCandidate, VISIBLE_FUTURE_BLEND


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _mean(values) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


def _word_count(text: str) -> int:
    return len([part for part in text.split() if part.strip()])


# ---------------------------------------------------------------------------
# Token-key helpers (coverage bookkeeping)
# ---------------------------------------------------------------------------

def _target_token_keys(candidate: ParagraphHybridCandidate) -> set[tuple[str, int]]:
    explicit_keys = candidate.metadata.get("coverage_target_token_keys")
    if explicit_keys:
        normalized: set[tuple[str, int]] = set()
        for item in explicit_keys:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                normalized.add((str(item[0]), int(item[1])))
        if normalized:
            return normalized
    start, end = int(candidate.target_span[0]), int(candidate.target_span[1])
    if end < start:
        return set()
    return {(candidate.scope_id, index) for index in range(start, end + 1)}


def _source_token_keys(candidate: ParagraphHybridCandidate) -> set[tuple[str, int]]:
    explicit_keys = candidate.metadata.get("coverage_source_token_keys")
    if explicit_keys:
        normalized: set[tuple[str, int]] = set()
        for item in explicit_keys:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                normalized.add((str(item[0]), int(item[1])))
        if normalized:
            return normalized
    start, end = int(candidate.source_span[0]), int(candidate.source_span[1])
    if end < start:
        return set()
    return {(candidate.scope_id, index) for index in range(start, end + 1)}


def _covered_target_tokens(candidates: list[ParagraphHybridCandidate]) -> set[tuple[str, int]]:
    covered: set[tuple[str, int]] = set()
    for candidate in candidates:
        covered.update(_target_token_keys(candidate))
    return covered


# ---------------------------------------------------------------------------
# Mass / word-count helpers
# ---------------------------------------------------------------------------

def _candidate_simple_mass(candidate: ParagraphHybridCandidate) -> float:
    return max(1.0, float(candidate.metadata.get("source_word_count", _word_count(candidate.source_text))))


def _target_span_word_count(candidate: ParagraphHybridCandidate) -> int:
    return max(1, int(candidate.metadata.get("target_word_count", _word_count(candidate.target_text))))


def _candidate_marginal_target_simple_mass(
    candidate: ParagraphHybridCandidate,
    *,
    covered_target_tokens: set[tuple[str, int]],
) -> float:
    token_keys = _target_token_keys(candidate)
    if not token_keys:
        return 0.0
    uncovered = token_keys - covered_target_tokens
    return float(len(uncovered))



# ---------------------------------------------------------------------------
# Aggregate mass over a candidate list
# (visible mass depends on text similarity — lives in candidate_scoring.py)
# ---------------------------------------------------------------------------

def _selected_simple_mass(candidates: list[ParagraphHybridCandidate]) -> float:
    return sum(_candidate_simple_mass(candidate) for candidate in candidates)


def _selected_marginal_target_simple_mass(candidates: list[ParagraphHybridCandidate]) -> float:
    covered: set[tuple[str, int]] = set()
    total = 0.0
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.granularity != "paragraph",
            item.granularity != "sentence",
            -float(_target_span_word_count(item)),
            -item.score,
            item.candidate_id,
        ),
    ):
        total += _candidate_marginal_target_simple_mass(candidate, covered_target_tokens=covered)
        covered.update(_target_token_keys(candidate))
    return total


# ---------------------------------------------------------------------------
# Czechness ratios
# ---------------------------------------------------------------------------

def _visible_czechness(visible_mass: float, total_word_count: int) -> float:
    if total_word_count <= 0:
        return 0.0
    return min(1.0, max(0.0, visible_mass / total_word_count))


def _simple_czechness(simple_mass: float, total_word_count: int) -> float:
    if total_word_count <= 0:
        return 0.0
    return min(1.0, max(0.0, simple_mass / total_word_count))


def _visible_target(progress: float, power: float) -> float:
    progress = max(0.0, min(1.0, progress))
    return min(0.92, 0.02 + (0.90 * (progress**power)))


def _future_czechness(
    remaining_after: Counter[str],
    introduced_families: set[str],
    *,
    family_weight_lookup: dict[str, float] | None = None,
) -> float:
    family_weight_lookup = family_weight_lookup or {}
    total = sum(count * family_weight_lookup.get(fid, 1.0) for fid, count in remaining_after.items())
    if total <= 0:
        return 1.0
    covered = sum(
        count * family_weight_lookup.get(fid, 1.0)
        for fid, count in remaining_after.items()
        if fid in introduced_families
    )
    return covered / total


def _target_future_czechness(
    remaining_target_after: Counter[str],
    introduced_target_lemmas: set[str],
) -> float:
    total = sum(int(count) for count in remaining_target_after.values())
    if total <= 0:
        return 1.0
    covered = sum(int(count) for lemma, count in remaining_target_after.items() if lemma in introduced_target_lemmas)
    return covered / total

