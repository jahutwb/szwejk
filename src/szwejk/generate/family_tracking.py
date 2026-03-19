"""Family-trust / timeline / schedule annotation for hybridization units."""
from __future__ import annotations

from collections import Counter, defaultdict

from szwejk.generate.candidate_model import ParagraphHybridCandidate, FUNCTIONAL_UPOS


# ---------------------------------------------------------------------------
# Helpers (forward-declared here; real impls imported from unit_builder once
# that module exists — for now they are imported lazily to avoid circular deps)
# ---------------------------------------------------------------------------

def _text_similarity_local(left: str, right: str) -> float:
    """SequenceMatcher ratio used only for family trust scoring."""
    from difflib import SequenceMatcher
    return SequenceMatcher(a=left.strip().lower(), b=right.strip().lower()).ratio()


def _covered_target_lemmas(candidate: ParagraphHybridCandidate) -> set[str]:
    """Target lemmas covered by this candidate (derived from family_id right side of '::')."""
    covered: set[str] = set()
    for family_id in _cost_family_ids(candidate):
        if "::" not in family_id:
            continue
        _source, target = family_id.split("::", 1)
        target = target.strip().lower()
        if target:
            covered.add(target)
    return covered


def _cost_family_ids(candidate: ParagraphHybridCandidate) -> list[str]:
    content = [str(item) for item in candidate.metadata.get("content_family_ids", []) if str(item).strip()]
    if content:
        return content
    return list(candidate.family_ids)


def _candidate_cost_target_lemmas(candidate: ParagraphHybridCandidate) -> list[str]:
    lemmas = {
        str(item).strip()
        for bucket in (
            candidate.metadata.get("candidate_target_content_lemmas", []),
            candidate.metadata.get("candidate_target_functional_lemmas", []),
        )
        for item in bucket
        if str(item).strip()
    }
    return sorted(lemmas)


# ---------------------------------------------------------------------------
# Family trust
# ---------------------------------------------------------------------------

def _annotate_family_trust(units: list[dict[str, object]]) -> None:
    family_stats: dict[str, dict[str, float]] = {}
    for unit in units:
        for candidate in unit.get("candidates", []):
            if candidate.granularity != "token" or not candidate.family_ids:
                continue
            family_id = candidate.family_ids[0]
            bucket = family_stats.setdefault(
                family_id,
                {"count": 0.0, "score_sum": 0.0, "sim_sum": 0.0, "best_score": 0.0},
            )
            similarity = _text_similarity_local(candidate.source_text, candidate.target_text)
            bucket["count"] += 1.0
            bucket["score_sum"] += float(candidate.score)
            bucket["sim_sum"] += similarity
            bucket["best_score"] = max(bucket["best_score"], float(candidate.score))

    for unit in units:
        for candidate in unit.get("candidates", []):
            if candidate.granularity != "token" or not candidate.family_ids:
                continue
            family_id = candidate.family_ids[0]
            stats = family_stats.get(family_id)
            if not stats:
                continue
            count = stats["count"]
            avg_score = stats["score_sum"] / max(count, 1.0)
            avg_similarity = stats["sim_sum"] / max(count, 1.0)
            count_factor = min(1.0, count / 4.0)
            family_trust = (0.45 * count_factor) + (0.30 * avg_score) + (0.25 * avg_similarity)
            candidate.metadata["family_trust"] = round(family_trust, 6)
            candidate.metadata["family_count"] = int(count)


# ---------------------------------------------------------------------------
# Family timeline
# ---------------------------------------------------------------------------

def _annotate_family_timeline(units: list[dict[str, object]]) -> None:
    family_occurrence_totals: Counter[str] = Counter()
    family_occurrence_progress: defaultdict[str, int] = defaultdict(int)
    total_occurrence_count = sum(int(unit["source_occurrence_count"]) for unit in units) or 1
    target_lemma_totals: Counter[str] = Counter()
    target_lemma_progress: defaultdict[str, int] = defaultdict(int)
    total_target_lemma_count = sum(sum(unit["target_lemma_occurrences"].values()) for unit in units) or 1

    for unit in units:
        family_occurrence_totals.update(unit["family_occurrences"])
        target_lemma_totals.update(unit["target_lemma_occurrences"])

    for unit in units:
        unit_index = int(unit["unit_index"])
        family_stats: dict[str, dict[str, float]] = {}
        target_lemma_stats: dict[str, dict[str, float]] = {}
        for family_id, count in unit["family_occurrences"].items():
            seen_before = family_occurrence_progress[family_id]
            total_count = int(family_occurrence_totals[family_id])
            remaining_after_unit = int(total_count - seen_before - count)
            urgency = 1.0 / (1.0 + remaining_after_unit)
            future_gain = remaining_after_unit / total_occurrence_count
            popularity = min(1.0, total_count / 12.0)
            remaining_ratio = remaining_after_unit / max(total_count, 1)
            deferrability = popularity * remaining_ratio
            family_stats[family_id] = {
                "remaining_after_unit": float(remaining_after_unit),
                "global_count": float(total_count),
                "urgency": urgency,
                "deferrability": deferrability,
                "future_gain_if_introduced_here": future_gain,
                "unit_index": float(unit_index),
            }
        for lemma_id, count in unit["target_lemma_occurrences"].items():
            seen_before = target_lemma_progress[lemma_id]
            total_count = int(target_lemma_totals[lemma_id])
            remaining_after_unit = int(total_count - seen_before - count)
            urgency = 1.0 / (1.0 + remaining_after_unit)
            future_gain = remaining_after_unit / total_target_lemma_count
            popularity = min(1.0, total_count / 12.0)
            remaining_ratio = remaining_after_unit / max(total_count, 1)
            deferrability = popularity * remaining_ratio
            target_lemma_stats[lemma_id] = {
                "remaining_after_unit": float(remaining_after_unit),
                "global_count": float(total_count),
                "urgency": urgency,
                "deferrability": deferrability,
                "future_gain_if_introduced_here": future_gain,
                "unit_index": float(unit_index),
            }
        for candidate in unit.get("candidates", []):
            if not candidate.family_ids:
                continue
            candidate_stats = [family_stats[fid] for fid in _cost_family_ids(candidate) if fid in family_stats]
            if candidate_stats:
                candidate.metadata["family_remaining_after_unit"] = round(sum(s["remaining_after_unit"] for s in candidate_stats) / len(candidate_stats), 6)
                candidate.metadata["family_urgency"] = round(sum(s["urgency"] for s in candidate_stats) / len(candidate_stats), 6)
                candidate.metadata["family_deferrability"] = round(sum(s["deferrability"] for s in candidate_stats) / len(candidate_stats), 6)
                candidate.metadata["family_global_count"] = round(sum(s["global_count"] for s in candidate_stats) / len(candidate_stats), 6)
                candidate.metadata["family_future_gain_if_introduced_here"] = round(sum(s["future_gain_if_introduced_here"] for s in candidate_stats) / len(candidate_stats), 8)
            target_stats = [target_lemma_stats[lid] for lid in _candidate_cost_target_lemmas(candidate) if lid in target_lemma_stats]
            if target_stats:
                candidate.metadata["target_lemma_remaining_after_unit"] = round(sum(s["remaining_after_unit"] for s in target_stats) / len(target_stats), 6)
                candidate.metadata["target_lemma_urgency"] = round(sum(s["urgency"] for s in target_stats) / len(target_stats), 6)
                candidate.metadata["target_lemma_deferrability"] = round(sum(s["deferrability"] for s in target_stats) / len(target_stats), 6)
                candidate.metadata["target_lemma_global_count"] = round(sum(s["global_count"] for s in target_stats) / len(target_stats), 6)
                candidate.metadata["target_lemma_future_gain_if_introduced_here"] = round(sum(s["future_gain_if_introduced_here"] for s in target_stats) / len(target_stats), 8)
        for family_id, count in unit["family_occurrences"].items():
            family_occurrence_progress[family_id] += int(count)
        for lemma_id, count in unit["target_lemma_occurrences"].items():
            target_lemma_progress[lemma_id] += int(count)


# ---------------------------------------------------------------------------
# Family schedule
# ---------------------------------------------------------------------------

def _annotate_family_schedule(
    units: list[dict[str, object]],
    *,
    lemma_schedule_payload: dict[str, object] | None,
) -> None:
    if not lemma_schedule_payload:
        return
    intro_lookup = {
        str(item.get("family_id", "")): int(item.get("intro_unit_index", 0) or 0)
        for item in lemma_schedule_payload.get("selected_families", [])
        if str(item.get("family_id", "")).strip()
    }
    if not intro_lookup:
        return
    for unit in units:
        for candidate in unit.get("candidates", []):
            scheduled_units = [intro_lookup[fid] for fid in _cost_family_ids(candidate) if fid in intro_lookup]
            if not scheduled_units:
                continue
            candidate.metadata["family_schedule_intro_unit_min"] = min(scheduled_units)
            candidate.metadata["family_schedule_intro_unit_avg"] = round(sum(scheduled_units) / len(scheduled_units), 6)
            candidate.metadata["family_schedule_covered_count"] = len(scheduled_units)


# ---------------------------------------------------------------------------
# Family weight lookup
# ---------------------------------------------------------------------------

def _build_family_weight_lookup(units: list[dict[str, object]]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for unit in units:
        for candidate in unit.get("candidates", []):
            if candidate.granularity != "token" or not candidate.family_ids:
                continue
            family_id = candidate.family_ids[0]
            source_upos = str(candidate.metadata.get("source_upos", "")).upper()
            target_upos = str(candidate.metadata.get("target_upos", "")).upper()
            weights.setdefault(family_id, 0.0 if _family_is_functional(source_upos, target_upos) else 1.0)
    return weights


def _family_is_functional(source_upos: str, target_upos: str) -> bool:
    return source_upos in FUNCTIONAL_UPOS or target_upos in FUNCTIONAL_UPOS

