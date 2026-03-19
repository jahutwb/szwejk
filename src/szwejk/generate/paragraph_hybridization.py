"""Paragraph-by-paragraph hybridization planning from a full alignment artifact."""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import json

# --- extracted sub-modules ---
from szwejk.generate.candidate_model import (  # noqa: F401 (re-exported)
    ParagraphHybridCandidate,
    candidate_from_dict as _candidate_from_dict,
    _is_span_granularity,
    _display_granularity,
    FALLBACK_RELATIONS,
    TOKEN_CONFIDENCE_PENALTY,
    STRUCTURAL_CONFIDENCE_PENALTY,
    FUNCTIONAL_UPOS,
    DEFAULT_BLOCKED_STANDALONE_UPOS,
    FALLBACK_STOPWORDS,
    SURFACE_FUNCTION_WORDS,
    IDIOMATICITY_PENALTY_WEIGHT,
    VISIBLE_FUTURE_BLEND,
)
from szwejk.generate.plan_metrics import (  # noqa: F401 (re-exported)
    _mean,
    _word_count,
    _target_token_keys,
    _source_token_keys,
    _covered_target_tokens,
    _candidate_simple_mass,
    _target_span_word_count,
    _candidate_marginal_target_simple_mass,
    _selected_simple_mass,
    _selected_marginal_target_simple_mass,
    _visible_czechness,
    _simple_czechness,
    _visible_target,
    _future_czechness,
    _target_future_czechness,
)
from szwejk.generate.family_tracking import (  # noqa: F401 (re-exported)
    _annotate_family_trust,
    _annotate_family_timeline,
    _annotate_family_schedule,
    _build_family_weight_lookup,
    _family_is_functional,
    _cost_family_ids,
    _candidate_cost_target_lemmas,
    _covered_target_lemmas,
)
from szwejk.generate.candidate_scoring import (  # noqa: F401 (re-exported)
    _text_similarity,
    _candidate_visible_mass,
    _selected_visible_mass,
    _candidate_idiomaticity,
    _semantic_confidence,
    _structural_support,
    _candidate_allowed_by_safety,
    _confidence_penalty,
    _late_visible_bonus,
    _effective_new_family_count,
    _new_family_penalty,
    _known_context_mass,
    _known_context_credit,
    _early_carrier_penalty,
    _local_scope_penalty,
    _uncovered_content_penalty,
    _lemma_urgency_bonus,
    _lemma_deferrability_penalty,
    _target_lemma_penalty,
    _family_schedule_penalty,
    _family_schedule_bonus,
    _idiomaticity_penalty,
    _granularity_progress_penalty,
    _smooth_acceptance_slack,
)
from szwejk.generate.candidate_filters import (  # noqa: F401 (re-exported)
    _spans_overlap,
    _span_contains,
    _candidate_contains,
    _conflicts,
    _candidate_allowed_as_surface_carrier,
    _candidate_has_better_narrower_carrier,
    _token_can_stand_alone,
    _candidate_allowed_by_progress,
    _candidate_allowed_by_promotion_safety,
    _candidate_allowed_by_promotion_shape,
    _candidate_blocked_by_family_schedule,
    _promotion_score_floor,
)
from szwejk.generate.selection import (  # noqa: F401 (re-exported)
    _promotion_known_lemma_threshold,
    _select_known_candidates,
    _select_introducing_candidates,
    _select_candidates_three_phase,
    _promote_large_carriers,
    _apply_chapter_simple_boost,
    _refresh_plan_metrics,
)


def load_alignment_artifact(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_paragraph_hybridization_plan(
    alignment_artifact: dict[str, object],
    *,
    target_power: float = 1.0,
    blocked_standalone_upos: frozenset[str] = DEFAULT_BLOCKED_STANDALONE_UPOS,
    idiomaticity_penalty_weight: float = 0.0,
    granularity_policy: str = "staircase",
    chapter_simple_target_power: float | None = None,
    chapter_simple_target_max: float = 0.85,
    chapter_simple_boost_start: float = 0.55,
    lemma_schedule_payload: dict[str, object] | None = None,
    wiktionary_lookup: dict[str, list[str]] | None = None,
    pl_to_cs_lookup: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    units = _build_units(alignment_artifact, blocked_standalone_upos=blocked_standalone_upos)
    family_weight_lookup = _build_family_weight_lookup(units)
    _annotate_family_trust(units)
    _annotate_family_timeline(units)
    _annotate_family_schedule(units, lemma_schedule_payload=lemma_schedule_payload)
    total_occurrence_count = sum(unit["source_occurrence_count"] for unit in units) or 1
    total_word_count = sum(unit["source_word_count"] for unit in units) or 1
    total_units = len(units) or 1
    remaining_occurrences = Counter()
    remaining_target_lemmas = Counter()
    for unit in units:
        remaining_occurrences.update(unit["family_occurrences"])
        remaining_target_lemmas.update(unit["target_lemma_occurrences"])

    introduced_families: set[str] = set()
    introduced_target_lemmas: set[str] = set()
    cumulative_occurrence_count = 0
    cumulative_word_count = 0
    cumulative_simple_mass = 0.0
    cumulative_visible_mass = 0.0
    cumulative_target_simple_mass = 0.0
    plan_units: list[dict[str, object]] = []

    for unit in units:
        cumulative_occurrence_count += unit["source_occurrence_count"]
        cumulative_word_count += unit["source_word_count"]
        if granularity_policy in {"visible-smooth", "cumulative-simple"}:
            progress = cumulative_word_count / total_word_count
        else:
            progress = cumulative_occurrence_count / total_occurrence_count
        if granularity_policy in {"visible-smooth", "cumulative-simple"}:
            target_after = round(_visible_target(progress, target_power), 6)
        else:
            target_after = round(progress**target_power, 6)
        if granularity_policy == "cumulative-simple":
            cumulative_target_simple_mass += float(target_after) * float(unit["source_word_count"])
            cumulative_target_after = round(
                cumulative_target_simple_mass / max(float(cumulative_word_count), 1.0),
                6,
            )
        else:
            cumulative_target_after = target_after
        remaining_after = remaining_occurrences.copy()
        remaining_after.subtract(unit["family_occurrences"])
        remaining_after = Counter({key: value for key, value in remaining_after.items() if value > 0})
        remaining_target_after = remaining_target_lemmas.copy()
        remaining_target_after.subtract(unit["target_lemma_occurrences"])
        remaining_target_after = Counter({key: value for key, value in remaining_target_after.items() if value > 0})

        filtered_candidates = [
            candidate
            for candidate in unit["candidates"]
            if not candidate.is_fallback and _candidate_cost_target_lemmas(candidate)
        ]
        simple_before_unit = _simple_czechness(cumulative_simple_mass, cumulative_word_count)
        visible_before_unit = cumulative_visible_mass

        if granularity_policy == "cumulative-simple":
            # ── New three-phase algorithm ────────────────────────────────────
            selected_candidates = _select_candidates_three_phase(
                filtered_candidates,
                introduced_families=introduced_families,
                introduced_target_lemmas=introduced_target_lemmas,
                cumulative_simple_mass=cumulative_simple_mass,
                seen_word_count_after=cumulative_word_count,
                target_cumulative_after=cumulative_target_after,
                progress=progress,
                wiktionary_lookup=wiktionary_lookup,
                pl_to_cs_lookup=pl_to_cs_lookup,
            )
            selected_candidates = _promote_large_carriers(
                selected_candidates=selected_candidates,
                candidates=filtered_candidates,
                introduced_target_lemmas=introduced_target_lemmas,
                progress=progress,
            )
        else:
            # ── Legacy algorithm for other policies ──────────────────────────
            selected_known, blocked_ids = _select_known_candidates(
                filtered_candidates,
                introduced_families,
                introduced_target_lemmas=introduced_target_lemmas,
                granularity_policy=granularity_policy,
                target_after=target_after,
                current_simple_mass=cumulative_simple_mass,
                seen_word_count_after=cumulative_word_count,
            )
            paragraph_families = set(introduced_families)
            paragraph_target_lemmas = set(introduced_target_lemmas)
            selected_candidates = list(selected_known)
            for candidate in selected_known:
                paragraph_families.update(candidate.family_ids)
                paragraph_target_lemmas.update(_candidate_cost_target_lemmas(candidate))
            simple_after_known = cumulative_simple_mass + _selected_simple_mass(selected_known)
            visible_after_known = cumulative_visible_mass + _selected_visible_mass(selected_known)
            introducing_candidates = [
                candidate
                for candidate in filtered_candidates
                if candidate.candidate_id not in blocked_ids
                and any(lemma not in introduced_target_lemmas for lemma in _candidate_cost_target_lemmas(candidate))
            ]
            selected_new, removed_known_ids = _select_introducing_candidates(
                introducing_candidates,
                selected_candidates=selected_candidates,
                introduced_families=introduced_families,
                introduced_target_lemmas=introduced_target_lemmas,
                remaining_after=remaining_after,
                remaining_target_after=remaining_target_after,
                family_weight_lookup=family_weight_lookup,
                target_after=target_after,
                progress=progress,
                idiomaticity_penalty_weight=idiomaticity_penalty_weight,
                granularity_policy=granularity_policy,
                current_simple_mass=simple_after_known,
                current_visible_mass=visible_after_known,
                seen_word_count_after=cumulative_word_count,
            )
            if removed_known_ids:
                selected_candidates = [c for c in selected_candidates if c.candidate_id not in removed_known_ids]
            selected_candidates.extend(selected_new)

        current_actual = _future_czechness(remaining_after, introduced_families, family_weight_lookup=family_weight_lookup)
        for candidate in selected_candidates:
            introduced_families.update(candidate.family_ids)
            introduced_target_lemmas.update(_candidate_cost_target_lemmas(candidate))

        actual_after = _future_czechness(remaining_after, introduced_families, family_weight_lookup=family_weight_lookup)
        if granularity_policy == "cumulative-simple":
            cumulative_simple_mass += _selected_simple_mass(selected_candidates)
        else:
            cumulative_simple_mass = simple_after_known + _selected_simple_mass(selected_new)
        cumulative_visible_mass = cumulative_visible_mass + _selected_visible_mass(selected_candidates)
        actual_simple_after = _simple_czechness(cumulative_simple_mass, cumulative_word_count)
        actual_visible_after = _visible_czechness(cumulative_visible_mass, cumulative_word_count)
        remaining_occurrences = remaining_after
        remaining_target_lemmas = remaining_target_after

        plan_units.append(
            {
                "unit_index": unit["unit_index"],
                "chapter_pair": list(unit["chapter_pair"]),
                "source_paragraph_range": list(unit["source_paragraph_range"]),
                "target_paragraph_range": list(unit["target_paragraph_range"]),
                "source_preview": unit["source_preview"],
                "target_preview": unit["target_preview"],
                "source_occurrence_count": unit["source_occurrence_count"],
                "source_word_count": unit["source_word_count"],
                "candidate_count": len(filtered_candidates),
                "selected_count": len(selected_candidates),
                "selected_granularity_counts": dict(
                    Counter(_display_granularity(candidate.granularity) for candidate in selected_candidates)
                ),
                "target_future_czechness_after": target_after,
                "target_local_simple_czechness_after": target_after if granularity_policy == "cumulative-simple" else None,
                "target_cumulative_simple_czechness_after": cumulative_target_after if granularity_policy == "cumulative-simple" else None,
                "actual_future_czechness_before": round(current_actual, 6),
                "actual_future_czechness_after": round(actual_after, 6),
                "actual_simple_czechness_before": round(simple_before_unit, 6),
                "actual_simple_czechness_after": round(actual_simple_after, 6),
                "actual_visible_czechness_before": round(_visible_czechness(visible_before_unit, cumulative_word_count), 6),
                "actual_visible_czechness_after": round(actual_visible_after, 6),
                "simple_mass_after": round(cumulative_simple_mass, 6),
                "visible_mass_after": round(cumulative_visible_mass, 6),
                "distance_after": round(
                    abs(actual_simple_after - cumulative_target_after)
                    if granularity_policy == "cumulative-simple"
                    else abs(actual_after - target_after),
                    6,
                ),
                "introduced_family_count": len(introduced_families),
                "selected_candidates": [candidate.to_dict() for candidate in selected_candidates],
            }
        )

    if chapter_simple_target_power is not None:
        _apply_chapter_simple_boost(
            plan_units=plan_units,
            units=units,
            target_power=chapter_simple_target_power,
            target_max=chapter_simple_target_max,
            boost_start=chapter_simple_boost_start,
        )
        _refresh_plan_metrics(
            plan_units=plan_units,
            units=units,
            total_occurrence_count=total_occurrence_count,
            total_word_count=total_word_count,
            family_weight_lookup=family_weight_lookup,
        )

    summary_family_ids = {
        family_id
        for unit in plan_units
        for candidate in unit["selected_candidates"]
        for family_id in candidate.get("family_ids", [])
    }
    final_future = float(plan_units[-1]["actual_future_czechness_after"]) if plan_units else 0.0
    final_visible = float(plan_units[-1]["actual_visible_czechness_after"]) if plan_units else 0.0

    return {
        "mode": "paragraph-by-paragraph-alignment-only-v1",
        "target_power": target_power,
        "blocked_standalone_upos": sorted(blocked_standalone_upos),
        "idiomaticity_penalty_weight": idiomaticity_penalty_weight,
        "granularity_policy": granularity_policy,
        "chapter_simple_target_power": chapter_simple_target_power,
        "chapter_simple_target_max": chapter_simple_target_max,
        "chapter_simple_boost_start": chapter_simple_boost_start,
        "lemma_schedule_mode": None if not lemma_schedule_payload else str(lemma_schedule_payload.get("mode", "")),
        "chapter_pair_count": int(alignment_artifact.get("chapter_pair_count", len(alignment_artifact.get("pair_reports", [])))),
        "unit_count": total_units,
        "summary": {
            "total_occurrence_count": total_occurrence_count,
            "total_word_count": total_word_count,
            "introduced_family_count": len(summary_family_ids),
            "final_future_czechness": round(final_future, 6),
            "final_visible_czechness": round(final_visible, 6),
            "selected_granularity_counts": dict(
                Counter(
                    _display_granularity(str(candidate["granularity"]))
                    for unit in plan_units
                    for candidate in unit["selected_candidates"]
                )
            ),
        },
        "units": plan_units,
    }


def _build_units(
    alignment_artifact: dict[str, object],
    *,
    blocked_standalone_upos: frozenset[str],
) -> list[dict[str, object]]:
    units: list[dict[str, object]] = []
    unit_index = 0
    for pair_report in alignment_artifact.get("pair_reports", []):
        chapter_pair = (int(pair_report["source_chapter_index"]), int(pair_report["target_chapter_index"]))
        span_lookup = _sentence_span_lookup(pair_report.get("sentence_rows", []))
        item_lookup = defaultdict(list)
        for item in pair_report["enrichment_bundle"]["items"]:
            if item.get("enrichment_status") == "sentence_safe":
                span_info = _consume_span_info(span_lookup, item)
                if span_info is not None:
                    item = dict(item)
                    item["sentence_span_alignment"] = span_info
            item_lookup[(int(item["source_paragraph_index"]), int(item["target_paragraph_index"]))].append(item)
        for block in pair_report["paragraph_alignment"]["matched_blocks"]:
            unit_index += 1
            source_start, source_end = int(block["source_range"][0]), int(block["source_range"][1])
            target_start, target_end = int(block["target_range"][0]), int(block["target_range"][1])
            block_items = []
            for source_idx in range(source_start, source_end + 1):
                for target_idx in range(target_start, target_end + 1):
                    block_items.extend(item_lookup.get((source_idx, target_idx), []))

            token_pairs = [
                pair
                for item in block_items
                if item["enrichment_status"] == "sentence_safe"
                for pair in item.get("token_pairs", [])
                if _family_id_for_token_pair(pair)
            ]
            family_occurrences = Counter(_family_id_for_token_pair(pair) for pair in token_pairs)
            candidates = _block_candidates(
                block_items=block_items,
                chapter_pair=chapter_pair,
                unit_index=unit_index,
                source_range=(source_start, source_end),
                target_range=(target_start, target_end),
                source_preview=str(block.get("source_preview", "")),
                target_preview=str(block.get("target_preview", "")),
                blocked_standalone_upos=blocked_standalone_upos,
            )
            units.append(
                {
                    "unit_index": unit_index,
                    "chapter_pair": chapter_pair,
                    "source_paragraph_range": (source_start, source_end),
                    "target_paragraph_range": (target_start, target_end),
                    "source_preview": str(block.get("source_preview", "")),
                    "target_preview": str(block.get("target_preview", "")),
                    "family_occurrences": family_occurrences,
                    "target_lemma_occurrences": _target_lemma_occurrences(block_items),
                    "source_occurrence_count": sum(family_occurrences.values()),
                    "source_word_count": sum(
                        1
                        for item in block_items
                        for token in item.get("source_tokens", [])
                        if token.get("kind") == "word"
                    ),
                    "candidates": candidates,
                }
            )
    return units


def _sentence_span_lookup(sentence_rows: list[dict[str, object]]) -> dict[tuple[int, int, str, str], list[dict[str, object]]]:
    lookup: dict[tuple[int, int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in sentence_rows:
        span = row["sentence_span_alignment"]
        key = (
            int(row["source_paragraph_index"]),
            int(row["target_paragraph_index"]),
            str(span["source_text"]),
            str(span["target_text"]),
        )
        lookup[key].append(span)
    return lookup


def _consume_span_info(
    span_lookup: dict[tuple[int, int, str, str], list[dict[str, object]]],
    item: dict[str, object],
) -> dict[str, object] | None:
    alignment = item.get("sentence_alignment", {})
    key = (
        int(item["source_paragraph_index"]),
        int(item["target_paragraph_index"]),
        str(alignment.get("source_text", "")),
        str(alignment.get("target_text", "")),
    )
    bucket = span_lookup.get(key)
    if not bucket:
        return None
    return bucket.pop(0)


def _block_candidates(
    *,
    block_items: list[dict[str, object]],
    chapter_pair: tuple[int, int],
    unit_index: int,
    source_range: tuple[int, int],
    target_range: tuple[int, int],
    source_preview: str,
    target_preview: str,
    blocked_standalone_upos: frozenset[str],
) -> list[ParagraphHybridCandidate]:
    candidates: list[ParagraphHybridCandidate] = []
    paragraph_family_ids = sorted(
        {
            family_id
            for item in block_items
            if item["enrichment_status"] == "sentence_safe"
            for pair in item.get("token_pairs", [])
            for family_id in [_family_id_for_token_pair(pair)]
            if family_id
        }
    )
    if paragraph_family_ids and source_range[0] == source_range[1] and target_range[0] == target_range[1]:
        paragraph_source_tokens = [token for item in block_items for token in item.get("source_tokens", [])]
        paragraph_target_tokens = [token for item in block_items for token in item.get("target_tokens", [])]
        paragraph_source_span = _token_index_span(paragraph_source_tokens) or source_range
        paragraph_target_span = _token_index_span(paragraph_target_tokens) or target_range
        paragraph_coverage_source_token_keys = [
            (f"sentence:{sentence_index}", int(token.get("index", 0) or 0))
            for sentence_index, item in enumerate(block_items, start=1)
            for token in item.get("source_tokens", [])
            if int(token.get("index", 0) or 0) > 0
        ]
        paragraph_coverage_target_token_keys = [
            (f"sentence:{sentence_index}", int(token.get("index", 0) or 0))
            for sentence_index, item in enumerate(block_items, start=1)
            for token in item.get("target_tokens", [])
            if int(token.get("index", 0) or 0) > 0
        ]
        candidates.append(
            ParagraphHybridCandidate(
                candidate_id=f"u{unit_index}:paragraph",
                chapter_pair=chapter_pair,
                unit_index=unit_index,
                granularity="paragraph",
                scope_id="block",
                source_span=paragraph_source_span,
                target_span=paragraph_target_span,
                source_text=source_preview,
                target_text=target_preview,
                family_ids=paragraph_family_ids,
                score=_mean(
                    float(item["sentence_alignment"].get("score", 0.0))
                    for item in block_items
                    if item["enrichment_status"] == "sentence_safe"
                ),
                relation="paragraph_block",
                metadata=_candidate_text_metadata(
                    source_preview,
                    target_preview,
                    source_tokens=paragraph_source_tokens,
                    target_tokens=paragraph_target_tokens,
                )
                | {
                    "sentence_count": sum(
                        1 for item in block_items if item.get("enrichment_status") == "sentence_safe"
                    ),
                    "coverage_source_token_keys": paragraph_coverage_source_token_keys,
                    "coverage_target_token_keys": paragraph_coverage_target_token_keys,
                },
            )
        )

    for sentence_index, item in enumerate(block_items, start=1):
        if item["enrichment_status"] != "sentence_safe":
            continue
        sentence_alignment = item.get("sentence_span_alignment") or item["sentence_alignment"]
        sentence_source_span = _token_index_span(item.get("source_tokens", [])) or (
            int(sentence_alignment["source_span"][0]),
            int(sentence_alignment["source_span"][1]),
        )
        sentence_target_span = _token_index_span(item.get("target_tokens", [])) or (
            int(sentence_alignment["target_span"][0]),
            int(sentence_alignment["target_span"][1]),
        )
        sentence_metadata = {
            "source_sentence_text": str(sentence_alignment.get("source_text", "")),
            "target_sentence_text": str(sentence_alignment.get("target_text", "")),
            "source_sentence_tokens": _sentence_token_metadata(item.get("source_tokens", [])),
        }
        token_pairs = item.get("token_pairs", [])
        token_pair_by_source_index = {
            int(pair["source_token_index"]): pair
            for pair in token_pairs
            if _family_id_for_token_pair(pair)
        }
        supported_pairs = [pair for pair in token_pairs if _family_id_for_token_pair(pair) and str(pair.get("relation", "")) not in FALLBACK_RELATIONS]
        token_actions = []
        for pair in token_pairs:
            pair_with_context = {
                **pair,
                "prev_source_upos": str(token_pair_by_source_index.get(int(pair["source_token_index"]) - 1, {}).get("source_upos", "")),
                "next_source_upos": str(token_pair_by_source_index.get(int(pair["source_token_index"]) + 1, {}).get("source_upos", "")),
            }
            family_id = _family_id_for_token_pair(pair)
            if not family_id:
                continue
            if not _token_can_stand_alone(pair_with_context, blocked_standalone_upos=blocked_standalone_upos):
                continue
            token_actions.append(
                ParagraphHybridCandidate(
                    candidate_id=f"u{unit_index}:s{sentence_index}:t{pair['source_token_index']}",
                    chapter_pair=chapter_pair,
                    unit_index=unit_index,
                    granularity="token",
                    scope_id=f"sentence:{sentence_index}",
                    source_span=(int(pair["source_token_index"]), int(pair["source_token_index"])),
                    target_span=(int(pair["target_token_index"]), int(pair["target_token_index"])),
                    source_text=str(pair["source_text"]),
                    target_text=str(pair["target_text"]),
                    family_ids=[family_id],
                    score=float(pair["score"]),
                    relation=str(pair.get("relation", "")),
                    metadata={
                        "signals": dict(pair.get("signals", {})),
                        "source_upos": str(pair.get("source_upos", "")),
                        "target_upos": str(pair.get("target_upos", "")),
                        "source_token_index": int(pair["source_token_index"]),
                        "target_token_index": int(pair["target_token_index"]),
                        "prev_source_upos": str(pair_with_context.get("prev_source_upos", "")),
                        "next_source_upos": str(pair_with_context.get("next_source_upos", "")),
                        "source_word_count": 1,
                        "target_word_count": 1,
                        **_candidate_text_metadata(
                            str(pair["source_text"]),
                            str(pair["target_text"]),
                            token_pairs=item.get("token_pairs", []),
                            source_span=(int(pair["source_token_index"]), int(pair["source_token_index"])),
                            target_span=(int(pair["target_token_index"]), int(pair["target_token_index"])),
                            source_scope_id=f"sentence:{sentence_index}",
                            target_scope_id=f"sentence:{sentence_index}",
                        ),
                        **sentence_metadata,
                    },
                )
            )
        token_family_by_span = {
            action.source_span[0]: action.family_ids[0]
            for action in token_actions
        }
        token_family_meta = {
            action.family_ids[0]: {
                "source_upos": str(action.metadata.get("source_upos", "")).upper(),
                "target_upos": str(action.metadata.get("target_upos", "")).upper(),
                "is_functional": _family_is_functional(
                    str(action.metadata.get("source_upos", "")).upper(),
                    str(action.metadata.get("target_upos", "")).upper(),
                ),
            }
            for action in token_actions
            if action.family_ids
        }
        candidates.extend(token_actions)

        sentence_family_ids = sorted({family_id for family_id in token_family_by_span.values() if family_id})
        if sentence_family_ids:
            alignment = sentence_alignment
            candidates.append(
                ParagraphHybridCandidate(
                    candidate_id=f"u{unit_index}:s{sentence_index}:sentence",
                    chapter_pair=chapter_pair,
                    unit_index=unit_index,
                    granularity="sentence",
                    scope_id=f"sentence:{sentence_index}",
                    source_span=sentence_source_span,
                    target_span=sentence_target_span,
                    source_text=str(alignment["source_text"]),
                    target_text=str(alignment["target_text"]),
                    family_ids=sentence_family_ids,
                    score=float(alignment["score"]),
                    relation="sentence_span",
                    metadata={
                        "signals": dict(alignment.get("signals", {})),
                        "content_family_ids": [
                            family_id for family_id in sentence_family_ids
                            if not token_family_meta.get(family_id, {}).get("is_functional", False)
                        ],
                        "functional_family_ids": [
                            family_id for family_id in sentence_family_ids
                            if token_family_meta.get(family_id, {}).get("is_functional", False)
                        ],
                        "source_word_count": _word_count(str(alignment["source_text"])),
                        "target_word_count": _word_count(str(alignment["target_text"])),
                        **_candidate_text_metadata(
                            str(alignment["source_text"]),
                            str(alignment["target_text"]),
                            source_tokens=item.get("source_tokens", []),
                            target_tokens=item.get("target_tokens", []),
                            token_pairs=item.get("token_pairs", []),
                            source_span=(int(alignment["source_span"][0]), int(alignment["source_span"][1])),
                            target_span=(int(alignment["target_span"][0]), int(alignment["target_span"][1])),
                            source_scope_id=f"sentence:{sentence_index}",
                            target_scope_id=f"sentence:{sentence_index}",
                        ),
                        **sentence_metadata,
                    },
                )
            )

        for idx, candidate in enumerate(item.get("subtree_candidates", []), start=1):
            family_ids = _family_ids_for_span(token_family_by_span, int(candidate["source_span"][0]), int(candidate["source_span"][1]))
            if not family_ids:
                continue
            support = _span_token_support(
                token_pairs=supported_pairs,
                source_tokens=item["source_tokens"],
                target_tokens=item["target_tokens"],
                source_span=(int(candidate["source_span"][0]), int(candidate["source_span"][1])),
                target_span=(int(candidate["target_span"][0]), int(candidate["target_span"][1])),
            )
            candidates.append(
                ParagraphHybridCandidate(
                    candidate_id=f"u{unit_index}:s{sentence_index}:subtree:{idx}",
                    chapter_pair=chapter_pair,
                    unit_index=unit_index,
                    granularity="subtree",
                    scope_id=f"sentence:{sentence_index}",
                    source_span=(int(candidate["source_span"][0]), int(candidate["source_span"][1])),
                    target_span=(int(candidate["target_span"][0]), int(candidate["target_span"][1])),
                    source_text=str(candidate["source_text"]),
                    target_text=str(candidate["target_text"]),
                    family_ids=family_ids,
                    score=float(candidate["score"]),
                    relation=str(candidate.get("relation", "")),
                    metadata={
                        "signals": dict(candidate.get("signals", {})),
                        "content_family_ids": [
                            family_id for family_id in family_ids
                            if not token_family_meta.get(family_id, {}).get("is_functional", False)
                        ],
                        "functional_family_ids": [
                            family_id for family_id in family_ids
                            if token_family_meta.get(family_id, {}).get("is_functional", False)
                        ],
                        **support,
                        **_candidate_text_metadata(
                            str(candidate["source_text"]),
                            str(candidate["target_text"]),
                            source_tokens=item.get("source_tokens", []),
                            target_tokens=item.get("target_tokens", []),
                            token_pairs=item.get("token_pairs", []),
                            source_span=(int(candidate["source_span"][0]), int(candidate["source_span"][1])),
                            target_span=(int(candidate["target_span"][0]), int(candidate["target_span"][1])),
                            source_scope_id=f"sentence:{sentence_index}",
                            target_scope_id=f"sentence:{sentence_index}",
                        ),
                        **sentence_metadata,
                    },
                )
            )

        for idx, candidate in enumerate(item.get("phrase_candidates", []), start=1):
            candidate_source_span = (int(candidate["source_span"][0]), int(candidate["source_span"][1]))
            candidate_target_span = (int(candidate["target_span"][0]), int(candidate["target_span"][1]))
            if candidate_source_span == sentence_source_span and candidate_target_span == sentence_target_span:
                continue
            phrase_source_text = _span_text(item["source_tokens"], candidate_source_span[0], candidate_source_span[1])
            phrase_target_text = _span_text(item["target_tokens"], candidate_target_span[0], candidate_target_span[1])
            if (
                _normalize_carrier_text(phrase_source_text) == _normalize_carrier_text(str(sentence_alignment.get("source_text", "")))
                and _normalize_carrier_text(phrase_target_text) == _normalize_carrier_text(str(sentence_alignment.get("target_text", "")))
            ):
                continue
            group_family_id = str(candidate.get("group_family_id", "")).strip()
            if group_family_id:
                family_ids = [group_family_id]
            else:
                family_ids = _family_ids_for_span(token_family_by_span, candidate_source_span[0], candidate_source_span[1])
            if not family_ids:
                continue
            support = _span_token_support(
                token_pairs=supported_pairs,
                source_tokens=item["source_tokens"],
                target_tokens=item["target_tokens"],
                source_span=candidate_source_span,
                target_span=candidate_target_span,
            )
            candidates.append(
                ParagraphHybridCandidate(
                    candidate_id=f"u{unit_index}:s{sentence_index}:phrase:{idx}",
                    chapter_pair=chapter_pair,
                    unit_index=unit_index,
                    granularity="phrase",
                    scope_id=f"sentence:{sentence_index}",
                    source_span=candidate_source_span,
                    target_span=candidate_target_span,
                    source_text=phrase_source_text,
                    target_text=phrase_target_text,
                    family_ids=family_ids,
                    score=float(candidate["score"]),
                    relation=str(candidate.get("relation", "")),
                    metadata={
                        "signals": dict(candidate.get("signals", {})),
                        "content_family_ids": [
                            family_id for family_id in family_ids
                            if not token_family_meta.get(family_id, {}).get("is_functional", False)
                        ],
                        "functional_family_ids": [
                            family_id for family_id in family_ids
                            if token_family_meta.get(family_id, {}).get("is_functional", False)
                        ],
                        **support,
                        **_candidate_text_metadata(
                            phrase_source_text,
                            phrase_target_text,
                            source_tokens=item.get("source_tokens", []),
                            target_tokens=item.get("target_tokens", []),
                            token_pairs=item.get("token_pairs", []),
                            source_span=candidate_source_span,
                            target_span=candidate_target_span,
                            source_scope_id=f"sentence:{sentence_index}",
                            target_scope_id=f"sentence:{sentence_index}",
                        ),
                        **sentence_metadata,
                    },
                )
            )

    return candidates


def _candidate_text_metadata(
    source_text: str,
    target_text: str,
    *,
    source_tokens: list[dict[str, object]] | None = None,
    target_tokens: list[dict[str, object]] | None = None,
    token_pairs: list[dict[str, object]] | None = None,
    source_span: tuple[int, int] | None = None,
    target_span: tuple[int, int] | None = None,
    source_scope_id: str | None = None,
    target_scope_id: str | None = None,
) -> dict[str, object]:
    source_content_lemmas = _token_span_lemmas(source_tokens or [], source_span, functional=False)
    target_content_lemmas = _token_span_lemmas(target_tokens or [], target_span, functional=False)
    target_functional_lemmas = _token_span_lemmas(target_tokens or [], target_span, functional=True)
    source_pair_lemmas = _pair_span_lemmas(token_pairs or [], source_span, side="source", functional=False)
    target_pair_lemmas = _pair_span_lemmas(token_pairs or [], target_span, side="target", functional=False)
    target_pair_functional = _pair_span_lemmas(token_pairs or [], target_span, side="target", functional=True)
    source_content_lemmas = sorted(set(source_content_lemmas) | set(source_pair_lemmas))
    target_content_lemmas = sorted(set(target_content_lemmas) | set(target_pair_lemmas))
    target_functional_lemmas = sorted(set(target_functional_lemmas) | set(target_pair_functional))
    has_source_lemma_data = _tokens_have_lemma_data(source_tokens or [], source_span)
    has_target_lemma_data = _tokens_have_lemma_data(target_tokens or [], target_span)
    if not has_source_lemma_data:
        source_content_lemmas = sorted(set(source_content_lemmas) | set(_fallback_span_lemmas(
            source_tokens or [],
            source_span,
            token_pairs or [],
            language="pl",
            functional=False,
            side="source",
        )))
    source_content_lemmas = sorted(set(source_content_lemmas) | set(_fallback_text_lemmas(source_text, language="pl", functional=False)))
    if not has_target_lemma_data:
        target_content_lemmas = sorted(set(target_content_lemmas) | set(_fallback_span_lemmas(
            target_tokens or [],
            target_span,
            token_pairs or [],
            language="cs",
            functional=False,
            side="target",
        )))
        target_functional_lemmas = sorted(set(target_functional_lemmas) | set(_fallback_span_lemmas(
            target_tokens or [],
            target_span,
            token_pairs or [],
            language="cs",
            functional=True,
            side="target",
        )))
    target_content_lemmas = sorted(set(target_content_lemmas) | set(_fallback_text_lemmas(target_text, language="cs", functional=False)))
    target_functional_lemmas = sorted(set(target_functional_lemmas) | set(_fallback_text_lemmas(target_text, language="cs", functional=True)))
    metadata = {
        "candidate_source_content_lemmas": source_content_lemmas,
        "candidate_target_content_lemmas": target_content_lemmas,
        "candidate_target_functional_lemmas": target_functional_lemmas,
    }
    if source_scope_id:
        metadata["coverage_source_token_keys"] = _token_keys_for_span(
            source_tokens or [],
            source_span,
            scope_id=source_scope_id,
        )
    if target_scope_id:
        metadata["coverage_target_token_keys"] = _token_keys_for_span(
            target_tokens or [],
            target_span,
            scope_id=target_scope_id,
        )
    return metadata


def _token_keys_for_span(
    tokens: list[dict[str, object]],
    span: tuple[int, int] | None,
    *,
    scope_id: str,
) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    for token in tokens:
        index = int(token.get("index", 0) or 0)
        if index <= 0:
            continue
        if span is not None and not (span[0] <= index <= span[1]):
            continue
        result.append((scope_id, index))
    return result


def _sentence_token_metadata(tokens: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "index": int(token.get("index", 0) or 0),
            "text": str(token.get("text", "")),
            "kind": str(token.get("kind", "")),
        }
        for token in tokens
        if int(token.get("index", 0) or 0) > 0
    ]


def _pair_span_lemmas(
    token_pairs: list[dict[str, object]],
    span: tuple[int, int] | None,
    *,
    side: str,
    functional: bool,
) -> list[str]:
    result: list[str] = []
    index_key = "source_token_index" if side == "source" else "target_token_index"
    lemma_key = "source_lemma" if side == "source" else "target_lemma"
    upos_key = "source_upos" if side == "source" else "target_upos"
    for pair in token_pairs:
        index = int(pair.get(index_key, 0) or 0)
        if span is not None and not (span[0] <= index <= span[1]):
            continue
        lemma = str(pair.get(lemma_key) or "").strip().lower()
        upos = str(pair.get(upos_key) or "").upper()
        if not lemma:
            continue
        is_functional = upos in FUNCTIONAL_UPOS
        if functional != is_functional:
            continue
        result.append(lemma)
    return sorted(set(result))


def _fallback_span_lemmas(
    tokens: list[dict[str, object]],
    span: tuple[int, int] | None,
    token_pairs: list[dict[str, object]],
    *,
    language: str,
    functional: bool,
    side: str,
) -> list[str]:
    covered_indices = {
        int(pair.get("source_token_index" if side == "source" else "target_token_index", 0) or 0)
        for pair in token_pairs
    }
    result: list[str] = []
    stopwords = FALLBACK_STOPWORDS.get(language, set())
    for token in tokens:
        if token.get("kind") != "word":
            continue
        index = int(token.get("index", 0) or 0)
        if span is not None and not (span[0] <= index <= span[1]):
            continue
        if index in covered_indices:
            continue
        lemma = str(token.get("text") or "").strip(",.!?;:\"“”„()[]{}").lower()
        if not lemma:
            continue
        is_functional = lemma in stopwords
        if functional != is_functional:
            continue
        result.append(lemma)
    return sorted(set(result))


def _fallback_text_lemmas(text: str, *, language: str, functional: bool) -> list[str]:
    result: list[str] = []
    stopwords = FALLBACK_STOPWORDS.get(language, set())
    for part in text.split():
        lemma = part.strip(",.!?;:\"“”„()[]{}").lower()
        if not lemma:
            continue
        is_functional = lemma in stopwords
        if functional != is_functional:
            continue
        result.append(lemma)
    return sorted(set(result))


def _token_span_lemmas(
    tokens: list[dict[str, object]],
    span: tuple[int, int] | None,
    *,
    functional: bool,
) -> list[str]:
    result: list[str] = []
    for token in tokens:
        if token.get("kind") != "word":
            continue
        index = int(token.get("index", 0) or 0)
        if span is not None and not (span[0] <= index <= span[1]):
            continue
        upos = str(token.get("upos") or "").upper()
        lemma = str(token.get("lemma") or "").strip().lower()
        if not lemma:
            continue
        is_functional = upos in FUNCTIONAL_UPOS
        if functional != is_functional:
            continue
        result.append(lemma)
    return sorted(set(result))


def _tokens_have_lemma_data(tokens: list[dict[str, object]], span: tuple[int, int] | None) -> bool:
    for token in tokens:
        if token.get("kind") != "word":
            continue
        index = int(token.get("index", 0) or 0)
        if span is not None and not (span[0] <= index <= span[1]):
            continue
        if str(token.get("lemma") or "").strip():
            return True
    return False


def _token_target_lemma(token: dict[str, object]) -> str | None:
    if token.get("kind") != "word":
        return None
    lemma = str(token.get("lemma") or "").strip().lower()
    if not lemma:
        return None
    return lemma


def _target_lemma_occurrences(block_items: list[dict[str, object]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in block_items:
        if item.get("enrichment_status") != "sentence_safe":
            continue
        for token in item.get("target_tokens", []):
            lemma = _token_target_lemma(token)
            if lemma:
                counts[lemma] += 1
    return counts


def _family_id_for_token_pair(pair: dict[str, object]) -> str | None:
    source_lemma = str(pair.get("source_lemma", "")).strip().lower()
    target_lemma = str(pair.get("target_lemma", "")).strip().lower()
    if not source_lemma or not target_lemma:
        return None
    return f"{source_lemma}::{target_lemma}"


def _family_ids_for_span(token_family_by_span: dict[int, str], start: int, end: int) -> list[str]:
    return sorted({family_id for index, family_id in token_family_by_span.items() if start <= index <= end})


def _span_text(tokens: list[dict[str, object]], start: int, end: int) -> str:
    rendered_parts: list[str] = []
    for token in tokens:
        index = int(token.get("index", 0) or 0)
        if not (start <= index <= end):
            continue
        text = str(token.get("text", ""))
        if not text:
            continue
        if token.get("kind") == "word" and _should_attach_left(token) and rendered_parts:
            rendered_parts[-1] = f"{rendered_parts[-1]}{text}"
        else:
            rendered_parts.append(text)
    return _normalize_rendered_text(rendered_parts)


def _token_index_span(tokens: list[dict[str, object]]) -> tuple[int, int] | None:
    indexes = [
        int(token.get("index", 0) or 0)
        for token in tokens
        if int(token.get("index", 0) or 0) > 0
    ]
    if not indexes:
        return None
    return (min(indexes), max(indexes))


def _normalize_rendered_text(parts: list[str]) -> str:
    text = " ".join(part for part in parts if part)
    text = text.replace(" ,", ",").replace(" .", ".").replace(" ;", ";").replace(" :", ":")
    text = text.replace(" !", "!").replace(" ?", "?").replace(" )", ")").replace("( ", "(")
    text = text.replace("” ", "”").replace("“ ", "“")
    return " ".join(text.split())


def _normalize_carrier_text(text: str) -> str:
    normalized = " ".join(text.strip().split())
    return normalized.strip(" \t\n\r,.;:!?\"'„”“‚‘’()-—–«»")


def _should_attach_left(token: dict[str, object]) -> bool:
    if token.get("kind") != "word":
        return False
    feats = str(token.get("feats") or "")
    deprel = str(token.get("deprel") or "")
    upos = str(token.get("upos") or "")
    if "Variant=Short" in feats and upos in {"AUX", "PART"}:
        return True
    if deprel == "aux:clitic":
        return True
    if upos in {"PART", "AUX"} and len(str(token.get("text", ""))) <= 2:
        return True
    return False


def _span_token_support(
    *,
    token_pairs: list[dict[str, object]],
    source_tokens: list[dict[str, object]],
    target_tokens: list[dict[str, object]],
    source_span: tuple[int, int],
    target_span: tuple[int, int],
) -> dict[str, object]:
    aligned_pair_count = sum(
        1
        for pair in token_pairs
        if source_span[0] <= int(pair["source_token_index"]) <= source_span[1]
        and target_span[0] <= int(pair["target_token_index"]) <= target_span[1]
    )
    source_word_count = sum(
        1 for token in source_tokens if token.get("kind") == "word" and source_span[0] <= int(token["index"]) <= source_span[1]
    )
    target_word_count = sum(
        1 for token in target_tokens if token.get("kind") == "word" and target_span[0] <= int(token["index"]) <= target_span[1]
    )
    span_word_count = max(source_word_count, target_word_count, 1)
    return {
        "aligned_token_pair_count": aligned_pair_count,
        "source_word_count": source_word_count,
        "target_word_count": target_word_count,
        "token_support_ratio": round(aligned_pair_count / span_word_count, 6),
    }
