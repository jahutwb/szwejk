"""Paragraph-by-paragraph hybridization planning from a full alignment artifact."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import json

FALLBACK_RELATIONS = {"embedding_assisted", "dictionary", "manual_forced"}
TOKEN_CONFIDENCE_PENALTY = 0.025
STRUCTURAL_CONFIDENCE_PENALTY = 0.008
FUNCTIONAL_UPOS = frozenset({"SCONJ", "CCONJ", "PART", "ADP", "PRON", "DET", "AUX"})
DEFAULT_BLOCKED_STANDALONE_UPOS = FUNCTIONAL_UPOS
FALLBACK_STOPWORDS = {
    "pl": {"a", "ale", "bo", "by", "co", "czy", "do", "go", "i", "ich", "jego", "jej", "jak", "już", "na", "nie", "o", "od", "po", "przed", "przy", "się", "swoim", "ten", "to", "w", "za", "z", "że"},
    "cs": {"a", "ale", "bo", "by", "co", "do", "ho", "i", "jak", "je", "jeho", "její", "již", "na", "ne", "o", "od", "po", "před", "pri", "se", "svým", "ten", "to", "u", "už", "v", "za", "z", "že"},
}
SURFACE_FUNCTION_WORDS = FALLBACK_STOPWORDS["pl"] | FALLBACK_STOPWORDS["cs"] | {
    "mu", "mi", "mnie", "mně", "tě", "ci", "pana", "pan", "pani", "ją", "jąż", "go", "ho", "me", "mě", "jsem", "jsi", "jest", "je", "było", "byl", "była", "bylo", "byli",
}
IDIOMATICITY_PENALTY_WEIGHT = 0.06
VISIBLE_FUTURE_BLEND = 0.18


@dataclass(slots=True)
class ParagraphHybridCandidate:
    candidate_id: str
    chapter_pair: tuple[int, int]
    unit_index: int
    granularity: str
    scope_id: str
    source_span: tuple[int, int]
    target_span: tuple[int, int]
    source_text: str
    target_text: str
    family_ids: list[str]
    score: float
    relation: str
    metadata: dict[str, object]

    @property
    def is_fallback(self) -> bool:
        return self.relation in FALLBACK_RELATIONS

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "chapter_pair": list(self.chapter_pair),
            "unit_index": self.unit_index,
            "granularity": self.granularity,
            "scope_id": self.scope_id,
            "source_span": list(self.source_span),
            "target_span": list(self.target_span),
            "source_text": self.source_text,
            "target_text": self.target_text,
            "family_ids": list(self.family_ids),
            "score": round(self.score, 6),
            "relation": self.relation,
            "metadata": self.metadata,
        }


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
        selected_known, blocked_ids = _select_known_candidates(
            filtered_candidates,
            introduced_families,
            introduced_target_lemmas=introduced_target_lemmas,
            granularity_policy=granularity_policy,
            target_after=cumulative_target_after if granularity_policy == "cumulative-simple" else target_after,
            current_simple_mass=cumulative_simple_mass,
            seen_word_count_after=cumulative_word_count,
        )
        paragraph_families = set(introduced_families)
        paragraph_target_lemmas = set(introduced_target_lemmas)
        selected_candidates = list(selected_known)
        for candidate in selected_known:
            paragraph_families.update(candidate.family_ids)
            paragraph_target_lemmas.update(_candidate_cost_target_lemmas(candidate))

        current_actual = _future_czechness(remaining_after, paragraph_families, family_weight_lookup=family_weight_lookup)
        simple_before_unit = _simple_czechness(cumulative_simple_mass, cumulative_word_count)
        visible_before_unit = cumulative_visible_mass
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
            target_after=cumulative_target_after if granularity_policy == "cumulative-simple" else target_after,
            progress=progress,
            idiomaticity_penalty_weight=idiomaticity_penalty_weight,
            granularity_policy=granularity_policy,
            current_simple_mass=simple_after_known,
            current_visible_mass=visible_after_known,
            seen_word_count_after=cumulative_word_count,
        )
        if removed_known_ids:
            selected_candidates = [candidate for candidate in selected_candidates if candidate.candidate_id not in removed_known_ids]
        selected_candidates.extend(selected_new)
        if granularity_policy == "cumulative-simple":
            selected_candidates = _promote_large_carriers(
                selected_candidates=selected_candidates,
                candidates=filtered_candidates,
                introduced_target_lemmas=introduced_target_lemmas,
                progress=progress,
            )
        for candidate in selected_candidates:
            introduced_families.update(candidate.family_ids)
            introduced_target_lemmas.update(_candidate_cost_target_lemmas(candidate))

        actual_after = _future_czechness(remaining_after, introduced_families, family_weight_lookup=family_weight_lookup)
        if granularity_policy == "cumulative-simple":
            cumulative_simple_mass += _selected_marginal_target_simple_mass(selected_candidates)
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
                "selected_granularity_counts": dict(Counter(candidate.granularity for candidate in selected_candidates)),
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
                    candidate["granularity"]
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
                | {"coverage_target_token_keys": paragraph_coverage_target_token_keys},
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
                        ),
                        **sentence_metadata,
                    },
                )
            )

    return candidates


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
            similarity = _text_similarity(candidate.source_text, candidate.target_text)
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
            candidate_stats = [family_stats[family_id] for family_id in _cost_family_ids(candidate) if family_id in family_stats]
            if not candidate_stats:
                continue
            candidate.metadata["family_remaining_after_unit"] = round(
                sum(item["remaining_after_unit"] for item in candidate_stats) / len(candidate_stats), 6
            )
            candidate.metadata["family_urgency"] = round(
                sum(item["urgency"] for item in candidate_stats) / len(candidate_stats), 6
            )
            candidate.metadata["family_deferrability"] = round(
                sum(item["deferrability"] for item in candidate_stats) / len(candidate_stats), 6
            )
            candidate.metadata["family_global_count"] = round(
                sum(item["global_count"] for item in candidate_stats) / len(candidate_stats), 6
            )
            candidate.metadata["family_future_gain_if_introduced_here"] = round(
                sum(item["future_gain_if_introduced_here"] for item in candidate_stats) / len(candidate_stats), 8
            )
            target_stats = [
                target_lemma_stats[lemma_id]
                for lemma_id in _candidate_cost_target_lemmas(candidate)
                if lemma_id in target_lemma_stats
            ]
            if target_stats:
                candidate.metadata["target_lemma_remaining_after_unit"] = round(
                    sum(item["remaining_after_unit"] for item in target_stats) / len(target_stats), 6
                )
                candidate.metadata["target_lemma_urgency"] = round(
                    sum(item["urgency"] for item in target_stats) / len(target_stats), 6
                )
                candidate.metadata["target_lemma_deferrability"] = round(
                    sum(item["deferrability"] for item in target_stats) / len(target_stats), 6
                )
                candidate.metadata["target_lemma_global_count"] = round(
                    sum(item["global_count"] for item in target_stats) / len(target_stats), 6
                )
                candidate.metadata["target_lemma_future_gain_if_introduced_here"] = round(
                    sum(item["future_gain_if_introduced_here"] for item in target_stats) / len(target_stats), 8
                )
        for family_id, count in unit["family_occurrences"].items():
            family_occurrence_progress[family_id] += int(count)
        for lemma_id, count in unit["target_lemma_occurrences"].items():
            target_lemma_progress[lemma_id] += int(count)


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
            scheduled_units = [
                intro_lookup[family_id]
                for family_id in _cost_family_ids(candidate)
                if family_id in intro_lookup
            ]
            if not scheduled_units:
                continue
            candidate.metadata["family_schedule_intro_unit_min"] = min(scheduled_units)
            candidate.metadata["family_schedule_intro_unit_avg"] = round(sum(scheduled_units) / len(scheduled_units), 6)
            candidate.metadata["family_schedule_covered_count"] = len(scheduled_units)


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


def _candidate_text_metadata(
    source_text: str,
    target_text: str,
    *,
    source_tokens: list[dict[str, object]] | None = None,
    target_tokens: list[dict[str, object]] | None = None,
    token_pairs: list[dict[str, object]] | None = None,
    source_span: tuple[int, int] | None = None,
    target_span: tuple[int, int] | None = None,
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
    return {
        "candidate_source_content_lemmas": source_content_lemmas,
        "candidate_target_content_lemmas": target_content_lemmas,
        "candidate_target_functional_lemmas": target_functional_lemmas,
    }


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


def _covered_target_lemmas(candidate: ParagraphHybridCandidate) -> set[str]:
    covered: set[str] = set()
    for family_id in _cost_family_ids(candidate):
        if "::" not in family_id:
            continue
        _source, target = family_id.split("::", 1)
        target = target.strip().lower()
        if target:
            covered.add(target)
    return covered


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


def _cost_family_ids(candidate: ParagraphHybridCandidate) -> list[str]:
    content = [str(item) for item in candidate.metadata.get("content_family_ids", []) if str(item).strip()]
    if content:
        return content
    return list(candidate.family_ids)


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


def _mean(values) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


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


def _select_known_candidates(
    candidates: list[ParagraphHybridCandidate],
    introduced_families: set[str],
    *,
    introduced_target_lemmas: set[str] | None = None,
    granularity_policy: str = "staircase",
    target_after: float = 0.0,
    current_simple_mass: float = 0.0,
    seen_word_count_after: int = 0,
) -> tuple[list[ParagraphHybridCandidate], set[str]]:
    introduced_target_lemmas = introduced_target_lemmas or set()
    selected: list[ParagraphHybridCandidate] = []
    blocked_ids: set[str] = set()
    known_candidates = [
        item
        for item in candidates
        if (
            _candidate_cost_target_lemmas(item)
            and all(lemma in introduced_target_lemmas for lemma in _candidate_cost_target_lemmas(item))
        )
    ]
    def _known_sort_key(item: ParagraphHybridCandidate) -> tuple[float, float, str]:
        if granularity_policy == "cumulative-simple":
            return (
                -float(_target_span_word_count(item)),
                -item.score,
                item.candidate_id,
            )
        return (
            -item.score,
            0.0,
            item.candidate_id,
        )

    for candidate in sorted(known_candidates, key=_known_sort_key):
        if not _candidate_allowed_as_surface_carrier(candidate):
            blocked_ids.add(candidate.candidate_id)
            continue
        if any(_conflicts(candidate, other) for other in selected):
            blocked_ids.add(candidate.candidate_id)
            continue
        selected.append(candidate)
    return selected, {item.candidate_id for item in selected} | blocked_ids


def _select_introducing_candidates(
    candidates: list[ParagraphHybridCandidate],
    *,
    selected_candidates: list[ParagraphHybridCandidate],
    introduced_families: set[str],
    introduced_target_lemmas: set[str],
    remaining_after: Counter[str],
    remaining_target_after: Counter[str],
    family_weight_lookup: dict[str, float],
    target_after: float,
    progress: float,
    idiomaticity_penalty_weight: float,
    granularity_policy: str,
    current_simple_mass: float,
    current_visible_mass: float,
    seen_word_count_after: int,
) -> tuple[list[ParagraphHybridCandidate], set[str]]:
    chosen: list[ParagraphHybridCandidate] = []
    removed_ids: set[str] = set()
    paragraph_target_lemmas = set(introduced_target_lemmas)
    paragraph_families = set(introduced_families)
    if granularity_policy == "visible-smooth":
        current_distance = abs(_visible_czechness(current_visible_mass, seen_word_count_after) - target_after)
    elif granularity_policy == "cumulative-simple":
        current_distance = abs(_simple_czechness(current_simple_mass, seen_word_count_after) - target_after)
    else:
        current_distance = abs(_target_future_czechness(remaining_target_after, paragraph_target_lemmas) - target_after)
    current_target_lemma_distance = abs(_target_future_czechness(remaining_target_after, paragraph_target_lemmas) - target_after)
    simple_mass = current_simple_mass
    available = list(candidates)
    visible_mass = current_visible_mass
    current_selected = list(selected_candidates)
    initially_selected_ids = {candidate.candidate_id for candidate in selected_candidates}

    while available:
        ranked = []
        for candidate in available:
            replaceable_conflicts: list[ParagraphHybridCandidate] = []
            hard_conflict = False
            for other in current_selected:
                if not _conflicts(candidate, other):
                    continue
                if granularity_policy == "cumulative-simple" and _candidate_contains(candidate, other):
                    replaceable_conflicts.append(other)
                    continue
                hard_conflict = True
                break
            if hard_conflict:
                continue
            if not _candidate_allowed_as_surface_carrier(candidate):
                continue
            if _candidate_has_better_narrower_carrier(
                candidate,
                candidates=available,
                paragraph_target_lemmas=paragraph_target_lemmas,
            ):
                continue
            if not _candidate_allowed_by_safety(candidate):
                continue
            if _candidate_blocked_by_family_schedule(candidate, paragraph_families=paragraph_families):
                continue
            new_target_lemmas = [lemma for lemma in _candidate_cost_target_lemmas(candidate) if lemma not in paragraph_target_lemmas]
            if not new_target_lemmas:
                continue
            after_target_lemmas = paragraph_target_lemmas | set(new_target_lemmas)
            target_lemma_distance = abs(_target_future_czechness(remaining_target_after, after_target_lemmas) - target_after)
            effective_new_family_count = float(len(set(new_target_lemmas)))
            known_context_mass = _known_context_mass(candidate, paragraph_target_lemmas=paragraph_target_lemmas)
            contextual_credit = _known_context_credit(
                candidate,
                known_context_mass=known_context_mass,
                effective_new_family_count=effective_new_family_count,
                progress=progress,
            )
            early_carrier_penalty = _early_carrier_penalty(
                candidate,
                progress=progress,
                effective_new_family_count=effective_new_family_count,
                known_context_mass=known_context_mass,
            )
            deferrability_penalty = _lemma_deferrability_penalty(candidate, progress=progress)
            local_scope_penalty = _local_scope_penalty(
                candidate,
                candidates=available,
                paragraph_target_lemmas=paragraph_target_lemmas,
            )
            weighted_distance = target_lemma_distance + _confidence_penalty(candidate) + _idiomaticity_penalty(
                candidate,
                progress=progress,
                idiomaticity_penalty_weight=idiomaticity_penalty_weight,
            ) + early_carrier_penalty + deferrability_penalty + local_scope_penalty - _lemma_urgency_bonus(candidate, progress=progress) - contextual_credit
            if granularity_policy == "smooth":
                weighted_distance += _granularity_progress_penalty(candidate, progress=progress)
            elif granularity_policy == "visible-smooth":
                visible_after = _visible_czechness(visible_mass + _candidate_visible_mass(candidate), seen_word_count_after)
                visible_distance = abs(visible_after - target_after)
                weighted_distance = (
                    visible_distance
                    + (0.35 * target_lemma_distance)
                    + _confidence_penalty(candidate)
                    + _idiomaticity_penalty(
                        candidate,
                        progress=progress,
                        idiomaticity_penalty_weight=idiomaticity_penalty_weight,
                    )
                    + early_carrier_penalty
                    + deferrability_penalty
                    + local_scope_penalty
                    + _granularity_progress_penalty(candidate, progress=progress)
                    - _lemma_urgency_bonus(candidate, progress=progress)
                    - contextual_credit
                )
            elif granularity_policy == "cumulative-simple":
                covered_target_tokens = _covered_target_tokens(
                    [item for item in current_selected if item not in replaceable_conflicts]
                )
                marginal_simple_gain = _candidate_marginal_target_simple_mass(
                    candidate,
                    covered_target_tokens=covered_target_tokens,
                )
                if marginal_simple_gain <= 0.0:
                    continue
                simple_after = _simple_czechness(simple_mass + marginal_simple_gain, seen_word_count_after)
                simple_distance = abs(simple_after - target_after)
                # v15: for cumulative-simple, rank almost purely by fit to cumulative
                # Czechness and use novelty as the explicit secondary criterion.
                weighted_distance = simple_distance
            elif granularity_policy == "reader-visible-late":
                weighted_distance = (
                    target_lemma_distance
                    + _confidence_penalty(candidate)
                    + _idiomaticity_penalty(
                        candidate,
                        progress=progress,
                        idiomaticity_penalty_weight=idiomaticity_penalty_weight,
                    )
                    + early_carrier_penalty
                    + deferrability_penalty
                    + local_scope_penalty
                    - _late_visible_bonus(candidate, progress=progress)
                    - _lemma_urgency_bonus(candidate, progress=progress)
                    - contextual_credit
                )
            ranked.append(
                (
                    weighted_distance,
                    target_lemma_distance,
                    effective_new_family_count,
                    -known_context_mass,
                    -candidate.score,
                    tuple(item.candidate_id for item in replaceable_conflicts),
                    candidate,
                )
            )
        if not ranked:
            break
        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5], item[6].candidate_id))
        best_weighted_distance, best_target_lemma_distance, _eff_new, _known_mass, _score, replaceable_ids, best_candidate = ranked[0]
        if granularity_policy in {"smooth", "visible-smooth", "reader-visible-late", "cumulative-simple"}:
            slack = _smooth_acceptance_slack(best_candidate, progress=progress)
            if granularity_policy == "visible-smooth":
                slack = max(slack, 0.02 + ((1.0 - progress) * 0.035))
            elif granularity_policy == "cumulative-simple":
                slack = 0.0
            elif granularity_policy == "reader-visible-late":
                slack = max(slack, 0.004 + ((progress**2.1) * 0.028))
            current_acceptance_distance = current_distance if granularity_policy in {"visible-smooth", "cumulative-simple"} else current_target_lemma_distance
            if granularity_policy == "cumulative-simple":
                if best_weighted_distance >= current_acceptance_distance:
                    break
            elif best_weighted_distance > current_acceptance_distance + slack:
                break
        elif best_target_lemma_distance > current_distance:
                break
        if granularity_policy == "cumulative-simple" and replaceable_ids:
            removed = [candidate for candidate in current_selected if candidate.candidate_id in set(replaceable_ids)]
            removed_ids.update(item.candidate_id for item in removed if item.candidate_id in initially_selected_ids)
            current_selected = [candidate for candidate in current_selected if candidate.candidate_id not in set(replaceable_ids)]
        chosen.append(best_candidate)
        current_selected.append(best_candidate)
        paragraph_families.update(best_candidate.family_ids)
        paragraph_target_lemmas.update(_candidate_cost_target_lemmas(best_candidate))
        if granularity_policy == "visible-smooth":
            visible_mass += _candidate_visible_mass(best_candidate)
            current_distance = abs(_visible_czechness(visible_mass, seen_word_count_after) - target_after)
        elif granularity_policy == "cumulative-simple":
            covered_target_tokens = _covered_target_tokens(
                [candidate for candidate in current_selected if candidate.candidate_id != best_candidate.candidate_id]
            )
            simple_mass += _candidate_marginal_target_simple_mass(
                best_candidate,
                covered_target_tokens=covered_target_tokens,
            )
            current_distance = abs(_simple_czechness(simple_mass, seen_word_count_after) - target_after)
        else:
            current_distance = best_target_lemma_distance
        current_target_lemma_distance = best_target_lemma_distance
        available = [candidate for candidate in available if candidate.candidate_id != best_candidate.candidate_id]

    return chosen, removed_ids


def _future_czechness(
    remaining_after: Counter[str],
    introduced_families: set[str],
    *,
    family_weight_lookup: dict[str, float] | None = None,
) -> float:
    family_weight_lookup = family_weight_lookup or {}
    total = sum(count * family_weight_lookup.get(family_id, 1.0) for family_id, count in remaining_after.items())
    if total <= 0:
        return 1.0
    covered = sum(
        count * family_weight_lookup.get(family_id, 1.0)
        for family_id, count in remaining_after.items()
        if family_id in introduced_families
    )
    return covered / total


def _target_future_czechness(
    remaining_target_after: Counter[str],
    introduced_target_lemmas: set[str],
) -> float:
    total = sum(int(count) for count in remaining_target_after.values())
    if total <= 0:
        return 1.0
    covered = sum(
        int(count)
        for lemma, count in remaining_target_after.items()
        if lemma in introduced_target_lemmas
    )
    return covered / total


def _visible_czechness(visible_mass: float, total_word_count: int) -> float:
    if total_word_count <= 0:
        return 0.0
    return min(1.0, max(0.0, visible_mass / total_word_count))


def _visible_target(progress: float, power: float) -> float:
    progress = max(0.0, min(1.0, progress))
    return min(0.92, 0.02 + (0.90 * (progress**power)))


def _selected_visible_mass(candidates: list[ParagraphHybridCandidate]) -> float:
    return sum(_candidate_visible_mass(candidate) for candidate in candidates)


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


def _simple_czechness(simple_mass: float, total_word_count: int) -> float:
    if total_word_count <= 0:
        return 0.0
    return min(1.0, max(0.0, simple_mass / total_word_count))


def _candidate_simple_mass(candidate: ParagraphHybridCandidate) -> float:
    return max(1.0, float(candidate.metadata.get("source_word_count", _word_count(candidate.source_text))))


def _target_span_word_count(candidate: ParagraphHybridCandidate) -> int:
    return max(1, int(candidate.metadata.get("target_word_count", _word_count(candidate.target_text))))


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


def _covered_target_tokens(candidates: list[ParagraphHybridCandidate]) -> set[tuple[str, int]]:
    covered: set[tuple[str, int]] = set()
    for candidate in candidates:
        covered.update(_target_token_keys(candidate))
    return covered


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


def _promote_large_carriers(
    *,
    selected_candidates: list[ParagraphHybridCandidate],
    candidates: list[ParagraphHybridCandidate],
    introduced_target_lemmas: set[str],
    progress: float,
) -> list[ParagraphHybridCandidate]:
    current = list(selected_candidates)
    by_id = {candidate.candidate_id: candidate for candidate in current}
    candidate_pool = sorted(
        [
            candidate
            for candidate in candidates
            if candidate.granularity in {"sentence", "paragraph"}
            and candidate.candidate_id not in by_id
            and _candidate_allowed_as_surface_carrier(candidate)
        ],
        key=lambda item: (
            item.scope_id,
            item.granularity != "paragraph",
            -float(_target_span_word_count(item)),
            -item.score,
            item.candidate_id,
        ),
    )

    for candidate in candidate_pool:
        if candidate.granularity == "paragraph":
            pool_selected = list(current)
        else:
            pool_selected = [item for item in current if item.scope_id == candidate.scope_id]
        if len(pool_selected) < 1:
            continue
        conflicts = [item for item in pool_selected if _conflicts(candidate, item)]
        if len(conflicts) < 1:
            continue
        if any(not _candidate_contains(candidate, item) for item in conflicts):
            continue
        covered = _covered_target_tokens(conflicts)
        target_tokens = _target_token_keys(candidate)
        if not target_tokens:
            continue
        coverage_ratio = len(covered & target_tokens) / max(len(target_tokens), 1)
        if coverage_ratio < _promotion_coverage_threshold(progress):
            continue
        base_lemmas = set(introduced_target_lemmas)
        for item in conflicts:
            base_lemmas.update(_candidate_cost_target_lemmas(item))
        conflict_ids = {item.candidate_id for item in conflicts}
        current = [item for item in current if item.candidate_id not in conflict_ids]
        current.append(candidate)
    current.sort(key=lambda item: item.candidate_id)
    return current


def _promotion_coverage_threshold(progress: float) -> float:
    _ = progress
    return 0.89


def _candidate_visible_mass(candidate: ParagraphHybridCandidate) -> float:
    source_words = max(1, int(candidate.metadata.get("source_word_count", _word_count(candidate.source_text))))
    similarity = _text_similarity(candidate.source_text, candidate.target_text)
    score = max(0.0, min(1.0, float(candidate.score)))
    visibility = max(0.0, 1.0 - similarity)
    confidence = 0.55 + (0.45 * score)
    return source_words * visibility * confidence


def _late_visible_bonus(candidate: ParagraphHybridCandidate, *, progress: float) -> float:
    progress = max(0.0, min(1.0, progress))
    visible_mass = _candidate_visible_mass(candidate)
    normalized_mass = min(1.0, visible_mass / 4.0)
    if candidate.granularity == "token":
        scale = 0.035
    elif candidate.granularity in {"phrase", "subtree"}:
        scale = 0.085
    elif candidate.granularity == "sentence":
        scale = 0.12
    elif candidate.granularity == "paragraph":
        scale = 0.16
    else:
        scale = 0.04
    return normalized_mass * (progress**2.4) * scale


def _candidate_idiomaticity(candidate: ParagraphHybridCandidate) -> float:
    if candidate.granularity not in {"phrase", "subtree", "sentence", "paragraph"}:
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


def _effective_new_family_count(candidate: ParagraphHybridCandidate, *, new_families: list[str]) -> float:
    unique_count = float(len(set(new_families)))
    if unique_count <= 0.0:
        return 0.0
    idiomaticity = _candidate_idiomaticity(candidate)
    if idiomaticity >= 0.45 and candidate.granularity in {"phrase", "subtree", "sentence"}:
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
    if candidate.granularity in {"phrase", "subtree", "sentence", "paragraph"} and effective_count <= 1.0:
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
        candidate.granularity in {"phrase", "subtree"}
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
    if candidate.granularity not in {"phrase", "subtree", "sentence", "paragraph"}:
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


def _candidate_allowed_by_safety(candidate: ParagraphHybridCandidate) -> bool:
    if candidate.granularity == "token":
        return True
    score = float(candidate.score)
    support = _structural_support(candidate)
    semantic = _semantic_confidence(candidate)
    similarity = _text_similarity(candidate.source_text, candidate.target_text)
    width = max(_word_count(candidate.source_text), _word_count(candidate.target_text), 1)
    if candidate.granularity in {"phrase", "subtree"}:
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
    if candidate.granularity in {"phrase", "subtree", "sentence", "paragraph"}:
        uncovered = [lemma for lemma in unique_lemmas if lemma not in _covered_target_lemmas(candidate)]
        if uncovered:
            penalty += 0.012 * len(uncovered)
    return penalty


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




def _apply_chapter_simple_boost(
    *,
    plan_units: list[dict[str, object]],
    units: list[dict[str, object]],
    target_power: float,
    target_max: float,
    boost_start: float,
) -> None:
    unit_lookup = {int(unit["unit_index"]): unit for unit in units}
    chapter_groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for plan_unit in plan_units:
        chapter_groups[int(plan_unit["chapter_pair"][0])].append(plan_unit)

    chapter_indices = sorted(chapter_groups)
    total_chapters = max(len(chapter_indices), 1)

    for ordinal, chapter_index in enumerate(chapter_indices, start=1):
        progress = ordinal / total_chapters
        if progress < boost_start:
            continue
        target_simple = min(1.0, target_max * (progress**target_power))
        chapter_units = chapter_groups[chapter_index]
        chapter_word_count = sum(int(unit.get("source_word_count", 0)) for unit in chapter_units)
        if chapter_word_count <= 0:
            continue

        current_simple_mass = sum(
            _candidate_simple_mass(_candidate_from_dict(candidate))
            for unit in chapter_units
            for candidate in unit.get("selected_candidates", [])
        )
        current_simple = current_simple_mass / chapter_word_count
        if current_simple >= target_simple:
            continue

        while current_simple < target_simple:
            best_pick: tuple[float, dict[str, object], ParagraphHybridCandidate, list[ParagraphHybridCandidate]] | None = None
            for plan_unit in chapter_units:
                unit_data = unit_lookup.get(int(plan_unit["unit_index"]))
                if unit_data is None:
                    continue
                selected = [_candidate_from_dict(item) for item in plan_unit.get("selected_candidates", [])]
                selected_ids = {candidate.candidate_id for candidate in selected}
                selected_target_lemmas = {
                    lemma
                    for selected_candidate in selected
                    for lemma in _candidate_cost_target_lemmas(selected_candidate)
                }
                for candidate in unit_data.get("candidates", []):
                    if candidate.is_fallback or not candidate.family_ids or candidate.candidate_id in selected_ids:
                        continue
                    if not _candidate_allowed_by_progress(candidate, progress=max(progress, boost_start)):
                        continue
                    conflicting = [other for other in selected if _conflicts(candidate, other)]
                    gain = _candidate_simple_mass(candidate) - sum(_candidate_simple_mass(other) for other in conflicting)
                    visible_delta = _candidate_visible_mass(candidate) - sum(_candidate_visible_mass(other) for other in conflicting)
                    if gain <= 0 and visible_delta <= 0:
                        continue
                    penalty = (
                        _confidence_penalty(candidate)
                        + _uncovered_content_penalty(
                            candidate,
                            paragraph_target_lemmas=selected_target_lemmas,
                            progress=progress,
                        )
                        + _local_scope_penalty(
                            candidate,
                            candidates=unit_data.get("candidates", []),
                            paragraph_families={family_id for selected_candidate in selected for family_id in selected_candidate.family_ids},
                            paragraph_target_lemmas=selected_target_lemmas,
                        )
                        + (0.5 * _granularity_progress_penalty(candidate, progress=progress))
                    )
                    value = gain + (0.18 * visible_delta) - (9.0 * penalty)
                    if candidate.granularity == "paragraph" and progress < 0.82:
                        value -= 0.7
                    if candidate.granularity == "sentence" and progress < 0.72:
                        value -= 0.2
                    if best_pick is None or value > best_pick[0]:
                        best_pick = (value, plan_unit, candidate, conflicting)
            if best_pick is None or best_pick[0] <= 0.0:
                break
            _value, plan_unit, candidate, conflicting = best_pick
            conflicting_ids = {item.candidate_id for item in conflicting}
            selected_payloads = [
                item for item in plan_unit.get("selected_candidates", [])
                if str(item["candidate_id"]) not in conflicting_ids
            ]
            selected_payloads.append(candidate.to_dict())
            plan_unit["selected_candidates"] = selected_payloads
            plan_unit["selected_count"] = len(plan_unit["selected_candidates"])
            plan_unit["selected_granularity_counts"] = dict(
                Counter(item["granularity"] for item in plan_unit["selected_candidates"])
            )
            current_simple_mass += _candidate_simple_mass(candidate) - sum(_candidate_simple_mass(other) for other in conflicting)
            current_simple = current_simple_mass / chapter_word_count


def _refresh_plan_metrics(
    *,
    plan_units: list[dict[str, object]],
    units: list[dict[str, object]],
    total_occurrence_count: int,
    total_word_count: int,
    family_weight_lookup: dict[str, float],
) -> None:
    unit_lookup = {int(unit["unit_index"]): unit for unit in units}
    remaining_occurrences = Counter()
    for unit in units:
        remaining_occurrences.update(unit["family_occurrences"])

    introduced_families: set[str] = set()
    cumulative_occurrence_count = 0
    cumulative_word_count = 0
    cumulative_visible_mass = 0.0

    for plan_unit in plan_units:
        unit_data = unit_lookup[int(plan_unit["unit_index"])]
        cumulative_occurrence_count += int(unit_data["source_occurrence_count"])
        cumulative_word_count += int(unit_data["source_word_count"])
        remaining_after = remaining_occurrences.copy()
        remaining_after.subtract(unit_data["family_occurrences"])
        remaining_after = Counter({key: value for key, value in remaining_after.items() if value > 0})
        actual_before = _future_czechness(remaining_after, introduced_families, family_weight_lookup=family_weight_lookup)
        visible_before = _visible_czechness(cumulative_visible_mass, cumulative_word_count)
        selected_candidates = [_candidate_from_dict(item) for item in plan_unit.get("selected_candidates", [])]
        for candidate in selected_candidates:
            introduced_families.update(candidate.family_ids)
        cumulative_visible_mass += _selected_visible_mass(selected_candidates)
        actual_after = _future_czechness(remaining_after, introduced_families, family_weight_lookup=family_weight_lookup)
        visible_after = _visible_czechness(cumulative_visible_mass, cumulative_word_count)
        plan_unit["selected_count"] = len(selected_candidates)
        plan_unit["selected_granularity_counts"] = dict(Counter(candidate.granularity for candidate in selected_candidates))
        plan_unit["actual_future_czechness_before"] = round(actual_before, 6)
        plan_unit["actual_future_czechness_after"] = round(actual_after, 6)
        plan_unit["actual_visible_czechness_before"] = round(visible_before, 6)
        plan_unit["actual_visible_czechness_after"] = round(visible_after, 6)
        plan_unit["visible_mass_after"] = round(cumulative_visible_mass, 6)
        plan_unit["distance_after"] = round(abs(actual_after - float(plan_unit["target_future_czechness_after"])), 6)
        plan_unit["introduced_family_count"] = len(introduced_families)
        remaining_occurrences = remaining_after


def _candidate_from_dict(payload: dict[str, object]) -> ParagraphHybridCandidate:
    return ParagraphHybridCandidate(
        candidate_id=str(payload["candidate_id"]),
        chapter_pair=(int(payload["chapter_pair"][0]), int(payload["chapter_pair"][1])),
        unit_index=int(payload["unit_index"]),
        granularity=str(payload["granularity"]),
        scope_id=str(payload["scope_id"]),
        source_span=(int(payload["source_span"][0]), int(payload["source_span"][1])),
        target_span=(int(payload["target_span"][0]), int(payload["target_span"][1])),
        source_text=str(payload["source_text"]),
        target_text=str(payload["target_text"]),
        family_ids=[str(item) for item in payload.get("family_ids", [])],
        score=float(payload.get("score", 0.0)),
        relation=str(payload.get("relation", "")),
        metadata=dict(payload.get("metadata", {})),
    )


def _conflicts(left: ParagraphHybridCandidate, right: ParagraphHybridCandidate) -> bool:
    if left.scope_id == "block" or right.scope_id == "block":
        return True
    if left.scope_id != right.scope_id:
        return False
    if left.granularity == "sentence" or right.granularity == "sentence":
        return True
    return _spans_overlap(left.source_span, right.source_span) or _spans_overlap(left.target_span, right.target_span)


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return not (left[1] < right[0] or right[1] < left[0])


def _span_contains(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] >= inner[1]


def _candidate_contains(outer: ParagraphHybridCandidate, inner: ParagraphHybridCandidate) -> bool:
    return _span_contains(outer.source_span, inner.source_span) and _span_contains(outer.target_span, inner.target_span)


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
    if candidate.granularity in {"phrase", "subtree", "sentence", "paragraph"} and width <= 1.0:
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
    if candidate.granularity in {"phrase", "subtree"} and target_content_lemmas <= 1 and target_functional_context > 0:
        penalty *= 0.35
    return penalty


def _candidate_allowed_by_progress(candidate: ParagraphHybridCandidate, *, progress: float) -> bool:
    progress = max(0.0, min(1.0, progress))
    if candidate.granularity == "token":
        return _token_allowed_by_progress(candidate, progress=progress)
    if candidate.granularity in {"phrase", "subtree"}:
        return _structural_candidate_allowed(candidate, progress=progress)
    if candidate.granularity == "sentence":
        if progress < 0.55:
            return False
        return candidate.score >= 0.62
    if candidate.granularity == "paragraph":
        if progress < 0.78:
            return False
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


def _idiomaticity_penalty(
    candidate: ParagraphHybridCandidate,
    *,
    progress: float,
    idiomaticity_penalty_weight: float,
) -> float:
    if idiomaticity_penalty_weight <= 0:
        return 0.0
    if candidate.granularity not in {"phrase", "subtree"}:
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

    if candidate.granularity in {"phrase", "subtree"}:
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
    if candidate.granularity in {"phrase", "subtree"}:
        width = max(_word_count(candidate.source_text), _word_count(candidate.target_text), 1)
        compactness = max(0.0, 1.0 - ((width - 1) / 6.0))
        return 0.0005 + (compactness * (phase**1.45) * 0.003) + ((progress**1.8) * 0.012)
    if candidate.granularity == "sentence":
        return 0.001 + ((progress**2.0) * 0.014)
    if candidate.granularity == "paragraph":
        return 0.001 + ((progress**2.2) * 0.02)
    return 0.001


def _text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(a=left.strip().lower(), b=right.strip().lower()).ratio()


def _word_count(text: str) -> int:
    return len([part for part in text.split() if part.strip()])
