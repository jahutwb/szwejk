"""Local czechness metrics derived from paragraph hybridization plans."""

from __future__ import annotations

from collections import Counter, defaultdict

from szwejk.generate.paragraph_hybridization import ParagraphHybridCandidate, _candidate_idiomaticity, _candidate_visible_mass


def build_czechness_report(
    alignment_artifact: dict[str, object],
    paragraph_plan: dict[str, object],
) -> dict[str, object]:
    block_metrics = _block_metrics(alignment_artifact)
    introduced: set[str] = set()
    paragraph_reports: list[dict[str, object]] = []
    chapter_buckets: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)

    for unit in paragraph_plan.get("units", []):
        chapter_pair = (int(unit["chapter_pair"][0]), int(unit["chapter_pair"][1]))
        key = (chapter_pair, int(unit["unit_index"]))
        metrics = block_metrics.get(key, {"occurrences": Counter(), "source_word_count": 0})
        occurrences = metrics["occurrences"]
        total = sum(occurrences.values())
        before = _coverage_ratio(occurrences, introduced)
        for candidate in unit.get("selected_candidates", []):
            introduced.update(str(item) for item in candidate.get("family_ids", []))
        after = _coverage_ratio(occurrences, introduced)
        source_word_count = int(metrics["source_word_count"])
        granularity_visible_masses = _granularity_visible_masses(unit.get("selected_candidates", []))
        granularity_simple_masses = _granularity_simple_masses(unit.get("selected_candidates", []))
        idiom_visible_mass, idiom_simple_mass, idiom_count = _idiom_metrics(unit.get("selected_candidates", []))
        total_visible_mass = sum(granularity_visible_masses.values())
        total_simple_mass = sum(granularity_simple_masses.values())
        paragraph_report = {
            "unit_index": int(unit["unit_index"]),
            "chapter_pair": list(chapter_pair),
            "target_future_czechness_after": float(unit["target_future_czechness_after"]),
            "actual_future_czechness_after": float(unit["actual_future_czechness_after"]),
            "local_czechness_before": round(before, 6),
            "local_czechness_after": round(after, 6),
            "source_word_count": source_word_count,
            "local_simple_czechness_after": round(_simple_ratio(total_simple_mass, source_word_count), 6),
            "local_visible_czechness_after": round(_visible_ratio(total_visible_mass, source_word_count), 6),
            "simple_mass": round(total_simple_mass, 6),
            "simple_mass_by_granularity": {key: round(value, 6) for key, value in granularity_simple_masses.items()},
            "idiom_simple_mass": round(idiom_simple_mass, 6),
            "visible_mass": round(total_visible_mass, 6),
            "visible_mass_by_granularity": {key: round(value, 6) for key, value in granularity_visible_masses.items()},
            "idiom_visible_mass": round(idiom_visible_mass, 6),
            "idiom_selected_count": idiom_count,
            "family_occurrence_count": total,
            "selected_count": int(unit["selected_count"]),
            "selected_granularity_counts": dict(unit["selected_granularity_counts"]),
        }
        paragraph_reports.append(paragraph_report)
        chapter_buckets[chapter_pair].append(paragraph_report)

    chapter_reports = []
    for chapter_pair, reports in sorted(chapter_buckets.items()):
        total_occurrences = sum(int(item["family_occurrence_count"]) for item in reports)
        total_words = sum(int(item["source_word_count"]) for item in reports)
        weighted_before = _weighted_average(reports, "local_czechness_before", "family_occurrence_count")
        weighted_after = _weighted_average(reports, "local_czechness_after", "family_occurrence_count")
        simple_after = _weighted_average(reports, "local_simple_czechness_after", "source_word_count")
        visible_after = _weighted_average(reports, "local_visible_czechness_after", "source_word_count")
        simple_mass_total = sum(float(item["simple_mass"]) for item in reports)
        visible_mass_total = sum(float(item["visible_mass"]) for item in reports)
        idiom_simple_total = sum(float(item.get("idiom_simple_mass", 0.0)) for item in reports)
        idiom_visible_total = sum(float(item.get("idiom_visible_mass", 0.0)) for item in reports)
        idiom_selected_count = sum(int(item.get("idiom_selected_count", 0)) for item in reports)
        simple_by_granularity = Counter()
        visible_by_granularity = Counter()
        for item in reports:
            simple_by_granularity.update(dict(item["simple_mass_by_granularity"]))
            visible_by_granularity.update(dict(item["visible_mass_by_granularity"]))
        chapter_reports.append(
            {
                "chapter_pair": list(chapter_pair),
                "paragraph_block_count": len(reports),
                "family_occurrence_count": total_occurrences,
                "source_word_count": total_words,
                "local_czechness_before": round(weighted_before, 6),
                "local_czechness_after": round(weighted_after, 6),
                "local_simple_czechness_after": round(simple_after, 6),
                "local_visible_czechness_after": round(visible_after, 6),
                "simple_mass": round(simple_mass_total, 6),
                "simple_mass_by_granularity": {key: round(value, 6) for key, value in sorted(simple_by_granularity.items())},
                "simple_share_by_granularity": {
                    key: round(value / simple_mass_total, 6) if simple_mass_total > 0 else 0.0
                    for key, value in sorted(simple_by_granularity.items())
                },
                "idiom_simple_mass": round(idiom_simple_total, 6),
                "idiom_simple_share": round(idiom_simple_total / simple_mass_total, 6) if simple_mass_total > 0 else 0.0,
                "visible_mass": round(visible_mass_total, 6),
                "visible_mass_by_granularity": {key: round(value, 6) for key, value in sorted(visible_by_granularity.items())},
                "visible_share_by_granularity": {
                    key: round(value / visible_mass_total, 6) if visible_mass_total > 0 else 0.0
                    for key, value in sorted(visible_by_granularity.items())
                },
                "idiom_visible_mass": round(idiom_visible_total, 6),
                "idiom_visible_share": round(idiom_visible_total / visible_mass_total, 6) if visible_mass_total > 0 else 0.0,
                "idiom_selected_count": idiom_selected_count,
                "avg_target_future_czechness_after": round(sum(float(item["target_future_czechness_after"]) for item in reports) / max(len(reports), 1), 6),
                "avg_actual_future_czechness_after": round(sum(float(item["actual_future_czechness_after"]) for item in reports) / max(len(reports), 1), 6),
            }
        )

    return {
        "paragraph_reports": paragraph_reports,
        "chapter_reports": chapter_reports,
    }


def _block_metrics(alignment_artifact: dict[str, object]) -> dict[tuple[tuple[int, int], int], dict[str, object]]:
    buckets: dict[tuple[tuple[int, int], int], dict[str, object]] = {}
    unit_index = 0
    for pair_report in alignment_artifact.get("pair_reports", []):
        chapter_pair = (int(pair_report["source_chapter_index"]), int(pair_report["target_chapter_index"]))
        item_lookup = defaultdict(list)
        for item in pair_report["enrichment_bundle"]["items"]:
            item_lookup[(int(item["source_paragraph_index"]), int(item["target_paragraph_index"]))].append(item)
        for block in pair_report["paragraph_alignment"]["matched_blocks"]:
            unit_index += 1
            source_start, source_end = int(block["source_range"][0]), int(block["source_range"][1])
            target_start, target_end = int(block["target_range"][0]), int(block["target_range"][1])
            counter: Counter[str] = Counter()
            source_word_count = 0
            for source_idx in range(source_start, source_end + 1):
                for target_idx in range(target_start, target_end + 1):
                    for item in item_lookup.get((source_idx, target_idx), []):
                        if item["enrichment_status"] != "sentence_safe":
                            continue
                        source_word_count += sum(1 for token in item.get("source_tokens", []) if token.get("kind") == "word")
                        for pair in item.get("token_pairs", []):
                            family_id = _family_id_for_token_pair(pair)
                            if family_id:
                                counter[family_id] += 1
            buckets[(chapter_pair, unit_index)] = {
                "occurrences": counter,
                "source_word_count": source_word_count,
            }
    return buckets


def _family_id_for_token_pair(pair: dict[str, object]) -> str | None:
    source_lemma = str(pair.get("source_lemma", "")).strip().lower()
    target_lemma = str(pair.get("target_lemma", "")).strip().lower()
    if not source_lemma or not target_lemma:
        return None
    return f"{source_lemma}::{target_lemma}"


def _coverage_ratio(occurrences: Counter[str], introduced: set[str]) -> float:
    total = sum(occurrences.values())
    if total <= 0:
        return 0.0
    covered = sum(count for family_id, count in occurrences.items() if family_id in introduced)
    return covered / total


def _weighted_average(items: list[dict[str, object]], value_key: str, weight_key: str) -> float:
    total_weight = sum(float(item[weight_key]) for item in items)
    if total_weight <= 0:
        return 0.0
    return sum(float(item[value_key]) * float(item[weight_key]) for item in items) / total_weight


def _granularity_visible_masses(selected_candidates: list[dict[str, object]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for candidate in selected_candidates:
        wrapped = ParagraphHybridCandidate(
            candidate_id=str(candidate["candidate_id"]),
            chapter_pair=(int(candidate["chapter_pair"][0]), int(candidate["chapter_pair"][1])),
            unit_index=int(candidate["unit_index"]),
            granularity=str(candidate["granularity"]),
            scope_id=str(candidate["scope_id"]),
            source_span=(int(candidate["source_span"][0]), int(candidate["source_span"][1])),
            target_span=(int(candidate["target_span"][0]), int(candidate["target_span"][1])),
            source_text=str(candidate["source_text"]),
            target_text=str(candidate["target_text"]),
            family_ids=[str(item) for item in candidate.get("family_ids", [])],
            score=float(candidate.get("score", 0.0)),
            relation=str(candidate.get("relation", "")),
            metadata=dict(candidate.get("metadata", {})),
        )
        counter[wrapped.granularity] += _candidate_visible_mass(wrapped)
    return counter


def _granularity_simple_masses(selected_candidates: list[dict[str, object]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for candidate in selected_candidates:
        granularity = str(candidate["granularity"])
        source_words = max(1, int(candidate.get("metadata", {}).get("source_word_count", 0) or _word_count(str(candidate["source_text"]))))
        counter[granularity] += source_words
    return counter


def _idiom_metrics(selected_candidates: list[dict[str, object]]) -> tuple[float, float, int]:
    visible_mass = 0.0
    simple_mass = 0.0
    count = 0
    for candidate in selected_candidates:
        wrapped = ParagraphHybridCandidate(
            candidate_id=str(candidate["candidate_id"]),
            chapter_pair=(int(candidate["chapter_pair"][0]), int(candidate["chapter_pair"][1])),
            unit_index=int(candidate["unit_index"]),
            granularity=str(candidate["granularity"]),
            scope_id=str(candidate["scope_id"]),
            source_span=(int(candidate["source_span"][0]), int(candidate["source_span"][1])),
            target_span=(int(candidate["target_span"][0]), int(candidate["target_span"][1])),
            source_text=str(candidate["source_text"]),
            target_text=str(candidate["target_text"]),
            family_ids=[str(item) for item in candidate.get("family_ids", [])],
            score=float(candidate.get("score", 0.0)),
            relation=str(candidate.get("relation", "")),
            metadata=dict(candidate.get("metadata", {})),
        )
        if _candidate_idiomaticity(wrapped) < 0.45:
            continue
        count += 1
        visible_mass += _candidate_visible_mass(wrapped)
        simple_mass += max(1.0, float(wrapped.metadata.get("source_word_count", _word_count(wrapped.source_text))))
    return visible_mass, simple_mass, count


def _visible_ratio(visible_mass: float, source_word_count: int) -> float:
    if source_word_count <= 0:
        return 0.0
    return min(1.0, max(0.0, visible_mass / source_word_count))


def _simple_ratio(simple_mass: float, source_word_count: int) -> float:
    if source_word_count <= 0:
        return 0.0
    return min(1.0, max(0.0, simple_mass / source_word_count))


def _word_count(text: str) -> int:
    return len([part for part in text.split() if part.strip()])
