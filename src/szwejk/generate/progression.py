"""Continuous progression engine over a global stream of aligned candidates."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from bisect import bisect_left
import re

from szwejk.align.enrichment import build_sentence_enrichment_bundle
from szwejk.align.sentences import align_chapter_sentences, align_chapter_sentences_with_paragraph_blocks
from szwejk.schemas import CanonicalBook

from szwejk.policy.actions import build_action_inventory
from .corpus_scope import scoped_source_chapter


FREE_RELATIONS = {"exact", "lemma_like"}
MIN_ALIGNMENT_CONFIDENCE = {
    "token": 0.5,
    "phrase": 0.74,
    "subtree": 0.6,
}
CONFIDENCE_PENALTY_WEIGHT = {
    "token": 0.14,
    "phrase": 0.08,
    "subtree": 0.08,
}
CONTENT_UPOS_WEIGHTS = {
    "NOUN": 1.0,
    "VERB": 1.0,
    "PROPN": 1.0,
    "ADJ": 0.7,
    "ADV": 0.7,
    "NUM": 0.65,
    "PRON": 0.45,
    "ADP": 0.25,
    "CCONJ": 0.15,
    "SCONJ": 0.15,
    "PART": 0.1,
    "AUX": 0.35,
    "DET": 0.2,
    "INTJ": 0.2,
}


@dataclass(slots=True)
class ProgressionMetrics:
    surface: float = 0.0
    novelty: float = 0.0
    difficulty: float = 0.0

    def add(self, other: "ProgressionMetrics") -> "ProgressionMetrics":
        return ProgressionMetrics(
            surface=self.surface + other.surface,
            novelty=self.novelty + other.novelty,
            difficulty=self.difficulty + other.difficulty,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "surface": round(self.surface, 6),
            "novelty": round(self.novelty, 6),
            "difficulty": round(self.difficulty, 6),
        }


@dataclass(slots=True)
class ProgressionCandidate:
    chapter_pair: tuple[int, int]
    sentence_index: int
    granularity: str
    source_text: str
    target_text: str
    source_span: tuple[int, int]
    target_span: tuple[int, int]
    global_source_start: int
    global_source_end: int
    global_target_start: int
    global_target_end: int
    family_ids: list[str]
    score: float
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "chapter_pair": list(self.chapter_pair),
            "sentence_index": self.sentence_index,
            "granularity": self.granularity,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "source_span": list(self.source_span),
            "target_span": list(self.target_span),
            "global_source_start": self.global_source_start,
            "global_source_end": self.global_source_end,
            "global_target_start": self.global_target_start,
            "global_target_end": self.global_target_end,
            "family_ids": self.family_ids,
            "score": round(self.score, 6),
            "metadata": self.metadata,
        }


def build_progression_report(
    source_book: CanonicalBook,
    target_book: CanonicalBook,
    *,
    chapter_pairs: list[tuple[int, int]],
    paragraph_window: int = 3,
    analysis_mode: str = "stanza",
    paragraph_pair_reports: dict[tuple[int, int], dict[str, object]] | None = None,
) -> dict[str, object]:
    candidates, unmatched_regions, total_source_positions, total_target_positions = build_progression_candidates(
        source_book,
        target_book,
        chapter_pairs=chapter_pairs,
        paragraph_window=paragraph_window,
        analysis_mode=analysis_mode,
        paragraph_pair_reports=paragraph_pair_reports,
    )
    steps, family_weights, cursor_gaps = select_progression_steps(
        candidates,
        total_source_positions=total_source_positions,
        total_target_positions=total_target_positions,
    )
    return {
        "chapter_pairs": [list(item) for item in chapter_pairs],
        "analysis_mode": analysis_mode,
        "total_source_positions": total_source_positions,
        "total_target_positions": total_target_positions,
        "candidate_count": len(candidates),
        "selected_step_count": len(steps),
        "unmatched_regions": unmatched_regions,
        "cursor_gaps": cursor_gaps,
        "family_weights": {key: round(value, 6) for key, value in family_weights.items()},
        "steps": steps,
    }


def build_progression_candidates(
    source_book: CanonicalBook,
    target_book: CanonicalBook,
    *,
    chapter_pairs: list[tuple[int, int]],
    paragraph_window: int = 3,
    analysis_mode: str = "stanza",
    paragraph_pair_reports: dict[tuple[int, int], dict[str, object]] | None = None,
) -> tuple[list[ProgressionCandidate], list[dict[str, object]], int, int]:
    candidates: list[ProgressionCandidate] = []
    unmatched_regions: list[dict[str, object]] = []
    source_chapter_offsets, total_source_positions = _chapter_word_offsets(
        source_book,
        included_chapters=[source for source, _target in chapter_pairs],
        scope_source=True,
    )
    target_chapter_offsets, total_target_positions = _chapter_word_offsets(
        target_book,
        included_chapters=[target for _source, target in chapter_pairs],
        scope_source=False,
    )

    for source_chapter_index, target_chapter_index in chapter_pairs:
        global_source_cursor = source_chapter_offsets[source_chapter_index]
        global_target_cursor = target_chapter_offsets[target_chapter_index]
        source_chapter = scoped_source_chapter(source_book.chapters[source_chapter_index - 1])
        target_chapter = target_book.chapters[target_chapter_index - 1]
        paragraph_pair_report = None if paragraph_pair_reports is None else paragraph_pair_reports.get((source_chapter_index, target_chapter_index))
        if paragraph_pair_report is not None:
            chapter_alignments = align_chapter_sentences_with_paragraph_blocks(
                source_chapter,
                target_chapter,
                paragraph_blocks=list(paragraph_pair_report.get("matched_blocks", [])),
            )
        else:
            chapter_alignments = align_chapter_sentences(
                source_chapter,
                target_chapter,
                paragraph_window=paragraph_window,
            )
        enrichment_bundle = build_sentence_enrichment_bundle(
            chapter_alignments=chapter_alignments,
            source_language=source_book.metadata.language,
            target_language=target_book.metadata.language,
            analysis_mode=analysis_mode,
        )
        action_inventory = build_action_inventory(enrichment_bundle)
        by_sentence_structural = _actions_by_sentence(action_inventory)
        safe_items = [item for item in enrichment_bundle["items"] if item["enrichment_status"] == "sentence_safe"]

        for sentence_index, item in enumerate(safe_items, start=1):
            source_words = [token for token in item["source_tokens"] if token["kind"] == "word"]
            target_words = [token for token in item["target_tokens"] if token["kind"] == "word"]
            source_global_lookup = {
                int(token["index"]): global_source_cursor + offset
                for offset, token in enumerate(source_words)
            }
            target_global_lookup = {
                int(token["index"]): global_target_cursor + offset
                for offset, token in enumerate(target_words)
            }

            sentence_actions = by_sentence_structural.get(sentence_index, [])
            if not sentence_actions:
                unmatched_regions.append(
                    {
                        "chapter_pair": [source_chapter_index, target_chapter_index],
                        "sentence_index": sentence_index,
                        "source_text": item["sentence_alignment"]["source_text"],
                        "target_text": item["sentence_alignment"]["target_text"],
                        "reason": "no_structural_candidates",
                        "global_source_start": global_source_cursor,
                        "global_source_end": global_source_cursor + max(len(source_words) - 1, 0),
                    }
                )

            for action in sentence_actions:
                if action["granularity"] in {"phrase", "subtree"} and not action["metadata"].get("family_ids"):
                    continue
                source_start = int(action["source_span"][0])
                source_end = int(action["source_span"][1])
                target_start = int(action["target_span"][0])
                target_end = int(action["target_span"][1])
                if source_start not in source_global_lookup or source_end not in source_global_lookup:
                    continue
                if target_start not in target_global_lookup or target_end not in target_global_lookup:
                    continue
                candidates.append(
                    ProgressionCandidate(
                        chapter_pair=(source_chapter_index, target_chapter_index),
                        sentence_index=sentence_index,
                        granularity=str(action["granularity"]),
                        source_text=str(action["source_text"]),
                        target_text=str(action["target_text"]),
                        source_span=(source_start, source_end),
                        target_span=(target_start, target_end),
                        global_source_start=source_global_lookup[source_start],
                        global_source_end=source_global_lookup[source_end],
                        global_target_start=target_global_lookup[target_start],
                        global_target_end=target_global_lookup[target_end],
                        family_ids=list(action["metadata"].get("family_ids", [])),
                        score=float(action["score"]),
                        metadata=dict(action["metadata"]),
                    )
                )

            global_source_cursor += len(source_words)
            global_target_cursor += len(target_words)

    return (
        sorted(candidates, key=lambda item: (item.global_source_start, item.global_source_end, item.granularity)),
        unmatched_regions,
        total_source_positions,
        total_target_positions,
    )


def select_progression_steps(
    candidates: list[ProgressionCandidate],
    *,
    total_source_positions: int,
    total_target_positions: int,
) -> tuple[list[dict[str, object]], dict[str, float], list[dict[str, object]]]:
    grouped: dict[int, list[ProgressionCandidate]] = defaultdict(list)
    family_occurrences = Counter()
    family_free = {}
    family_difficulty_profiles: dict[str, float] = {}
    family_occurrence_positions: dict[str, list[int]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.global_source_start].append(candidate)
        if candidate.granularity == "token":
            for family_id in candidate.family_ids:
                family_occurrence_positions[family_id].append(candidate.global_source_start)
        for family_id in candidate.family_ids:
            family_occurrences[family_id] += 1
            family_free.setdefault(family_id, _family_is_free(candidate, family_id))
            difficulty_profile = _candidate_difficulty_profile(candidate)
            previous_profile = family_difficulty_profiles.get(family_id)
            family_difficulty_profiles[family_id] = (
                difficulty_profile
                if previous_profile is None
                else min(previous_profile, difficulty_profile)
            )
    weighted_families = {
        family_id: count
        for family_id, count in family_occurrences.items()
        if not family_free.get(family_id, False)
    }
    total_weight = sum(weighted_families.values()) or 1.0
    family_weights = {family_id: count / total_weight for family_id, count in weighted_families.items()}
    weighted_difficulties = {
        family_id: weighted_families[family_id] * family_difficulty_profiles.get(family_id, 0.0)
        for family_id in weighted_families
    }
    total_difficulty_weight = sum(weighted_difficulties.values()) or 1.0
    family_difficulty_weights = {
        family_id: weighted_difficulties[family_id] / total_difficulty_weight
        for family_id in weighted_difficulties
    }
    for positions in family_occurrence_positions.values():
        positions.sort()

    introduced_families: set[str] = set()
    steps: list[dict[str, object]] = []
    cursor_gaps: list[dict[str, object]] = []
    actual = ProgressionMetrics()
    source_cursor = 1
    target_cursor = 1
    max_source_position = total_source_positions or 0
    max_target_position = total_target_positions or 0

    while source_cursor <= max_source_position and target_cursor <= max_target_position:
        options = _eligible_options(grouped, source_cursor=source_cursor, target_cursor=target_cursor)
        if not options:
            source_cursor += 1
            target_cursor += 1
            continue

        current_target = _target_metrics(
            source_cursor=source_cursor,
            target_cursor=target_cursor,
            total_source_positions=max_source_position,
            total_target_positions=max_target_position,
        )
        current_distance = _metrics_distance(actual, current_target)
        carry_options = []
        introducing_options = []
        for candidate in options:
            alignment_confidence = _alignment_confidence(candidate)
            if alignment_confidence < MIN_ALIGNMENT_CONFIDENCE.get(candidate.granularity, 0.6):
                continue
            new_families = [family_id for family_id in candidate.family_ids if family_id not in introduced_families]
            delta = ProgressionMetrics(
                surface=_candidate_surface_delta(candidate, total_source_positions=max_source_position),
                novelty=sum(family_weights.get(family_id, 0.0) for family_id in new_families),
                difficulty=sum(family_difficulty_weights.get(family_id, 0.0) for family_id in new_families),
            )
            after = actual.add(delta)
            confidence_penalty = 1.0 - alignment_confidence
            distance_after = _metrics_distance(after, current_target) + (
                CONFIDENCE_PENALTY_WEIGHT.get(candidate.granularity, 0.1) * confidence_penalty
            )
            option = {
                "candidate": candidate,
                "new_families": new_families,
                "delta": delta,
                "after": after,
                "distance_after": distance_after,
                "alignment_confidence": alignment_confidence,
            }
            if _is_free_carry_option(candidate, new_families=new_families, delta=delta):
                carry_options.append(option)
            else:
                introducing_options.append(option)

        chosen = None
        if carry_options:
            chosen = max(
                carry_options,
                key=lambda item: (
                    _granularity_rank(item["candidate"].granularity),
                    item["candidate"].global_source_end - item["candidate"].global_source_start,
                    item["candidate"].score,
                ),
            )
        elif introducing_options:
            best_option = min(
                introducing_options,
                key=lambda item: (
                    item["distance_after"],
                    -_granularity_rank(item["candidate"].granularity),
                    -item["candidate"].score,
                ),
            )
            if best_option["distance_after"] < current_distance:
                chosen = best_option

        if chosen is None:
            source_cursor += 1
            target_cursor += 1
            continue

        candidate = chosen["candidate"]
        if candidate.global_source_start > source_cursor or candidate.global_target_start > target_cursor:
            cursor_gaps.append(
                {
                    "source_start": source_cursor,
                    "source_end": candidate.global_source_start - 1,
                    "target_start": target_cursor,
                    "target_end": candidate.global_target_start - 1,
                    "reason": "advanced_to_next_viable_candidate",
                }
            )
        actual = chosen["after"]
        introduced_families.update(chosen["new_families"])
        steps.append(
            {
                "global_source_start": candidate.global_source_start,
                "global_source_end": candidate.global_source_end,
                "global_target_start": candidate.global_target_start,
                "global_target_end": candidate.global_target_end,
                "source_span": list(candidate.source_span),
                "target_span": list(candidate.target_span),
                "granularity": candidate.granularity,
                "source_text": candidate.source_text,
                "target_text": candidate.target_text,
                "chapter_pair": list(candidate.chapter_pair),
                "sentence_index": candidate.sentence_index,
                "delta": chosen["delta"].to_dict(),
                "actual_after": actual.to_dict(),
                "target_here": current_target.to_dict(),
                "distance_after": round(chosen["distance_after"], 6),
                "distance_before": round(current_distance, 6),
                "alignment_confidence": round(chosen["alignment_confidence"], 6),
                "future_coverage_after": round(
                    _future_family_coverage(
                        family_occurrence_positions,
                        introduced_families,
                        source_cursor=candidate.global_source_end + 1,
                    ),
                    6,
                ),
                "new_families": chosen["new_families"],
                "all_families": candidate.family_ids,
            }
        )
        source_cursor = candidate.global_source_end + 1
        target_cursor = candidate.global_target_end + 1

    return steps, family_weights, cursor_gaps


def _actions_by_sentence(action_inventory: dict[str, object]) -> dict[int, list[dict[str, object]]]:
    mapping: dict[int, list[dict[str, object]]] = defaultdict(list)
    for action in action_inventory["token_actions"]:
        mapping[int(action["sentence_index"])].append(action)
    for action in action_inventory["phrase_actions"]:
        mapping[int(action["sentence_index"])].append(action)
    for action in action_inventory["subtree_actions"]:
        mapping[int(action["sentence_index"])].append(action)
    return mapping


def _family_is_free(candidate: ProgressionCandidate, family_id: str) -> bool:
    relation = str(candidate.metadata.get("relation", ""))
    if candidate.granularity == "token" and relation in FREE_RELATIONS:
        return True
    return False


def _granularity_rank(granularity: str) -> int:
    return {"token": 1, "phrase": 2, "subtree": 3}.get(granularity, 0)


def _eligible_options(
    grouped: dict[int, list[ProgressionCandidate]],
    *,
    source_cursor: int,
    target_cursor: int,
) -> list[ProgressionCandidate]:
    exact_source = [
        item
        for item in grouped.get(source_cursor, [])
        if item.global_target_start >= target_cursor
    ]
    if exact_source:
        min_target = min(item.global_target_start for item in exact_source)
        return [item for item in exact_source if item.global_target_start == min_target]

    future_candidates: list[ProgressionCandidate] = []
    for start in sorted(index for index in grouped.keys() if index > source_cursor):
        matching = [item for item in grouped[start] if item.global_target_start >= target_cursor]
        if matching:
            min_target = min(item.global_target_start for item in matching)
            return [item for item in matching if item.global_target_start == min_target]
    return future_candidates


def _target_metrics(
    *,
    source_cursor: int,
    target_cursor: int,
    total_source_positions: int,
    total_target_positions: int,
) -> ProgressionMetrics:
    source_progress = (source_cursor - 1) / total_source_positions if total_source_positions else 0.0
    target_progress = (target_cursor - 1) / total_target_positions if total_target_positions else 0.0
    progress = (source_progress + target_progress) / 2
    return ProgressionMetrics(surface=progress, novelty=progress, difficulty=progress)


def _metrics_distance(actual: ProgressionMetrics, target: ProgressionMetrics) -> float:
    return (
        abs(target.surface - actual.surface)
        + abs(target.novelty - actual.novelty)
        + abs(target.difficulty - actual.difficulty)
    )


def _candidate_surface_delta(candidate: ProgressionCandidate, *, total_source_positions: int) -> float:
    if total_source_positions <= 0:
        return 0.0
    visible_change = _visible_change(candidate.source_text, candidate.target_text)
    span_weight = _candidate_span_weight(candidate)
    return (visible_change * span_weight) / total_source_positions


def _candidate_difficulty_profile(candidate: ProgressionCandidate) -> float:
    opacity = _visible_change(candidate.source_text, candidate.target_text)
    relation = str(candidate.metadata.get("relation", ""))
    relation_risk = {
        "exact": 0.0,
        "lemma_like": 0.05,
        "cognate": 0.15,
        "parallel_span": 0.2,
    }.get(relation, 0.25)
    structural_shift = {
        "token": 0.0,
        "phrase": 0.2,
        "subtree": 0.35,
    }.get(candidate.granularity, 0.2)
    source_span_len = max(candidate.source_span[1] - candidate.source_span[0] + 1, 1)
    target_span_len = max(candidate.target_span[1] - candidate.target_span[0] + 1, 1)
    length_shift = min(abs(source_span_len - target_span_len) / max(source_span_len, target_span_len), 1.0)
    return min(
        (0.45 * opacity) + (0.25 * relation_risk) + (0.2 * structural_shift) + (0.1 * length_shift),
        1.0,
    )


def _candidate_span_weight(candidate: ProgressionCandidate) -> float:
    if candidate.granularity == "token":
        upos = str(candidate.metadata.get("source_upos") or "")
        return CONTENT_UPOS_WEIGHTS.get(upos, 0.5)
    family_count = len(candidate.family_ids)
    if family_count:
        return float(family_count)
    return float(max(candidate.source_span[1] - candidate.source_span[0] + 1, 1))


def _visible_change(source_text: str, target_text: str) -> float:
    normalized_source = " ".join(source_text.lower().split())
    normalized_target = " ".join(target_text.lower().split())
    if not normalized_source and not normalized_target:
        return 0.0
    return 1.0 - SequenceMatcher(None, normalized_source, normalized_target).ratio()


def _future_family_coverage(
    family_occurrence_positions: dict[str, list[int]],
    introduced_families: set[str],
    *,
    source_cursor: int,
) -> float:
    total_remaining = 0
    covered_remaining = 0
    for family_id, positions in family_occurrence_positions.items():
        start_index = bisect_left(positions, source_cursor)
        remaining = len(positions) - start_index
        if remaining <= 0:
            continue
        total_remaining += remaining
        if family_id in introduced_families:
            covered_remaining += remaining
    if total_remaining == 0:
        return 1.0
    return covered_remaining / total_remaining


def _alignment_confidence(candidate: ProgressionCandidate) -> float:
    score = max(0.0, min(candidate.score, 1.0))
    relation = str(candidate.metadata.get("relation", ""))
    relation_boost = {
        "exact": 0.08,
        "lemma_like": 0.06,
        "exact_span": 0.05,
        "parallel_span": 0.0,
        "dependency_span": 0.02,
        "dependency_anchor": 0.03,
        "upos_cognate": -0.02,
        "cognate": -0.04,
    }.get(relation, -0.02)
    family_support = 0.0
    if candidate.granularity in {"phrase", "subtree"}:
        family_count = len(candidate.family_ids)
        if family_count:
            family_support = min(0.06, 0.02 * family_count)
        else:
            family_support = -0.08
    return max(0.0, min(1.0, score + relation_boost + family_support))


def _is_free_carry_option(
    candidate: ProgressionCandidate,
    *,
    new_families: list[str],
    delta: ProgressionMetrics,
) -> bool:
    if candidate.granularity == "token" and delta.novelty == 0.0 and delta.difficulty == 0.0:
        return True
    if candidate.family_ids and not new_families:
        return True
    return False


def _chapter_word_offsets(
    book: CanonicalBook,
    *,
    included_chapters: list[int] | None = None,
    scope_source: bool = False,
) -> tuple[dict[int, int], int]:
    offsets: dict[int, int] = {}
    running = 1
    chapter_filter = set(included_chapters or [chapter.index for chapter in book.chapters])
    for chapter in book.chapters:
        if chapter.index not in chapter_filter:
            continue
        scoped_chapter = scoped_source_chapter(chapter) if scope_source else chapter
        offsets[chapter.index] = running
        running += _count_chapter_words(scoped_chapter)
    return offsets, max(running - 1, 0)


def _count_chapter_words(chapter: object) -> int:
    count = 0
    for paragraph in chapter.paragraphs:
        for sentence in paragraph.sentences:
            count += len(_word_re.findall(sentence.text))
    return count


_word_re = re.compile(r"\w+", re.UNICODE)
