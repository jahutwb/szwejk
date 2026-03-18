"""Deterministic structural candidate generation for PL-CS alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from szwejk.common.ids import slugify_fragment
from szwejk.schemas import CanonicalBook, CanonicalChapter, CanonicalParagraph


@dataclass(slots=True)
class ChapterCandidate:
    source_chapter_id: str
    source_chapter_index: int
    target_chapter_id: str
    target_chapter_index: int
    score: float
    signals: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_chapter_id": self.source_chapter_id,
            "source_chapter_index": self.source_chapter_index,
            "target_chapter_id": self.target_chapter_id,
            "target_chapter_index": self.target_chapter_index,
            "score": round(self.score, 6),
            "signals": {key: round(value, 6) for key, value in self.signals.items()},
        }


@dataclass(slots=True)
class ParagraphCandidate:
    source_paragraph_id: str
    source_paragraph_index: int
    target_paragraph_id: str
    target_paragraph_index: int
    score: float
    signals: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_paragraph_id": self.source_paragraph_id,
            "source_paragraph_index": self.source_paragraph_index,
            "target_paragraph_id": self.target_paragraph_id,
            "target_paragraph_index": self.target_paragraph_index,
            "score": round(self.score, 6),
            "signals": {key: round(value, 6) for key, value in self.signals.items()},
        }


@dataclass(slots=True)
class AlignmentCandidates:
    chapter_candidates: dict[str, list[ChapterCandidate]] = field(default_factory=dict)
    paragraph_candidates: dict[tuple[str, str], dict[str, list[ParagraphCandidate]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        chapter_candidates = {
            chapter_id: [candidate.to_dict() for candidate in candidates]
            for chapter_id, candidates in self.chapter_candidates.items()
        }
        paragraph_candidates = {
            f"{source_id}::{target_id}": {
                paragraph_id: [candidate.to_dict() for candidate in candidates]
                for paragraph_id, candidates in mapping.items()
            }
            for (source_id, target_id), mapping in self.paragraph_candidates.items()
        }
        return {
            "chapter_candidates": chapter_candidates,
            "paragraph_candidates": paragraph_candidates,
        }


def build_monotonic_chapter_alignment(
    source_book: CanonicalBook,
    target_book: CanonicalBook,
    *,
    min_match_score: float = 0.72,
    skip_penalty: float = 0.2,
) -> dict[str, object]:
    source_total = len(source_book.chapters)
    target_total = len(target_book.chapters)
    pair_scores = {
        (source_index, target_index): score_chapter_pair(
            source_book.chapters[source_index - 1],
            target_book.chapters[target_index - 1],
            source_total=source_total,
            target_total=target_total,
        )
        for source_index in range(1, source_total + 1)
        for target_index in range(1, target_total + 1)
    }

    dp: dict[tuple[int, int], tuple[float, list[tuple[str, int, int | None]]]] = {(0, 0): (0.0, [])}
    for source_done in range(0, source_total + 1):
        for target_done in range(0, target_total + 1):
            state = (source_done, target_done)
            if state not in dp:
                continue
            current_score, current_path = dp[state]

            if source_done < source_total:
                next_state = (source_done + 1, target_done)
                candidate = (current_score - skip_penalty, current_path + [("skip_source", source_done + 1, None)])
                previous = dp.get(next_state)
                if previous is None or candidate[0] > previous[0]:
                    dp[next_state] = candidate

            if target_done < target_total:
                next_state = (source_done, target_done + 1)
                candidate = (current_score - skip_penalty, current_path + [("skip_target", target_done + 1, None)])
                previous = dp.get(next_state)
                if previous is None or candidate[0] > previous[0]:
                    dp[next_state] = candidate

            if source_done < source_total and target_done < target_total:
                source_index = source_done + 1
                target_index = target_done + 1
                pair_score, _signals = pair_scores[(source_index, target_index)]
                if pair_score >= min_match_score:
                    next_state = (source_index, target_index)
                    candidate = (current_score + pair_score, current_path + [("match", source_index, target_index)])
                    previous = dp.get(next_state)
                    if previous is None or candidate[0] > previous[0]:
                        dp[next_state] = candidate

    path = dp.get((source_total, target_total), (0.0, []))[1]
    matches = []
    unmatched_source = []
    unmatched_target = []
    for action, left_index, right_index in path:
        if action == "match" and right_index is not None:
            score, signals = pair_scores[(left_index, right_index)]
            source_chapter = source_book.chapters[left_index - 1]
            target_chapter = target_book.chapters[right_index - 1]
            matches.append(
                {
                    "source_chapter_id": source_chapter.id,
                    "source_chapter_index": source_chapter.index,
                    "source_title": source_chapter.title,
                    "target_chapter_id": target_chapter.id,
                    "target_chapter_index": target_chapter.index,
                    "target_title": target_chapter.title,
                    "score": round(score, 6),
                    "signals": {key: round(value, 6) for key, value in signals.items()},
                }
            )
        elif action == "skip_source":
            source_chapter = source_book.chapters[left_index - 1]
            unmatched_source.append(
                {
                    "source_chapter_id": source_chapter.id,
                    "source_chapter_index": source_chapter.index,
                    "source_title": source_chapter.title,
                }
            )
        elif action == "skip_target":
            target_chapter = target_book.chapters[left_index - 1]
            unmatched_target.append(
                {
                    "target_chapter_id": target_chapter.id,
                    "target_chapter_index": target_chapter.index,
                    "target_title": target_chapter.title,
                }
            )

    return {
        "source_chapter_count": source_total,
        "target_chapter_count": target_total,
        "min_match_score": min_match_score,
        "skip_penalty": skip_penalty,
        "matches": matches,
        "unmatched_source_chapters": unmatched_source,
        "unmatched_target_chapters": unmatched_target,
    }


def build_alignment_candidates(
    source_book: CanonicalBook,
    target_book: CanonicalBook,
    *,
    chapter_window: int = 1,
    paragraph_window: int = 3,
    top_k: int = 3,
) -> AlignmentCandidates:
    chapter_candidates = generate_chapter_candidates(source_book, target_book, chapter_window=chapter_window, top_k=top_k)
    paragraph_candidates: dict[tuple[str, str], dict[str, list[ParagraphCandidate]]] = {}

    target_lookup = {chapter.id: chapter for chapter in target_book.chapters}
    for source_chapter in source_book.chapters:
        top_chapter = chapter_candidates[source_chapter.id][0]
        target_chapter = target_lookup[top_chapter.target_chapter_id]
        paragraph_candidates[(source_chapter.id, target_chapter.id)] = generate_paragraph_candidates(
            source_chapter, target_chapter, window=paragraph_window, top_k=top_k
        )

    return AlignmentCandidates(chapter_candidates=chapter_candidates, paragraph_candidates=paragraph_candidates)


def generate_chapter_candidates(
    source_book: CanonicalBook,
    target_book: CanonicalBook,
    *,
    chapter_window: int = 1,
    top_k: int = 3,
) -> dict[str, list[ChapterCandidate]]:
    source_total = len(source_book.chapters)
    target_total = len(target_book.chapters)
    results: dict[str, list[ChapterCandidate]] = {}

    for source_chapter in source_book.chapters:
        expected_index = _expected_target_index(source_chapter.index, source_total, target_total)
        candidate_targets = _window(target_book.chapters, expected_index, chapter_window)
        candidates: list[ChapterCandidate] = []
        for target_chapter in candidate_targets:
            score, signals = score_chapter_pair(
                source_chapter,
                target_chapter,
                source_total=source_total,
                target_total=target_total,
            )
            candidates.append(
                ChapterCandidate(
                    source_chapter_id=source_chapter.id,
                    source_chapter_index=source_chapter.index,
                    target_chapter_id=target_chapter.id,
                    target_chapter_index=target_chapter.index,
                    score=score,
                    signals=signals,
                )
            )
        results[source_chapter.id] = sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k]
    return results


def generate_paragraph_candidates(
    source_chapter: CanonicalChapter,
    target_chapter: CanonicalChapter,
    *,
    window: int = 3,
    top_k: int = 3,
) -> dict[str, list[ParagraphCandidate]]:
    source_total = len(source_chapter.paragraphs)
    target_total = len(target_chapter.paragraphs)
    results: dict[str, list[ParagraphCandidate]] = {}

    for source_paragraph in source_chapter.paragraphs:
        expected_index = _expected_target_index(source_paragraph.index, source_total, target_total)
        candidate_targets = _window(target_chapter.paragraphs, expected_index, window)
        paragraph_candidates: list[ParagraphCandidate] = []
        for target_paragraph in candidate_targets:
            position_signal = _normalized_index_similarity(source_paragraph.index, source_total, target_paragraph.index, target_total)
            length_signal = _count_similarity(len(source_paragraph.text), len(target_paragraph.text))
            trigram_signal = _char_trigram_jaccard(source_paragraph.text, target_paragraph.text)
            score = (0.50 * position_signal) + (0.30 * length_signal) + (0.20 * trigram_signal)
            paragraph_candidates.append(
                ParagraphCandidate(
                    source_paragraph_id=source_paragraph.id,
                    source_paragraph_index=source_paragraph.index,
                    target_paragraph_id=target_paragraph.id,
                    target_paragraph_index=target_paragraph.index,
                    score=score,
                    signals={
                        "position": position_signal,
                        "length": length_signal,
                        "char_trigram": trigram_signal,
                    },
                )
            )
        results[source_paragraph.id] = sorted(paragraph_candidates, key=lambda item: item.score, reverse=True)[:top_k]
    return results


def _expected_target_index(source_index: int, source_total: int, target_total: int) -> int:
    if source_total <= 1 or target_total <= 1:
        return 1
    normalized = (source_index - 1) / max(source_total - 1, 1)
    return round(normalized * (target_total - 1)) + 1


def score_chapter_pair(
    source_chapter: CanonicalChapter,
    target_chapter: CanonicalChapter,
    *,
    source_total: int,
    target_total: int,
) -> tuple[float, dict[str, float]]:
    position_signal = _normalized_index_similarity(source_chapter.index, source_total, target_chapter.index, target_total)
    paragraph_signal = _count_similarity(len(source_chapter.paragraphs), len(target_chapter.paragraphs))
    title_signal = _title_hint_similarity(source_chapter.title, target_chapter.title)
    score = (0.55 * position_signal) + (0.35 * paragraph_signal) + (0.10 * title_signal)
    return score, {
        "position": position_signal,
        "paragraph_count": paragraph_signal,
        "title_hint": title_signal,
    }


def _window(items: list[CanonicalChapter] | list[CanonicalParagraph], center_index: int, radius: int) -> list:
    start = max(1, center_index - radius)
    end = min(len(items), center_index + radius)
    return items[start - 1 : end]


def _normalized_index_similarity(source_index: int, source_total: int, target_index: int, target_total: int) -> float:
    if source_total <= 1 and target_total <= 1:
        return 1.0
    source_position = (source_index - 1) / max(source_total - 1, 1)
    target_position = (target_index - 1) / max(target_total - 1, 1)
    return max(0.0, 1.0 - abs(source_position - target_position))


def _count_similarity(left: int, right: int) -> float:
    if left <= 0 or right <= 0:
        return 0.0
    return min(left, right) / max(left, right)


def _title_hint_similarity(source_title: str, target_title: str) -> float:
    special = _special_title_hint_similarity(source_title, target_title)
    if special > 0.0:
        return special
    source_slug = slugify_fragment(source_title)
    target_slug = slugify_fragment(target_title)
    if source_slug == target_slug:
        return 1.0
    source_tokens = set(_tokenize_slug(source_slug))
    target_tokens = set(_tokenize_slug(target_slug))
    if not source_tokens or not target_tokens:
        return 0.0
    overlap = len(source_tokens & target_tokens)
    union = len(source_tokens | target_tokens)
    return overlap / union if union else 0.0


def _special_title_hint_similarity(source_title: str, target_title: str) -> float:
    source_slug = slugify_fragment(source_title)
    target_slug = slugify_fragment(target_title)
    intro_groups = {
        "wstep",
        "uvod",
    }
    if source_slug in intro_groups and target_slug in intro_groups:
        return 1.0
    if _extract_roman(source_slug) and _extract_roman(source_slug) == _extract_roman(target_slug):
        return 0.8
    return 0.0


def _extract_roman(slug: str) -> str | None:
    roman_values = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv"}
    for token in _tokenize_slug(slug):
        if token in roman_values:
            return token
    return None


def _tokenize_slug(slug: str) -> list[str]:
    return [token for token in slug.split("-") if token]


def _char_trigram_jaccard(left: str, right: str) -> float:
    left_grams = set(_char_ngrams(left))
    right_grams = set(_char_ngrams(right))
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def _char_ngrams(text: str, n: int = 3) -> Iterable[str]:
    slug = slugify_fragment(text).replace("-", "")
    if len(slug) < n:
        if slug:
            yield slug
        return
    for index in range(0, len(slug) - n + 1):
        yield slug[index : index + n]
