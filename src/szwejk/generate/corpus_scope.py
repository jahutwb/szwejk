"""Scope overrides for the comparable PL-CS corpus."""

from __future__ import annotations

from szwejk.schemas import CanonicalChapter


DEFAULT_SOURCE_PARAGRAPH_CUTOFFS = {
    29: 393,
}

DEFAULT_TAIL_CHAPTER_PAIRS = [
    (27, 27),
    (28, 28),
    (29, 29),
]


def apply_default_tail_chapter_overrides(
    pairs: list[tuple[int, int]],
    *,
    source_chapter_count: int,
    target_chapter_count: int,
) -> list[tuple[int, int]]:
    if source_chapter_count < 29 or target_chapter_count < 29:
        return pairs
    kept = [(left, right) for left, right in pairs if left < 27 and right < 27]
    return kept + list(DEFAULT_TAIL_CHAPTER_PAIRS)


def source_paragraph_cutoff_for_chapter(chapter_index: int) -> int | None:
    return DEFAULT_SOURCE_PARAGRAPH_CUTOFFS.get(chapter_index)


def scoped_source_chapter(chapter: CanonicalChapter) -> CanonicalChapter:
    cutoff = source_paragraph_cutoff_for_chapter(chapter.index)
    if cutoff is None:
        return chapter
    return CanonicalChapter(
        id=chapter.id,
        index=chapter.index,
        title=chapter.title,
        paragraphs=list(chapter.paragraphs[:cutoff]),
        label=chapter.label,
        source_path=chapter.source_path,
    )
