"""Book-level monotonic many-to-many paragraph alignment."""

from __future__ import annotations

from dataclasses import dataclass

from szwejk.align.candidates import build_monotonic_chapter_alignment
from szwejk.align.embedding_eval import HeuristicEmbeddingEncoder, SentenceEncoder
from szwejk.align.tail_volume4 import (
    ParagraphItem,
    align_monotonic_many_to_many,
    attach_embeddings,
    collect_chapter_paragraphs,
)
from szwejk.generate.corpus_scope import apply_default_tail_chapter_overrides, scoped_source_chapter
from szwejk.schemas import CanonicalBook, CanonicalChapter


@dataclass(slots=True)
class ParagraphAlignmentConfig:
    max_source_span: int = 4
    max_target_span: int = 4
    skip_source_penalty: float = 0.55
    skip_target_penalty: float = 0.65
    pair_bonus: float = 0.04
    position_slack: float = 0.08


def build_book_paragraph_alignment_report(
    source_book: CanonicalBook,
    target_book: CanonicalBook,
    *,
    encoder: SentenceEncoder | None = None,
    model_name: str = "heuristic",
    chapter_pairs: list[tuple[int, int]] | None = None,
    min_match_score: float = 0.72,
    skip_penalty: float = 0.2,
    config: ParagraphAlignmentConfig | None = None,
) -> dict[str, object]:
    if chapter_pairs is None:
        chapter_alignment = build_monotonic_chapter_alignment(
            source_book,
            target_book,
            min_match_score=min_match_score,
            skip_penalty=skip_penalty,
        )
        selected_pairs = apply_default_tail_chapter_overrides(
            [
                (item["source_chapter_index"], item["target_chapter_index"])
                for item in chapter_alignment["matches"]
            ],
            source_chapter_count=source_book.chapter_count,
            target_chapter_count=target_book.chapter_count,
        )
    else:
        chapter_alignment = None
        selected_pairs = apply_default_tail_chapter_overrides(
            chapter_pairs,
            source_chapter_count=source_book.chapter_count,
            target_chapter_count=target_book.chapter_count,
        )
    resolved_encoder = encoder or HeuristicEmbeddingEncoder()
    resolved_config = config or ParagraphAlignmentConfig()
    pair_reports = _build_pair_reports(
        source_book,
        target_book,
        chapter_pairs=selected_pairs,
        encoder=resolved_encoder,
        config=resolved_config,
    )
    return {
        "model_name": model_name,
        "chapter_alignment": chapter_alignment,
        "matched_pair_count": len(selected_pairs),
        "pair_reports": pair_reports,
        "global_metrics": _aggregate_global_metrics(pair_reports),
        "config": {
            "max_source_span": resolved_config.max_source_span,
            "max_target_span": resolved_config.max_target_span,
            "skip_source_penalty": resolved_config.skip_source_penalty,
            "skip_target_penalty": resolved_config.skip_target_penalty,
            "pair_bonus": resolved_config.pair_bonus,
            "position_slack": resolved_config.position_slack,
        },
    }


def _build_pair_reports(
    source_book: CanonicalBook,
    target_book: CanonicalBook,
    *,
    chapter_pairs: list[tuple[int, int]],
    encoder: SentenceEncoder,
    config: ParagraphAlignmentConfig,
) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for source_index, target_index in chapter_pairs:
        source_chapter = scoped_source_chapter(source_book.chapters[source_index - 1])
        target_chapter = target_book.chapters[target_index - 1]
        report = build_chapter_paragraph_alignment(
            source_chapter,
            target_chapter,
            encoder=encoder,
            config=config,
        )
        reports.append(
            {
                "source_chapter_index": source_index,
                "target_chapter_index": target_index,
                "source_title": source_chapter.title,
                "target_title": target_chapter.title,
                **report,
            }
        )
    return reports


def build_chapter_paragraph_alignment(
    source_chapter: CanonicalChapter,
    target_chapter: CanonicalChapter,
    *,
    encoder: SentenceEncoder,
    config: ParagraphAlignmentConfig | None = None,
) -> dict[str, object]:
    resolved_config = config or ParagraphAlignmentConfig()
    source_items = collect_chapter_paragraphs_from_chapter(source_chapter)
    target_items = collect_chapter_paragraphs_from_chapter(target_chapter)
    _attach_paragraph_embeddings(source_items, target_items, encoder=encoder)
    report = align_monotonic_many_to_many(
        source_items,
        target_items,
        max_source_span=resolved_config.max_source_span,
        max_target_span=resolved_config.max_target_span,
        skip_source_penalty=resolved_config.skip_source_penalty,
        skip_target_penalty=resolved_config.skip_target_penalty,
        pair_bonus=resolved_config.pair_bonus,
        position_slack=resolved_config.position_slack,
    )
    matched_blocks = report["matched_blocks"]
    unmatched_source = report["unmatched_source"]
    unmatched_target = report["unmatched_target"]
    return {
        "source_total": len(source_items),
        "target_total": len(target_items),
        "matched_block_count": len(matched_blocks),
        "unmatched_source_block_count": len(unmatched_source),
        "unmatched_target_block_count": len(unmatched_target),
        "matched_blocks": matched_blocks,
        "unmatched_source": unmatched_source,
        "unmatched_target": unmatched_target,
        "metrics": {
            "source_paragraph_coverage": report["source_coverage"],
            "target_paragraph_coverage": report["target_coverage"],
        },
    }


def collect_chapter_paragraphs_from_chapter(chapter: CanonicalChapter) -> list[ParagraphItem]:
    book = CanonicalBookProxy(chapter)
    return collect_chapter_paragraphs(book, 1)


class CanonicalBookProxy:
    """Minimal adapter so the existing collector can operate on a single chapter."""

    def __init__(self, chapter: CanonicalChapter) -> None:
        self.chapters = [chapter]


def _attach_paragraph_embeddings(
    source_items: list[ParagraphItem],
    target_items: list[ParagraphItem],
    *,
    encoder: SentenceEncoder,
) -> None:
    texts = ["passage: " + item.text for item in source_items + target_items]
    vectors = encoder.encode(texts)
    source_count = len(source_items)
    attach_embeddings(source_items, _as_float_matrix(vectors[:source_count]))
    attach_embeddings(target_items, _as_float_matrix(vectors[source_count:]))


def _as_float_matrix(vectors: list[list[float]]) -> list[list[float]]:
    return [[float(value) for value in row] for row in vectors]


def _aggregate_global_metrics(pair_reports: list[dict[str, object]]) -> dict[str, float]:
    if not pair_reports:
        return {
            "source_paragraph_coverage": 0.0,
            "target_paragraph_coverage": 0.0,
            "matched_block_count": 0,
            "unmatched_source_block_count": 0,
            "unmatched_target_block_count": 0,
        }

    source_total = sum(int(report["source_total"]) for report in pair_reports)
    target_total = sum(int(report["target_total"]) for report in pair_reports)
    matched_source = sum(
        int(block["source_span"])
        for report in pair_reports
        for block in report["matched_blocks"]
    )
    matched_target = sum(
        int(block["target_span"])
        for report in pair_reports
        for block in report["matched_blocks"]
    )
    return {
        "source_paragraph_coverage": round(matched_source / max(source_total, 1), 6),
        "target_paragraph_coverage": round(matched_target / max(target_total, 1), 6),
        "matched_block_count": sum(int(report["matched_block_count"]) for report in pair_reports),
        "unmatched_source_block_count": sum(int(report["unmatched_source_block_count"]) for report in pair_reports),
        "unmatched_target_block_count": sum(int(report["unmatched_target_block_count"]) for report in pair_reports),
    }
