"""Adjudication-ready review bundle generation for alignment QA."""

from __future__ import annotations

from collections import Counter

from szwejk.align.diagnostics import augment_sentence_alignment, classify_confidence
from szwejk.align.sentences import ChapterSentenceAlignment, align_chapter_sentences
from szwejk.schemas import CanonicalChapter


def determine_safe_alignment_depth(confidence_label: str) -> str:
    if confidence_label in {"high", "medium"}:
        return "sentence"
    return "paragraph"


def _preview_text(text: str, limit: int = 220) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def build_review_bundle_for_chapter_pair(
    source_chapter: CanonicalChapter,
    target_chapter: CanonicalChapter,
    *,
    paragraph_window: int = 3,
) -> dict[str, object]:
    chapter_alignments = align_chapter_sentences(
        source_chapter,
        target_chapter,
        paragraph_window=paragraph_window,
    )
    return build_review_bundle(
        source_chapter=source_chapter,
        target_chapter=target_chapter,
        source_title=source_chapter.title,
        target_title=target_chapter.title,
        chapter_alignments=chapter_alignments,
    )


def build_review_bundle(
    *,
    source_chapter: CanonicalChapter,
    target_chapter: CanonicalChapter,
    source_title: str,
    target_title: str,
    chapter_alignments: list[ChapterSentenceAlignment],
) -> dict[str, object]:
    items: list[dict[str, object]] = []
    adjudication_items: list[dict[str, object]] = []
    safe_depth_counts: Counter[str] = Counter()
    source_paragraphs = {paragraph.id: paragraph for paragraph in source_chapter.paragraphs}
    target_paragraphs = {paragraph.id: paragraph for paragraph in target_chapter.paragraphs}

    for row in chapter_alignments:
        sentence_alignment = augment_sentence_alignment(row.sentence_alignment)
        safe_depth = determine_safe_alignment_depth(sentence_alignment["confidence_label"])
        source_paragraph = source_paragraphs[row.source_paragraph_id]
        target_paragraph = target_paragraphs[row.target_paragraph_id]
        item = {
            "source_chapter_id": source_chapter.id,
            "target_chapter_id": target_chapter.id,
            "source_title": source_title,
            "target_title": target_title,
            "source_paragraph_id": row.source_paragraph_id,
            "target_paragraph_id": row.target_paragraph_id,
            "source_paragraph_index": row.source_paragraph_index,
            "target_paragraph_index": row.target_paragraph_index,
            "source_paragraph_text": source_paragraph.text,
            "target_paragraph_text": target_paragraph.text,
            "source_paragraph_preview": _preview_text(source_paragraph.text),
            "target_paragraph_preview": _preview_text(target_paragraph.text),
            "safe_alignment_depth": safe_depth,
            "sentence_alignment": sentence_alignment,
        }
        items.append(item)
        safe_depth_counts[safe_depth] += 1
        if sentence_alignment["confidence_label"] == "low":
            adjudication_items.append(item)

    exception_queue = [
        {
            "queue_reason": "low_confidence_sentence_alignment",
            "safe_alignment_depth": item["safe_alignment_depth"],
            "source_chapter_id": item["source_chapter_id"],
            "target_chapter_id": item["target_chapter_id"],
            "source_paragraph_index": item["source_paragraph_index"],
            "target_paragraph_index": item["target_paragraph_index"],
            "source_paragraph_preview": item["source_paragraph_preview"],
            "target_paragraph_preview": item["target_paragraph_preview"],
            "sentence_alignment": item["sentence_alignment"],
        }
        for item in adjudication_items
    ]

    return {
        "summary": {
            "source_title": source_title,
            "target_title": target_title,
            "total_items": len(items),
            "adjudication_items": len(adjudication_items),
            "safe_depth_counts": dict(safe_depth_counts),
        },
        "items": items,
        "adjudication_items": adjudication_items,
        "exception_queue": exception_queue,
    }
