"""Confidence and diagnostics helpers for alignment backbone review."""

from __future__ import annotations

import statistics

from szwejk.align.sentences import SentenceAlignment, align_chapter_sentences
from szwejk.schemas import CanonicalBook


def classify_confidence(score: float) -> str:
    if score >= 0.65:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def build_sentence_rationale(alignment: SentenceAlignment) -> list[str]:
    rationale: list[str] = []
    signals = alignment.signals
    if signals.get("char_trigram", 0.0) >= 0.18:
        rationale.append("strong character-overlap signal")
    if signals.get("length", 0.0) >= 0.85:
        rationale.append("similar sentence length")
    if signals.get("punctuation", 0.0) >= 0.75:
        rationale.append("matching punctuation shape")
    if signals.get("digit_pattern", 0.0) >= 1.0:
        rationale.append("same digit pattern")
    if not rationale:
        rationale.append("weak lexical and structural evidence")
    return rationale


def augment_sentence_alignment(alignment: SentenceAlignment) -> dict[str, object]:
    confidence_value = round(alignment.score, 6)
    confidence_label = classify_confidence(confidence_value)
    return {
        **alignment.to_dict(),
        "confidence_value": confidence_value,
        "confidence_label": confidence_label,
        "rationale": build_sentence_rationale(alignment),
    }


def summarize_sentence_alignments(alignments: list[SentenceAlignment]) -> dict[str, object]:
    scores = [alignment.score for alignment in alignments]
    if not scores:
        return {
            "count": 0,
            "score_distribution": {"min": None, "median": None, "max": None, "mean": None},
            "confidence_counts": {"high": 0, "medium": 0, "low": 0},
            "difficult_cases": [],
        }

    confidence_counts = {"high": 0, "medium": 0, "low": 0}
    difficult_cases: list[dict[str, object]] = []
    for alignment in alignments:
        label = classify_confidence(alignment.score)
        confidence_counts[label] += 1
        if label == "low":
            difficult_cases.append(augment_sentence_alignment(alignment))

    return {
        "count": len(scores),
        "score_distribution": {
            "min": round(min(scores), 6),
            "median": round(statistics.median(scores), 6),
            "max": round(max(scores), 6),
            "mean": round(statistics.fmean(scores), 6),
        },
        "confidence_counts": confidence_counts,
        "difficult_cases": difficult_cases[:20],
    }


def build_benchmark_report(
    source_book: CanonicalBook,
    target_book: CanonicalBook,
    *,
    chapter_pairs: list[tuple[int, int]],
    paragraph_window: int = 3,
) -> dict[str, object]:
    pair_reports: list[dict[str, object]] = []
    all_sentence_alignments: list[SentenceAlignment] = []

    for source_chapter_index, target_chapter_index in chapter_pairs:
        chapter_alignments = align_chapter_sentences(
            source_book.chapters[source_chapter_index - 1],
            target_book.chapters[target_chapter_index - 1],
            paragraph_window=paragraph_window,
        )
        sentence_alignments = [row.sentence_alignment for row in chapter_alignments]
        all_sentence_alignments.extend(sentence_alignments)
        summary = summarize_sentence_alignments(sentence_alignments)
        pair_reports.append(
            {
                "source_chapter_index": source_chapter_index,
                "target_chapter_index": target_chapter_index,
                "source_title": source_book.chapters[source_chapter_index - 1].title,
                "target_title": target_book.chapters[target_chapter_index - 1].title,
                "alignment_count": len(sentence_alignments),
                "score_distribution": summary["score_distribution"],
                "confidence_counts": summary["confidence_counts"],
                "difficult_cases": summary["difficult_cases"],
                "sample_alignments": [augment_sentence_alignment(alignment) for alignment in sentence_alignments[:5]],
            }
        )

    global_summary = summarize_sentence_alignments(all_sentence_alignments)
    return {
        "chapter_pair_count": len(chapter_pairs),
        "pair_reports": pair_reports,
        "global_score_distribution": global_summary["score_distribution"],
        "global_confidence_counts": global_summary["confidence_counts"],
    }
