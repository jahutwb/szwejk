"""Hierarchical alignment report driven by paragraph skeleton and sentence/subtree refinement."""

from __future__ import annotations

from szwejk.align.embedding_eval import HeuristicEmbeddingEncoder, SentenceEncoder
from szwejk.align.enrichment import build_sentence_enrichment_bundle
from szwejk.align.paragraphs import build_book_paragraph_alignment_report
from szwejk.align.sentence_blocks import align_chapter_sentence_spans_with_paragraph_blocks
from szwejk.align.sentences import ChapterSentenceAlignment, SentenceAlignment
from szwejk.generate.corpus_scope import apply_default_tail_chapter_overrides, scoped_source_chapter
from szwejk.schemas import CanonicalBook


def build_hierarchical_alignment_report(
    source_book: CanonicalBook,
    target_book: CanonicalBook,
    *,
    chapter_pairs: list[tuple[int, int]] | None = None,
    paragraph_encoder: SentenceEncoder | None = None,
    sentence_encoder: SentenceEncoder | None = None,
    analysis_mode: str = "stanza",
) -> dict[str, object]:
    normalized_pairs = apply_default_tail_chapter_overrides(
        chapter_pairs
        or [(index, index) for index in range(1, min(source_book.chapter_count, target_book.chapter_count) + 1)],
        source_chapter_count=source_book.chapter_count,
        target_chapter_count=target_book.chapter_count,
    )
    resolved_paragraph_encoder = paragraph_encoder or HeuristicEmbeddingEncoder()
    paragraph_report = build_book_paragraph_alignment_report(
        source_book,
        target_book,
        chapter_pairs=normalized_pairs,
        encoder=resolved_paragraph_encoder,
        model_name="heuristic" if isinstance(resolved_paragraph_encoder, HeuristicEmbeddingEncoder) else "embedding",
    )
    pair_lookup = {
        (int(item["source_chapter_index"]), int(item["target_chapter_index"])): item
        for item in paragraph_report["pair_reports"]
    }
    pair_reports = []
    for source_index, target_index in normalized_pairs:
        source_chapter = scoped_source_chapter(source_book.chapters[source_index - 1])
        target_chapter = target_book.chapters[target_index - 1]
        paragraph_pair_report = pair_lookup[(source_index, target_index)]
        span_rows = align_chapter_sentence_spans_with_paragraph_blocks(
            source_chapter,
            target_chapter,
            paragraph_blocks=list(paragraph_pair_report["matched_blocks"]),
            source_language=source_book.metadata.language,
            target_language=target_book.metadata.language,
            analysis_mode=analysis_mode,
            encoder=sentence_encoder,
        )
        enrichment_bundle = build_sentence_enrichment_bundle(
            chapter_alignments=[_to_chapter_sentence_alignment(row) for row in span_rows],
            source_language=source_book.metadata.language,
            target_language=target_book.metadata.language,
            analysis_mode=analysis_mode,
            subtree_encoder=sentence_encoder,
        )
        pair_reports.append(
            {
                "source_chapter_index": source_index,
                "target_chapter_index": target_index,
                "source_title": source_chapter.title,
                "target_title": target_chapter.title,
                "paragraph_metrics": paragraph_pair_report["metrics"],
                "sentence_metrics": _sentence_metrics(source_chapter, target_chapter, span_rows),
                "deep_metrics": _deep_metrics(enrichment_bundle),
            }
        )
    return {
        "chapter_pair_count": len(normalized_pairs),
        "analysis_mode": analysis_mode,
        "paragraph_global_metrics": paragraph_report["global_metrics"],
        "pair_reports": pair_reports,
        "global_metrics": _aggregate_hierarchical_metrics(pair_reports),
    }


def _sentence_metrics(source_chapter, target_chapter, span_rows) -> dict[str, float]:
    source_total = sum(len(paragraph.sentences) for paragraph in source_chapter.paragraphs)
    target_total = sum(len(paragraph.sentences) for paragraph in target_chapter.paragraphs)
    source_covered = {sentence_id for row in span_rows for sentence_id in row.source_sentence_ids}
    target_covered = {sentence_id for row in span_rows for sentence_id in row.target_sentence_ids}
    one_to_one = sum(1 for row in span_rows if row.sentence_span_alignment.alignment_type == "1:1")
    split_merge = len(span_rows) - one_to_one
    return {
        "source_sentence_coverage": round(len(source_covered) / max(source_total, 1), 6),
        "target_sentence_coverage": round(len(target_covered) / max(target_total, 1), 6),
        "one_to_one_row_rate": round(one_to_one / max(len(span_rows), 1), 6),
        "split_merge_row_rate": round(split_merge / max(len(span_rows), 1), 6),
    }


def _deep_metrics(enrichment_bundle: dict[str, object]) -> dict[str, float]:
    safe_items = [item for item in enrichment_bundle["items"] if item["enrichment_status"] == "sentence_safe"]
    source_word_total = 0
    target_word_total = 0
    phrase_source_covered: set[tuple[int, int]] = set()
    phrase_target_covered: set[tuple[int, int]] = set()
    dependency_source_covered: set[tuple[int, int]] = set()
    dependency_target_covered: set[tuple[int, int]] = set()
    subtree_source_covered: set[tuple[int, int]] = set()
    subtree_target_covered: set[tuple[int, int]] = set()

    for item_index, item in enumerate(safe_items):
        source_words = [token for token in item["source_tokens"] if token["kind"] == "word"]
        target_words = [token for token in item["target_tokens"] if token["kind"] == "word"]
        source_word_indices = {int(token["index"]) for token in source_words}
        target_word_indices = {int(token["index"]) for token in target_words}
        source_word_total += len(source_words)
        target_word_total += len(target_words)
        for candidate in item["phrase_candidates"]:
            _mark_span_tokens(phrase_source_covered, item_index, candidate["source_span"], source_word_indices)
            _mark_span_tokens(phrase_target_covered, item_index, candidate["target_span"], target_word_indices)
        for candidate in item["dependency_candidates"]:
            _mark_span_tokens(dependency_source_covered, item_index, candidate["source_span"], source_word_indices)
            _mark_span_tokens(dependency_target_covered, item_index, candidate["target_span"], target_word_indices)
        for candidate in item["subtree_candidates"]:
            _mark_span_tokens(subtree_source_covered, item_index, candidate["source_span"], source_word_indices)
            _mark_span_tokens(subtree_target_covered, item_index, candidate["target_span"], target_word_indices)

    return {
        "token_row_rate": round(sum(1 for item in safe_items if item["token_pairs"]) / max(len(safe_items), 1), 6),
        "phrase_row_rate": round(sum(1 for item in safe_items if item["phrase_candidates"]) / max(len(safe_items), 1), 6),
        "dependency_row_rate": round(sum(1 for item in safe_items if item["dependency_candidates"]) / max(len(safe_items), 1), 6),
        "subtree_row_rate": round(sum(1 for item in safe_items if item["subtree_candidates"]) / max(len(safe_items), 1), 6),
        "phrase_source_token_coverage": round(len(phrase_source_covered) / max(source_word_total, 1), 6),
        "phrase_target_token_coverage": round(len(phrase_target_covered) / max(target_word_total, 1), 6),
        "dependency_source_token_coverage": round(len(dependency_source_covered) / max(source_word_total, 1), 6),
        "dependency_target_token_coverage": round(len(dependency_target_covered) / max(target_word_total, 1), 6),
        "subtree_source_token_coverage": round(len(subtree_source_covered) / max(source_word_total, 1), 6),
        "subtree_target_token_coverage": round(len(subtree_target_covered) / max(target_word_total, 1), 6),
    }


def _aggregate_hierarchical_metrics(pair_reports: list[dict[str, object]]) -> dict[str, float]:
    if not pair_reports:
        return {}
    keys = (
        "source_paragraph_coverage",
        "target_paragraph_coverage",
        "source_sentence_coverage",
        "target_sentence_coverage",
        "one_to_one_row_rate",
        "split_merge_row_rate",
        "token_row_rate",
        "phrase_row_rate",
        "dependency_row_rate",
        "subtree_row_rate",
        "phrase_source_token_coverage",
        "phrase_target_token_coverage",
        "dependency_source_token_coverage",
        "dependency_target_token_coverage",
        "subtree_source_token_coverage",
        "subtree_target_token_coverage",
    )
    totals = {key: 0.0 for key in keys}
    for report in pair_reports:
        totals["source_paragraph_coverage"] += float(report["paragraph_metrics"]["source_paragraph_coverage"])
        totals["target_paragraph_coverage"] += float(report["paragraph_metrics"]["target_paragraph_coverage"])
        for key, value in report["sentence_metrics"].items():
            totals[key] += float(value)
        for key, value in report["deep_metrics"].items():
            totals[key] += float(value)
    count = len(pair_reports)
    return {key: round(value / count, 6) for key, value in totals.items()}


def _mark_span_tokens(
    bucket: set[tuple[int, int]],
    item_index: int,
    span: list[int],
    valid_indices: set[int],
) -> None:
    start = int(span[0])
    end = int(span[1])
    for token_index in range(start, end + 1):
        if token_index in valid_indices:
            bucket.add((item_index, token_index))


def _to_chapter_sentence_alignment(row) -> ChapterSentenceAlignment:
    span = row.sentence_span_alignment
    return ChapterSentenceAlignment(
        source_paragraph_id=row.source_paragraph_id,
        target_paragraph_id=row.target_paragraph_id,
        source_paragraph_index=row.source_paragraph_index,
        target_paragraph_index=row.target_paragraph_index,
        sentence_alignment=SentenceAlignment(
            source_index=span.source_span[0],
            target_index=span.target_span[0],
            source_text=span.source_text,
            target_text=span.target_text,
            score=span.score,
            signals=span.signals,
        ),
    )
