"""Coverage reporting for sentence-span and deeper alignment."""

from __future__ import annotations

from szwejk.align.candidates import build_monotonic_chapter_alignment
from szwejk.align.enrichment import build_sentence_enrichment_bundle
from szwejk.align.sentences import align_chapter_sentence_spans
from szwejk.schemas import CanonicalBook
from szwejk.generate.corpus_scope import apply_default_tail_chapter_overrides, scoped_source_chapter


def build_sentence_span_coverage_report(
    source_book: CanonicalBook,
    target_book: CanonicalBook,
    *,
    chapter_pairs: list[tuple[int, int]],
    paragraph_window: int = 3,
    analysis_mode: str = "stanza",
) -> dict[str, object]:
    normalized_pairs = apply_default_tail_chapter_overrides(
        chapter_pairs,
        source_chapter_count=source_book.chapter_count,
        target_chapter_count=target_book.chapter_count,
    )
    pair_reports = _build_pair_reports(
        source_book,
        target_book,
        chapter_pairs=normalized_pairs,
        paragraph_window=paragraph_window,
        analysis_mode=analysis_mode,
    )
    return {
        "chapter_pair_count": len(normalized_pairs),
        "analysis_mode": analysis_mode,
        "pair_reports": pair_reports,
        "global_metrics": _aggregate_global_metrics(pair_reports),
    }


def build_book_chapter_span_report(
    source_book: CanonicalBook,
    target_book: CanonicalBook,
    *,
    paragraph_window: int = 3,
    analysis_mode: str = "stanza",
    min_match_score: float = 0.72,
    skip_penalty: float = 0.2,
) -> dict[str, object]:
    chapter_alignment = build_monotonic_chapter_alignment(
        source_book,
        target_book,
        min_match_score=min_match_score,
        skip_penalty=skip_penalty,
    )
    chapter_pairs = apply_default_tail_chapter_overrides(
        [
        (item["source_chapter_index"], item["target_chapter_index"])
        for item in chapter_alignment["matches"]
        ],
        source_chapter_count=source_book.chapter_count,
        target_chapter_count=target_book.chapter_count,
    )
    pair_reports = _build_pair_reports(
        source_book,
        target_book,
        chapter_pairs=chapter_pairs,
        paragraph_window=paragraph_window,
        analysis_mode=analysis_mode,
    )
    return {
        "analysis_mode": analysis_mode,
        "paragraph_window": paragraph_window,
        "chapter_alignment": chapter_alignment,
        "matched_pair_count": len(chapter_pairs),
        "pair_reports": pair_reports,
        "global_metrics": _aggregate_global_metrics(pair_reports),
    }


def _build_pair_reports(
    source_book: CanonicalBook,
    target_book: CanonicalBook,
    *,
    chapter_pairs: list[tuple[int, int]],
    paragraph_window: int,
    analysis_mode: str,
) -> list[dict[str, object]]:
    pair_reports: list[dict[str, object]] = []
    for source_index, target_index in chapter_pairs:
        source_chapter = scoped_source_chapter(source_book.chapters[source_index - 1])
        target_chapter = target_book.chapters[target_index - 1]
        span_rows = align_chapter_sentence_spans(
            source_chapter,
            target_chapter,
            paragraph_window=paragraph_window,
            source_language=source_book.metadata.language,
            target_language=target_book.metadata.language,
            analysis_mode=analysis_mode,
        )
        enrichment = build_sentence_enrichment_bundle(
            chapter_alignments=[_to_chapter_sentence_alignment(row) for row in span_rows],
            source_language=source_book.metadata.language,
            target_language=target_book.metadata.language,
            analysis_mode=analysis_mode,
        )
        pair_reports.append(
            _build_pair_report(
                source_index=source_index,
                target_index=target_index,
                source_title=source_chapter.title,
                target_title=target_chapter.title,
                span_rows=span_rows,
                enrichment_bundle=enrichment,
                source_sentence_total=sum(len(paragraph.sentences) for paragraph in source_chapter.paragraphs),
                target_sentence_total=sum(len(paragraph.sentences) for paragraph in target_chapter.paragraphs),
                source_sentence_id_to_text={sentence.id: sentence.text for paragraph in source_chapter.paragraphs for sentence in paragraph.sentences},
                target_sentence_id_to_text={sentence.id: sentence.text for paragraph in target_chapter.paragraphs for sentence in paragraph.sentences},
            )
        )
    return pair_reports


def _build_pair_report(
    *,
    source_index: int,
    target_index: int,
    source_title: str,
    target_title: str,
    span_rows,
    enrichment_bundle: dict[str, object],
    source_sentence_total: int,
    target_sentence_total: int,
    source_sentence_id_to_text: dict[str, str],
    target_sentence_id_to_text: dict[str, str],
) -> dict[str, object]:
    source_covered: set[str] = set()
    target_covered: set[str] = set()
    one_to_one_rows = 0
    split_merge_rows = 0
    for row in span_rows:
        span = row.sentence_span_alignment
        source_covered.update(row.source_sentence_ids)
        target_covered.update(row.target_sentence_ids)
        if span.alignment_type == "1:1":
            one_to_one_rows += 1
        else:
            split_merge_rows += 1

    source_chars_total = sum(len(text) for text in source_sentence_id_to_text.values())
    target_chars_total = sum(len(text) for text in target_sentence_id_to_text.values())
    source_chars_covered = sum(len(source_sentence_id_to_text[sentence_id]) for sentence_id in source_covered)
    target_chars_covered = sum(len(target_sentence_id_to_text[sentence_id]) for sentence_id in target_covered)

    safe_items = [item for item in enrichment_bundle["items"] if item["enrichment_status"] == "sentence_safe"]
    phrase_row_count = sum(1 for item in safe_items if item["phrase_candidates"])
    token_row_count = sum(1 for item in safe_items if item["token_pairs"])
    dependency_row_count = sum(1 for item in safe_items if item["dependency_candidates"])
    subtree_row_count = sum(1 for item in safe_items if item.get("subtree_candidates"))
    counts = {
        "source_sentence_total": source_sentence_total,
        "target_sentence_total": target_sentence_total,
        "source_sentence_covered": len(source_covered),
        "target_sentence_covered": len(target_covered),
        "source_chars_total": source_chars_total,
        "target_chars_total": target_chars_total,
        "source_chars_covered": source_chars_covered,
        "target_chars_covered": target_chars_covered,
        "row_count": len(span_rows),
        "one_to_one_row_count": one_to_one_rows,
        "split_merge_row_count": split_merge_rows,
        "sentence_safe_row_count": len(safe_items),
        "phrase_row_count": phrase_row_count,
        "token_row_count": token_row_count,
        "dependency_row_count": dependency_row_count,
        "subtree_row_count": subtree_row_count,
    }
    metrics = {
        "source_sentence_coverage": round(len(source_covered) / source_sentence_total, 6) if source_sentence_total else 0.0,
        "target_sentence_coverage": round(len(target_covered) / target_sentence_total, 6) if target_sentence_total else 0.0,
        "one_to_one_row_rate": round(one_to_one_rows / len(span_rows), 6) if span_rows else 0.0,
        "split_merge_row_rate": round(split_merge_rows / len(span_rows), 6) if span_rows else 0.0,
        "source_char_coverage": round(source_chars_covered / source_chars_total, 6) if source_chars_total else 0.0,
        "target_char_coverage": round(target_chars_covered / target_chars_total, 6) if target_chars_total else 0.0,
        "phrase_row_rate": round(phrase_row_count / len(safe_items), 6) if safe_items else 0.0,
        "token_row_rate": round(token_row_count / len(safe_items), 6) if safe_items else 0.0,
        "dependency_row_rate": round(dependency_row_count / len(safe_items), 6) if safe_items else 0.0,
        "subtree_row_rate": round(subtree_row_count / len(safe_items), 6) if safe_items else 0.0,
    }
    return {
        "source_chapter_index": source_index,
        "target_chapter_index": target_index,
        "source_title": source_title,
        "target_title": target_title,
        "row_count": len(span_rows),
        "counts": counts,
        "metrics": metrics,
    }


def _aggregate_global_metrics(pair_reports: list[dict[str, object]]) -> dict[str, float]:
    if not pair_reports:
        return {
            "source_sentence_coverage": 0.0,
            "target_sentence_coverage": 0.0,
            "one_to_one_row_rate": 0.0,
            "split_merge_row_rate": 0.0,
            "source_char_coverage": 0.0,
            "target_char_coverage": 0.0,
            "phrase_row_rate": 0.0,
            "token_row_rate": 0.0,
            "dependency_row_rate": 0.0,
            "subtree_row_rate": 0.0,
        }

    source_sentence_total = sum(item["counts"]["source_sentence_total"] for item in pair_reports)
    target_sentence_total = sum(item["counts"]["target_sentence_total"] for item in pair_reports)
    source_sentence_covered = sum(item["counts"]["source_sentence_covered"] for item in pair_reports)
    target_sentence_covered = sum(item["counts"]["target_sentence_covered"] for item in pair_reports)
    source_chars_total = sum(item["counts"]["source_chars_total"] for item in pair_reports)
    target_chars_total = sum(item["counts"]["target_chars_total"] for item in pair_reports)
    source_chars_covered = sum(item["counts"]["source_chars_covered"] for item in pair_reports)
    target_chars_covered = sum(item["counts"]["target_chars_covered"] for item in pair_reports)
    row_count = sum(item["counts"]["row_count"] for item in pair_reports)
    one_to_one_row_count = sum(item["counts"]["one_to_one_row_count"] for item in pair_reports)
    split_merge_row_count = sum(item["counts"]["split_merge_row_count"] for item in pair_reports)
    sentence_safe_row_count = sum(item["counts"]["sentence_safe_row_count"] for item in pair_reports)
    phrase_row_count = sum(item["counts"]["phrase_row_count"] for item in pair_reports)
    token_row_count = sum(item["counts"]["token_row_count"] for item in pair_reports)
    dependency_row_count = sum(item["counts"]["dependency_row_count"] for item in pair_reports)
    subtree_row_count = sum(item["counts"]["subtree_row_count"] for item in pair_reports)
    return {
        "source_sentence_coverage": round(source_sentence_covered / source_sentence_total, 6) if source_sentence_total else 0.0,
        "target_sentence_coverage": round(target_sentence_covered / target_sentence_total, 6) if target_sentence_total else 0.0,
        "one_to_one_row_rate": round(one_to_one_row_count / row_count, 6) if row_count else 0.0,
        "split_merge_row_rate": round(split_merge_row_count / row_count, 6) if row_count else 0.0,
        "source_char_coverage": round(source_chars_covered / source_chars_total, 6) if source_chars_total else 0.0,
        "target_char_coverage": round(target_chars_covered / target_chars_total, 6) if target_chars_total else 0.0,
        "phrase_row_rate": round(phrase_row_count / sentence_safe_row_count, 6) if sentence_safe_row_count else 0.0,
        "token_row_rate": round(token_row_count / sentence_safe_row_count, 6) if sentence_safe_row_count else 0.0,
        "dependency_row_rate": round(dependency_row_count / sentence_safe_row_count, 6) if sentence_safe_row_count else 0.0,
        "subtree_row_rate": round(subtree_row_count / sentence_safe_row_count, 6) if sentence_safe_row_count else 0.0,
    }


def _to_chapter_sentence_alignment(row):
    from szwejk.align.sentences import ChapterSentenceAlignment, SentenceAlignment

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
