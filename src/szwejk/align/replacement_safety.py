"""Replacement safety metrics for token, phrase, and subtree candidates."""

from __future__ import annotations

from collections import Counter

from szwejk.align.diagnostics import classify_confidence
from szwejk.align.embedding_eval import HeuristicEmbeddingEncoder, SentenceEncoder
from szwejk.align.enrichment import build_sentence_enrichment_bundle
from szwejk.align.hierarchical import build_hierarchical_alignment_report
from szwejk.align.paragraphs import build_book_paragraph_alignment_report
from szwejk.align.sentence_blocks import align_chapter_sentence_spans_with_paragraph_blocks
from szwejk.align.sentences import ChapterSentenceAlignment, SentenceAlignment
from szwejk.generate.corpus_scope import apply_default_tail_chapter_overrides, scoped_source_chapter
from szwejk.schemas import CanonicalBook


PAIRING_BUCKETS = (
    "high",
    "medium",
    "low",
    "dictionary",
    "embedding_assisted",
    "manual_forced",
)


def build_replacement_safety_report(
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
                "token_coverage": _token_coverage_metrics(enrichment_bundle),
                "confidence_distribution": _confidence_distribution(enrichment_bundle),
            }
        )
    return {
        "chapter_pair_count": len(normalized_pairs),
        "analysis_mode": analysis_mode,
        "pair_reports": pair_reports,
        "global_metrics": _aggregate_metrics(pair_reports),
    }


def _token_coverage_metrics(enrichment_bundle: dict[str, object]) -> dict[str, float]:
    source_word_total = 0
    target_word_total = 0
    source_word_matched = 0
    target_word_matched = 0
    unresolved_source = 0
    unresolved_target = 0

    for item in enrichment_bundle["items"]:
        if item["enrichment_status"] != "sentence_safe":
            continue
        source_words = [token for token in item["source_tokens"] if token["kind"] == "word"]
        target_words = [token for token in item["target_tokens"] if token["kind"] == "word"]
        source_word_total += len(source_words)
        target_word_total += len(target_words)
        matched_source_indices = {int(pair["source_token_index"]) for pair in item["token_pairs"]}
        matched_target_indices = {int(pair["target_token_index"]) for pair in item["token_pairs"]}
        source_word_matched += len(matched_source_indices)
        target_word_matched += len(matched_target_indices)
        unresolved_source += len([token for token in source_words if int(token["index"]) not in matched_source_indices])
        unresolved_target += len([token for token in target_words if int(token["index"]) not in matched_target_indices])

    return {
        "source_token_coverage": round(source_word_matched / max(source_word_total, 1), 6),
        "target_token_coverage": round(target_word_matched / max(target_word_total, 1), 6),
        "source_unresolved_token_rate": round(unresolved_source / max(source_word_total, 1), 6),
        "target_unresolved_token_rate": round(unresolved_target / max(target_word_total, 1), 6),
    }


def _confidence_distribution(enrichment_bundle: dict[str, object]) -> dict[str, dict[str, int]]:
    token_counter = Counter({bucket: 0 for bucket in PAIRING_BUCKETS})
    phrase_counter = Counter({bucket: 0 for bucket in PAIRING_BUCKETS})
    subtree_counter = Counter({bucket: 0 for bucket in PAIRING_BUCKETS})

    for item in enrichment_bundle["items"]:
        if item["enrichment_status"] != "sentence_safe":
            continue
        for pair in item["token_pairs"]:
            token_counter[_pair_bucket(float(pair["score"]), relation=str(pair.get("relation", "")))] += 1
        for candidate in item["phrase_candidates"]:
            phrase_counter[_pair_bucket(float(candidate["score"]), relation=str(candidate.get("relation", "")))] += 1
        for candidate in item["subtree_candidates"]:
            subtree_counter[_pair_bucket(float(candidate["score"]), relation=str(candidate.get("relation", "")))] += 1

    return {
        "token": dict(token_counter),
        "phrase": dict(phrase_counter),
        "subtree": dict(subtree_counter),
    }


def _pair_bucket(score: float, *, relation: str) -> str:
    if relation == "dictionary":
        return "dictionary"
    if relation == "embedding_assisted":
        return "embedding_assisted"
    if relation == "manual_forced":
        return "manual_forced"
    return classify_confidence(score)


def _aggregate_metrics(pair_reports: list[dict[str, object]]) -> dict[str, object]:
    if not pair_reports:
        return {}
    token_coverage_keys = (
        "source_token_coverage",
        "target_token_coverage",
        "source_unresolved_token_rate",
        "target_unresolved_token_rate",
    )
    token_totals = {key: 0.0 for key in token_coverage_keys}
    confidence_totals = {
        "token": Counter({bucket: 0 for bucket in PAIRING_BUCKETS}),
        "phrase": Counter({bucket: 0 for bucket in PAIRING_BUCKETS}),
        "subtree": Counter({bucket: 0 for bucket in PAIRING_BUCKETS}),
    }
    for report in pair_reports:
        for key in token_coverage_keys:
            token_totals[key] += float(report["token_coverage"][key])
        for level in confidence_totals:
            confidence_totals[level].update(report["confidence_distribution"][level])
    count = len(pair_reports)
    return {
        "token_coverage": {key: round(value / count, 6) for key, value in token_totals.items()},
        "confidence_distribution": {
            level: dict(counter)
            for level, counter in confidence_totals.items()
        },
    }


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
