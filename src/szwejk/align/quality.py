"""Quality reporting for safe-depth and smaller-unit alignment coverage."""

from __future__ import annotations

from statistics import fmean

from szwejk.align.enrichment import build_sentence_enrichment_bundle
from szwejk.align.sentences import align_chapter_sentences
from szwejk.schemas import CanonicalBook


DEFAULT_QUALITY_THRESHOLDS = {
    "sentence_safe_ratio_min": 0.80,
    "blocked_ratio_max": 0.20,
    "source_token_match_rate_min": 0.15,
    "target_token_match_rate_min": 0.15,
    "phrase_row_rate_min": 0.15,
}


def evaluate_quality_thresholds(
    metrics: dict[str, float],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    active = thresholds or DEFAULT_QUALITY_THRESHOLDS
    checks = [
        _make_check("sentence_safe_ratio", metrics["sentence_safe_ratio"], ">=", active["sentence_safe_ratio_min"]),
        _make_check("blocked_ratio", metrics["blocked_ratio"], "<=", active["blocked_ratio_max"]),
        _make_check("source_token_match_rate", metrics["source_token_match_rate"], ">=", active["source_token_match_rate_min"]),
        _make_check("target_token_match_rate", metrics["target_token_match_rate"], ">=", active["target_token_match_rate_min"]),
        _make_check("phrase_row_rate", metrics["phrase_row_rate"], ">=", active["phrase_row_rate_min"]),
    ]
    return {
        "overall_status": "pass" if all(item["status"] == "pass" for item in checks) else "fail",
        "checks": checks,
    }


def build_pair_quality_report(
    *,
    source_chapter_index: int,
    target_chapter_index: int,
    source_title: str,
    target_title: str,
    enrichment_bundle: dict[str, object],
    thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    metrics = _compute_granularity_metrics(enrichment_bundle)
    return {
        "source_chapter_index": source_chapter_index,
        "target_chapter_index": target_chapter_index,
        "source_title": source_title,
        "target_title": target_title,
        "safe_depth": {
            "total_items": enrichment_bundle["summary"]["total_items"],
            "sentence_safe_items": enrichment_bundle["summary"]["sentence_safe_items"],
            "blocked_items": enrichment_bundle["summary"]["blocked_items"],
        },
        "analysis_mode": enrichment_bundle.get("analysis_mode", "heuristic"),
        "granularity": metrics,
        "threshold_verdict": evaluate_quality_thresholds(metrics, thresholds=thresholds),
    }


def build_alignment_quality_report(
    source_book: CanonicalBook,
    target_book: CanonicalBook,
    *,
    chapter_pairs: list[tuple[int, int]],
    paragraph_window: int = 3,
    thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    pair_reports: list[dict[str, object]] = []
    pair_metrics: list[dict[str, float]] = []

    for source_chapter_index, target_chapter_index in chapter_pairs:
        chapter_alignments = align_chapter_sentences(
            source_book.chapters[source_chapter_index - 1],
            target_book.chapters[target_chapter_index - 1],
            paragraph_window=paragraph_window,
        )
        enrichment_bundle = build_sentence_enrichment_bundle(
            chapter_alignments=chapter_alignments,
            source_language=source_book.metadata.language,
            target_language=target_book.metadata.language,
            analysis_mode="heuristic",
        )
        report = build_pair_quality_report(
            source_chapter_index=source_chapter_index,
            target_chapter_index=target_chapter_index,
            source_title=source_book.chapters[source_chapter_index - 1].title,
            target_title=target_book.chapters[target_chapter_index - 1].title,
            enrichment_bundle=enrichment_bundle,
            thresholds=thresholds,
        )
        pair_reports.append(report)
        pair_metrics.append(report["granularity"])

    global_metrics = {
        "sentence_safe_ratio": round(fmean(item["sentence_safe_ratio"] for item in pair_metrics), 6),
        "blocked_ratio": round(fmean(item["blocked_ratio"] for item in pair_metrics), 6),
        "source_token_match_rate": round(fmean(item["source_token_match_rate"] for item in pair_metrics), 6),
        "target_token_match_rate": round(fmean(item["target_token_match_rate"] for item in pair_metrics), 6),
        "phrase_row_rate": round(fmean(item["phrase_row_rate"] for item in pair_metrics), 6),
        "dependency_row_rate": round(fmean(item["dependency_row_rate"] for item in pair_metrics), 6),
    }
    return {
        "chapter_pair_count": len(chapter_pairs),
        "pair_reports": pair_reports,
        "global_granularity": global_metrics,
        "global_threshold_verdict": evaluate_quality_thresholds(global_metrics, thresholds=thresholds),
    }


def _compute_granularity_metrics(enrichment_bundle: dict[str, object]) -> dict[str, float]:
    total_items = int(enrichment_bundle["summary"]["total_items"])
    sentence_safe_items = int(enrichment_bundle["summary"]["sentence_safe_items"])
    blocked_items = int(enrichment_bundle["summary"]["blocked_items"])

    source_word_total = 0
    target_word_total = 0
    source_word_matched = 0
    target_word_matched = 0
    phrase_rows = 0
    dependency_rows = 0

    for item in enrichment_bundle["items"]:
        if item["enrichment_status"] != "sentence_safe":
            continue
        source_words = [token for token in item["source_tokens"] if token["kind"] == "word"]
        target_words = [token for token in item["target_tokens"] if token["kind"] == "word"]
        source_word_total += len(source_words)
        target_word_total += len(target_words)
        source_word_matched += len({pair["source_token_index"] for pair in item["token_pairs"]})
        target_word_matched += len({pair["target_token_index"] for pair in item["token_pairs"]})
        if item["phrase_candidates"]:
            phrase_rows += 1
        if item["dependency_candidates"]:
            dependency_rows += 1

    sentence_safe_ratio = (sentence_safe_items / total_items) if total_items else 0.0
    blocked_ratio = (blocked_items / total_items) if total_items else 0.0
    source_token_match_rate = (source_word_matched / source_word_total) if source_word_total else 0.0
    target_token_match_rate = (target_word_matched / target_word_total) if target_word_total else 0.0
    phrase_row_rate = (phrase_rows / sentence_safe_items) if sentence_safe_items else 0.0
    dependency_row_rate = (dependency_rows / sentence_safe_items) if sentence_safe_items else 0.0

    return {
        "sentence_safe_ratio": round(sentence_safe_ratio, 6),
        "blocked_ratio": round(blocked_ratio, 6),
        "source_token_match_rate": round(source_token_match_rate, 6),
        "target_token_match_rate": round(target_token_match_rate, 6),
        "phrase_row_rate": round(phrase_row_rate, 6),
        "dependency_row_rate": round(dependency_row_rate, 6),
    }


def _make_check(metric: str, actual: float, operator: str, threshold: float) -> dict[str, object]:
    if operator == ">=":
        status = "pass" if actual >= threshold else "fail"
    else:
        status = "pass" if actual <= threshold else "fail"
    return {
        "metric": metric,
        "actual": round(actual, 6),
        "operator": operator,
        "threshold": round(threshold, 6),
        "status": status,
    }
