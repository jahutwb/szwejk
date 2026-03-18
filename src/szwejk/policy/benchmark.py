"""Policy tuning reports across multiple chapter pairs."""

from __future__ import annotations

from collections import Counter

from szwejk.schemas import CanonicalBook

from .evaluate import build_chapter_policy_plan
from .review import build_policy_review_report
from .schema import HybridizationPolicy


def build_policy_benchmark_report(
    source_book: CanonicalBook,
    target_book: CanonicalBook,
    *,
    policy: HybridizationPolicy,
    level_id: str,
    chapter_pairs: list[tuple[int, int]],
    paragraph_window: int = 3,
    analysis_mode: str = "stanza",
) -> dict[str, object]:
    pair_reports: list[dict[str, object]] = []
    selected_candidate_total = 0
    selected_family_total = 0
    blocked_candidate_total = 0
    actual_surface_total = 0.0
    target_surface_total = 0.0
    blocked_reasons = Counter()
    recommendations = Counter()

    for source_chapter_index, target_chapter_index in chapter_pairs:
        plan = build_chapter_policy_plan(
            source_book,
            target_book,
            source_chapter_index=source_chapter_index,
            target_chapter_index=target_chapter_index,
            policy=policy,
            level_id=level_id,
            paragraph_window=paragraph_window,
            analysis_mode=analysis_mode,
        )
        review = build_policy_review_report(plan)
        pair_reports.append(review)

        summary = review["summary"]
        selected_candidate_total += int(summary["selected_candidate_count"])
        selected_family_total += int(summary["selected_family_count"])
        blocked_candidate_total += int(summary["blocked_candidate_count"])
        actual_surface_total += float(summary["actual_surface_czechness"])
        target_surface_total += float(summary["target_surface_czechness"])
        blocked_reasons.update(review["blocked_breakdown"]["reason_counts"])
        recommendations.update(review["recommendations"])

    pair_count = len(pair_reports)
    under_target_pairs = [
        {
            "source_chapter_index": item["source_chapter_index"],
            "target_chapter_index": item["target_chapter_index"],
            "source_title": item["source_title"],
            "target_title": item["target_title"],
            "actual_surface_czechness": item["summary"]["actual_surface_czechness"],
            "target_surface_czechness": item["summary"]["target_surface_czechness"],
            "gap_to_target": item["summary"]["gap_to_target"],
        }
        for item in pair_reports
        if float(item["summary"]["actual_surface_czechness"]) < (float(item["summary"]["target_surface_czechness"]) * 0.5)
    ]

    return {
        "policy_id": policy.id,
        "level_id": level_id,
        "analysis_mode": analysis_mode,
        "chapter_pair_count": pair_count,
        "pair_reports": pair_reports,
        "global_summary": {
            "avg_actual_surface_czechness": round(actual_surface_total / pair_count, 6) if pair_count else 0.0,
            "avg_target_surface_czechness": round(target_surface_total / pair_count, 6) if pair_count else 0.0,
            "selected_candidate_total": selected_candidate_total,
            "selected_family_total": selected_family_total,
            "blocked_candidate_total": blocked_candidate_total,
            "blocked_reason_counts": dict(blocked_reasons),
            "recommendation_counts": dict(recommendations),
            "under_target_pair_count": len(under_target_pairs),
        },
        "under_target_pairs": under_target_pairs,
    }
