"""Review-oriented summaries for chapter-level policy plans."""

from __future__ import annotations

from collections import Counter

from .evaluate import PolicyChapterPlan


def build_policy_review_report(plan: PolicyChapterPlan | dict[str, object]) -> dict[str, object]:
    payload = plan.to_dict() if isinstance(plan, PolicyChapterPlan) else plan
    selected = list(payload.get("selected_candidates", []))
    blocked = list(payload.get("blocked_candidates", []))

    unique_families = Counter(item["family_id"] for item in selected)
    blocked_reasons = Counter(item["reason"] for item in blocked)
    selected_reasons = Counter(item["selection_reason"] for item in selected)
    support_depths = Counter(item["support_depth"] for item in selected)
    unit_classes = Counter(item["unit_class"] for item in selected)
    granularities = Counter(item.get("granularity", "token") for item in selected)

    actual_surface = float(payload["actual_surface_czechness"])
    target_surface = float(payload["target_surface_czechness"])
    gap_to_target = max(target_surface - actual_surface, 0.0)

    top_families = [
        {
            "family_id": family_id,
            "occurrences": count,
            "example_source": next(item["source_text"] for item in selected if item["family_id"] == family_id),
            "example_target": next(item["target_text"] for item in selected if item["family_id"] == family_id),
        }
        for family_id, count in unique_families.most_common(12)
    ]
    recommendations = _build_recommendations(
        actual_surface=actual_surface,
        target_surface=target_surface,
        blocked_reasons=blocked_reasons,
        support_depths=support_depths,
    )

    return {
        "policy_id": payload["policy_id"],
        "level_id": payload["level_id"],
        "source_chapter_index": payload["source_chapter_index"],
        "target_chapter_index": payload["target_chapter_index"],
        "source_title": payload["source_title"],
        "target_title": payload["target_title"],
        "summary": {
            "actual_surface_czechness": round(actual_surface, 6),
            "target_surface_czechness": round(target_surface, 6),
            "gap_to_target": round(gap_to_target, 6),
            "selected_candidate_count": len(selected),
            "selected_family_count": len(unique_families),
            "blocked_candidate_count": len(blocked),
        },
        "selected_breakdown": {
            "selection_reasons": dict(selected_reasons),
            "granularities": dict(granularities),
            "support_depths": dict(support_depths),
            "unit_classes": dict(unit_classes),
            "top_families": top_families,
        },
        "blocked_breakdown": {
            "reason_counts": dict(blocked_reasons),
            "sample_blocked": blocked[:12],
        },
        "recommendations": recommendations,
    }


def _build_recommendations(
    *,
    actual_surface: float,
    target_surface: float,
    blocked_reasons: Counter[str],
    support_depths: Counter[str],
) -> list[str]:
    recommendations: list[str] = []
    if actual_surface < (target_surface * 0.5):
        recommendations.append("planner is materially below target surface; review gates before generation")
    if blocked_reasons.get("alignment_confidence_below_threshold", 0) > max(sum(blocked_reasons.values()) * 0.6, 0):
        recommendations.append("alignment confidence is the main limiter; calibrate thresholds or widen accepted high-confidence rows")
    if blocked_reasons.get("content_weight_below_0.45", 0) > 0:
        recommendations.append("low-content function words are being filtered from L1 as intended")
    if support_depths and support_depths.get("subtree", 0) == sum(support_depths.values()):
        recommendations.append("all accepted L1 substitutions currently rely on subtree support, which is conservative but stable")
    if not recommendations:
        recommendations.append("current policy plan is balanced enough to move into text materialization experiments")
    return recommendations
