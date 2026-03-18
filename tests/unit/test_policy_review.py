from __future__ import annotations

import unittest

from szwejk.policy import build_policy_review_report


class PolicyReviewUnitTest(unittest.TestCase):
    def test_build_policy_review_report_summarizes_gap_and_reasons(self) -> None:
        payload = {
            "policy_id": "p1",
            "level_id": "l1",
            "source_chapter_index": 2,
            "target_chapter_index": 2,
            "source_title": "PL",
            "target_title": "CS",
            "actual_surface_czechness": 0.02,
            "target_surface_czechness": 0.12,
            "selected_candidates": [
                {
                    "granularity": "token",
                    "family_id": "pan::pan",
                    "source_text": "pana",
                    "target_text": "pana",
                    "selection_reason": "best_positive_candidate",
                    "support_depth": "subtree",
                    "unit_class": "A",
                },
                {
                    "granularity": "token",
                    "family_id": "pan::pan",
                    "source_text": "panu",
                    "target_text": "panu",
                    "selection_reason": "family_propagation",
                    "support_depth": "subtree",
                    "unit_class": "A",
                },
            ],
            "blocked_candidates": [
                {"family_id": "my::my", "source_text": "nam", "target_text": "nám", "reason": "alignment_confidence_below_threshold"},
                {"family_id": "a::a", "source_text": "a", "target_text": "a", "reason": "content_weight_below_0.45"},
            ],
        }

        report = build_policy_review_report(payload)

        self.assertEqual(report["summary"]["selected_family_count"], 1)
        self.assertEqual(report["summary"]["gap_to_target"], 0.1)
        self.assertEqual(report["selected_breakdown"]["selection_reasons"]["family_propagation"], 1)
        self.assertEqual(report["selected_breakdown"]["granularities"]["token"], 2)
        self.assertIn("planner is materially below target surface; review gates before generation", report["recommendations"])
