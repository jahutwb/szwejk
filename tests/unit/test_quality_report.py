from __future__ import annotations

import unittest

from szwejk.align.quality import (
    DEFAULT_QUALITY_THRESHOLDS,
    build_pair_quality_report,
    evaluate_quality_thresholds,
)


class QualityReportUnitTest(unittest.TestCase):
    def test_evaluate_quality_thresholds_marks_pass_and_fail_metrics(self) -> None:
        metrics = {
            "sentence_safe_ratio": 0.82,
            "blocked_ratio": 0.18,
            "source_token_match_rate": 0.19,
            "target_token_match_rate": 0.17,
            "phrase_row_rate": 0.21,
        }

        verdict = evaluate_quality_thresholds(metrics, thresholds=DEFAULT_QUALITY_THRESHOLDS)

        self.assertEqual(verdict["overall_status"], "pass")
        self.assertTrue(all(item["status"] == "pass" for item in verdict["checks"]))

    def test_build_pair_quality_report_summarizes_safe_depth_and_granularity(self) -> None:
        enrichment_bundle = {
            "analysis_mode": "heuristic",
            "summary": {
                "total_items": 4,
                "sentence_safe_items": 3,
                "blocked_items": 1,
            },
            "items": [
                {
                    "enrichment_status": "sentence_safe",
                    "source_tokens": [{"index": 1, "kind": "word"}, {"index": 2, "kind": "word"}],
                    "target_tokens": [{"index": 1, "kind": "word"}, {"index": 2, "kind": "word"}],
                    "token_pairs": [{"source_token_index": 1, "target_token_index": 1}],
                    "phrase_candidates": [{"source_span": [1, 2], "target_span": [1, 2]}],
                    "dependency_candidates": [],
                },
                {
                    "enrichment_status": "sentence_safe",
                    "source_tokens": [{"index": 1, "kind": "word"}],
                    "target_tokens": [{"index": 1, "kind": "word"}],
                    "token_pairs": [{"source_token_index": 1, "target_token_index": 1}],
                    "phrase_candidates": [],
                    "dependency_candidates": [],
                },
                {
                    "enrichment_status": "sentence_safe",
                    "source_tokens": [{"index": 1, "kind": "word"}],
                    "target_tokens": [{"index": 1, "kind": "word"}],
                    "token_pairs": [],
                    "phrase_candidates": [],
                    "dependency_candidates": [],
                },
                {
                    "enrichment_status": "blocked_by_safe_depth",
                    "source_tokens": [{"index": 1, "kind": "word"}],
                    "target_tokens": [{"index": 1, "kind": "word"}],
                    "token_pairs": [],
                    "phrase_candidates": [],
                    "dependency_candidates": [],
                },
            ],
        }

        report = build_pair_quality_report(
            source_chapter_index=2,
            target_chapter_index=2,
            source_title="Rozdział I",
            target_title="Zasáhnutí",
            enrichment_bundle=enrichment_bundle,
            thresholds=DEFAULT_QUALITY_THRESHOLDS,
        )

        self.assertEqual(report["safe_depth"]["sentence_safe_items"], 3)
        self.assertEqual(report["safe_depth"]["blocked_items"], 1)
        self.assertEqual(report["analysis_mode"], "heuristic")
        self.assertAlmostEqual(report["granularity"]["sentence_safe_ratio"], 0.75)
        self.assertAlmostEqual(report["granularity"]["phrase_row_rate"], 1 / 3, places=6)
        self.assertAlmostEqual(report["granularity"]["dependency_row_rate"], 0.0)
        self.assertIn("threshold_verdict", report)


if __name__ == "__main__":
    unittest.main()
