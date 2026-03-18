from __future__ import annotations

import unittest

from szwejk.policy.benchmark import build_policy_benchmark_report
from szwejk.policy.schema import load_policy_from_json
from szwejk.review import load_book_from_json


class PolicyBenchmarkUnitTest(unittest.TestCase):
    def test_build_policy_benchmark_report_aggregates_pair_results(self) -> None:
        policy = load_policy_from_json("data/policies/progressive_czechization_v1.json").policy
        source_book = load_book_from_json("data/corpus/pl_szwejk.json")
        target_book = load_book_from_json("data/corpus/cs_szwejk.json")

        report = build_policy_benchmark_report(
            source_book,
            target_book,
            policy=policy,
            level_id="l1-lexical-onboarding",
            chapter_pairs=[(1, 1)],
            analysis_mode="heuristic",
        )

        self.assertEqual(report["chapter_pair_count"], 1)
        self.assertIn("global_summary", report)
        self.assertIn("pair_reports", report)
        self.assertEqual(report["pair_reports"][0]["level_id"], "l1-lexical-onboarding")
