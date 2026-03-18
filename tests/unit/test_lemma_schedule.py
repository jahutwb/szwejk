from __future__ import annotations

import unittest

from szwejk.generate.lemma_schedule import build_family_schedule_report, compute_prefix_error, simulate_block


class LemmaScheduleTest(unittest.TestCase):
    def test_simulate_block_adds_gain_only_from_intro_onwards(self) -> None:
        result = simulate_block("x", 3, [0.0, 0.0, 0.0, 0.0], gain=0.2)
        self.assertEqual(result, [0.0, 0.0, 0.2, 0.2])

    def test_compute_prefix_error_respects_weights(self) -> None:
        error = compute_prefix_error([0.0, 0.2, 0.2], [0.1, 0.1, 0.3], [2.0, 1.0, 1.0])
        self.assertAlmostEqual(error, 0.4, places=8)

    def test_residual_greedy_can_shift_choice_when_early_prefix_weight_is_high(self) -> None:
        timeline = {
            "total_occurrence_count": 10,
            "families": [
                {
                    "family_id": "rare::vzacny",
                    "total_count": 1,
                    "last_unit_index": 2,
                    "occurrences": [
                        {
                            "unit_index": 2,
                            "future_gain_if_introduced_here": 0.1,
                            "remaining_after_occurrence": 0,
                            "urgency": 1.0,
                        }
                    ],
                },
                {
                    "family_id": "common::spolecny",
                    "total_count": 3,
                    "last_unit_index": 5,
                    "occurrences": [
                        {
                            "unit_index": 1,
                            "future_gain_if_introduced_here": 0.3,
                            "remaining_after_occurrence": 2,
                            "urgency": 0.333333,
                        },
                        {
                            "unit_index": 3,
                            "future_gain_if_introduced_here": 0.1,
                            "remaining_after_occurrence": 0,
                            "urgency": 1.0,
                        },
                    ],
                },
            ],
        }

        report = build_family_schedule_report(
            timeline,
            target_power=1.0,
            early_weight=3.0,
            max_steps=2,
            min_improvement=1e-12,
        )
        selected = {item["family_id"]: item["intro_unit_index"] for item in report["selected_families"]}

        self.assertEqual(selected["rare::vzacny"], 2)
        self.assertEqual(selected["common::spolecny"], 3)
        self.assertEqual(report["scheduled_family_count"], 2)
        self.assertLess(report["early_section_error_20"], report["total_prefix_l1_error"])


if __name__ == "__main__":
    unittest.main()
