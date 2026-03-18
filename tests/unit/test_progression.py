from __future__ import annotations

import unittest

from szwejk.generate.progression import select_progression_steps, ProgressionCandidate


class ProgressionUnitTest(unittest.TestCase):
    def test_select_progression_prefers_already_introduced_family_candidate(self) -> None:
        candidates = [
            ProgressionCandidate(
                chapter_pair=(1, 1),
                sentence_index=1,
                granularity="token",
                source_text="nam",
                target_text="nám",
                source_span=(1, 1),
                target_span=(1, 1),
                global_source_start=1,
                global_source_end=1,
                global_target_start=1,
                global_target_end=1,
                family_ids=["my::my"],
                score=0.92,
                metadata={"relation": "lemma_like", "source_upos": "PRON"},
            ),
            ProgressionCandidate(
                chapter_pair=(1, 1),
                sentence_index=1,
                granularity="phrase",
                source_text="nam zabili Ferdynanda",
                target_text="nám zabili Ferdinanda",
                source_span=(1, 3),
                target_span=(1, 3),
                global_source_start=1,
                global_source_end=3,
                global_target_start=1,
                global_target_end=3,
                family_ids=["my::my", "zabic::zabit", "ferdynand::ferdinand"],
                score=0.88,
                metadata={"relation": "parallel_span"},
            ),
        ]

        steps, family_weights, cursor_gaps = select_progression_steps(
            candidates,
            total_source_positions=3,
            total_target_positions=3,
        )

        self.assertEqual(steps[0]["granularity"], "token")
        self.assertEqual(steps[0]["delta"]["novelty"], 0.0)
        self.assertEqual(steps[0]["delta"]["difficulty"], 0.0)
        self.assertTrue(family_weights)
        self.assertEqual(cursor_gaps, [])

    def test_select_progression_advances_both_source_and_target_after_span(self) -> None:
        candidates = [
            ProgressionCandidate(
                chapter_pair=(1, 1),
                sentence_index=1,
                granularity="phrase",
                source_text="nam zabili Ferdynanda",
                target_text="nám zabili Ferdinanda",
                source_span=(1, 3),
                target_span=(2, 4),
                global_source_start=3,
                global_source_end=5,
                global_target_start=3,
                global_target_end=5,
                family_ids=["zabic::zabit", "ferdynand::ferdinand"],
                score=0.88,
                metadata={"relation": "parallel_span"},
            ),
            ProgressionCandidate(
                chapter_pair=(1, 1),
                sentence_index=2,
                granularity="token",
                source_text="dalej",
                target_text="dál",
                source_span=(4, 4),
                target_span=(5, 5),
                global_source_start=6,
                global_source_end=6,
                global_target_start=6,
                global_target_end=6,
                family_ids=["dalej::dal"],
                score=0.7,
                metadata={"relation": "cognate"},
            ),
        ]

        steps, _family_weights, cursor_gaps = select_progression_steps(
            candidates,
            total_source_positions=6,
            total_target_positions=6,
        )

        self.assertEqual(steps[0]["global_source_start"], 3)
        self.assertEqual(steps[0]["global_source_end"], 5)
        self.assertEqual(steps[0]["global_target_end"], 5)
        self.assertEqual(steps[1]["global_source_start"], 6)
        self.assertEqual(steps[1]["global_target_start"], 6)
        self.assertEqual(cursor_gaps, [])

    def test_select_progression_uses_absolute_distance_to_target_vector(self) -> None:
        candidates = [
            ProgressionCandidate(
                chapter_pair=(1, 1),
                sentence_index=2,
                granularity="token",
                source_text="pole",
                target_text="pól",
                source_span=(2, 2),
                target_span=(2, 2),
                global_source_start=2,
                global_source_end=2,
                global_target_start=2,
                global_target_end=2,
                family_ids=["pole::pul"],
                score=0.8,
                metadata={"relation": "cognate", "source_upos": "NOUN"},
            ),
            ProgressionCandidate(
                chapter_pair=(1, 1),
                sentence_index=2,
                granularity="subtree",
                source_text="na wielkim polu",
                target_text="na velikém poli",
                source_span=(2, 4),
                target_span=(2, 4),
                global_source_start=2,
                global_source_end=4,
                global_target_start=2,
                global_target_end=4,
                family_ids=["na::na", "wielki::veliky", "pole::pole"],
                score=0.95,
                metadata={"relation": "parallel_span"},
            ),
        ]

        steps, _family_weights, _cursor_gaps = select_progression_steps(
            candidates,
            total_source_positions=4,
            total_target_positions=4,
        )

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["granularity"], "token")
        self.assertLess(steps[0]["distance_after"], steps[0]["distance_before"])

    def test_select_progression_filters_low_confidence_candidate(self) -> None:
        candidates = [
            ProgressionCandidate(
                chapter_pair=(1, 1),
                sentence_index=1,
                granularity="token",
                source_text="domu",
                target_text="domu",
                source_span=(1, 1),
                target_span=(1, 1),
                global_source_start=1,
                global_source_end=1,
                global_target_start=1,
                global_target_end=1,
                family_ids=["dom::dum"],
                score=0.3,
                metadata={"relation": "weak", "source_upos": "NOUN"},
            )
        ]

        steps, _family_weights, _cursor_gaps = select_progression_steps(
            candidates,
            total_source_positions=2,
            total_target_positions=2,
        )

        self.assertEqual(steps, [])
