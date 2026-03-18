from __future__ import annotations

import unittest

import numpy as np

from szwejk.align.tail_volume4 import ParagraphItem, align_monotonic_many_to_many


class TailVolume4UnitTest(unittest.TestCase):
    def test_many_to_many_alignment_prefers_two_to_one_block_when_semantically_stronger(self) -> None:
        source_items = [
            ParagraphItem(
                seq=1,
                chapter_index=29,
                chapter_title="III",
                paragraph_index=1,
                paragraph_id="pl-1",
                text="A",
                char_len=10,
                sentence_count=1,
                foreign_ratio=0.0,
                normalized_pos=0.0,
                embedding=np.array([1.0, 0.0], dtype=float),
            ),
            ParagraphItem(
                seq=2,
                chapter_index=29,
                chapter_title="III",
                paragraph_index=2,
                paragraph_id="pl-2",
                text="B",
                char_len=10,
                sentence_count=1,
                foreign_ratio=0.0,
                normalized_pos=0.5,
                embedding=np.array([1.0, 0.0], dtype=float),
            ),
        ]
        target_items = [
            ParagraphItem(
                seq=1,
                chapter_index=29,
                chapter_title="Marškumpanie",
                paragraph_index=1,
                paragraph_id="cs-1",
                text="AB",
                char_len=20,
                sentence_count=2,
                foreign_ratio=0.0,
                normalized_pos=0.0,
                embedding=np.array([1.0, 0.0], dtype=float),
            )
        ]

        report = align_monotonic_many_to_many(source_items, target_items, max_source_span=2, max_target_span=1)

        self.assertEqual(report["source_coverage"], 1.0)
        self.assertEqual(report["target_coverage"], 1.0)
        self.assertEqual(len(report["matched_blocks"]), 1)
        self.assertEqual(report["matched_blocks"][0]["source_range"], [1, 2])
        self.assertEqual(report["matched_blocks"][0]["target_range"], [1, 1])
        self.assertEqual(report["unmatched_source"], [])
        self.assertEqual(report["unmatched_target"], [])


if __name__ == "__main__":
    unittest.main()
