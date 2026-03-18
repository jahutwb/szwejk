from __future__ import annotations

import unittest

from szwejk.align.benchmark import DEFAULT_ALIGNMENT_BENCHMARK_PAIRS, format_chapter_pairs


class BenchmarkConfigUnitTest(unittest.TestCase):
    def test_default_alignment_benchmark_pairs_cover_four_curated_pairs(self) -> None:
        self.assertEqual(DEFAULT_ALIGNMENT_BENCHMARK_PAIRS, [(1, 1), (2, 2), (10, 10), (15, 15)])

    def test_format_chapter_pairs_renders_cli_friendly_string(self) -> None:
        self.assertEqual(format_chapter_pairs(DEFAULT_ALIGNMENT_BENCHMARK_PAIRS), "1:1,2:2,10:10,15:15")


if __name__ == "__main__":
    unittest.main()
