from __future__ import annotations

import unittest

from szwejk.align.diagnostics import (
    classify_confidence,
    summarize_sentence_alignments,
)
from szwejk.align.sentences import SentenceAlignment


class AlignmentDiagnosticsUnitTest(unittest.TestCase):
    def test_classify_confidence_thresholds(self) -> None:
        self.assertEqual(classify_confidence(0.72), "high")
        self.assertEqual(classify_confidence(0.55), "medium")
        self.assertEqual(classify_confidence(0.32), "low")

    def test_summarize_sentence_alignments_emits_distribution_and_low_confidence_cases(self) -> None:
        alignments = [
            SentenceAlignment(1, 1, "a", "a", 0.72, {"length": 1.0, "char_trigram": 0.7, "punctuation": 1.0, "digit_pattern": 1.0, "total": 0.72}),
            SentenceAlignment(2, 2, "b", "b", 0.51, {"length": 0.8, "char_trigram": 0.4, "punctuation": 1.0, "digit_pattern": 1.0, "total": 0.51}),
            SentenceAlignment(3, 4, "c", "d", 0.28, {"length": 0.4, "char_trigram": 0.1, "punctuation": 0.2, "digit_pattern": 1.0, "total": 0.28}),
        ]

        summary = summarize_sentence_alignments(alignments)

        self.assertEqual(summary["count"], 3)
        self.assertIn("score_distribution", summary)
        self.assertEqual(summary["confidence_counts"]["high"], 1)
        self.assertEqual(summary["confidence_counts"]["medium"], 1)
        self.assertEqual(summary["confidence_counts"]["low"], 1)
        self.assertEqual(len(summary["difficult_cases"]), 1)


if __name__ == "__main__":
    unittest.main()
