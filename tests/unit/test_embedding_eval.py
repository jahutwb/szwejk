from __future__ import annotations

import unittest

from szwejk.align.embedding_eval import (
    HeuristicEmbeddingEncoder,
    build_candidate_set,
    evaluate_encoder_on_samples,
)
from szwejk.align.sentences import SentenceAlignment


class EmbeddingEvalUnitTest(unittest.TestCase):
    def test_build_candidate_set_includes_positive_and_bounded_distractors(self) -> None:
        target_sentences = ["zero", "dobry wojak", "zly wojak", "inna fraza"]
        candidates = build_candidate_set(target_sentences, positive_index=2, distractor_window=1)

        self.assertEqual(candidates[0], "dobry wojak")
        self.assertIn("zero", candidates)
        self.assertIn("zly wojak", candidates)
        self.assertNotIn("inna fraza", candidates)

    def test_evaluate_encoder_on_samples_reports_top1_and_mrr(self) -> None:
        encoder = HeuristicEmbeddingEncoder()
        samples = [
            {
                "source_text": "dobry wojak",
                "positive_text": "dobry vojak",
                "candidate_texts": ["dobry vojak", "zla veta", "jina veta"],
                "baseline_score": 0.7,
            },
            {
                "source_text": "pan ferdynand",
                "positive_text": "pan ferdinand",
                "candidate_texts": ["jiny text", "pan ferdinand", "uplne jine"],
                "baseline_score": 0.52,
            },
        ]

        report = evaluate_encoder_on_samples(
            encoder=encoder,
            model_name="heuristic",
            samples=samples,
        )

        self.assertEqual(report["model_name"], "heuristic")
        self.assertEqual(report["sample_count"], 2)
        self.assertGreaterEqual(report["top1_accuracy"], 0.5)
        self.assertGreater(report["mrr"], 0.5)
        self.assertIn("hard_case_count", report)


if __name__ == "__main__":
    unittest.main()
