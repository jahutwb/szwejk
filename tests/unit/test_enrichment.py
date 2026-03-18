from __future__ import annotations

import unittest

from szwejk.align.enrichment import (
    build_sentence_enrichment,
    build_sentence_enrichment_bundle,
    tokenize_text,
)
from szwejk.align.sentences import ChapterSentenceAlignment, SentenceAlignment


class _FakeEncoder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        mapping = {
            "query: alpha": [1.0, 0.0, 0.0],
            "query: beta": [0.0, 1.0, 0.0],
            "passage: uno": [1.0, 0.0, 0.0],
            "passage: dos": [0.0, 1.0, 0.0],
        }
        return [mapping.get(text, [0.0, 0.0, 1.0]) for text in texts]


class EnrichmentUnitTest(unittest.TestCase):
    def test_tokenize_text_emits_word_and_punctuation_metadata(self) -> None:
        tokens = tokenize_text("„Tak nám zabili Ferdinanda,“", language="cs")

        self.assertEqual(tokens[0]["text"], "„")
        self.assertEqual(tokens[0]["kind"], "punct")
        self.assertEqual(tokens[1]["text"], "Tak")
        self.assertEqual(tokens[1]["kind"], "word")
        self.assertEqual(tokens[1]["normalized"], "tak")
        self.assertEqual(tokens[1]["lemma_source"], "heuristic")
        self.assertIsNone(tokens[1]["upos"])

    def test_build_sentence_enrichment_creates_token_pairs_and_phrase_candidates(self) -> None:
        enrichment = build_sentence_enrichment(
            source_text="Pan Josef siedzi spokojnie.",
            target_text="Pan Josef sedí pokojně.",
            source_language="pl",
            target_language="cs",
        )

        self.assertTrue(enrichment["token_pairs"])
        self.assertTrue(enrichment["phrase_candidates"])
        self.assertIn("dependency_candidates", enrichment)
        self.assertIn("source_tokens", enrichment)
        self.assertIn("target_tokens", enrichment)
        first_pair = enrichment["token_pairs"][0]
        self.assertIn(first_pair["relation"], {"exact", "lemma_like", "cognate", "upos_cognate"})

    def test_build_sentence_enrichment_bundle_respects_safe_depth_gate(self) -> None:
        alignments = [
            ChapterSentenceAlignment(
                source_paragraph_id="pl-p1",
                target_paragraph_id="cs-p1",
                source_paragraph_index=1,
                target_paragraph_index=1,
                sentence_alignment=SentenceAlignment(
                    source_index=1,
                    target_index=1,
                    source_text="Pan Ferdynand siedzi.",
                    target_text="Pan Ferdinand sedí.",
                    score=0.71,
                    signals={"length": 0.9, "char_trigram": 0.45, "punctuation": 1.0, "digit_pattern": 1.0, "total": 0.71},
                ),
            ),
            ChapterSentenceAlignment(
                source_paragraph_id="pl-p2",
                target_paragraph_id="cs-p2",
                source_paragraph_index=2,
                target_paragraph_index=2,
                sentence_alignment=SentenceAlignment(
                    source_index=1,
                    target_index=1,
                    source_text="A gdzie to się przytrafiło?",
                    target_text="Jel tam s tou svou arcikněžnou v automobilu.",
                    score=0.39,
                    signals={"length": 0.87, "char_trigram": 0.05, "punctuation": 0.0, "digit_pattern": 1.0, "total": 0.39},
                ),
            ),
        ]

        bundle = build_sentence_enrichment_bundle(
            chapter_alignments=alignments,
            source_language="pl",
            target_language="cs",
        )

        self.assertEqual(bundle["summary"]["sentence_safe_items"], 1)
        self.assertEqual(bundle["summary"]["blocked_items"], 1)
        self.assertEqual(bundle["items"][0]["enrichment_status"], "sentence_safe")
        self.assertEqual(bundle["items"][1]["enrichment_status"], "blocked_by_safe_depth")
        self.assertFalse(bundle["items"][1]["token_pairs"])

    def test_build_sentence_enrichment_with_stanza_emits_dependency_candidates(self) -> None:
        enrichment = build_sentence_enrichment(
            source_text="Pan Ferdynand siedzi spokojnie.",
            target_text="Pan Ferdinand sedí pokojně.",
            source_language="pl",
            target_language="cs",
            analysis_mode="stanza",
        )

        self.assertEqual(enrichment["source_analysis"]["mode"], "stanza")
        self.assertEqual(enrichment["target_analysis"]["mode"], "stanza")
        self.assertTrue(any(token["upos"] for token in enrichment["source_tokens"]))
        self.assertTrue(enrichment["dependency_candidates"])

    def test_build_sentence_enrichment_marks_embedding_fallback_pairs(self) -> None:
        enrichment = build_sentence_enrichment(
            source_text="alpha beta",
            target_text="uno dos",
            source_language="pl",
            target_language="cs",
            analysis_mode="heuristic",
            subtree_encoder=_FakeEncoder(),
        )

        self.assertTrue(any(pair["relation"] == "embedding_assisted" for pair in enrichment["token_pairs"]))
        self.assertTrue(any(candidate["relation"] == "embedding_assisted" for candidate in enrichment["phrase_candidates"]))
        self.assertTrue(any(candidate["relation"] == "embedding_assisted" for candidate in enrichment["subtree_candidates"]))


if __name__ == "__main__":
    unittest.main()
