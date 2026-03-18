from __future__ import annotations

import unittest

from szwejk.align.sentences import (
    SentenceSpanAlignment,
    SentenceAlignment,
    align_sentence_spans,
    align_sentence_sequences,
    punctuation_shape_similarity,
    sentence_similarity_signals,
)


class SentenceAlignmentUnitTest(unittest.TestCase):
    def test_punctuation_shape_similarity_prefers_matching_quotes(self) -> None:
        left = "— A to nam zabili Ferdynanda — rzekła posługaczka."
        right = "„Tak nám zabili Ferdinanda,“ řekla posluhovačka."
        other = "Veliká doba žádá velké lidi."

        self.assertGreater(punctuation_shape_similarity(left, right), punctuation_shape_similarity(left, other))

    def test_sentence_similarity_signals_reward_parallel_sentences(self) -> None:
        left = "Prócz tego zajęcia dotknięty był reumatyzmem i właśnie nacierał sobie kolana opodeldokiem."
        right = "Kromě tohoto zaměstnání byl stižen revmatismem a mazal si právě kolena opodeldokem."
        wrong = "Veliká doba žádá velké lidi."

        good = sentence_similarity_signals(left, right)
        bad = sentence_similarity_signals(left, wrong)

        self.assertGreater(good["total"], bad["total"])
        self.assertGreater(good["char_trigram"], bad["char_trigram"])

    def test_align_sentence_sequences_returns_monotonic_pairs(self) -> None:
        source_sentences = [
            "„Tak nám zabili Ferdinanda,“ řekla posluhovačka panu Švejkovi.",
            "Kromě tohoto zaměstnání byl stižen revmatismem.",
            "V hospodě U kalicha seděl jen jeden host.",
        ]
        target_sentences = [
            "— A to nam zabili Ferdynanda — rzekła posługaczka do pana Szwejka.",
            "Prócz tego zajęcia dotknięty był reumatyzmem.",
            "W gospodzie Pod kielichem siedział tylko jeden gość.",
        ]

        alignments = align_sentence_sequences(source_sentences, target_sentences)

        self.assertEqual(len(alignments), 3)
        self.assertTrue(all(isinstance(item, SentenceAlignment) for item in alignments))
        self.assertEqual([item.source_index for item in alignments], [1, 2, 3])
        self.assertEqual([item.target_index for item in alignments], [1, 2, 3])
        self.assertTrue(all(item.score > 0.3 for item in alignments))

    def test_align_sentence_spans_supports_split_target(self) -> None:
        source_sentences = [
            "Pan Ferdynand siedzi spokojnie i patrzy w okno.",
            "Potem wstaje.",
        ]
        target_sentences = [
            "Pan Ferdinand sedí pokojně.",
            "A dívá se z okna.",
            "Pak vstane.",
        ]

        alignments = align_sentence_spans(source_sentences, target_sentences, source_language="pl", target_language="cs", analysis_mode="stanza")

        self.assertEqual(len(alignments), 2)
        self.assertTrue(all(isinstance(item, SentenceSpanAlignment) for item in alignments))
        self.assertEqual(alignments[0].source_span, (1, 1))
        self.assertEqual(alignments[0].target_span, (1, 2))
        self.assertEqual(alignments[0].alignment_type, "1:2")
        self.assertEqual(alignments[1].source_span, (2, 2))
        self.assertEqual(alignments[1].target_span, (3, 3))

    def test_align_sentence_spans_supports_split_source(self) -> None:
        source_sentences = [
            "Pan Ferdynand siedzi spokojnie.",
            "I patrzy w okno.",
            "Potem wstaje.",
        ]
        target_sentences = [
            "Pan Ferdinand sedí pokojně a dívá se z okna.",
            "Pak vstane.",
        ]

        alignments = align_sentence_spans(source_sentences, target_sentences, source_language="pl", target_language="cs", analysis_mode="stanza")

        self.assertEqual(len(alignments), 2)
        self.assertEqual(alignments[0].source_span, (1, 2))
        self.assertEqual(alignments[0].target_span, (1, 1))
        self.assertEqual(alignments[0].alignment_type, "2:1")
        self.assertEqual(alignments[1].alignment_type, "1:1")


if __name__ == "__main__":
    unittest.main()
