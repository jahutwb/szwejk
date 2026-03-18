from __future__ import annotations

import unittest

from szwejk.align.embedding_eval import HeuristicEmbeddingEncoder
from szwejk.align.sentence_blocks import SentenceBlockItem, align_sentence_block_spans


class SentenceBlocksUnitTest(unittest.TestCase):
    def test_align_sentence_block_spans_supports_split_target_with_encoder(self) -> None:
        source_items = [
            SentenceBlockItem(seq=1, paragraph_id="pl-p1", paragraph_index=1, sentence_id="pl-s1", sentence_index=1, text="Pan Ferdynand siedzi spokojnie i patrzy w okno."),
            SentenceBlockItem(seq=2, paragraph_id="pl-p1", paragraph_index=1, sentence_id="pl-s2", sentence_index=2, text="Potem wstaje."),
        ]
        target_items = [
            SentenceBlockItem(seq=1, paragraph_id="cs-p1", paragraph_index=1, sentence_id="cs-s1", sentence_index=1, text="Pan Ferdinand sedí pokojně."),
            SentenceBlockItem(seq=2, paragraph_id="cs-p1", paragraph_index=1, sentence_id="cs-s2", sentence_index=2, text="A dívá se z okna."),
            SentenceBlockItem(seq=3, paragraph_id="cs-p1", paragraph_index=1, sentence_id="cs-s3", sentence_index=3, text="Pak vstane."),
        ]

        alignments = align_sentence_block_spans(
            source_items,
            target_items,
            source_language="pl",
            target_language="cs",
            analysis_mode="stanza",
            encoder=HeuristicEmbeddingEncoder(),
            max_source_span=2,
            max_target_span=2,
            skip_source_penalty=0.42,
            skip_target_penalty=0.42,
        )

        self.assertEqual(len(alignments), 2)
        self.assertEqual(alignments[0].alignment_type, "1:2")
        self.assertEqual(alignments[1].alignment_type, "1:1")


if __name__ == "__main__":
    unittest.main()
