from __future__ import annotations

import unittest

from szwejk.generate.book import GLOBAL_LINEAR_LEVEL_ID, _plan_payload_from_progression_steps
from szwejk.generate.corpus_scope import apply_default_tail_chapter_overrides, scoped_source_chapter
from szwejk.review import load_book_from_json


PL_CORPUS = "/home/jahu/PycharmProjects/szwejk/data/corpus/pl_szwejk.json"


class HybridBookUnitTest(unittest.TestCase):
    def test_plan_payload_from_progression_steps_preserves_spans(self) -> None:
        payload = _plan_payload_from_progression_steps(
            source_chapter_index=2,
            target_chapter_index=2,
            source_title="Rozdział I",
            target_title="Zasáhnutí",
            chapter_steps=[
                {
                    "granularity": "phrase",
                    "sentence_index": 3,
                    "source_span": [1, 3],
                    "target_span": [1, 3],
                    "source_text": "nam zabili Ferdynanda",
                    "target_text": "nám zabili Ferdinanda",
                    "new_families": ["my::my", "zabic::zabit"],
                }
            ],
        )

        self.assertEqual(payload["level_id"], GLOBAL_LINEAR_LEVEL_ID)
        self.assertEqual(payload["selected_candidates"][0]["source_span"], [1, 3])
        self.assertEqual(payload["selected_candidates"][0]["target_span"], [1, 3])
        self.assertEqual(payload["selected_candidates"][0]["family_id"], "my::my::zabic::zabit")

    def test_tail_pair_overrides_replace_automatic_tail_mapping(self) -> None:
        pairs = [(index, index) for index in range(1, 27)] + [(27, 27), (30, 28), (32, 29)]

        overridden = apply_default_tail_chapter_overrides(
            pairs,
            source_chapter_count=33,
            target_chapter_count=29,
        )

        self.assertEqual(overridden[-3:], [(27, 27), (28, 28), (29, 29)])
        self.assertEqual(len(overridden), 29)

    def test_scoped_source_chapter_trims_polish_tail_chapter_29(self) -> None:
        book = load_book_from_json(PL_CORPUS)

        chapter = scoped_source_chapter(book.chapters[28])

        self.assertEqual(chapter.index, 29)
        self.assertEqual(len(chapter.paragraphs), 393)
        self.assertEqual(chapter.paragraphs[-1].index, 393)
