from __future__ import annotations

import unittest

from szwejk.generate.debug_subset import apply_selected_candidates


class DebugSubsetRendererUnitTest(unittest.TestCase):
    def test_apply_selected_candidates_matches_phrase_despite_punctuation(self) -> None:
        paragraph = (
            "— W gazetach piszą, proszę pana, że pan arcyksiążę był podziurawiony jak sito. "
            "Wystrzelił do niego wszystkie naboje."
        )
        candidates = [
            {
                "candidate_id": "u15:s1:phrase:1",
                "granularity": "phrase",
                "source_text": "W gazetach piszą proszę pana że pan arcyksiążę był podziurawiony jak sito",
                "target_text": "Noviny píšou milostpane že pan arcivévoda byl jako řešeto",
            },
            {
                "candidate_id": "u15:s2:phrase:1",
                "granularity": "phrase",
                "source_text": "Wystrzelił do niego wszystkie naboje",
                "target_text": "Vystřílel do něho všechny patrony",
            },
        ]
        notes = {
            "u15:s1:phrase:1": {
                "candidate_id": "u15:s1:phrase:1",
                "note_text": "Noviny píšou milostpane že pan arcivévoda byl jako řešeto: W gazetach piszą proszę pana że pan arcyksiążę był podziurawiony jak sito.",
            }
        }

        rendered, _, used_notes = apply_selected_candidates(paragraph, candidates, notes)

        self.assertIn(r"\cznote{Noviny píšou milostpane że pan arcivévoda byl jako řešeto}", rendered)
        self.assertIn(r"\czplain{Vystřílel do něho všechny patrony}", rendered)
        self.assertTrue(rendered.startswith("— "))
        self.assertEqual(len(used_notes), 1)

    def test_apply_selected_candidates_replaces_whole_paragraph(self) -> None:
        rendered, _, _ = apply_selected_candidates(
            "— Podobno więcej ich tam było, proszę pana.",
            [
                {
                    "candidate_id": "u11:paragraph",
                    "granularity": "paragraph",
                    "source_text": "— Podobno więcej ich tam było, proszę pana.",
                    "target_text": "„Vono prej jich bylo víc, milostpane.“",
                }
            ],
            {},
        )

        self.assertEqual(rendered, r"\czplain{„Vono prej jich bylo víc, milostpane.“}")


if __name__ == "__main__":
    unittest.main()