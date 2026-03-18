from __future__ import annotations

import unittest

from szwejk.generate.hybrid import _apply_actions_to_tokens


class HybridGenerationUnitTest(unittest.TestCase):
    def test_apply_actions_to_tokens_replaces_phrase_span(self) -> None:
        tokens = [
            {"index": 1, "kind": "word", "text": "nam"},
            {"index": 2, "kind": "word", "text": "zabili"},
            {"index": 3, "kind": "word", "text": "Ferdynanda"},
            {"index": 4, "kind": "punct", "text": ","},
        ]
        actions = [
            {
                "source_span": [1, 3],
                "target_span": [1, 3],
                "target_text": "nám zabili Ferdinanda",
            }
        ]

        rendered = _apply_actions_to_tokens(tokens, actions)

        self.assertEqual(rendered, "nám zabili Ferdinanda,")

    def test_apply_actions_to_tokens_keeps_clitic_attached(self) -> None:
        tokens = [
            {"index": 1, "kind": "punct", "text": "—"},
            {"index": 2, "kind": "word", "text": "Nie", "upos": "PART"},
            {"index": 3, "kind": "word", "text": "zapieraj", "upos": "VERB"},
            {"index": 4, "kind": "word", "text": "się", "upos": "PRON"},
            {"index": 5, "kind": "punct", "text": ","},
            {"index": 6, "kind": "word", "text": "bo", "upos": "SCONJ"},
            {"index": 7, "kind": "word", "text": "ś", "upos": "AUX", "feats": "Variant=Short", "deprel": "aux:clitic"},
            {"index": 8, "kind": "word", "text": "go", "upos": "PRON"},
            {"index": 9, "kind": "word", "text": "zeżarł", "upos": "VERB"},
            {"index": 10, "kind": "punct", "text": "."},
        ]
        actions = [
            {
                "source_span": [9, 9],
                "target_span": [7, 7],
                "target_text": "sežral",
            }
        ]

        rendered = _apply_actions_to_tokens(tokens, actions)

        self.assertEqual(rendered, "— Nie zapieraj się, boś go sežral.")
