from __future__ import annotations

import unittest

from szwejk.policy import build_action_inventory


class PolicyActionsUnitTest(unittest.TestCase):
    def test_build_action_inventory_exposes_mixed_granularity_candidates(self) -> None:
        enrichment_bundle = {
            "summary": {"total_items": 1, "sentence_safe_items": 1, "blocked_items": 0},
            "items": [
                {
                    "enrichment_status": "sentence_safe",
                    "source_tokens": [
                        {"index": 1, "kind": "word", "text": "nam"},
                        {"index": 2, "kind": "word", "text": "zabili"},
                        {"index": 3, "kind": "word", "text": "Ferdynanda"},
                    ],
                    "target_tokens": [
                        {"index": 1, "kind": "word", "text": "nám"},
                        {"index": 2, "kind": "word", "text": "zabili"},
                        {"index": 3, "kind": "word", "text": "Ferdinanda"},
                    ],
                    "token_pairs": [
                        {
                            "source_token_index": 1,
                            "target_token_index": 1,
                            "source_text": "nam",
                            "target_text": "nám",
                            "source_lemma": "my",
                            "target_lemma": "my",
                            "relation": "lemma_like",
                            "score": 0.92,
                            "source_upos": "PRON",
                            "target_upos": "PRON",
                        }
                    ],
                    "phrase_candidates": [
                        {"source_span": [1, 3], "target_span": [1, 3], "token_count": 3, "score": 0.88, "relation": "parallel_span"}
                    ],
                    "subtree_candidates": [
                        {
                            "source_root_index": 2,
                            "target_root_index": 2,
                            "source_span": [1, 3],
                            "target_span": [1, 3],
                            "source_text": "nam zabili Ferdynanda",
                            "target_text": "nám zabili Ferdinanda",
                            "score": 0.93,
                        }
                    ],
                }
            ],
        }

        inventory = build_action_inventory(enrichment_bundle)

        self.assertEqual(inventory["summary"]["token_action_count"], 1)
        self.assertEqual(inventory["summary"]["phrase_action_count"], 1)
        self.assertEqual(inventory["summary"]["subtree_action_count"], 1)
        self.assertEqual(inventory["phrase_actions"][0]["source_text"], "nam zabili Ferdynanda")
        self.assertEqual(inventory["subtree_actions"][0]["support_depth"], "subtree")
