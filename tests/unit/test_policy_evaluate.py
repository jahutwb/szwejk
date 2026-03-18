from __future__ import annotations

import unittest

from szwejk.policy import evaluate_policy_on_enrichment_bundle, load_policy_from_json


class PolicyEvaluateUnitTest(unittest.TestCase):
    def test_l1_selects_transparent_family_and_blocks_low_score_pair(self) -> None:
        policy = load_policy_from_json("data/policies/progressive_czechization_v1.json").policy
        level = policy.levels[0]
        enrichment_bundle = {
            "summary": {"total_items": 1, "sentence_safe_items": 1, "blocked_items": 0},
            "items": [
                {
                    "safe_alignment_depth": "token",
                    "enrichment_status": "sentence_safe",
                    "sentence_alignment": {"score": 0.92},
                    "source_tokens": [
                        {"index": 1, "kind": "word", "upos": "NOUN", "lemma": "skromny", "text": "skromny"},
                        {"index": 2, "kind": "word", "upos": "CCONJ", "lemma": "i", "text": "i"},
                    ],
                    "target_tokens": [
                        {"index": 1, "kind": "word", "upos": "NOUN", "lemma": "skromny", "text": "skromny"},
                        {"index": 2, "kind": "word", "upos": "CCONJ", "lemma": "a", "text": "a"},
                    ],
                    "token_pairs": [
                        {
                            "source_token_index": 1,
                            "target_token_index": 1,
                            "source_text": "skromny",
                            "target_text": "skromny",
                            "source_lemma": "skromny",
                            "target_lemma": "skromny",
                            "relation": "exact",
                            "score": 1.0,
                            "source_upos": "NOUN",
                            "target_upos": "NOUN",
                            "source_deprel": "root",
                            "target_deprel": "root",
                        },
                        {
                            "source_token_index": 2,
                            "target_token_index": 2,
                            "source_text": "i",
                            "target_text": "a",
                            "source_lemma": "i",
                            "target_lemma": "a",
                            "relation": "weak",
                            "score": 0.4,
                            "source_upos": "CCONJ",
                            "target_upos": "CCONJ",
                            "source_deprel": "cc",
                            "target_deprel": "cc",
                        },
                    ],
                    "phrase_candidates": [],
                    "dependency_candidates": [{"source_span": [1, 1], "target_span": [1, 1]}],
                }
            ],
        }

        plan = evaluate_policy_on_enrichment_bundle(
            enrichment_bundle,
            policy=policy,
            level=level,
            source_chapter_index=1,
            target_chapter_index=1,
            source_title="PL",
            target_title="CS",
            introduced_families=set(),
        )

        self.assertEqual(len(plan.selected_candidates), 1)
        self.assertEqual(plan.selected_candidates[0].granularity, "token")
        self.assertEqual(plan.selected_candidates[0].family_id, "skromny::skromny")
        self.assertEqual(plan.selected_candidates[0].support_depth, "subtree")
        self.assertTrue(plan.blocked_candidates)

    def test_gate_blocks_candidate_when_safe_depth_is_too_low(self) -> None:
        policy = load_policy_from_json("data/policies/progressive_czechization_v1.json").policy
        level = policy.levels[1]
        enrichment_bundle = {
            "summary": {"total_items": 1, "sentence_safe_items": 1, "blocked_items": 0},
            "items": [
                {
                    "safe_alignment_depth": "sentence",
                    "enrichment_status": "sentence_safe",
                    "sentence_alignment": {"score": 0.92},
                    "source_tokens": [{"index": 1, "kind": "word", "upos": "NOUN", "lemma": "ucitel", "text": "ucitel"}],
                    "target_tokens": [{"index": 1, "kind": "word", "upos": "NOUN", "lemma": "ucitel", "text": "ucitel"}],
                    "token_pairs": [
                        {
                            "source_token_index": 1,
                            "target_token_index": 1,
                            "source_text": "ucitel",
                            "target_text": "ucitel",
                            "source_lemma": "ucitel",
                            "target_lemma": "ucitel",
                            "relation": "exact",
                            "score": 1.0,
                            "source_upos": "NOUN",
                            "target_upos": "NOUN",
                            "source_deprel": "root",
                            "target_deprel": "root",
                        }
                    ],
                    "phrase_candidates": [],
                    "dependency_candidates": [],
                }
            ],
        }

        plan = evaluate_policy_on_enrichment_bundle(
            enrichment_bundle,
            policy=policy,
            level=level,
            source_chapter_index=1,
            target_chapter_index=1,
            source_title="PL",
            target_title="CS",
            introduced_families=set(),
        )

        self.assertEqual(plan.selected_candidates, [])
        self.assertTrue(any(item["reason"] == "support_depth_below_phrase" for item in plan.blocked_candidates))

    def test_l2_selects_phrase_action_instead_of_token_only(self) -> None:
        policy = load_policy_from_json("data/policies/progressive_czechization_v1.json").policy
        level = policy.levels[1]
        enrichment_bundle = {
            "summary": {"total_items": 1, "sentence_safe_items": 1, "blocked_items": 0},
            "items": [
                {
                    "safe_alignment_depth": "sentence",
                    "enrichment_status": "sentence_safe",
                    "sentence_alignment": {"score": 0.9},
                    "source_tokens": [
                        {"index": 1, "kind": "word", "upos": "PRON", "lemma": "my", "text": "nam"},
                        {"index": 2, "kind": "word", "upos": "VERB", "lemma": "zabic", "text": "zabili"},
                        {"index": 3, "kind": "word", "upos": "PROPN", "lemma": "ferdynand", "text": "Ferdynanda"},
                    ],
                    "target_tokens": [
                        {"index": 1, "kind": "word", "upos": "PRON", "lemma": "my", "text": "nám"},
                        {"index": 2, "kind": "word", "upos": "VERB", "lemma": "zabit", "text": "zabili"},
                        {"index": 3, "kind": "word", "upos": "PROPN", "lemma": "ferdinand", "text": "Ferdinanda"},
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
                            "source_deprel": "obj",
                            "target_deprel": "obj",
                        }
                    ],
                    "phrase_candidates": [
                        {"source_span": [1, 3], "target_span": [1, 3], "token_count": 3, "score": 0.88, "relation": "parallel_span"}
                    ],
                    "dependency_candidates": [],
                    "subtree_candidates": [],
                }
            ],
        }

        plan = evaluate_policy_on_enrichment_bundle(
            enrichment_bundle,
            policy=policy,
            level=level,
            source_chapter_index=1,
            target_chapter_index=1,
            source_title="PL",
            target_title="CS",
            introduced_families=set(),
        )

        self.assertEqual(len(plan.selected_candidates), 1)
        self.assertEqual(plan.selected_candidates[0].granularity, "phrase")
        self.assertEqual(plan.selected_candidates[0].source_text, "nam zabili Ferdynanda")
