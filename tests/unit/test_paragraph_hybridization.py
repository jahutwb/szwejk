from __future__ import annotations

import unittest

from szwejk.generate.paragraph_hybridization import _block_candidates, build_paragraph_hybridization_plan
from szwejk.generate.lemma_timeline import build_lemma_timeline_report


class ParagraphHybridizationUnitTest(unittest.TestCase):
    def test_phrase_target_text_preserves_punctuation(self) -> None:
        artifact = {
            "chapter_pair_count": 1,
            "pair_reports": [
                {
                    "source_chapter_index": 1,
                    "target_chapter_index": 1,
                    "paragraph_alignment": {
                        "matched_blocks": [
                            {
                                "source_range": [1, 1],
                                "target_range": [1, 1],
                                "source_preview": "Na przykład, gdybyśmy panu powiedzieli, że jest pan ondatra. Mógłby się pan gniewać?",
                                "target_preview": "Například, jestli bychom vám řekli, že jste ondatra. Mohl byste se zlobit?",
                            }
                        ]
                    },
                    "enrichment_bundle": {
                        "items": [
                            {
                                "source_paragraph_index": 1,
                                "target_paragraph_index": 1,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 12],
                                    "target_span": [1, 13],
                                    "source_text": "Na przykład, gdybyśmy panu powiedzieli, że jest pan ondatra. Mógłby się pan gniewać?",
                                    "target_text": "Například, jestli bychom vám řekli, že jste ondatra. Mohl byste se zlobit?",
                                    "score": 0.8,
                                    "signals": {"total": 0.8},
                                },
                                "source_tokens": [
                                    {"index": 1, "text": "Na", "kind": "word"},
                                    {"index": 2, "text": "przykład", "kind": "word"},
                                    {"index": 3, "text": ",", "kind": "punct"},
                                    {"index": 4, "text": "gdybyśmy", "kind": "word"},
                                    {"index": 5, "text": "panu", "kind": "word"},
                                    {"index": 6, "text": "powiedzieli", "kind": "word"},
                                    {"index": 7, "text": ",", "kind": "punct"},
                                    {"index": 8, "text": "że", "kind": "word"},
                                    {"index": 9, "text": "jest", "kind": "word"},
                                    {"index": 10, "text": "pan", "kind": "word"},
                                    {"index": 11, "text": "ondatra", "kind": "word"},
                                    {"index": 12, "text": ".", "kind": "punct"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "text": "Například", "kind": "word", "lemma": "například", "upos": "ADV"},
                                    {"index": 2, "text": ",", "kind": "punct"},
                                    {"index": 3, "text": "jestli", "kind": "word", "lemma": "jestli", "upos": "SCONJ"},
                                    {"index": 4, "text": "bychom", "kind": "word", "lemma": "být", "upos": "AUX"},
                                    {"index": 5, "text": "vám", "kind": "word", "lemma": "vy", "upos": "PRON"},
                                    {"index": 6, "text": "řekli", "kind": "word", "lemma": "říci", "upos": "VERB"},
                                    {"index": 7, "text": ",", "kind": "punct"},
                                    {"index": 8, "text": "že", "kind": "word", "lemma": "že", "upos": "SCONJ"},
                                    {"index": 9, "text": "jste", "kind": "word", "lemma": "být", "upos": "AUX"},
                                    {"index": 10, "text": "ondatra", "kind": "word", "lemma": "ondatra", "upos": "NOUN"},
                                    {"index": 11, "text": ".", "kind": "punct"},
                                    {"index": 12, "text": "Mohl", "kind": "word", "lemma": "moci", "upos": "AUX"},
                                    {"index": 13, "text": "?", "kind": "punct"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": 2,
                                        "target_token_index": 1,
                                        "source_text": "przykład",
                                        "target_text": "Například",
                                        "source_lemma": "przykład",
                                        "target_lemma": "například",
                                        "source_upos": "NOUN",
                                        "target_upos": "ADV",
                                        "relation": "lemma_like",
                                        "score": 0.82,
                                        "signals": {"norm": 0.8, "total": 0.82},
                                    },
                                    {
                                        "source_token_index": 11,
                                        "target_token_index": 10,
                                        "source_text": "ondatra",
                                        "target_text": "ondatra",
                                        "source_lemma": "ondatra",
                                        "target_lemma": "ondatra",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "relation": "lemma_like",
                                        "score": 0.9,
                                        "signals": {"norm": 0.9, "total": 0.9},
                                    },
                                ],
                                "phrase_candidates": [
                                    {
                                        "source_span": [1, 11],
                                        "target_span": [1, 10],
                                        "target_text": "IGNORED",
                                        "score": 0.84,
                                        "signals": {"total": 0.84},
                                    }
                                ],
                                "subtree_candidates": [],
                            }
                        ]
                    },
                }
            ],
        }

        item = artifact["pair_reports"][0]["enrichment_bundle"]["items"][0]
        candidates = _block_candidates(
            block_items=[item],
            chapter_pair=(1, 1),
            unit_index=1,
            source_range=(1, 1),
            target_range=(1, 1),
            source_preview=str(artifact["pair_reports"][0]["paragraph_alignment"]["matched_blocks"][0]["source_preview"]),
            target_preview=str(artifact["pair_reports"][0]["paragraph_alignment"]["matched_blocks"][0]["target_preview"]),
            blocked_standalone_upos=frozenset({"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"}),
        )
        phrase = next(candidate for candidate in candidates if candidate.granularity == "phrase")
        self.assertEqual(phrase.target_text, "Například, jestli bychom vám řekli, že jste ondatra")

    def test_phrase_candidate_identical_to_sentence_span_is_filtered_out(self) -> None:
        item = {
            "source_paragraph_index": 1,
            "target_paragraph_index": 1,
            "enrichment_status": "sentence_safe",
            "sentence_alignment": {
                "source_span": [1, 3],
                "target_span": [1, 3],
                "source_text": "Host jako host.",
                "target_text": "Host jako host.",
                "score": 0.8,
                "signals": {"total": 0.8},
            },
            "source_tokens": [
                {"index": 1, "text": "Host", "kind": "word"},
                {"index": 2, "text": "jako", "kind": "word"},
                {"index": 3, "text": "host", "kind": "word"},
                {"index": 4, "text": ".", "kind": "punct"},
            ],
            "target_tokens": [
                {"index": 1, "text": "Host", "kind": "word", "lemma": "host", "upos": "NOUN"},
                {"index": 2, "text": "jako", "kind": "word", "lemma": "jako", "upos": "SCONJ"},
                {"index": 3, "text": "host", "kind": "word", "lemma": "host", "upos": "NOUN"},
                {"index": 4, "text": ".", "kind": "punct"},
            ],
            "token_pairs": [
                {
                    "source_token_index": 1,
                    "target_token_index": 1,
                    "source_text": "Host",
                    "target_text": "Host",
                    "source_lemma": "host",
                    "target_lemma": "host",
                    "source_upos": "NOUN",
                    "target_upos": "NOUN",
                    "relation": "exact",
                    "score": 1.0,
                    "signals": {"norm": 1.0, "total": 1.0},
                },
                {
                    "source_token_index": 2,
                    "target_token_index": 2,
                    "source_text": "jako",
                    "target_text": "jako",
                    "source_lemma": "jako",
                    "target_lemma": "jako",
                    "source_upos": "SCONJ",
                    "target_upos": "SCONJ",
                    "relation": "exact",
                    "score": 1.0,
                    "signals": {"norm": 1.0, "total": 1.0},
                },
                {
                    "source_token_index": 3,
                    "target_token_index": 3,
                    "source_text": "host",
                    "target_text": "host",
                    "source_lemma": "host",
                    "target_lemma": "host",
                    "source_upos": "NOUN",
                    "target_upos": "NOUN",
                    "relation": "exact",
                    "score": 1.0,
                    "signals": {"norm": 1.0, "total": 1.0},
                },
            ],
            "phrase_candidates": [
                {
                    "source_span": [1, 3],
                    "target_span": [1, 3],
                    "score": 0.95,
                    "relation": "parallel_span",
                    "signals": {"total": 0.95},
                }
            ],
            "subtree_candidates": [],
        }
        candidates = _block_candidates(
            block_items=[item],
            chapter_pair=(1, 1),
            unit_index=1,
            source_range=(1, 1),
            target_range=(1, 1),
            source_preview="Host jako host.",
            target_preview="Host jako host.",
            blocked_standalone_upos=frozenset({"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"}),
        )
        granularities = [candidate.granularity for candidate in candidates]
        self.assertIn("sentence", granularities)
        self.assertNotIn("phrase", granularities)

    def test_phrase_candidate_equal_to_sentence_without_final_punctuation_is_filtered_out(self) -> None:
        item = {
            "source_paragraph_index": 1,
            "target_paragraph_index": 1,
            "enrichment_status": "sentence_safe",
            "sentence_alignment": {
                "source_span": [1, 3],
                "target_span": [1, 3],
                "source_text": "Host jako host.",
                "target_text": "Host jako host!",
                "score": 0.8,
                "signals": {"total": 0.8},
            },
            "source_tokens": [
                {"index": 1, "text": "Host", "kind": "word"},
                {"index": 2, "text": "jako", "kind": "word"},
                {"index": 3, "text": "host", "kind": "word"},
            ],
            "target_tokens": [
                {"index": 1, "text": "Host", "kind": "word", "lemma": "host", "upos": "NOUN"},
                {"index": 2, "text": "jako", "kind": "word", "lemma": "jako", "upos": "SCONJ"},
                {"index": 3, "text": "host", "kind": "word", "lemma": "host", "upos": "NOUN"},
            ],
            "token_pairs": [
                {
                    "source_token_index": 1,
                    "target_token_index": 1,
                    "source_text": "Host",
                    "target_text": "Host",
                    "source_lemma": "host",
                    "target_lemma": "host",
                    "source_upos": "NOUN",
                    "target_upos": "NOUN",
                    "relation": "exact",
                    "score": 1.0,
                    "signals": {"norm": 1.0, "total": 1.0},
                },
                {
                    "source_token_index": 2,
                    "target_token_index": 2,
                    "source_text": "jako",
                    "target_text": "jako",
                    "source_lemma": "jako",
                    "target_lemma": "jako",
                    "source_upos": "SCONJ",
                    "target_upos": "SCONJ",
                    "relation": "exact",
                    "score": 1.0,
                    "signals": {"norm": 1.0, "total": 1.0},
                },
                {
                    "source_token_index": 3,
                    "target_token_index": 3,
                    "source_text": "host",
                    "target_text": "host",
                    "source_lemma": "host",
                    "target_lemma": "host",
                    "source_upos": "NOUN",
                    "target_upos": "NOUN",
                    "relation": "exact",
                    "score": 1.0,
                    "signals": {"norm": 1.0, "total": 1.0},
                },
            ],
            "phrase_candidates": [
                {
                    "source_span": [1, 3],
                    "target_span": [1, 3],
                    "score": 0.95,
                    "relation": "parallel_span",
                    "signals": {"total": 0.95},
                }
            ],
            "subtree_candidates": [],
        }
        candidates = _block_candidates(
            block_items=[item],
            chapter_pair=(1, 1),
            unit_index=1,
            source_range=(1, 1),
            target_range=(1, 1),
            source_preview="Host jako host.",
            target_preview="Host jako host!",
            blocked_standalone_upos=frozenset({"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"}),
        )
        granularities = [candidate.granularity for candidate in candidates]
        self.assertIn("sentence", granularities)
        self.assertNotIn("phrase", granularities)

    def test_lemma_schedule_prior_can_defer_family_to_later_unit(self) -> None:
        artifact = {
            "chapter_pair_count": 1,
            "pair_reports": [
                {
                    "source_chapter_index": 1,
                    "target_chapter_index": 1,
                    "paragraph_alignment": {
                        "matched_blocks": [
                            {
                                "source_range": [1, 1],
                                "target_range": [1, 1],
                                "source_preview": "kapral",
                                "target_preview": "desátník",
                            },
                            {
                                "source_range": [2, 2],
                                "target_range": [2, 2],
                                "source_preview": "kapral",
                                "target_preview": "desátník",
                            },
                        ]
                    },
                    "enrichment_bundle": {
                        "items": [
                            {
                                "source_paragraph_index": 1,
                                "target_paragraph_index": 1,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 1],
                                    "target_span": [1, 1],
                                    "source_text": "kapral",
                                    "target_text": "desátník",
                                    "score": 0.84,
                                    "signals": {"total": 0.84},
                                },
                                "source_tokens": [{"index": 1, "text": "kapral", "kind": "word"}],
                                "target_tokens": [{"index": 1, "text": "desátník", "kind": "word"}],
                                "token_pairs": [
                                    {
                                        "source_token_index": 1,
                                        "target_token_index": 1,
                                        "source_text": "kapral",
                                        "target_text": "desátník",
                                        "source_lemma": "kapral",
                                        "target_lemma": "desátník",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "relation": "lemma_like",
                                        "score": 0.84,
                                        "signals": {"norm": 0.8, "total": 0.84},
                                    }
                                ],
                                "phrase_candidates": [],
                                "subtree_candidates": [],
                            },
                            {
                                "source_paragraph_index": 2,
                                "target_paragraph_index": 2,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 1],
                                    "target_span": [1, 1],
                                    "source_text": "kapral",
                                    "target_text": "desátník",
                                    "score": 0.84,
                                    "signals": {"total": 0.84},
                                },
                                "source_tokens": [{"index": 1, "text": "kapral", "kind": "word"}],
                                "target_tokens": [{"index": 1, "text": "desátník", "kind": "word"}],
                                "token_pairs": [
                                    {
                                        "source_token_index": 1,
                                        "target_token_index": 1,
                                        "source_text": "kapral",
                                        "target_text": "desátník",
                                        "source_lemma": "kapral",
                                        "target_lemma": "desátník",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "relation": "lemma_like",
                                        "score": 0.84,
                                        "signals": {"norm": 0.8, "total": 0.84},
                                    }
                                ],
                                "phrase_candidates": [],
                                "subtree_candidates": [],
                            },
                        ]
                    },
                }
            ],
        }

        scheduled = {
            "mode": "lemma-family-geometry-v3-residual-greedy",
            "selected_families": [
                {"family_id": "kapral::desátník", "intro_unit_index": 2}
            ],
        }

        payload = build_paragraph_hybridization_plan(
            artifact,
            target_power=1.0,
            lemma_schedule_payload=scheduled,
        )
        first_selected = payload["units"][0]["selected_candidates"]
        second_selected = payload["units"][1]["selected_candidates"]
        self.assertFalse(first_selected)
        self.assertTrue(second_selected)

    def test_family_timeline_marks_rare_family_as_more_urgent_than_deferrable_common_family(self) -> None:
        artifact = {
            "chapter_pair_count": 1,
            "pair_reports": [
                {
                    "source_chapter_index": 1,
                    "target_chapter_index": 1,
                    "paragraph_alignment": {
                        "matched_blocks": [
                            {
                                "source_range": [1, 1],
                                "target_range": [1, 1],
                                "source_preview": "rzadki wspolny",
                                "target_preview": "vzácný společný",
                            },
                            {
                                "source_range": [2, 2],
                                "target_range": [2, 2],
                                "source_preview": "wspolny wspolny wspolny wspolny wspolny",
                                "target_preview": "společný společný společný společný společný",
                            },
                        ]
                    },
                    "enrichment_bundle": {
                        "items": [
                            {
                                "source_paragraph_index": 1,
                                "target_paragraph_index": 1,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 2],
                                    "target_span": [1, 2],
                                    "source_text": "rzadki wspolny",
                                    "target_text": "vzácný společný",
                                    "score": 0.9,
                                    "signals": {"total": 0.9},
                                },
                                "source_tokens": [
                                    {"index": 1, "text": "rzadki", "kind": "word"},
                                    {"index": 2, "text": "wspolny", "kind": "word"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "text": "vzácný", "kind": "word"},
                                    {"index": 2, "text": "společný", "kind": "word"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": 1,
                                        "target_token_index": 1,
                                        "source_text": "rzadki",
                                        "target_text": "vzácný",
                                        "source_lemma": "rzadki",
                                        "target_lemma": "vzácný",
                                        "source_upos": "ADJ",
                                        "target_upos": "ADJ",
                                        "relation": "lemma_like",
                                        "score": 0.88,
                                        "signals": {"norm": 0.9, "total": 0.88},
                                    },
                                    {
                                        "source_token_index": 2,
                                        "target_token_index": 2,
                                        "source_text": "wspolny",
                                        "target_text": "společný",
                                        "source_lemma": "wspolny",
                                        "target_lemma": "společný",
                                        "source_upos": "ADJ",
                                        "target_upos": "ADJ",
                                        "relation": "lemma_like",
                                        "score": 0.9,
                                        "signals": {"norm": 0.9, "total": 0.9},
                                    },
                                ],
                                "phrase_candidates": [],
                                "subtree_candidates": [],
                            },
                            {
                                "source_paragraph_index": 2,
                                "target_paragraph_index": 2,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 5],
                                    "target_span": [1, 5],
                                    "source_text": "wspolny wspolny wspolny wspolny wspolny",
                                    "target_text": "společný společný společný společný společný",
                                    "score": 0.9,
                                    "signals": {"total": 0.9},
                                },
                                "source_tokens": [{"index": i, "text": "wspolny", "kind": "word"} for i in range(1, 6)],
                                "target_tokens": [{"index": i, "text": "společný", "kind": "word"} for i in range(1, 6)],
                                "token_pairs": [
                                    {
                                        "source_token_index": i,
                                        "target_token_index": i,
                                        "source_text": "wspolny",
                                        "target_text": "společný",
                                        "source_lemma": "wspolny",
                                        "target_lemma": "společný",
                                        "source_upos": "ADJ",
                                        "target_upos": "ADJ",
                                        "relation": "lemma_like",
                                        "score": 0.9,
                                        "signals": {"norm": 0.9, "total": 0.9},
                                    }
                                    for i in range(1, 6)
                                ],
                                "phrase_candidates": [],
                                "subtree_candidates": [],
                            },
                        ]
                    },
                }
            ],
        }

        timeline = build_lemma_timeline_report(
            artifact,
            blocked_standalone_upos=frozenset({"SCONJ", "CCONJ", "PART", "ADP", "PRON", "DET", "AUX"}),
        )
        families = {item["family_id"]: item for item in timeline["families"]}
        rare = families["rzadki::vzácný"]["occurrences"][0]
        common = families["wspolny::společný"]["occurrences"][0]
        self.assertGreater(rare["urgency"], common["urgency"])
        self.assertLess(rare["future_gain_if_introduced_here"], common["future_gain_if_introduced_here"])
    def test_reader_visible_late_can_expand_into_phrase_with_single_new_family(self) -> None:
        artifact = {
            "chapter_pair_count": 1,
            "pair_reports": [
                {
                    "source_chapter_index": 1,
                    "target_chapter_index": 1,
                    "paragraph_alignment": {
                        "matched_blocks": [
                            {
                                "source_range": [1, 1],
                                "target_range": [1, 1],
                                "source_preview": "Ferdynand miał reklamę przed swoim pogrzebem",
                                "target_preview": "Ferdinand měl reklamu před svým pohřbem",
                            },
                            {
                                "source_range": [2, 2],
                                "target_range": [2, 2],
                                "source_preview": "dom dom dom dom dom dom dom dom dom dom dom dom",
                                "target_preview": "dům dům dům dům dům dům dům dům dům dům dům dům",
                            }
                        ]
                    },
                    "enrichment_bundle": {
                        "items": [
                            {
                                "source_paragraph_index": 1,
                                "target_paragraph_index": 1,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 6],
                                    "target_span": [1, 6],
                                    "source_text": "Ferdynand miał reklamę przed swoim pogrzebem",
                                    "target_text": "Ferdinand měl reklamu před svým pohřbem",
                                    "score": 0.82,
                                    "signals": {"total": 0.82},
                                },
                                "source_tokens": [
                                    {"index": 1, "text": "Ferdynand", "kind": "word"},
                                    {"index": 2, "text": "miał", "kind": "word"},
                                    {"index": 3, "text": "reklamę", "kind": "word"},
                                    {"index": 4, "text": "przed", "kind": "word"},
                                    {"index": 5, "text": "swoim", "kind": "word"},
                                    {"index": 6, "text": "pogrzebem", "kind": "word"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "text": "Ferdinand", "kind": "word"},
                                    {"index": 2, "text": "měl", "kind": "word"},
                                    {"index": 3, "text": "reklamu", "kind": "word"},
                                    {"index": 4, "text": "před", "kind": "word"},
                                    {"index": 5, "text": "svým", "kind": "word"},
                                    {"index": 6, "text": "pohřbem", "kind": "word"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": 3,
                                        "target_token_index": 3,
                                        "source_text": "reklamę",
                                        "target_text": "reklamu",
                                        "source_lemma": "reklama",
                                        "target_lemma": "reklama",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "relation": "lemma_like",
                                        "score": 0.79,
                                        "signals": {"norm": 0.85, "total": 0.79},
                                    },
                                    {
                                        "source_token_index": 6,
                                        "target_token_index": 6,
                                        "source_text": "pogrzebem",
                                        "target_text": "pohřbem",
                                        "source_lemma": "pogrzeb",
                                        "target_lemma": "pohreb",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "relation": "lemma_like",
                                        "score": 0.82,
                                        "signals": {"norm": 0.7, "total": 0.82},
                                    },
                                ],
                                "phrase_candidates": [
                                    {
                                        "source_span": [4, 6],
                                        "target_span": [4, 6],
                                        "score": 0.705,
                                        "relation": "tree_phrase",
                                        "signals": {"embedding": 0.95, "lemma": 0.0},
                                    }
                                ],
                                "subtree_candidates": [],
                            },
                            {
                                "source_paragraph_index": 2,
                                "target_paragraph_index": 2,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 12],
                                    "target_span": [1, 12],
                                    "source_text": "dom dom dom dom dom dom dom dom dom dom dom dom",
                                    "target_text": "dům dům dům dům dům dům dům dům dům dům dům dům",
                                    "score": 0.9,
                                    "signals": {"total": 0.9},
                                },
                                "source_tokens": [
                                    {"index": idx, "text": "dom", "kind": "word"}
                                    for idx in range(1, 13)
                                ],
                                "target_tokens": [
                                    {"index": idx, "text": "dům", "kind": "word"}
                                    for idx in range(1, 13)
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": idx,
                                        "target_token_index": idx,
                                        "source_text": "dom",
                                        "target_text": "dům",
                                        "source_lemma": "dom",
                                        "target_lemma": "dům",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "relation": "lemma_like",
                                        "score": 0.9,
                                        "signals": {"norm": 0.4, "total": 0.9},
                                    }
                                    for idx in range(1, 13)
                                ],
                                "phrase_candidates": [],
                                "subtree_candidates": [],
                            }
                        ]
                    },
                }
            ],
        }

        payload = build_paragraph_hybridization_plan(
            artifact,
            target_power=1.4,
            granularity_policy="reader-visible-late",
        )
        selected = payload["units"][0]["selected_candidates"]
        texts = {(item["granularity"], item["source_text"], item["target_text"]) for item in selected}
        self.assertNotIn(("paragraph", "Ferdynand miał reklamę przed swoim pogrzebem", "Ferdinand měl reklamu před svým pohřbem"), texts)
        self.assertTrue(
            ("phrase", "przed swoim pogrzebem", "před svým pohřbem") in texts
            or ("token", "pogrzebem", "pohřbem") in texts
        )

    def test_plan_defers_paragraph_candidate_early_and_prefers_transparent_token(self) -> None:
        artifact = {
            "chapter_pair_count": 1,
            "pair_reports": [
                {
                    "source_chapter_index": 1,
                    "target_chapter_index": 1,
                    "paragraph_alignment": {
                        "matched_blocks": [
                            {
                                "source_range": [1, 1],
                                "target_range": [1, 1],
                                "source_preview": "pani Müllerowo",
                                "target_preview": "paní Müllerová",
                            },
                            {
                                "source_range": [2, 2],
                                "target_range": [2, 2],
                                "source_preview": "dom dom dom dom dom dom dom dom dom dom",
                                "target_preview": "dům dům dům dům dům dům dům dům dům dům",
                            },
                        ]
                    },
                    "enrichment_bundle": {
                        "items": [
                            {
                                "source_paragraph_index": 1,
                                "target_paragraph_index": 1,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 2],
                                    "target_span": [1, 2],
                                    "source_text": "pani Müllerowo",
                                    "target_text": "paní Müllerová",
                                    "score": 0.85,
                                    "signals": {"total": 0.85},
                                },
                                "source_tokens": [
                                    {"index": 1, "text": "pani", "kind": "word"},
                                    {"index": 2, "text": "Müllerowo", "kind": "word"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "text": "paní", "kind": "word"},
                                    {"index": 2, "text": "Müllerová", "kind": "word"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": 2,
                                        "target_token_index": 2,
                                        "source_text": "Müllerowo",
                                        "target_text": "Müllerová",
                                        "source_lemma": "müllerowa",
                                        "target_lemma": "müllerová",
                                        "source_upos": "PROPN",
                                        "target_upos": "PROPN",
                                        "relation": "lemma_like",
                                        "score": 0.62,
                                        "signals": {"norm": 0.8, "total": 0.62},
                                    }
                                ],
                                "phrase_candidates": [],
                                "subtree_candidates": [],
                            },
                            {
                                "source_paragraph_index": 2,
                                "target_paragraph_index": 2,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 10],
                                    "target_span": [1, 10],
                                    "source_text": "dom dom dom dom dom dom dom dom dom dom",
                                    "target_text": "dům dům dům dům dům dům dům dům dům dům",
                                    "score": 0.9,
                                    "signals": {"total": 0.9},
                                },
                                "source_tokens": [
                                    {"index": idx, "text": "dom", "kind": "word"}
                                    for idx in range(1, 11)
                                ],
                                "target_tokens": [
                                    {"index": idx, "text": "dům", "kind": "word"}
                                    for idx in range(1, 11)
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": idx,
                                        "target_token_index": idx,
                                        "source_text": "dom",
                                        "target_text": "dům",
                                        "source_lemma": "dom",
                                        "target_lemma": "dům",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "relation": "lemma_like",
                                        "score": 0.9,
                                        "signals": {"norm": 0.4, "total": 0.9},
                                    }
                                    for idx in range(1, 11)
                                ],
                                "phrase_candidates": [],
                                "subtree_candidates": [],
                            },
                        ]
                    },
                }
            ],
        }

        payload = build_paragraph_hybridization_plan(artifact, target_power=1.8)
        first_unit = payload["units"][0]
        selected = {(item["granularity"], item["source_text"], item["target_text"]) for item in first_unit["selected_candidates"]}
        self.assertTrue(
            ("paragraph", "pani Müllerowo", "paní Müllerová") in selected
            or ("token", "Müllerowo", "Müllerová") in selected
        )

    def test_sentence_candidate_conflicts_with_nested_token_in_same_scope(self) -> None:
        artifact = {
            "chapter_pair_count": 1,
            "pair_reports": [
                {
                    "source_chapter_index": 1,
                    "target_chapter_index": 1,
                    "paragraph_alignment": {
                        "matched_blocks": [
                            {
                                "source_range": [1, 1],
                                "target_range": [1, 1],
                                "source_preview": "Każdy gość jest dobry. Dla nas polityka nie ma znaczenia.",
                                "target_preview": "Host jako host. Pro nás politika nic neznamená.",
                            }
                        ]
                    },
                    "enrichment_bundle": {
                        "items": [
                            {
                                "source_paragraph_index": 1,
                                "target_paragraph_index": 1,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 2],
                                    "target_span": [1, 2],
                                    "source_text": "Każdy gość jest dobry. Dla nas polityka nie ma znaczenia.",
                                    "target_text": "Host jako host. Pro nás politika nic neznamená.",
                                    "score": 0.7,
                                    "signals": {"total": 0.7},
                                },
                                "source_tokens": [
                                    {"index": 1, "text": "Każdy", "kind": "word"},
                                    {"index": 2, "text": "gość", "kind": "word"},
                                    {"index": 3, "text": "jest", "kind": "word"},
                                    {"index": 4, "text": "dobry", "kind": "word"},
                                    {"index": 5, "text": "Dla", "kind": "word"},
                                    {"index": 6, "text": "nas", "kind": "word"},
                                    {"index": 7, "text": "polityka", "kind": "word"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "text": "Host", "kind": "word"},
                                    {"index": 2, "text": "jako", "kind": "word"},
                                    {"index": 3, "text": "host", "kind": "word"},
                                    {"index": 4, "text": "Pro", "kind": "word"},
                                    {"index": 5, "text": "nás", "kind": "word"},
                                    {"index": 6, "text": "politika", "kind": "word"},
                                    {"index": 7, "text": "neznamená", "kind": "word"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": 7,
                                        "target_token_index": 6,
                                        "source_text": "polityka",
                                        "target_text": "politika",
                                        "source_lemma": "polityka",
                                        "target_lemma": "politika",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "relation": "upos_cognate",
                                        "score": 0.93,
                                        "signals": {"norm": 0.9, "total": 0.93},
                                    }
                                ],
                                "phrase_candidates": [],
                                "subtree_candidates": [],
                            }
                        ]
                    },
                }
            ],
        }

        payload = build_paragraph_hybridization_plan(artifact, target_power=1.4)
        first_unit = payload["units"][0]["selected_candidates"]
        granularities = [item["granularity"] for item in first_unit]
        self.assertFalse("sentence" in granularities and "token" in granularities)

    def test_smooth_policy_prefers_transparent_token_early_over_large_unit(self) -> None:
        artifact = {
            "chapter_pair_count": 1,
            "pair_reports": [
                {
                    "source_chapter_index": 1,
                    "target_chapter_index": 1,
                    "paragraph_alignment": {
                        "matched_blocks": [
                            {
                                "source_range": [1, 1],
                                "target_range": [1, 1],
                                "source_preview": "pani Müllerowo przyszła szybko",
                                "target_preview": "paní Müllerová přišla rychle",
                            },
                            {
                                "source_range": [2, 2],
                                "target_range": [2, 2],
                                "source_preview": "dom dom dom dom dom dom dom dom",
                                "target_preview": "dům dům dům dům dům dům dům dům",
                            },
                        ]
                    },
                    "enrichment_bundle": {
                        "items": [
                            {
                                "source_paragraph_index": 1,
                                "target_paragraph_index": 1,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 4],
                                    "target_span": [1, 4],
                                    "source_text": "pani Müllerowo przyszła szybko",
                                    "target_text": "paní Müllerová přišla rychle",
                                    "score": 0.86,
                                    "signals": {"total": 0.86},
                                },
                                "source_tokens": [
                                    {"index": 1, "text": "pani", "kind": "word"},
                                    {"index": 2, "text": "Müllerowo", "kind": "word"},
                                    {"index": 3, "text": "przyszła", "kind": "word"},
                                    {"index": 4, "text": "szybko", "kind": "word"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "text": "paní", "kind": "word"},
                                    {"index": 2, "text": "Müllerová", "kind": "word"},
                                    {"index": 3, "text": "přišla", "kind": "word"},
                                    {"index": 4, "text": "rychle", "kind": "word"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": 2,
                                        "target_token_index": 2,
                                        "source_text": "Müllerowo",
                                        "target_text": "Müllerová",
                                        "source_lemma": "müllerowa",
                                        "target_lemma": "müllerová",
                                        "source_upos": "PROPN",
                                        "target_upos": "PROPN",
                                        "relation": "lemma_like",
                                        "score": 0.68,
                                        "signals": {"norm": 0.8, "total": 0.68},
                                    }
                                ],
                                "phrase_candidates": [],
                                "subtree_candidates": [],
                            },
                            {
                                "source_paragraph_index": 2,
                                "target_paragraph_index": 2,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 8],
                                    "target_span": [1, 8],
                                    "source_text": "dom dom dom dom dom dom dom dom",
                                    "target_text": "dům dům dům dům dům dům dům dům",
                                    "score": 0.92,
                                    "signals": {"total": 0.92},
                                },
                                "source_tokens": [{"index": i, "text": "dom", "kind": "word"} for i in range(1, 9)],
                                "target_tokens": [{"index": i, "text": "dům", "kind": "word"} for i in range(1, 9)],
                                "token_pairs": [
                                    {
                                        "source_token_index": i,
                                        "target_token_index": i,
                                        "source_text": "dom",
                                        "target_text": "dům",
                                        "source_lemma": "dom",
                                        "target_lemma": "dům",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "relation": "lemma_like",
                                        "score": 0.92,
                                        "signals": {"norm": 0.4, "total": 0.92},
                                    }
                                    for i in range(1, 9)
                                ],
                                "phrase_candidates": [],
                                "subtree_candidates": [],
                            },
                        ]
                    },
                }
            ],
        }

        payload = build_paragraph_hybridization_plan(
            artifact,
            target_power=1.4,
            granularity_policy="smooth",
        )
        selected = {(item["granularity"], item["source_text"], item["target_text"]) for item in payload["units"][0]["selected_candidates"]}
        self.assertNotIn(("paragraph", "pani Müllerowo przyszła szybko", "paní Müllerová přišla rychle"), selected)

    def test_plan_ignores_fallback_candidates_and_keeps_known_units(self) -> None:
        artifact = {
            "chapter_pair_count": 1,
            "pair_reports": [
                {
                    "source_chapter_index": 1,
                    "target_chapter_index": 1,
                    "paragraph_alignment": {
                        "matched_blocks": [
                            {
                                "source_range": [1, 1],
                                "target_range": [1, 1],
                                "source_preview": "nam zabili Ferdynanda",
                                "target_preview": "nám zabili Ferdinanda",
                            },
                            {
                                "source_range": [2, 2],
                                "target_range": [2, 2],
                                "source_preview": "stary dom",
                                "target_preview": "starý dům",
                            },
                        ]
                    },
                    "enrichment_bundle": {
                        "items": [
                            {
                                "source_paragraph_index": 1,
                                "target_paragraph_index": 1,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 3],
                                    "target_span": [1, 3],
                                    "source_text": "nam zabili Ferdynanda",
                                    "target_text": "nám zabili Ferdinanda",
                                    "score": 0.9,
                                    "signals": {"total": 0.9},
                                },
                                "source_tokens": [
                                    {"index": 1, "text": "nam", "kind": "word"},
                                    {"index": 2, "text": "zabili", "kind": "word"},
                                    {"index": 3, "text": "Ferdynanda", "kind": "word"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "text": "nám", "kind": "word"},
                                    {"index": 2, "text": "zabili", "kind": "word"},
                                    {"index": 3, "text": "Ferdinanda", "kind": "word"},
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
                                        "score": 0.9,
                                    },
                                    {
                                        "source_token_index": 2,
                                        "target_token_index": 2,
                                        "source_text": "zabili",
                                        "target_text": "zabili",
                                        "source_lemma": "zabić",
                                        "target_lemma": "zabit",
                                        "relation": "lemma_like",
                                        "score": 0.9,
                                    },
                                    {
                                        "source_token_index": 3,
                                        "target_token_index": 3,
                                        "source_text": "Ferdynanda",
                                        "target_text": "Ferdinanda",
                                        "source_lemma": "ferdynand",
                                        "target_lemma": "ferdinand",
                                        "relation": "embedding_assisted",
                                        "score": 0.7,
                                    },
                                ],
                                "phrase_candidates": [
                                    {
                                        "source_span": [1, 2],
                                        "target_span": [1, 2],
                                        "score": 0.82,
                                        "relation": "tree_phrase",
                                        "signals": {"total": 0.82},
                                    }
                                ],
                                "subtree_candidates": [
                                    {
                                        "source_span": [1, 2],
                                        "target_span": [1, 2],
                                        "source_text": "nam zabili",
                                        "target_text": "nám zabili",
                                        "score": 0.84,
                                        "relation": "tree_walk",
                                        "signals": {"total": 0.84},
                                    }
                                ],
                            },
                            {
                                "source_paragraph_index": 2,
                                "target_paragraph_index": 2,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 2],
                                    "target_span": [1, 2],
                                    "source_text": "stary dom",
                                    "target_text": "starý dům",
                                    "score": 0.91,
                                    "signals": {"total": 0.91},
                                },
                                "source_tokens": [
                                    {"index": 1, "text": "stary", "kind": "word"},
                                    {"index": 2, "text": "dom", "kind": "word"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "text": "starý", "kind": "word"},
                                    {"index": 2, "text": "dům", "kind": "word"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": 1,
                                        "target_token_index": 1,
                                        "source_text": "stary",
                                        "target_text": "starý",
                                        "source_lemma": "stary",
                                        "target_lemma": "starý",
                                        "relation": "lemma_like",
                                        "score": 0.92,
                                    },
                                    {
                                        "source_token_index": 2,
                                        "target_token_index": 2,
                                        "source_text": "dom",
                                        "target_text": "dům",
                                        "source_lemma": "dom",
                                        "target_lemma": "dům",
                                        "relation": "lemma_like",
                                        "score": 0.92,
                                    },
                                ],
                                "phrase_candidates": [],
                                "subtree_candidates": [
                                    {
                                        "source_span": [1, 2],
                                        "target_span": [1, 2],
                                        "source_text": "stary dom",
                                        "target_text": "starý dům",
                                        "score": 0.95,
                                        "relation": "tree_walk",
                                        "signals": {"total": 0.95},
                                    }
                                ],
                            },
                        ]
                    },
                }
            ],
        }

        payload = build_paragraph_hybridization_plan(artifact)

        self.assertEqual(payload["mode"], "paragraph-by-paragraph-alignment-only-v1")
        self.assertEqual(payload["unit_count"], 2)
        first_unit = payload["units"][0]
        relations = {candidate["relation"] for candidate in first_unit["selected_candidates"]}
        self.assertNotIn("embedding_assisted", relations)
        self.assertGreaterEqual(first_unit["selected_count"], 1)

        second_unit = payload["units"][1]
        selected_granularities = {candidate["granularity"] for candidate in second_unit["selected_candidates"]}
        self.assertTrue(selected_granularities)
        self.assertEqual(second_unit["actual_future_czechness_after"], 1.0)

    def test_plan_filters_low_similarity_upos_cognate_token(self) -> None:
        artifact = {
            "chapter_pair_count": 1,
            "pair_reports": [
                {
                    "source_chapter_index": 1,
                    "target_chapter_index": 1,
                    "paragraph_alignment": {
                        "matched_blocks": [
                            {
                                "source_range": [1, 1],
                                "target_range": [1, 1],
                                "source_preview": "pierwszego stycznia",
                                "target_preview": "na Silvestra",
                            },
                        ]
                    },
                    "enrichment_bundle": {
                        "items": [
                            {
                                "source_paragraph_index": 1,
                                "target_paragraph_index": 1,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 2],
                                    "target_span": [1, 2],
                                    "source_text": "pierwszego stycznia",
                                    "target_text": "na Silvestra",
                                    "score": 0.8,
                                    "signals": {"total": 0.8},
                                },
                                "source_tokens": [
                                    {"index": 1, "text": "pierwszego", "kind": "word"},
                                    {"index": 2, "text": "stycznia", "kind": "word"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "text": "na", "kind": "word"},
                                    {"index": 2, "text": "Silvestra", "kind": "word"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": 2,
                                        "target_token_index": 2,
                                        "source_text": "stycznia",
                                        "target_text": "Silvestra",
                                        "source_lemma": "styczen",
                                        "target_lemma": "silvestr",
                                        "relation": "upos_cognate",
                                        "score": 0.64,
                                        "signals": {"norm": 0.333333, "total": 0.64},
                                    }
                                ],
                                "phrase_candidates": [],
                                "subtree_candidates": [],
                            }
                        ]
                    },
                }
            ],
        }

        payload = build_paragraph_hybridization_plan(artifact, target_power=1.8)
        first_unit = payload["units"][0]
        selected_tokens = [item for item in first_unit["selected_candidates"] if item["granularity"] == "token"]
        self.assertEqual(selected_tokens, [])

    def test_plan_blocks_function_words_as_standalone_tokens(self) -> None:
        artifact = {
            "chapter_pair_count": 1,
            "pair_reports": [
                {
                    "source_chapter_index": 1,
                    "target_chapter_index": 1,
                    "paragraph_alignment": {
                        "matched_blocks": [
                            {
                                "source_range": [1, 1],
                                "target_range": [1, 1],
                                "source_preview": "jeśli przyjdzie",
                                "target_preview": "že přijde",
                            },
                        ]
                    },
                    "enrichment_bundle": {
                        "items": [
                            {
                                "source_paragraph_index": 1,
                                "target_paragraph_index": 1,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 2],
                                    "target_span": [1, 2],
                                    "source_text": "jeśli przyjdzie",
                                    "target_text": "že přijde",
                                    "score": 0.84,
                                    "signals": {"total": 0.84},
                                },
                                "source_tokens": [
                                    {"index": 1, "text": "jeśli", "kind": "word"},
                                    {"index": 2, "text": "przyjdzie", "kind": "word"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "text": "že", "kind": "word"},
                                    {"index": 2, "text": "přijde", "kind": "word"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": 1,
                                        "target_token_index": 1,
                                        "source_text": "jeśli",
                                        "target_text": "že",
                                        "source_lemma": "jeśli",
                                        "target_lemma": "že",
                                        "source_upos": "SCONJ",
                                        "target_upos": "SCONJ",
                                        "relation": "lemma_like",
                                        "score": 0.91,
                                        "signals": {"norm": 0.2, "total": 0.91},
                                    },
                                    {
                                        "source_token_index": 2,
                                        "target_token_index": 2,
                                        "source_text": "przyjdzie",
                                        "target_text": "přijde",
                                        "source_lemma": "przyjść",
                                        "target_lemma": "přijít",
                                        "source_upos": "VERB",
                                        "target_upos": "VERB",
                                        "relation": "lemma_like",
                                        "score": 0.91,
                                        "signals": {"norm": 0.4, "total": 0.91},
                                    },
                                ],
                                "phrase_candidates": [],
                                "subtree_candidates": [],
                            }
                        ]
                    },
                }
            ],
        }

        payload = build_paragraph_hybridization_plan(artifact, target_power=1.8)
        first_unit = payload["units"][0]
        selected_pairs = {
            (item["source_text"], item["target_text"])
            for item in first_unit["selected_candidates"]
            if item["granularity"] == "token"
        }
        self.assertNotIn(("jeśli", "že"), selected_pairs)

    def test_early_phase_gating_can_prefer_token_over_phrase(self) -> None:
        artifact = {
            "chapter_pair_count": 1,
            "pair_reports": [
                {
                    "source_chapter_index": 1,
                    "target_chapter_index": 1,
                    "paragraph_alignment": {
                        "matched_blocks": [
                            {
                                "source_range": [1, 1],
                                "target_range": [1, 1],
                                "source_preview": "pani Müllerowa przyszła tutaj szybko",
                                "target_preview": "paní Müllerová sem rychle přišla",
                            },
                            {
                                "source_range": [2, 2],
                                "target_range": [2, 2],
                                "source_preview": "dom dom dom dom dom",
                                "target_preview": "dům dům dům dům dům",
                            },
                        ]
                    },
                    "enrichment_bundle": {
                        "items": [
                            {
                                "source_paragraph_index": 1,
                                "target_paragraph_index": 1,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 4],
                                    "target_span": [1, 5],
                                    "source_text": "pani Müllerowa przyszła tutaj szybko",
                                    "target_text": "paní Müllerová sem rychle přišla",
                                    "score": 0.88,
                                    "signals": {"total": 0.88},
                                },
                                "source_tokens": [
                                    {"index": 1, "text": "pani", "kind": "word"},
                                    {"index": 2, "text": "Müllerowa", "kind": "word"},
                                    {"index": 3, "text": "przyszła", "kind": "word"},
                                    {"index": 4, "text": "tutaj", "kind": "word"},
                                    {"index": 5, "text": "szybko", "kind": "word"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "text": "paní", "kind": "word"},
                                    {"index": 2, "text": "Müllerová", "kind": "word"},
                                    {"index": 3, "text": "sem", "kind": "word"},
                                    {"index": 4, "text": "rychle", "kind": "word"},
                                    {"index": 5, "text": "přišla", "kind": "word"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": 2,
                                        "target_token_index": 2,
                                        "source_text": "Müllerowa",
                                        "target_text": "Müllerová",
                                        "source_lemma": "müllerowa",
                                        "target_lemma": "müllerová",
                                        "source_upos": "PROPN",
                                        "target_upos": "PROPN",
                                        "relation": "lemma_like",
                                        "score": 0.68,
                                        "signals": {"norm": 0.8, "total": 0.68},
                                    },
                                ],
                                "phrase_candidates": [
                                    {
                                        "source_span": [1, 5],
                                        "target_span": [1, 5],
                                        "score": 0.94,
                                        "relation": "tree_phrase",
                                        "signals": {"total": 0.94},
                                    }
                                ],
                                "subtree_candidates": [],
                            }
                            ,
                            {
                                "source_paragraph_index": 2,
                                "target_paragraph_index": 2,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 5],
                                    "target_span": [1, 5],
                                    "source_text": "dom dom dom dom dom",
                                    "target_text": "dům dům dům dům dům",
                                    "score": 0.9,
                                    "signals": {"total": 0.9},
                                },
                                "source_tokens": [
                                    {"index": 1, "text": "dom", "kind": "word"},
                                    {"index": 2, "text": "dom", "kind": "word"},
                                    {"index": 3, "text": "dom", "kind": "word"},
                                    {"index": 4, "text": "dom", "kind": "word"},
                                    {"index": 5, "text": "dom", "kind": "word"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "text": "dům", "kind": "word"},
                                    {"index": 2, "text": "dům", "kind": "word"},
                                    {"index": 3, "text": "dům", "kind": "word"},
                                    {"index": 4, "text": "dům", "kind": "word"},
                                    {"index": 5, "text": "dům", "kind": "word"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": idx,
                                        "target_token_index": idx,
                                        "source_text": "dom",
                                        "target_text": "dům",
                                        "source_lemma": "dom",
                                        "target_lemma": "dům",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "relation": "lemma_like",
                                        "score": 0.9,
                                        "signals": {"norm": 0.3, "total": 0.9},
                                    }
                                    for idx in range(1, 6)
                                ],
                                "phrase_candidates": [],
                                "subtree_candidates": [],
                            }
                        ]
                    },
                }
            ],
        }

        without_penalty = build_paragraph_hybridization_plan(
            artifact,
            target_power=1.8,
            idiomaticity_penalty_weight=0.0,
        )
        with_penalty = build_paragraph_hybridization_plan(
            artifact,
            target_power=1.8,
            idiomaticity_penalty_weight=1.0,
        )

        first_without = without_penalty["units"][0]["selected_candidates"]
        first_with = with_penalty["units"][0]["selected_candidates"]

        self.assertTrue(first_without)
        self.assertTrue(first_with)

    def test_plan_blocks_standalone_token_inside_compound_ordinal(self) -> None:
        artifact = {
            "chapter_pair_count": 1,
            "pair_reports": [
                {
                    "source_chapter_index": 1,
                    "target_chapter_index": 1,
                    "paragraph_alignment": {
                        "matched_blocks": [
                            {
                                "source_range": [1, 1],
                                "target_range": [1, 1],
                                "source_preview": "od roku siedemdziesiątego pierwszego",
                                "target_preview": "od jedenasedmdesátého roku",
                            }
                        ]
                    },
                    "enrichment_bundle": {
                        "items": [
                            {
                                "source_paragraph_index": 1,
                                "target_paragraph_index": 1,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 4],
                                    "target_span": [1, 3],
                                    "source_text": "od roku siedemdziesiątego pierwszego",
                                    "target_text": "od jedenasedmdesátého roku",
                                    "score": 0.82,
                                    "signals": {"total": 0.82},
                                },
                                "source_tokens": [
                                    {"index": 1, "text": "od", "kind": "word"},
                                    {"index": 2, "text": "roku", "kind": "word"},
                                    {"index": 3, "text": "siedemdziesiątego", "kind": "word"},
                                    {"index": 4, "text": "pierwszego", "kind": "word"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "text": "od", "kind": "word"},
                                    {"index": 2, "text": "jedenasedmdesátého", "kind": "word"},
                                    {"index": 3, "text": "roku", "kind": "word"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": 2,
                                        "target_token_index": 3,
                                        "source_text": "roku",
                                        "target_text": "roku",
                                        "source_lemma": "rok",
                                        "target_lemma": "rok",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "relation": "exact",
                                        "score": 1.0,
                                        "signals": {"norm": 1.0, "total": 1.0},
                                    },
                                    {
                                        "source_token_index": 3,
                                        "target_token_index": 2,
                                        "source_text": "siedemdziesiątego",
                                        "target_text": "jedenasedmdesátého",
                                        "source_lemma": "siedemdziesiąty",
                                        "target_lemma": "jedenasedmdesátý",
                                        "source_upos": "ADJ",
                                        "target_upos": "ADJ",
                                        "relation": "upos_cognate",
                                        "score": 0.75,
                                        "signals": {"norm": 0.53, "total": 0.75},
                                    },
                                    {
                                        "source_token_index": 4,
                                        "target_token_index": 2,
                                        "source_text": "pierwszego",
                                        "target_text": "jedenasedmdesátého",
                                        "source_lemma": "pierwszy",
                                        "target_lemma": "jedenasedmdesátý",
                                        "source_upos": "ADJ",
                                        "target_upos": "ADJ",
                                        "relation": "embedding_assisted",
                                        "score": 0.7,
                                        "signals": {"norm": 0.1, "total": 0.7},
                                    },
                                ],
                                "phrase_candidates": [],
                                "subtree_candidates": [],
                            }
                        ]
                    },
                }
            ],
        }

        payload = build_paragraph_hybridization_plan(artifact, target_power=1.4)
        selected = payload["units"][0]["selected_candidates"]
        self.assertFalse(any(item["source_text"] == "siedemdziesiątego" for item in selected))

    def test_visible_smooth_prefers_visibly_czech_phrase_over_identical_token(self) -> None:
        artifact = {
            "chapter_pair_count": 1,
            "pair_reports": [
                {
                    "source_chapter_index": 1,
                    "target_chapter_index": 1,
                    "paragraph_alignment": {
                        "matched_blocks": [
                            {
                                "source_range": [1, 1],
                                "target_range": [1, 1],
                                "source_preview": "Jak pan chce za ramię.",
                                "target_preview": "Jak pan chce za rameno.",
                            }
                        ]
                    },
                    "enrichment_bundle": {
                        "items": [
                            {
                                "source_paragraph_index": 1,
                                "target_paragraph_index": 1,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 5],
                                    "target_span": [1, 5],
                                    "source_text": "Jak pan chce za ramię",
                                    "target_text": "Jak pan chce za rameno",
                                    "score": 0.84,
                                    "signals": {"total": 0.84},
                                },
                                "source_tokens": [
                                    {"index": 1, "text": "Jak", "kind": "word"},
                                    {"index": 2, "text": "pan", "kind": "word"},
                                    {"index": 3, "text": "chce", "kind": "word"},
                                    {"index": 4, "text": "za", "kind": "word"},
                                    {"index": 5, "text": "ramię", "kind": "word"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "text": "Jak", "kind": "word"},
                                    {"index": 2, "text": "pan", "kind": "word"},
                                    {"index": 3, "text": "chce", "kind": "word"},
                                    {"index": 4, "text": "za", "kind": "word"},
                                    {"index": 5, "text": "rameno", "kind": "word"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": 1,
                                        "target_token_index": 1,
                                        "source_text": "Jak",
                                        "target_text": "Jak",
                                        "source_lemma": "jak",
                                        "target_lemma": "jak",
                                        "source_upos": "ADV",
                                        "target_upos": "ADV",
                                        "relation": "exact",
                                        "score": 1.0,
                                        "signals": {"norm": 1.0, "total": 1.0},
                                    },
                                    {
                                        "source_token_index": 5,
                                        "target_token_index": 5,
                                        "source_text": "ramię",
                                        "target_text": "rameno",
                                        "source_lemma": "ramię",
                                        "target_lemma": "rameno",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "relation": "lemma_like",
                                        "score": 0.78,
                                        "signals": {"norm": 0.55, "total": 0.78},
                                    },
                                ],
                                "phrase_candidates": [
                                    {
                                        "source_span": [4, 5],
                                        "target_span": [4, 5],
                                        "score": 0.88,
                                        "relation": "tree_phrase",
                                        "signals": {"embedding": 0.9, "lemma": 0.6},
                                    }
                                ],
                                "subtree_candidates": [],
                            }
                        ]
                    },
                }
            ],
        }

        payload = build_paragraph_hybridization_plan(
            artifact,
            target_power=1.4,
            granularity_policy="visible-smooth",
        )
        selected = payload["units"][0]["selected_candidates"]
        texts = {(item["granularity"], item["source_text"], item["target_text"]) for item in selected}
        self.assertIn(("phrase", "za ramię", "za rameno"), texts)

    def test_reader_visible_late_can_select_sentence_in_late_progress(self) -> None:
        items = []
        matched_blocks = []
        for idx in range(1, 13):
            matched_blocks.append(
                {
                    "source_range": [idx, idx],
                    "target_range": [idx, idx],
                    "source_preview": f"blok {idx}",
                    "target_preview": f"blok {idx}",
                }
            )
            source_text = "dom dom dom"
            target_text = "dům dům dům"
            token_pairs = [
                {
                    "source_token_index": j,
                    "target_token_index": j,
                    "source_text": "dom",
                    "target_text": "dům",
                    "source_lemma": "dom",
                    "target_lemma": "dům",
                    "source_upos": "NOUN",
                    "target_upos": "NOUN",
                    "relation": "lemma_like",
                    "score": 0.8,
                    "signals": {"norm": 0.4, "total": 0.8},
                }
                for j in range(1, 4)
            ]
            if idx == 12:
                source_text = "Za ramię go złapał"
                target_text = "Za rameno ho chytil"
                token_pairs = [
                    {
                        "source_token_index": 1,
                        "target_token_index": 1,
                        "source_text": "Za",
                        "target_text": "Za",
                        "source_lemma": "za",
                        "target_lemma": "za",
                        "source_upos": "ADP",
                        "target_upos": "ADP",
                        "relation": "exact",
                        "score": 1.0,
                        "signals": {"norm": 1.0, "total": 1.0},
                    },
                    {
                        "source_token_index": 2,
                        "target_token_index": 2,
                        "source_text": "ramię",
                        "target_text": "rameno",
                        "source_lemma": "ramię",
                        "target_lemma": "rameno",
                        "source_upos": "NOUN",
                        "target_upos": "NOUN",
                        "relation": "lemma_like",
                        "score": 0.8,
                        "signals": {"norm": 0.5, "total": 0.8},
                    },
                    {
                        "source_token_index": 4,
                        "target_token_index": 4,
                        "source_text": "złapał",
                        "target_text": "chytil",
                        "source_lemma": "złapać",
                        "target_lemma": "chytit",
                        "source_upos": "VERB",
                        "target_upos": "VERB",
                        "relation": "lemma_like",
                        "score": 0.83,
                        "signals": {"norm": 0.45, "total": 0.83},
                    },
                ]
            items.append(
                {
                    "source_paragraph_index": idx,
                    "target_paragraph_index": idx,
                    "enrichment_status": "sentence_safe",
                    "sentence_alignment": {
                        "source_span": [1, max(3, len(token_pairs))],
                        "target_span": [1, max(3, len(token_pairs))],
                        "source_text": source_text,
                        "target_text": target_text,
                        "score": 0.84,
                        "signals": {"total": 0.84},
                    },
                    "source_tokens": [
                        {"index": j, "text": token_pairs[j - 1]["source_text"], "kind": "word"}
                        for j in range(1, len(token_pairs) + 1)
                    ],
                    "target_tokens": [
                        {"index": j, "text": token_pairs[j - 1]["target_text"], "kind": "word"}
                        for j in range(1, len(token_pairs) + 1)
                    ],
                    "token_pairs": token_pairs,
                    "phrase_candidates": [
                        {
                            "source_span": [1, len(token_pairs)],
                            "target_span": [1, len(token_pairs)],
                            "score": 0.9,
                            "relation": "tree_phrase",
                            "signals": {"embedding": 0.91},
                        }
                    ]
                    if idx == 12
                    else [],
                    "subtree_candidates": [],
                }
            )
        artifact = {
            "chapter_pair_count": 1,
            "pair_reports": [
                {
                    "source_chapter_index": 1,
                    "target_chapter_index": 1,
                    "paragraph_alignment": {"matched_blocks": matched_blocks},
                    "enrichment_bundle": {"items": items},
                }
            ],
        }

        payload = build_paragraph_hybridization_plan(
            artifact,
            target_power=1.4,
            granularity_policy="reader-visible-late",
        )
        selected = payload["units"][-1]["selected_candidates"]
        self.assertTrue(any(item["granularity"] in {"phrase", "sentence"} for item in selected))

    def test_early_wide_phrase_is_not_used_as_carrier_for_single_new_family(self) -> None:
        artifact = {
            "chapter_pair_count": 1,
            "pair_reports": [
                {
                    "source_chapter_index": 1,
                    "target_chapter_index": 1,
                    "paragraph_alignment": {
                        "matched_blocks": [
                            {
                                "source_range": [1, 1],
                                "target_range": [1, 1],
                                "source_preview": "jego losy czasu wojny światowej",
                                "target_preview": "jeho osudy za světové války",
                            },
                            {
                                "source_range": [2, 2],
                                "target_range": [2, 2],
                                "source_preview": "dom dom dom dom dom dom dom dom dom dom",
                                "target_preview": "dům dům dům dům dům dům dům dům dům dům",
                            },
                        ]
                    },
                    "enrichment_bundle": {
                        "items": [
                            {
                                "source_paragraph_index": 1,
                                "target_paragraph_index": 1,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 5],
                                    "target_span": [1, 5],
                                    "source_text": "jego losy czasu wojny światowej",
                                    "target_text": "jeho osudy za světové války",
                                    "score": 0.61,
                                    "signals": {"total": 0.61},
                                },
                                "source_tokens": [
                                    {"index": 1, "text": "jego", "kind": "word"},
                                    {"index": 2, "text": "losy", "kind": "word"},
                                    {"index": 3, "text": "czasu", "kind": "word"},
                                    {"index": 4, "text": "wojny", "kind": "word"},
                                    {"index": 5, "text": "światowej", "kind": "word"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "text": "jeho", "kind": "word"},
                                    {"index": 2, "text": "osudy", "kind": "word"},
                                    {"index": 3, "text": "za", "kind": "word"},
                                    {"index": 4, "text": "světové", "kind": "word"},
                                    {"index": 5, "text": "války", "kind": "word"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": 2,
                                        "target_token_index": 2,
                                        "source_text": "losy",
                                        "target_text": "osudy",
                                        "source_lemma": "los",
                                        "target_lemma": "osud",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "relation": "lemma_like",
                                        "score": 0.74,
                                        "signals": {"norm": 0.7, "total": 0.74},
                                    }
                                ],
                                "phrase_candidates": [
                                    {
                                        "source_span": [1, 5],
                                        "target_span": [1, 5],
                                        "score": 0.604,
                                        "relation": "tree_phrase",
                                        "signals": {"embedding": 0.7, "lemma": 0.1, "upos": 0.2, "deprel": 0.2},
                                    }
                                ],
                                "subtree_candidates": [],
                            },
                            {
                                "source_paragraph_index": 2,
                                "target_paragraph_index": 2,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 10],
                                    "target_span": [1, 10],
                                    "source_text": "dom dom dom dom dom dom dom dom dom dom",
                                    "target_text": "dům dům dům dům dům dům dům dům dům dům",
                                    "score": 0.9,
                                    "signals": {"total": 0.9},
                                },
                                "source_tokens": [
                                    {"index": idx, "text": "dom", "kind": "word"}
                                    for idx in range(1, 11)
                                ],
                                "target_tokens": [
                                    {"index": idx, "text": "dům", "kind": "word"}
                                    for idx in range(1, 11)
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": idx,
                                        "target_token_index": idx,
                                        "source_text": "dom",
                                        "target_text": "dům",
                                        "source_lemma": "dom",
                                        "target_lemma": "dům",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "relation": "lemma_like",
                                        "score": 0.9,
                                        "signals": {"norm": 0.4, "total": 0.9},
                                    }
                                    for idx in range(1, 11)
                                ],
                                "phrase_candidates": [],
                                "subtree_candidates": [],
                            },
                        ]
                    },
                }
            ],
        }

        payload = build_paragraph_hybridization_plan(
            artifact,
            target_power=1.4,
            granularity_policy="reader-visible-late",
        )
        selected = {(item["granularity"], item["source_text"], item["target_text"]) for item in payload["units"][0]["selected_candidates"]}
        self.assertNotIn(("phrase", "jego losy czasu wojny światowej", "jeho osudy za světové války"), selected)

    def test_function_words_affect_target_future_but_remain_blocked_standalone(self) -> None:
        artifact = {
            "chapter_pair_count": 1,
            "pair_reports": [
                {
                    "source_chapter_index": 1,
                    "target_chapter_index": 1,
                    "paragraph_alignment": {
                        "matched_blocks": [
                            {
                                "source_range": [1, 1],
                                "target_range": [1, 1],
                                "source_preview": "i dom",
                                "target_preview": "a dům",
                            },
                            {
                                "source_range": [2, 2],
                                "target_range": [2, 2],
                                "source_preview": "i",
                                "target_preview": "a",
                            },
                        ]
                    },
                    "enrichment_bundle": {
                        "items": [
                            {
                                "source_paragraph_index": 1,
                                "target_paragraph_index": 1,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 2],
                                    "target_span": [1, 2],
                                    "source_text": "i dom",
                                    "target_text": "a dům",
                                    "score": 0.9,
                                    "signals": {"total": 0.9},
                                },
                                "source_tokens": [
                                    {"index": 1, "text": "i", "lemma": "i", "upos": "CCONJ", "kind": "word"},
                                    {"index": 2, "text": "dom", "lemma": "dom", "upos": "NOUN", "kind": "word"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "text": "a", "lemma": "a", "upos": "CCONJ", "kind": "word"},
                                    {"index": 2, "text": "dům", "lemma": "dům", "upos": "NOUN", "kind": "word"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": 1,
                                        "target_token_index": 1,
                                        "source_text": "i",
                                        "target_text": "a",
                                        "source_lemma": "i",
                                        "target_lemma": "a",
                                        "source_upos": "CCONJ",
                                        "target_upos": "CCONJ",
                                        "relation": "lemma_like",
                                        "score": 0.86,
                                        "signals": {"norm": 0.86, "total": 0.86},
                                    },
                                    {
                                        "source_token_index": 2,
                                        "target_token_index": 2,
                                        "source_text": "dom",
                                        "target_text": "dům",
                                        "source_lemma": "dom",
                                        "target_lemma": "dům",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "relation": "lemma_like",
                                        "score": 0.9,
                                        "signals": {"norm": 0.9, "total": 0.9},
                                    },
                                ],
                                "phrase_candidates": [],
                                "subtree_candidates": [],
                            },
                            {
                                "source_paragraph_index": 2,
                                "target_paragraph_index": 2,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 1],
                                    "target_span": [1, 1],
                                    "source_text": "i",
                                    "target_text": "a",
                                    "score": 0.86,
                                    "signals": {"total": 0.86},
                                },
                                "source_tokens": [
                                    {"index": 1, "text": "i", "lemma": "i", "upos": "CCONJ", "kind": "word"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "text": "a", "lemma": "a", "upos": "CCONJ", "kind": "word"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": 1,
                                        "target_token_index": 1,
                                        "source_text": "i",
                                        "target_text": "a",
                                        "source_lemma": "i",
                                        "target_lemma": "a",
                                        "source_upos": "CCONJ",
                                        "target_upos": "CCONJ",
                                        "relation": "lemma_like",
                                        "score": 0.86,
                                        "signals": {"norm": 0.86, "total": 0.86},
                                    },
                                ],
                                "phrase_candidates": [],
                                "subtree_candidates": [],
                            },
                        ]
                    },
                }
            ],
        }

        payload = build_paragraph_hybridization_plan(
            artifact,
            target_power=1.0,
            granularity_policy="reader-visible-late",
        )

        first_unit = payload["units"][0]
        second_unit = payload["units"][1]
        first_selected = {candidate["candidate_id"] for candidate in first_unit["selected_candidates"]}
        second_selected = {candidate["candidate_id"] for candidate in second_unit["selected_candidates"]}

        self.assertNotIn("u1:token:1", first_selected)
        self.assertTrue(first_selected)
        self.assertFalse(second_selected)
        self.assertGreater(second_unit["actual_future_czechness_before"], 0.0)

    def test_cumulative_simple_can_promote_known_phrase_into_sentence_with_one_new_lemma(self) -> None:
        artifact = {
            "chapter_pair_count": 1,
            "pair_reports": [
                {
                    "source_chapter_index": 1,
                    "target_chapter_index": 1,
                    "paragraph_alignment": {
                        "matched_blocks": [
                            {
                                "source_range": [1, 1],
                                "target_range": [1, 1],
                                "source_preview": "idzie stara droga",
                                "target_preview": "jde stará cesta",
                            },
                            {
                                "source_range": [2, 2],
                                "target_range": [2, 2],
                                "source_preview": "idzie stara droga dalej",
                                "target_preview": "jde starou cestou dál",
                            },
                        ]
                    },
                    "enrichment_bundle": {
                        "items": [
                            {
                                "source_paragraph_index": 1,
                                "target_paragraph_index": 1,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 3],
                                    "target_span": [1, 3],
                                    "source_text": "idzie stara droga",
                                    "target_text": "jde stará cesta",
                                    "score": 0.95,
                                    "signals": {"embedding": 0.95, "lemma": 0.95, "upos": 1.0, "deprel": 0.95},
                                },
                                "source_tokens": [
                                    {"index": 1, "text": "idzie", "kind": "word", "lemma": "iść", "upos": "VERB"},
                                    {"index": 2, "text": "stara", "kind": "word", "lemma": "stary", "upos": "ADJ"},
                                    {"index": 3, "text": "droga", "kind": "word", "lemma": "droga", "upos": "NOUN"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "text": "jde", "kind": "word", "lemma": "jít", "upos": "VERB"},
                                    {"index": 2, "text": "stará", "kind": "word", "lemma": "starý", "upos": "ADJ"},
                                    {"index": 3, "text": "cesta", "kind": "word", "lemma": "cesta", "upos": "NOUN"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": 1,
                                        "target_token_index": 1,
                                        "source_text": "idzie",
                                        "target_text": "jde",
                                        "source_lemma": "iść",
                                        "target_lemma": "jít",
                                        "source_upos": "VERB",
                                        "target_upos": "VERB",
                                        "relation": "lemma_like",
                                        "score": 0.95,
                                        "signals": {"embedding": 0.95, "lemma": 0.95, "upos": 1.0, "deprel": 0.95},
                                    },
                                    {
                                        "source_token_index": 2,
                                        "target_token_index": 2,
                                        "source_text": "stara",
                                        "target_text": "stará",
                                        "source_lemma": "stary",
                                        "target_lemma": "starý",
                                        "source_upos": "ADJ",
                                        "target_upos": "ADJ",
                                        "relation": "lemma_like",
                                        "score": 0.94,
                                        "signals": {"embedding": 0.94, "lemma": 0.94, "upos": 1.0, "deprel": 0.94},
                                    },
                                    {
                                        "source_token_index": 3,
                                        "target_token_index": 3,
                                        "source_text": "droga",
                                        "target_text": "cesta",
                                        "source_lemma": "droga",
                                        "target_lemma": "cesta",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "relation": "lemma_like",
                                        "score": 0.93,
                                        "signals": {"embedding": 0.93, "lemma": 0.93, "upos": 1.0, "deprel": 0.93},
                                    },
                                ],
                                "phrase_candidates": [],
                                "subtree_candidates": [],
                            },
                            {
                                "source_paragraph_index": 2,
                                "target_paragraph_index": 2,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 4],
                                    "target_span": [1, 4],
                                    "source_text": "idzie stara droga dalej",
                                    "target_text": "jde starou cestou dál",
                                    "score": 0.95,
                                    "signals": {"embedding": 0.95, "lemma": 0.9, "upos": 0.95, "deprel": 0.9},
                                },
                                "source_tokens": [
                                    {"index": 1, "text": "idzie", "kind": "word", "lemma": "iść", "upos": "VERB"},
                                    {"index": 2, "text": "stara", "kind": "word", "lemma": "stary", "upos": "ADJ"},
                                    {"index": 3, "text": "droga", "kind": "word", "lemma": "droga", "upos": "NOUN"},
                                    {"index": 4, "text": "dalej", "kind": "word", "lemma": "dalej", "upos": "ADV"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "text": "jde", "kind": "word", "lemma": "jít", "upos": "VERB"},
                                    {"index": 2, "text": "starou", "kind": "word", "lemma": "starý", "upos": "ADJ"},
                                    {"index": 3, "text": "cestou", "kind": "word", "lemma": "cesta", "upos": "NOUN"},
                                    {"index": 4, "text": "dál", "kind": "word", "lemma": "dál", "upos": "ADV"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": 1,
                                        "target_token_index": 1,
                                        "source_text": "idzie",
                                        "target_text": "jde",
                                        "source_lemma": "iść",
                                        "target_lemma": "jít",
                                        "source_upos": "VERB",
                                        "target_upos": "VERB",
                                        "relation": "lemma_like",
                                        "score": 0.95,
                                        "signals": {"embedding": 0.95, "lemma": 0.95, "upos": 1.0, "deprel": 0.95},
                                    },
                                    {
                                        "source_token_index": 2,
                                        "target_token_index": 2,
                                        "source_text": "stara",
                                        "target_text": "starou",
                                        "source_lemma": "stary",
                                        "target_lemma": "starý",
                                        "source_upos": "ADJ",
                                        "target_upos": "ADJ",
                                        "relation": "lemma_like",
                                        "score": 0.94,
                                        "signals": {"embedding": 0.94, "lemma": 0.94, "upos": 1.0, "deprel": 0.94},
                                    },
                                    {
                                        "source_token_index": 3,
                                        "target_token_index": 3,
                                        "source_text": "droga",
                                        "target_text": "cestou",
                                        "source_lemma": "droga",
                                        "target_lemma": "cesta",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "relation": "lemma_like",
                                        "score": 0.93,
                                        "signals": {"embedding": 0.93, "lemma": 0.93, "upos": 1.0, "deprel": 0.93},
                                    },
                                ],
                                "phrase_candidates": [
                                    {
                                        "source_span": [1, 3],
                                        "target_span": [1, 3],
                                        "score": 0.94,
                                        "relation": "phrase_like",
                                        "signals": {"embedding": 0.94, "lemma": 0.94, "upos": 1.0, "deprel": 0.94},
                                    }
                                ],
                                "subtree_candidates": [],
                            },
                        ]
                    },
                }
            ],
        }

        payload = build_paragraph_hybridization_plan(
            artifact,
            granularity_policy="cumulative-simple",
            target_power=1.4,
        )
        second_selected = payload["units"][1]["selected_candidates"]
        self.assertTrue(
            any(
                sel["granularity"] in {"sentence", "paragraph"} and sel["target_text"] == "jde starou cestou dál"
                for sel in second_selected
            )
        )
        self.assertFalse(
            any(sel["granularity"] == "phrase" and sel["target_text"] == "jde starou cestou" for sel in second_selected)
        )

    def test_cumulative_simple_does_not_promote_sentence_below_coverage_threshold(self) -> None:
        artifact = {
            "chapter_pair_count": 1,
            "pair_reports": [
                {
                    "source_chapter_index": 1,
                    "target_chapter_index": 1,
                    "paragraph_alignment": {
                        "matched_blocks": [
                            {
                                "source_range": [1, 1],
                                "target_range": [1, 1],
                                "source_preview": "stary dom",
                                "target_preview": "starý dům",
                            },
                            {
                                "source_range": [2, 2],
                                "target_range": [2, 2],
                                "source_preview": "stary dom stoi tutaj dziś",
                                "target_preview": "starý dům stojí tady dnes",
                            },
                        ]
                    },
                    "enrichment_bundle": {
                        "items": [
                            {
                                "source_paragraph_index": 1,
                                "target_paragraph_index": 1,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 2],
                                    "target_span": [1, 2],
                                    "source_text": "stary dom",
                                    "target_text": "starý dům",
                                    "score": 0.95,
                                    "signals": {"embedding": 0.95, "lemma": 0.95, "upos": 1.0, "deprel": 0.95},
                                },
                                "source_tokens": [
                                    {"index": 1, "text": "stary", "kind": "word", "lemma": "stary", "upos": "ADJ"},
                                    {"index": 2, "text": "dom", "kind": "word", "lemma": "dom", "upos": "NOUN"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "text": "starý", "kind": "word", "lemma": "starý", "upos": "ADJ"},
                                    {"index": 2, "text": "dům", "kind": "word", "lemma": "dům", "upos": "NOUN"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": 1,
                                        "target_token_index": 1,
                                        "source_text": "stary",
                                        "target_text": "starý",
                                        "source_lemma": "stary",
                                        "target_lemma": "starý",
                                        "source_upos": "ADJ",
                                        "target_upos": "ADJ",
                                        "relation": "lemma_like",
                                        "score": 0.94,
                                        "signals": {"norm": 0.9, "total": 0.94},
                                    },
                                    {
                                        "source_token_index": 2,
                                        "target_token_index": 2,
                                        "source_text": "dom",
                                        "target_text": "dům",
                                        "source_lemma": "dom",
                                        "target_lemma": "dům",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "relation": "lemma_like",
                                        "score": 0.96,
                                        "signals": {"norm": 0.92, "total": 0.96},
                                    },
                                ],
                                "phrase_candidates": [],
                                "subtree_candidates": [],
                            },
                            {
                                "source_paragraph_index": 2,
                                "target_paragraph_index": 2,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 9],
                                    "target_span": [1, 9],
                                    "source_text": "stary dom stoi tutaj dziś całkiem sobie sam",
                                    "target_text": "starý dům stojí tady dnes docela sobě sám",
                                    "score": 0.58,
                                    "signals": {"embedding": 0.58, "lemma": 0.58, "upos": 0.9, "deprel": 0.7},
                                },
                                "source_tokens": [
                                    {"index": 1, "text": "stary", "kind": "word", "lemma": "stary", "upos": "ADJ"},
                                    {"index": 2, "text": "dom", "kind": "word", "lemma": "dom", "upos": "NOUN"},
                                    {"index": 3, "text": "stoi", "kind": "word", "lemma": "stać", "upos": "VERB"},
                                    {"index": 4, "text": "tutaj", "kind": "word", "lemma": "tutaj", "upos": "ADV"},
                                    {"index": 5, "text": "dziś", "kind": "word", "lemma": "dziś", "upos": "ADV"},
                                    {"index": 6, "text": "całkiem", "kind": "word", "lemma": "całkiem", "upos": "ADV"},
                                    {"index": 7, "text": "sobie", "kind": "word", "lemma": "sobie", "upos": "PRON"},
                                    {"index": 8, "text": "sam", "kind": "word", "lemma": "sam", "upos": "ADJ"},
                                    {"index": 9, "text": ".", "kind": "punct", "upos": "PUNCT"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "text": "starý", "kind": "word", "lemma": "starý", "upos": "ADJ"},
                                    {"index": 2, "text": "dům", "kind": "word", "lemma": "dům", "upos": "NOUN"},
                                    {"index": 3, "text": "stojí", "kind": "word", "lemma": "stát", "upos": "VERB"},
                                    {"index": 4, "text": "tady", "kind": "word", "lemma": "tady", "upos": "ADV"},
                                    {"index": 5, "text": "dnes", "kind": "word", "lemma": "dnes", "upos": "ADV"},
                                    {"index": 6, "text": "docela", "kind": "word", "lemma": "docela", "upos": "ADV"},
                                    {"index": 7, "text": "sobě", "kind": "word", "lemma": "sebe", "upos": "PRON"},
                                    {"index": 8, "text": "sám", "kind": "word", "lemma": "sám", "upos": "ADJ"},
                                    {"index": 9, "text": ".", "kind": "punct", "upos": "PUNCT"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": 1,
                                        "target_token_index": 1,
                                        "source_text": "stary",
                                        "target_text": "starý",
                                        "source_lemma": "stary",
                                        "target_lemma": "starý",
                                        "source_upos": "ADJ",
                                        "target_upos": "ADJ",
                                        "relation": "lemma_like",
                                        "score": 0.95,
                                        "signals": {"norm": 0.9, "total": 0.95},
                                    },
                                    {
                                        "source_token_index": 2,
                                        "target_token_index": 2,
                                        "source_text": "dom",
                                        "target_text": "dům",
                                        "source_lemma": "dom",
                                        "target_lemma": "dům",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "relation": "lemma_like",
                                        "score": 0.96,
                                        "signals": {"norm": 0.92, "total": 0.96},
                                    },
                                    {
                                        "source_token_index": 3,
                                        "target_token_index": 3,
                                        "source_text": "stoi",
                                        "target_text": "stojí",
                                        "source_lemma": "stać",
                                        "target_lemma": "stát",
                                        "source_upos": "VERB",
                                        "target_upos": "VERB",
                                        "relation": "lemma_like",
                                        "score": 0.7,
                                        "signals": {"norm": 0.62, "total": 0.7},
                                    },
                                ],
                                "phrase_candidates": [
                                    {
                                        "source_span": [1, 8],
                                        "target_span": [1, 8],
                                        "score": 0.93,
                                        "signals": {"total": 0.93},
                                    },
                                ],
                                "subtree_candidates": [],
                            },
                        ]
                    },
                }
            ],
        }

        payload = build_paragraph_hybridization_plan(
            artifact,
            target_power=1.0,
            granularity_policy="cumulative-simple",
        )

        second_selected = payload["units"][1]["selected_candidates"]
        self.assertFalse(any(sel["granularity"] in {"sentence", "paragraph"} for sel in second_selected))

    def test_cumulative_simple_promotes_paragraph_over_sentence_when_covered(self) -> None:
        artifact = {
            "chapter_pair_count": 1,
            "pair_reports": [
                {
                    "source_chapter_index": 1,
                    "target_chapter_index": 1,
                    "paragraph_alignment": {
                        "matched_blocks": [
                            {
                                "source_range": [1, 1],
                                "target_range": [1, 1],
                                "source_preview": "stary dom stoi tutaj",
                                "target_preview": "starý dům stojí tady",
                            }
                        ]
                    },
                    "enrichment_bundle": {
                        "items": [
                            {
                                "source_paragraph_index": 1,
                                "target_paragraph_index": 1,
                                "enrichment_status": "sentence_safe",
                                "sentence_alignment": {
                                    "source_span": [1, 4],
                                    "target_span": [1, 4],
                                    "source_text": "stary dom stoi tutaj",
                                    "target_text": "starý dům stojí tady",
                                    "score": 0.58,
                                    "signals": {"embedding": 0.58, "lemma": 0.58, "upos": 0.9, "deprel": 0.7},
                                },
                                "source_tokens": [
                                    {"index": 1, "text": "stary", "kind": "word", "lemma": "stary", "upos": "ADJ"},
                                    {"index": 2, "text": "dom", "kind": "word", "lemma": "dom", "upos": "NOUN"},
                                    {"index": 3, "text": "stoi", "kind": "word", "lemma": "stać", "upos": "VERB"},
                                    {"index": 4, "text": "tutaj", "kind": "word", "lemma": "tutaj", "upos": "ADV"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "text": "starý", "kind": "word", "lemma": "starý", "upos": "ADJ"},
                                    {"index": 2, "text": "dům", "kind": "word", "lemma": "dům", "upos": "NOUN"},
                                    {"index": 3, "text": "stojí", "kind": "word", "lemma": "stát", "upos": "VERB"},
                                    {"index": 4, "text": "tady", "kind": "word", "lemma": "tady", "upos": "ADV"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": 1,
                                        "target_token_index": 1,
                                        "source_text": "stary",
                                        "target_text": "starý",
                                        "source_lemma": "stary",
                                        "target_lemma": "starý",
                                        "source_upos": "ADJ",
                                        "target_upos": "ADJ",
                                        "relation": "lemma_like",
                                        "score": 0.95,
                                        "signals": {"norm": 0.9, "total": 0.95},
                                    },
                                    {
                                        "source_token_index": 2,
                                        "target_token_index": 2,
                                        "source_text": "dom",
                                        "target_text": "dům",
                                        "source_lemma": "dom",
                                        "target_lemma": "dům",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "relation": "lemma_like",
                                        "score": 0.96,
                                        "signals": {"norm": 0.92, "total": 0.96},
                                    },
                                    {
                                        "source_token_index": 3,
                                        "target_token_index": 3,
                                        "source_text": "stoi",
                                        "target_text": "stojí",
                                        "source_lemma": "stać",
                                        "target_lemma": "stát",
                                        "source_upos": "VERB",
                                        "target_upos": "VERB",
                                        "relation": "lemma_like",
                                        "score": 0.7,
                                        "signals": {"norm": 0.62, "total": 0.7},
                                    },
                                    {
                                        "source_token_index": 4,
                                        "target_token_index": 4,
                                        "source_text": "tutaj",
                                        "target_text": "tady",
                                        "source_lemma": "tutaj",
                                        "target_lemma": "tady",
                                        "source_upos": "ADV",
                                        "target_upos": "ADV",
                                        "relation": "lemma_like",
                                        "score": 0.69,
                                        "signals": {"norm": 0.6, "total": 0.69},
                                    },
                                ],
                                "phrase_candidates": [
                                    {
                                        "source_span": [1, 2],
                                        "target_span": [1, 2],
                                        "score": 0.93,
                                        "signals": {"total": 0.93},
                                    }
                                ],
                                "subtree_candidates": [],
                            }
                        ]
                    },
                }
            ],
        }

        payload = build_paragraph_hybridization_plan(
            artifact,
            target_power=1.0,
            granularity_policy="cumulative-simple",
        )

        selected = payload["units"][0]["selected_candidates"]
        self.assertTrue(any(sel["granularity"] == "paragraph" for sel in selected))
        self.assertFalse(any(sel["granularity"] == "sentence" for sel in selected))
