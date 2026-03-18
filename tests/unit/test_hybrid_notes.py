from __future__ import annotations

import unittest

from szwejk.generate.notes import _render_note_text, build_didactic_note_bundle, build_paragraph_note_bundle
from szwejk.generate.paragraph_hybridization import ParagraphHybridCandidate


class HybridNotesUnitTest(unittest.TestCase):
    def test_render_note_text_explains_foreign_language_fragments(self) -> None:
        candidate = ParagraphHybridCandidate(
            candidate_id="c1",
            chapter_pair=(1, 1),
            unit_index=1,
            granularity="token",
            scope_id="sentence:1",
            source_span=(1, 1),
            target_span=(1, 1),
            source_text="Herr",
            target_text="Herr",
            family_ids=["herr::herr"],
            score=0.54,
            relation="exact",
            metadata={},
        )

        self.assertEqual(_render_note_text(candidate, "lexical"), 'Herr - niem. "pan".')

    def test_build_didactic_note_bundle_deduplicates_repeated_actions(self) -> None:
        document = {
            "source_chapter_index": 2,
            "target_chapter_index": 2,
            "level_id": "l2",
            "paragraphs": [
                {
                    "sentences": [
                        {
                            "sentence_index": 1,
                            "source_paragraph_id": "sp1",
                            "actions_applied": [
                                {
                                    "granularity": "phrase",
                                    "source_text": "pani Müllerowo",
                                    "target_text": "paní Müllerová",
                                }
                            ],
                        },
                        {
                            "sentence_index": 2,
                            "source_paragraph_id": "sp1",
                            "actions_applied": [
                                {
                                    "granularity": "phrase",
                                    "source_text": "pani Müllerowo",
                                    "target_text": "paní Müllerová",
                                }
                            ],
                        },
                    ]
                }
            ],
        }

        payload = build_didactic_note_bundle(document)

        self.assertEqual(payload["summary"]["note_count"], 1)
        self.assertEqual(payload["notes"][0]["kind"], "phrase")

    def test_build_paragraph_note_bundle_only_notes_selected_new_lemmas_and_prefers_phrase_anchor(self) -> None:
        alignment_artifact = {
            "pair_reports": [
                {
                    "source_chapter_index": 2,
                    "target_chapter_index": 2,
                    "sentence_rows": [],
                    "paragraph_alignment": {
                        "matched_blocks": [
                            {
                                "source_range": [1, 1],
                                "target_range": [1, 1],
                                "source_preview": "Ujął kupca za ramię.",
                                "target_preview": "Ujal obchodníka za rameno.",
                            }
                        ]
                    },
                    "enrichment_bundle": {
                        "items": [
                            {
                                "enrichment_status": "sentence_safe",
                                "source_paragraph_index": 1,
                                "target_paragraph_index": 1,
                                "sentence_alignment": {
                                    "source_span": [1, 4],
                                    "target_span": [1, 4],
                                    "source_text": "Ujął kupca za ramię",
                                    "target_text": "Ujal obchodníka za rameno",
                                    "score": 0.8,
                                    "signals": {},
                                },
                                "source_tokens": [
                                    {"index": 1, "kind": "word", "text": "Ujął"},
                                    {"index": 2, "kind": "word", "text": "kupca"},
                                    {"index": 3, "kind": "word", "text": "za"},
                                    {"index": 4, "kind": "word", "text": "ramię"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "kind": "word", "text": "Ujal"},
                                    {"index": 2, "kind": "word", "text": "obchodníka"},
                                    {"index": 3, "kind": "word", "text": "za"},
                                    {"index": 4, "kind": "word", "text": "rameno"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": 2,
                                        "target_token_index": 2,
                                        "source_text": "kupca",
                                        "target_text": "obchodníka",
                                        "source_lemma": "kupiec",
                                        "target_lemma": "obchodnik",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "score": 0.7,
                                        "relation": "lemma_like",
                                        "signals": {"norm": 0.3},
                                    },
                                    {
                                        "source_token_index": 4,
                                        "target_token_index": 4,
                                        "source_text": "ramię",
                                        "target_text": "rameno",
                                        "source_lemma": "ramię",
                                        "target_lemma": "rameno",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "score": 0.72,
                                        "relation": "lemma_like",
                                        "signals": {"norm": 0.4},
                                    },
                                ],
                                "subtree_candidates": [],
                                "phrase_candidates": [
                                    {
                                        "source_span": [3, 4],
                                        "target_span": [3, 4],
                                        "score": 0.85,
                                        "relation": "tree_phrase",
                                        "signals": {"embedding": 0.92, "lemma": 0.5},
                                    }
                                ],
                            }
                        ]
                    },
                }
            ]
        }
        plan_payload = {
            "blocked_standalone_upos": ["SCONJ", "CCONJ", "PART", "ADP", "PRON", "DET", "AUX"],
            "units": [
                {
                    "unit_index": 1,
                    "selected_candidates": [
                        {
                            "candidate_id": "u1:s1:token:2",
                            "chapter_pair": [2, 2],
                            "unit_index": 1,
                            "granularity": "token",
                            "scope_id": "sentence:1",
                            "source_span": [2, 2],
                            "target_span": [2, 2],
                            "source_text": "kupca",
                            "target_text": "obchodníka",
                            "family_ids": ["kupiec::obchodnik"],
                            "score": 0.7,
                            "relation": "lemma_like",
                            "metadata": {"signals": {"norm": 0.3}},
                        },
                        {
                            "candidate_id": "u1:s1:phrase:1",
                            "chapter_pair": [2, 2],
                            "unit_index": 1,
                            "granularity": "phrase",
                            "scope_id": "sentence:1",
                            "source_span": [3, 4],
                            "target_span": [3, 4],
                            "source_text": "za ramię",
                            "target_text": "za rameno",
                            "family_ids": ["ramię::rameno"],
                            "score": 0.85,
                            "relation": "tree_phrase",
                            "metadata": {"signals": {"embedding": 0.92, "lemma": 0.5}},
                        },
                        {
                            "candidate_id": "u1:s1:token:4",
                            "chapter_pair": [2, 2],
                            "unit_index": 1,
                            "granularity": "token",
                            "scope_id": "sentence:1",
                            "source_span": [4, 4],
                            "target_span": [4, 4],
                            "source_text": "ramię",
                            "target_text": "rameno",
                            "family_ids": ["ramię::rameno"],
                            "score": 0.72,
                            "relation": "lemma_like",
                            "metadata": {"signals": {"norm": 0.4}},
                        },
                    ],
                }
            ],
        }

        payload = build_paragraph_note_bundle(alignment_artifact, plan_payload)

        self.assertEqual(payload["summary"]["note_count"], 2)
        self.assertEqual(payload["notes"][0]["note_type"], "lexical")
        self.assertEqual(payload["notes"][0]["source_anchor_text"], "kupca")
        self.assertEqual(payload["notes"][1]["note_type"], "contextual")
        self.assertEqual(payload["notes"][1]["source_anchor_text"], "za ramię")
        self.assertEqual(payload["notes"][1]["target_anchor_text"], "za rameno")

    def test_build_paragraph_note_bundle_relaxed_suppresses_transparent_lexical_notes(self) -> None:
        alignment_artifact = {
            "pair_reports": [
                {
                    "source_chapter_index": 1,
                    "target_chapter_index": 1,
                    "sentence_rows": [],
                    "paragraph_alignment": {
                        "matched_blocks": [
                            {
                                "source_range": [1, 1],
                                "target_range": [1, 1],
                                "source_preview": "Park.",
                                "target_preview": "Park.",
                            }
                        ]
                    },
                    "enrichment_bundle": {
                        "items": [
                            {
                                "enrichment_status": "sentence_safe",
                                "source_paragraph_index": 1,
                                "target_paragraph_index": 1,
                                "sentence_alignment": {
                                    "source_span": [1, 1],
                                    "target_span": [1, 1],
                                    "source_text": "Park",
                                    "target_text": "Park",
                                    "score": 1.0,
                                    "signals": {},
                                },
                                "source_tokens": [{"index": 1, "kind": "word", "text": "park"}],
                                "target_tokens": [{"index": 1, "kind": "word", "text": "park"}],
                                "token_pairs": [
                                    {
                                        "source_token_index": 1,
                                        "target_token_index": 1,
                                        "source_text": "park",
                                        "target_text": "park",
                                        "source_lemma": "park",
                                        "target_lemma": "park",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "score": 1.0,
                                        "relation": "exact",
                                        "signals": {"norm": 1.0},
                                    }
                                ],
                                "subtree_candidates": [],
                                "phrase_candidates": [],
                            }
                        ]
                    },
                }
            ]
        }
        plan_payload = {
            "blocked_standalone_upos": [],
            "units": [
                {
                    "unit_index": 1,
                    "selected_candidates": [
                        {
                            "candidate_id": "u1:s1:token:1",
                            "chapter_pair": [1, 1],
                            "unit_index": 1,
                            "granularity": "token",
                            "scope_id": "sentence:1",
                            "source_span": [1, 1],
                            "target_span": [1, 1],
                            "source_text": "park",
                            "target_text": "park",
                            "family_ids": ["park::park"],
                            "score": 1.0,
                            "relation": "exact",
                            "metadata": {"signals": {"norm": 1.0}},
                        }
                    ],
                }
            ],
        }

        payload = build_paragraph_note_bundle(alignment_artifact, plan_payload, note_policy="relaxed")

        self.assertEqual(payload["summary"]["note_count"], 0)

    def test_build_paragraph_note_bundle_falls_back_to_token_anchor_for_broad_phrase(self) -> None:
        alignment_artifact = {
            "pair_reports": [
                {
                    "source_chapter_index": 2,
                    "target_chapter_index": 2,
                    "sentence_rows": [],
                    "paragraph_alignment": {
                        "matched_blocks": [
                            {
                                "source_range": [1, 1],
                                "target_range": [1, 1],
                                "source_preview": "albo nieboszczyk pan arcyksiążę",
                                "target_preview": "nebo nebožtík arcivévoda",
                            }
                        ]
                    },
                    "enrichment_bundle": {
                        "items": [
                            {
                                "enrichment_status": "sentence_safe",
                                "source_paragraph_index": 1,
                                "target_paragraph_index": 1,
                                "sentence_alignment": {
                                    "source_span": [1, 4],
                                    "target_span": [1, 3],
                                    "source_text": "albo nieboszczyk pan arcyksiążę",
                                    "target_text": "nebo nebožtík arcivévoda",
                                    "score": 0.7,
                                    "signals": {},
                                },
                                "source_tokens": [
                                    {"index": 1, "kind": "word", "text": "albo"},
                                    {"index": 2, "kind": "word", "text": "nieboszczyk"},
                                    {"index": 3, "kind": "word", "text": "pan"},
                                    {"index": 4, "kind": "word", "text": "arcyksiążę"},
                                ],
                                "target_tokens": [
                                    {"index": 1, "kind": "word", "text": "nebo"},
                                    {"index": 2, "kind": "word", "text": "nebožtík"},
                                    {"index": 3, "kind": "word", "text": "arcivévoda"},
                                ],
                                "token_pairs": [
                                    {
                                        "source_token_index": 2,
                                        "target_token_index": 2,
                                        "source_text": "nieboszczyk",
                                        "target_text": "nebožtík",
                                        "source_lemma": "nieboszczyk",
                                        "target_lemma": "nebožtík",
                                        "source_upos": "NOUN",
                                        "target_upos": "NOUN",
                                        "score": 0.8,
                                        "relation": "upos_cognate",
                                        "signals": {"norm": 0.4},
                                    }
                                ],
                                "subtree_candidates": [],
                                "phrase_candidates": [
                                    {
                                        "source_span": [1, 4],
                                        "target_span": [1, 3],
                                        "score": 0.68,
                                        "relation": "tree_phrase",
                                        "signals": {"embedding": 0.91, "lemma": 0.0},
                                    },
                                    {
                                        "source_span": [1, 2],
                                        "target_span": [1, 3],
                                        "score": 0.68,
                                        "relation": "tree_phrase",
                                        "signals": {"embedding": 0.91, "lemma": 0.0},
                                    },
                                ],
                            }
                        ]
                    },
                }
            ]
        }
        plan_payload = {
            "blocked_standalone_upos": [],
            "units": [
                {
                    "unit_index": 1,
                    "selected_candidates": [
                        {
                            "candidate_id": "u1:s1:phrase:2",
                            "chapter_pair": [2, 2],
                            "unit_index": 1,
                            "granularity": "phrase",
                            "scope_id": "sentence:1",
                            "source_span": [1, 2],
                            "target_span": [1, 3],
                            "source_text": "albo nieboszczyk",
                            "target_text": "nebo nebožtík arcivévoda",
                            "family_ids": ["nieboszczyk::nebožtík"],
                            "score": 0.68,
                            "relation": "tree_phrase",
                            "metadata": {"signals": {"embedding": 0.91, "lemma": 0.0}},
                        }
                    ],
                }
            ],
        }

        payload = build_paragraph_note_bundle(alignment_artifact, plan_payload)

        self.assertEqual(payload["summary"]["note_count"], 1)
        self.assertEqual(payload["notes"][0]["source_anchor_text"], "nieboszczyk")
        self.assertEqual(payload["notes"][0]["target_anchor_text"], "nebožtík")
