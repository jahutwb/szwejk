from __future__ import annotations

import unittest

from szwejk.generate.diff import build_hybrid_diff_payload


class HybridDiffUnitTest(unittest.TestCase):
    def test_build_hybrid_diff_payload_marks_changed_sentence(self) -> None:
        document = {
            "source_chapter_index": 2,
            "target_chapter_index": 2,
            "level_id": "l2",
            "paragraphs": [
                {
                    "source_paragraph_id": "sp1",
                    "target_paragraph_id": "tp1",
                    "source_text": "A to nam zabili Ferdynanda.",
                    "hybrid_text": "A to nám zabili Ferdinanda.",
                    "target_text": "Tak nám zabili Ferdinanda.",
                    "sentences": [
                        {
                            "sentence_index": 1,
                            "source_text": "A to nam zabili Ferdynanda.",
                            "hybrid_text": "A to nám zabili Ferdinanda.",
                            "target_text": "Tak nám zabili Ferdinanda.",
                        }
                    ],
                }
            ],
        }

        payload = build_hybrid_diff_payload(document)

        self.assertEqual(payload["summary"]["changed_sentence_count"], 1)
        row = payload["paragraphs"][0]["sentences"][0]
        self.assertTrue(any(segment["op"] != "equal" for segment in row["pl_to_hybrid"]))
