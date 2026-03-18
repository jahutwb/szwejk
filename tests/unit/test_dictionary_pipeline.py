from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from szwejk.generate.debug_subset import DebugSubstitution
from szwejk.generate.dictionary_pipeline import (
    build_dictionary_entries,
    extract_used_entries,
)


class DictionaryPipelineTest(unittest.TestCase):
    def test_extract_used_entries_deduplicates_and_prefers_cleaner_note_anchor(self) -> None:
        used = [
            DebugSubstitution(
                unit_index=1,
                chapter_pair=(1, 1),
                paragraph_index=1,
                candidate_id="c1",
                source_text="i Hercegowinę",
                target_text="Bosně a Hercegovině",
                granularity="phrase",
                note_id="n1",
                is_new_lemma=True,
            ),
            DebugSubstitution(
                unit_index=1,
                chapter_pair=(1, 1),
                paragraph_index=1,
                candidate_id="c1-dup",
                source_text="i Hercegowinę",
                target_text="Bosně a Hercegovině",
                granularity="phrase",
                note_id="n1",
                is_new_lemma=True,
            ),
        ]
        notes = {
            "notes": [
                {
                    "note_id": "n1",
                    "candidate_id": "c1",
                    "target_anchor_text": "Hercegovině",
                    "source_anchor_text": "Hercegowinę",
                }
            ]
        }

        entries = extract_used_entries(used, notes_payload=notes)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].czech, "Hercegovině")
        self.assertEqual(entries[0].polish_fallback, "i Hercegowinę")

    def test_build_dictionary_entries_prefers_note_translation(self) -> None:
        used = [
            DebugSubstitution(
                unit_index=2,
                chapter_pair=(1, 1),
                paragraph_index=1,
                candidate_id="c2",
                source_text="nieboszczyk",
                target_text="nebožtík",
                granularity="token",
                note_id="n2",
                is_new_lemma=True,
            )
        ]
        notes = {
            "notes": [
                {
                    "note_id": "n2",
                    "candidate_id": "c2",
                    "target_anchor_text": "nebožtík",
                    "source_anchor_text": "nieboszczyk",
                    "note_text": "nebožtík = nieboszczyk",
                    "note_type": "lexical",
                    "anchor_granularity": "token",
                    "alignment_score": 0.91,
                    "new_family_ids": ["nieboszczyk::nebožtík"],
                }
            ]
        }

        used_entries, final_entries = build_dictionary_entries(
            used_substitutions=used,
            notes_payload=notes,
        )

        self.assertEqual(len(used_entries), 1)
        self.assertEqual(len(final_entries), 1)
        self.assertEqual(final_entries[0].czech, "nebožtík")
        self.assertEqual(final_entries[0].polish, "nieboszczyk")
        self.assertEqual(final_entries[0].source, "note")

    def test_build_dictionary_entries_uses_cached_translation_before_alignment_fallback(self) -> None:
        used = [
            DebugSubstitution(
                unit_index=3,
                chapter_pair=(1, 1),
                paragraph_index=1,
                candidate_id="c3",
                source_text="odresuadresu",
                target_text="adresa",
                granularity="token",
                note_id=None,
                is_new_lemma=True,
            )
        ]
        notes = {"notes": []}

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "cache.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "adresa": {
                            "polish": "adres",
                            "source": "deepl",
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            _, final_entries = build_dictionary_entries(
                used_substitutions=used,
                notes_payload=notes,
                cache_path=cache_path,
            )

        self.assertEqual(len(final_entries), 1)
        self.assertEqual(final_entries[0].czech, "adresa")
        self.assertEqual(final_entries[0].polish, "adres")
        self.assertEqual(final_entries[0].source, "deepl")


if __name__ == "__main__":
    unittest.main()
