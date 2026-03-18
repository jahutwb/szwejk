from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from szwejk.generate.full_book_pdf import build_full_book_latex, write_full_book_artifacts


TEMPLATE = r"""
\documentclass{memoir}
\newcommand{\cznote}[2]{CZNOTE(#1|#2)}
\newcommand{\cznew}[1]{CZNEW(#1)}
\newcommand{\czplain}[1]{CZPLAIN(#1)}
\begin{document}
%%BOOK_TITLE%%
%%BOOK_AUTHOR%%
%%BOOK_SUBTITLE%%
%%BOOK_CONTENT%%
\chapter*{Słownik}
%%GLOSSARY_CONTENT%%
%%META_CONTENT%%
\end{document}
"""


class FullBookPdfRendererTest(unittest.TestCase):
    def _corpus(self) -> dict[str, object]:
        return {
            "metadata": {"title": "Test Book"},
            "chapters": [
                {
                    "index": 1,
                    "title": "WSTĘP",
                    "paragraphs": [
                        {
                            "index": 1,
                            "kind": "paragraph",
                            "text": "Wstęp bez zmian.",
                            "sentences": [{"index": 1, "text": "Wstęp bez zmian."}],
                        }
                    ],
                },
                {
                    "index": 2,
                    "title": "Rozdział I",
                    "paragraphs": [
                        {
                            "index": 1,
                            "kind": "paragraph",
                            "text": "Ala ma kota. Ala lubi kota.",
                            "sentences": [
                                {"index": 1, "text": "Ala ma kota."},
                                {"index": 2, "text": "Ala lubi kota."},
                            ],
                        },
                        {
                            "index": 2,
                            "kind": "paragraph",
                            "text": "Pies śpi.",
                            "sentences": [{"index": 1, "text": "Pies śpi."}],
                        },
                    ],
                },
            ],
        }

    def _plan(self) -> dict[str, object]:
        return {
            "mode": "paragraph-plan",
            "target_power": 1.2,
            "blocked_standalone_upos": ["PRON"],
            "summary": {
                "introduced_family_count": 2,
                "selected_granularity_counts": {"phrase": 1, "token": 1},
            },
            "units": [
                {
                    "unit_index": 1,
                    "chapter_pair": [2, 2],
                    "source_paragraph_range": [1, 1],
                    "selected_candidates": [
                        {
                            "candidate_id": "c1",
                            "chapter_pair": [2, 2],
                            "unit_index": 1,
                            "granularity": "phrase",
                            "scope_id": "sentence:1",
                            "source_span": [1, 3],
                            "target_span": [1, 2],
                            "source_text": "Ala ma kota",
                            "target_text": "Alena ma kocku",
                            "family_ids": ["ala::alena"],
                            "score": 0.9,
                            "relation": "phrase",
                            "metadata": {"source_sentence_text": "Ala ma kota."},
                        }
                    ],
                },
                {
                    "unit_index": 2,
                    "chapter_pair": [2, 2],
                    "source_paragraph_range": [2, 2],
                    "selected_candidates": [
                        {
                            "candidate_id": "c2",
                            "chapter_pair": [2, 2],
                            "unit_index": 2,
                            "granularity": "token",
                            "scope_id": "sentence:1",
                            "source_span": [1, 1],
                            "target_span": [1, 1],
                            "source_text": "Pies",
                            "target_text": "Pes",
                            "family_ids": ["pies::pes"],
                            "score": 0.8,
                            "relation": "token",
                            "metadata": {"source_sentence_text": "Pies śpi."},
                        }
                    ],
                },
            ],
        }

    def _notes(self) -> dict[str, object]:
        return {
            "notes": [
                {
                    "note_id": "note-1",
                    "unit_index": 1,
                    "chapter_pair": [2, 2],
                    "candidate_id": "c1",
                    "source_anchor_text": "Ala ma kota",
                    "target_anchor_text": "Alena ma kocku",
                    "note_text": "Alena ma kocku = Ala ma kota",
                    "note_type": "contextual",
                    "new_family_ids": ["ala::alena", "kot::kocka"],
                },
                {
                    "note_id": "note-2",
                    "unit_index": 2,
                    "chapter_pair": [2, 2],
                    "candidate_id": "c2",
                    "source_anchor_text": "Pies",
                    "target_anchor_text": "Pes",
                    "note_text": "Pes = Pies",
                    "note_type": "lexical",
                    "anchor_granularity": "token",
                    "alignment_score": 0.8,
                    "new_family_ids": ["pies::pes"],
                },
                {
                    "note_id": "note-3",
                    "unit_index": 2,
                    "chapter_pair": [2, 2],
                    "candidate_id": "c-foreign",
                    "source_anchor_text": "Herr",
                    "target_anchor_text": "Herr",
                    "note_text": "Herr - niem. \"pan\".",
                    "note_type": "lexical",
                    "new_family_ids": ["herr::herr"],
                },
                {
                    "note_id": "note-4",
                    "unit_index": 2,
                    "chapter_pair": [2, 2],
                    "candidate_id": "c-noisy",
                    "source_anchor_text": "bardzo zwyczajny",
                    "target_anchor_text": "zcela obycejny",
                    "note_text": "zcela obycejny = bardzo zwyczajny",
                    "note_type": "contextual",
                    "anchor_granularity": "phrase",
                    "new_family_ids": ["bardzo::afera"],
                }
            ]
        }

    def _write_test_epub(self, directory: Path) -> Path:
        epub_path = directory / "test.epub"
        toc = """<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <navMap>
    <navPoint id="title"><navLabel><text>Strona tytułowa</text></navLabel></navPoint>
    <navPoint id="book"><navLabel><text>Przygody dobrego wojaka Szwejka. podczas wojny światowej</text></navLabel></navPoint>
    <navPoint id="part1">
      <navLabel><text>Tom Pierwszy</text></navLabel>
      <navPoint id="ch1"><navLabel><text>Wstęp</text></navLabel></navPoint>
    </navPoint>
    <navPoint id="part2">
      <navLabel><text>Tom Drugi</text></navLabel>
      <navPoint id="ch2"><navLabel><text>Rozdział I</text></navLabel></navPoint>
    </navPoint>
  </navMap>
</ncx>
"""
        with ZipFile(epub_path, "w") as zf:
            zf.writestr("OPS/toc.ncx", toc)
        return epub_path

    @patch("szwejk.generate.full_book_pdf.build_pdf_meta_section", return_value="")
    def test_build_full_book_latex_skips_conflicting_candidates(self, _meta_mock) -> None:
        plan = self._plan()
        plan["units"][0]["selected_candidates"].append(
            {
                "candidate_id": "c1-conflict",
                "chapter_pair": [2, 2],
                "unit_index": 1,
                "granularity": "phrase",
                "scope_id": "sentence:1",
                "source_span": [1, 2],
                "target_span": [1, 1],
                "source_text": "Ala ma",
                "target_text": "Konflikt",
                "family_ids": ["conflict::candidate"],
                "score": 0.5,
                "relation": "phrase",
                "metadata": {"source_sentence_text": "Ala ma kota."},
            }
        )

        payload = build_full_book_latex(
            corpus_payload=self._corpus(),
            plan_payload=plan,
            notes_payload=self._notes(),
            template_text=TEMPLATE,
        )

        latex = payload["latex"]
        self.assertIn(r"\cznote{Alena ma kocku}{Alena ma kocku = Ala ma kota}", latex)
        self.assertNotIn("Konflikt", latex)
        self.assertEqual(len(payload["used_substitutions"]), 2)

    @patch("szwejk.generate.full_book_pdf.build_pdf_meta_section", return_value="\\chapter*{Meta}\nMeta body")
    def test_build_full_book_latex_renders_body_glossary_and_meta(self, meta_mock) -> None:
        payload = build_full_book_latex(
            corpus_payload=self._corpus(),
            plan_payload=self._plan(),
            notes_payload=self._notes(),
            template_text=TEMPLATE,
            alignment_artifact={"pair_reports": []},
            root=Path("/tmp/project"),
            meta_output_dir=Path("/tmp/project/output/meta"),
            meta_label="book",
        )

        latex = payload["latex"]
        self.assertIn(r"\chapter*{WSTĘP}", latex)
        self.assertIn(r"\chapter{Rozdział I}", latex)
        self.assertIn(r"\cznote{Alena ma kocku}{Alena ma kocku = Ala ma kota}", latex)
        self.assertIn(r"\cznote{Pes}{Pes = Pies}", latex)
        self.assertIn(r"\textbf{\czplain{Pes}} --- Pies\par", latex)
        self.assertNotIn(r"\textbf{Alena ma kocku} --- Ala ma kota\par", latex)
        self.assertNotIn(r"\textbf{Herr}", latex)
        self.assertNotIn(r"\textbf{afera}", latex)
        self.assertIn(r"\chapter*{Meta}", latex)
        self.assertEqual(payload["paragraph_count"], 3)
        self.assertEqual(len(payload["used_notes"]), 2)
        self.assertEqual(len(payload["used_substitutions"]), 2)
        meta_mock.assert_called_once()

    @patch("szwejk.generate.full_book_pdf.build_pdf_meta_section", return_value="")
    def test_build_full_book_latex_uses_epub_toc_for_parts_and_titles(self, _meta_mock) -> None:
        corpus = self._corpus()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            epub_path = self._write_test_epub(tmp_path)
            corpus["metadata"]["source_path"] = str(epub_path.relative_to(tmp_path))
            payload = build_full_book_latex(
                corpus_payload=corpus,
                plan_payload=self._plan(),
                notes_payload=self._notes(),
                template_text=TEMPLATE,
                root=tmp_path,
            )

        latex = payload["latex"]
        self.assertIn(r"\part{Tom Pierwszy}", latex)
        self.assertIn(r"\part{Tom Drugi}", latex)
        self.assertIn(r"\chapter*{Wstęp}", latex)
        self.assertNotIn(r"\chapter*{WSTĘP}", latex)
        self.assertIn(r"\chapter{Rozdział I}", latex)

    @patch("szwejk.generate.full_book_pdf.build_pdf_meta_section", return_value="")
    def test_write_full_book_artifacts_writes_expected_files(self, _meta_mock) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            payload = write_full_book_artifacts(
                corpus_payload=self._corpus(),
                plan_payload=self._plan(),
                notes_payload=self._notes(),
                template_text=TEMPLATE,
                output_dir=output_dir,
                label="hybrid_book",
            )

            tex_path = output_dir / "hybrid_book.tex"
            substitutions_path = output_dir / "hybrid_book_substitutions.json"
            notes_path = output_dir / "hybrid_book_notes.json"
            summary_path = output_dir / "hybrid_book_summary.json"

            self.assertTrue(tex_path.exists())
            self.assertTrue(substitutions_path.exists())
            self.assertTrue(notes_path.exists())
            self.assertTrue(summary_path.exists())

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["paragraph_count"], 3)
            self.assertEqual(summary["note_count"], 2)
            self.assertEqual(payload["tex_path"], tex_path)

    @patch("szwejk.generate.full_book_pdf.build_pdf_meta_section", return_value="")
    def test_build_full_book_latex_notes_only_anchor_inside_larger_czech_span(self, _meta_mock) -> None:
        corpus = {
            "metadata": {"title": "Test Book"},
            "chapters": [
                {
                    "index": 1,
                    "title": "Rozdział I",
                    "paragraphs": [
                        {
                            "index": 1,
                            "kind": "paragraph",
                            "text": "A co? Miał może czekać, aż go rozstrzelają?",
                            "sentences": [{"index": 1, "text": "A co? Miał może czekać, aż go rozstrzelają?"}],
                        }
                    ],
                }
            ],
        }
        plan = {
            "units": [
                {
                    "unit_index": 1,
                    "chapter_pair": [1, 1],
                    "source_paragraph_range": [1, 1],
                    "selected_candidates": [
                        {
                            "candidate_id": "c1",
                            "chapter_pair": [1, 1],
                            "unit_index": 1,
                            "granularity": "phrase",
                            "scope_id": "sentence:1",
                            "source_span": [6, 8],
                            "target_span": [6, 8],
                            "source_text": "aż go rozstrzelają",
                            "target_text": "až ho zastřelejí",
                            "family_ids": ["rozstrzelac::zastrelit"],
                            "score": 0.9,
                            "relation": "phrase",
                            "metadata": {"source_sentence_text": "A co? Miał może czekać, aż go rozstrzelają?"},
                        }
                    ],
                }
            ]
        }
        notes = {
            "notes": [
                {
                    "note_id": "n1",
                    "unit_index": 1,
                    "chapter_pair": [1, 1],
                    "candidate_id": "c1",
                    "source_anchor_text": "rozstrzelają",
                    "target_anchor_text": "zastřelejí",
                    "note_text": "zastřelejí = rozstrzelają",
                    "note_type": "lexical",
                    "new_family_ids": ["rozstrzelac::zastrelit"],
                }
            ]
        }

        payload = build_full_book_latex(
            corpus_payload=corpus,
            plan_payload=plan,
            notes_payload=notes,
            template_text=TEMPLATE,
        )

        latex = payload["latex"]
        self.assertIn(r"\czplain{až ho }\cznote{zastřelejí}{zastřelejí = rozstrzelają}", latex)
        self.assertNotIn(r"\cznote{až ho zastřelejí}{zastřelejí = rozstrzelają}", latex)

    @patch("szwejk.generate.full_book_pdf.build_pdf_meta_section", return_value="")
    def test_dictionary_uses_only_used_entries_and_sorts_by_czech(self, _meta_mock) -> None:
        payload = build_full_book_latex(
            corpus_payload=self._corpus(),
            plan_payload=self._plan(),
            notes_payload=self._notes(),
            template_text=TEMPLATE,
        )

        entries = payload["dictionary_entries"]
        rendered = [(entry.czech, entry.polish) for entry in entries]
        self.assertIn(("Pes", "Pies"), rendered)
        self.assertNotIn(("Alena ma kocku", "Ala ma kota"), rendered)
        self.assertNotIn(("Herr", "Herr"), rendered)
        self.assertEqual(rendered, sorted(rendered, key=lambda item: item[0].lower()) or rendered)
        self.assertIn(r"\begin{multicols}{2}", payload["latex"])


if __name__ == "__main__":
    unittest.main()
