from __future__ import annotations

import json
import unittest
from pathlib import Path

from szwejk.common.ids import make_book_id, make_chapter_id, make_paragraph_id, make_sentence_id
from szwejk.schemas import BookMetadata, CanonicalBook, CanonicalChapter, CanonicalParagraph, CanonicalSentence, SourceFragmentRef


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


class CorpusSchemaTest(unittest.TestCase):
    def test_manual_schema_construction(self) -> None:
        metadata = BookMetadata(
            id=make_book_id("pl", "Przygody dobrego wojaka Szwejka"),
            language="pl",
            title="Przygody dobrego wojaka Szwejka",
            source_path="raw_texts/pl/Przygody_dobrego_wojaka_Szwejka.epub",
        )
        chapter_id = make_chapter_id(metadata.id, 1, "Rozdzial I")
        paragraph_id = make_paragraph_id(chapter_id, 1)
        sentence_id = make_sentence_id(paragraph_id, 1)

        sentence = CanonicalSentence(id=sentence_id, index=1, text="A to nam zabili Ferdynanda.")
        paragraph = CanonicalParagraph(
            id=paragraph_id,
            index=1,
            text="A to nam zabili Ferdynanda.",
            sentences=[sentence],
            source_fragments=[
                SourceFragmentRef(
                    source_path="OPS/c3_Przygody_dobrego_wojaka_Szwejka_Tom_I_I.xhtml",
                    element_tag="p",
                    ordinal=0,
                )
            ],
        )
        chapter = CanonicalChapter(id=chapter_id, index=1, title="Rozdzial I", paragraphs=[paragraph])
        book = CanonicalBook(metadata=metadata, chapters=[chapter])

        self.assertEqual(book.chapter_count, 1)
        self.assertEqual(book.to_dict()["chapters"][0]["paragraphs"][0]["sentences"][0]["text"], "A to nam zabili Ferdynanda.")

    def test_sample_fixture_round_trip(self) -> None:
        payload = json.loads((FIXTURES_DIR / "sample_corpus.json").read_text(encoding="utf-8"))
        book = CanonicalBook.from_dict(payload)

        self.assertEqual(book.metadata.language, "pl")
        self.assertEqual(book.chapters[0].paragraphs[0].source_fragments[0].element_tag, "p")
        self.assertEqual(book.to_dict(), payload)

    def test_blank_sentence_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CanonicalSentence(id="x", index=0, text="   ")

    def test_negative_source_ordinal_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SourceFragmentRef(source_path="OPS/a.xhtml", element_tag="p", ordinal=-1)

    def test_real_epub_shape_fixtures_capture_markup_variation(self) -> None:
        pl_html = (FIXTURES_DIR / "pl_chapter_snippet.xhtml").read_text(encoding="utf-8")
        cs_html = (FIXTURES_DIR / "cs_chapter_snippet.xhtml").read_text(encoding="utf-8")

        self.assertIn('class="_tb"', pl_html)
        self.assertIn("<table", cs_html)
        self.assertIn('xml:lang="pl"', pl_html)
        self.assertIn('xml:lang="cs"', cs_html)


if __name__ == "__main__":
    unittest.main()
