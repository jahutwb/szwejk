from __future__ import annotations

import unittest

from szwejk.align.embedding_eval import HeuristicEmbeddingEncoder
from szwejk.align.replacement_safety import build_replacement_safety_report
from szwejk.common.ids import make_book_id, make_chapter_id, make_paragraph_id, make_sentence_id
from szwejk.schemas import BookMetadata, CanonicalBook, CanonicalChapter, CanonicalParagraph, CanonicalSentence, SourceFragmentRef


def _make_book(language: str, title: str, chapter_specs: list[tuple[str, list[str]]]) -> CanonicalBook:
    book_id = make_book_id(language, title)
    chapters = []
    for chapter_index, (chapter_title, paragraph_texts) in enumerate(chapter_specs, start=1):
        chapter_id = make_chapter_id(book_id, chapter_index, chapter_title)
        paragraphs = []
        for paragraph_index, paragraph_text in enumerate(paragraph_texts, start=1):
            paragraph_id = make_paragraph_id(chapter_id, paragraph_index)
            sentence = CanonicalSentence(
                id=make_sentence_id(paragraph_id, 1),
                index=1,
                text=paragraph_text,
            )
            paragraphs.append(
                CanonicalParagraph(
                    id=paragraph_id,
                    index=paragraph_index,
                    text=paragraph_text,
                    sentences=[sentence],
                    source_fragments=[
                        SourceFragmentRef(
                            source_path=f"OPS/{chapter_title}.xhtml",
                            element_tag="p",
                            ordinal=paragraph_index - 1,
                        )
                    ],
                )
            )
        chapters.append(
            CanonicalChapter(
                id=chapter_id,
                index=chapter_index,
                title=chapter_title,
                paragraphs=paragraphs,
                source_path=f"OPS/{chapter_title}.xhtml",
            )
        )
    return CanonicalBook(
        metadata=BookMetadata(id=book_id, language=language, title=title, source_path=f"raw_texts/{language}.epub"),
        chapters=chapters,
    )


class ReplacementSafetyUnitTest(unittest.TestCase):
    def test_report_exposes_token_coverage_and_confidence_buckets(self) -> None:
        pl_book = _make_book("pl", "PL", [("Rozdział I", ["Tak nam zabili Ferdynanda.", "Potem cisza."])])
        cs_book = _make_book("cs", "CS", [("Kapitola I", ["Tak nám zabili Ferdinanda.", "Potom ticho."])])

        report = build_replacement_safety_report(
            pl_book,
            cs_book,
            chapter_pairs=[(1, 1)],
            paragraph_encoder=HeuristicEmbeddingEncoder(),
            sentence_encoder=HeuristicEmbeddingEncoder(),
            analysis_mode="stanza",
        )

        self.assertEqual(report["chapter_pair_count"], 1)
        self.assertIn("token_coverage", report["global_metrics"])
        self.assertIn("confidence_distribution", report["global_metrics"])
        self.assertIn("source_token_coverage", report["global_metrics"]["token_coverage"])
        self.assertIn("token", report["global_metrics"]["confidence_distribution"])

    def test_report_counts_embedding_assisted_fallbacks(self) -> None:
        class FakeEncoder:
            def encode(self, texts: list[str]) -> list[list[float]]:
                mapping = {
                    "query: alpha": [1.0, 0.0, 0.0],
                    "query: beta": [0.0, 1.0, 0.0],
                    "passage: uno": [1.0, 0.0, 0.0],
                    "passage: dos": [0.0, 1.0, 0.0],
                }
                return [mapping.get(text, [0.0, 0.0, 1.0]) for text in texts]

        pl_book = _make_book("pl", "PL", [("Rozdział I", ["alpha beta"])])
        cs_book = _make_book("cs", "CS", [("Kapitola I", ["uno dos"])])

        report = build_replacement_safety_report(
            pl_book,
            cs_book,
            chapter_pairs=[(1, 1)],
            paragraph_encoder=HeuristicEmbeddingEncoder(),
            sentence_encoder=FakeEncoder(),
            analysis_mode="heuristic",
        )

        self.assertGreater(report["global_metrics"]["confidence_distribution"]["token"]["embedding_assisted"], 0)
        self.assertGreater(report["global_metrics"]["confidence_distribution"]["phrase"]["embedding_assisted"], 0)
        self.assertGreater(report["global_metrics"]["confidence_distribution"]["subtree"]["embedding_assisted"], 0)


if __name__ == "__main__":
    unittest.main()
