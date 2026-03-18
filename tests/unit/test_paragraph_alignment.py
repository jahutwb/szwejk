from __future__ import annotations

import unittest

from szwejk.align.paragraphs import (
    ParagraphAlignmentConfig,
    build_book_paragraph_alignment_report,
    build_chapter_paragraph_alignment,
)
from szwejk.align.embedding_eval import HeuristicEmbeddingEncoder
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


class ParagraphAlignmentUnitTest(unittest.TestCase):
    def test_chapter_paragraph_alignment_supports_many_to_one_blocks(self) -> None:
        source_book = _make_book(
            "pl",
            "PL",
            [("Rozdział I", ["Tak nam zabili.", "Ferdynanda.", "Potem cisza."])],
        )
        target_book = _make_book(
            "cs",
            "CS",
            [("Kapitola I", ["Tak nám zabili Ferdinanda.", "Potom ticho."])],
        )

        report = build_chapter_paragraph_alignment(
            source_book.chapters[0],
            target_book.chapters[0],
            encoder=HeuristicEmbeddingEncoder(),
            config=ParagraphAlignmentConfig(max_source_span=2, max_target_span=2, position_slack=0.4),
        )

        self.assertGreaterEqual(report["metrics"]["source_paragraph_coverage"], 1.0)
        self.assertGreaterEqual(report["metrics"]["target_paragraph_coverage"], 1.0)
        self.assertEqual(report["matched_blocks"][0]["source_span"], 2)
        self.assertEqual(report["matched_blocks"][0]["target_span"], 1)

    def test_book_paragraph_report_aggregates_global_coverage(self) -> None:
        pl_book = _make_book(
            "pl",
            "PL",
            [
                ("Rozdział I", ["A.", "B."]),
                ("Rozdział II", ["C."]),
            ],
        )
        cs_book = _make_book(
            "cs",
            "CS",
            [
                ("Kapitola I", ["A.", "B."]),
                ("Kapitola II", ["C."]),
            ],
        )

        report = build_book_paragraph_alignment_report(
            pl_book,
            cs_book,
            encoder=HeuristicEmbeddingEncoder(),
            model_name="heuristic",
            config=ParagraphAlignmentConfig(position_slack=0.4),
        )

        self.assertEqual(report["matched_pair_count"], 2)
        self.assertEqual(report["global_metrics"]["source_paragraph_coverage"], 1.0)
        self.assertEqual(report["global_metrics"]["target_paragraph_coverage"], 1.0)
        self.assertIn("pair_reports", report)


if __name__ == "__main__":
    unittest.main()
