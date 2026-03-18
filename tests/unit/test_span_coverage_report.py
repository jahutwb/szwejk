from __future__ import annotations

import unittest

from szwejk.align.coverage import build_book_chapter_span_report
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


class SpanCoverageReportUnitTest(unittest.TestCase):
    def test_book_span_report_uses_monotonic_chapter_pairs(self) -> None:
        pl_book = _make_book(
            "pl",
            "PL",
            [
                ("WSTĘP", ["To jest wstęp."] * 4),
                ("Rozdział I", ["A to nam zabili Ferdynanda."] * 86),
                ("Rozdział II", ["Pan Szwejk był w domu."] * 101),
                ("Dodatek", ["Krótki dodatek."] * 2),
            ],
        )
        cs_book = _make_book(
            "cs",
            "CS",
            [
                ("Úvod", ["Tohle je úvod."] * 3),
                ("Zasáhnutí", ["Tak nám zabili Ferdinanda."] * 86),
                ("Dobrý voják", ["Pan Švejk byl doma."] * 101),
            ],
        )

        report = build_book_chapter_span_report(
            pl_book,
            cs_book,
            analysis_mode="heuristic",
            min_match_score=0.7,
            skip_penalty=0.2,
        )

        self.assertEqual(report["matched_pair_count"], 3)
        self.assertEqual(
            [(item["source_chapter_index"], item["target_chapter_index"]) for item in report["chapter_alignment"]["matches"]],
            [(1, 1), (2, 2), (3, 3)],
        )
        self.assertEqual(
            [item["source_chapter_index"] for item in report["chapter_alignment"]["unmatched_source_chapters"]],
            [4],
        )
        self.assertIn("global_metrics", report)
        self.assertIn("token_row_rate", report["global_metrics"])


if __name__ == "__main__":
    unittest.main()
