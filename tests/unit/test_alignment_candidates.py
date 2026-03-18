from __future__ import annotations

import unittest

from szwejk.align.candidates import (
    build_monotonic_chapter_alignment,
    _count_similarity,
    _normalized_index_similarity,
    _special_title_hint_similarity,
    build_alignment_candidates,
)
from szwejk.common.ids import make_book_id, make_chapter_id, make_paragraph_id, make_sentence_id
from szwejk.schemas import BookMetadata, CanonicalBook, CanonicalChapter, CanonicalParagraph, CanonicalSentence, SourceFragmentRef


def _make_book(language: str, title: str, chapter_specs: list[tuple[str, int]]) -> CanonicalBook:
    book_id = make_book_id(language, title)
    chapters = []
    for chapter_index, (chapter_title, paragraph_count) in enumerate(chapter_specs, start=1):
        chapter_id = make_chapter_id(book_id, chapter_index, chapter_title)
        paragraphs = []
        for paragraph_index in range(1, paragraph_count + 1):
            paragraph_id = make_paragraph_id(chapter_id, paragraph_index)
            sentence = CanonicalSentence(
                id=make_sentence_id(paragraph_id, 1),
                index=1,
                text=f"{chapter_title} paragraph {paragraph_index}.",
            )
            paragraphs.append(
                CanonicalParagraph(
                    id=paragraph_id,
                    index=paragraph_index,
                    text=sentence.text,
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


class AlignmentCandidateUnitTest(unittest.TestCase):
    def test_normalized_index_similarity_rewards_parallel_positions(self) -> None:
        self.assertGreater(_normalized_index_similarity(2, 10, 2, 12), _normalized_index_similarity(2, 10, 9, 12))

    def test_count_similarity_prefers_similar_sizes(self) -> None:
        self.assertGreater(_count_similarity(86, 84), _count_similarity(86, 12))

    def test_special_title_hint_recognizes_intro_pair(self) -> None:
        self.assertEqual(_special_title_hint_similarity("WSTĘP", "Úvod"), 1.0)
        self.assertEqual(_special_title_hint_similarity("Rozdział I", "Úvod"), 0.0)

    def test_build_alignment_candidates_prefers_structural_matches(self) -> None:
        pl_book = _make_book("pl", "PL", [("WSTĘP", 4), ("Rozdział I", 86), ("Rozdział II", 101)])
        cs_book = _make_book("cs", "CS", [("Úvod", 3), ("Zasáhnutí", 86), ("Dobrý voják", 101)])

        alignment = build_alignment_candidates(pl_book, cs_book, chapter_window=1, paragraph_window=2, top_k=2)

        first_chapter_top = alignment.chapter_candidates[pl_book.chapters[0].id][0]
        second_chapter_top = alignment.chapter_candidates[pl_book.chapters[1].id][0]

        self.assertEqual(first_chapter_top.target_chapter_id, cs_book.chapters[0].id)
        self.assertEqual(second_chapter_top.target_chapter_id, cs_book.chapters[1].id)
        self.assertTrue(alignment.paragraph_candidates[(pl_book.chapters[1].id, cs_book.chapters[1].id)])

    def test_build_monotonic_chapter_alignment_keeps_unique_monotonic_matches(self) -> None:
        pl_book = _make_book(
            "pl",
            "PL",
            [
                ("WSTĘP", 4),
                ("Rozdział I", 86),
                ("Rozdział II", 101),
                ("Dodatek", 2),
            ],
        )
        cs_book = _make_book(
            "cs",
            "CS",
            [
                ("Úvod", 3),
                ("Zasáhnutí", 86),
                ("Dobrý voják", 101),
            ],
        )

        alignment = build_monotonic_chapter_alignment(
            pl_book,
            cs_book,
            min_match_score=0.7,
            skip_penalty=0.2,
        )

        matched_pairs = [(item["source_chapter_index"], item["target_chapter_index"]) for item in alignment["matches"]]
        unmatched_source = [item["source_chapter_index"] for item in alignment["unmatched_source_chapters"]]
        unmatched_target = [item["target_chapter_index"] for item in alignment["unmatched_target_chapters"]]

        self.assertEqual(matched_pairs, [(1, 1), (2, 2), (3, 3)])
        self.assertEqual(unmatched_source, [4])
        self.assertEqual(unmatched_target, [])


if __name__ == "__main__":
    unittest.main()
