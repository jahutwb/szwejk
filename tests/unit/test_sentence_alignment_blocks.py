from __future__ import annotations

import unittest

from szwejk.align.sentences import align_chapter_sentences_with_paragraph_blocks
from szwejk.common.ids import make_book_id, make_chapter_id, make_paragraph_id, make_sentence_id
from szwejk.schemas import BookMetadata, CanonicalBook, CanonicalChapter, CanonicalParagraph, CanonicalSentence, SourceFragmentRef


def _make_chapter(language: str, title: str, paragraph_texts: list[str]) -> CanonicalChapter:
    book_id = make_book_id(language, title)
    chapter_id = make_chapter_id(book_id, 1, title)
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
                        source_path=f"OPS/{title}.xhtml",
                        element_tag="p",
                        ordinal=paragraph_index - 1,
                    )
                ],
            )
        )
    return CanonicalChapter(
        id=chapter_id,
        index=1,
        title=title,
        paragraphs=paragraphs,
        source_path=f"OPS/{title}.xhtml",
    )


class SentenceAlignmentBlocksUnitTest(unittest.TestCase):
    def test_align_chapter_sentences_uses_paragraph_blocks_as_skeleton(self) -> None:
        source = _make_chapter("pl", "PL", ["Tak nam zabili.", "Ferdynanda.", "Potem cisza."])
        target = _make_chapter("cs", "CS", ["Tak nám zabili Ferdinanda.", "Potom ticho."])

        alignments = align_chapter_sentences_with_paragraph_blocks(
            source,
            target,
            paragraph_blocks=[
                {
                    "source_refs": [
                        {"paragraph_id": source.paragraphs[0].id},
                        {"paragraph_id": source.paragraphs[1].id},
                    ],
                    "target_refs": [
                        {"paragraph_id": target.paragraphs[0].id},
                    ],
                },
                {
                    "source_refs": [
                        {"paragraph_id": source.paragraphs[2].id},
                    ],
                    "target_refs": [
                        {"paragraph_id": target.paragraphs[1].id},
                    ],
                },
            ],
        )

        self.assertEqual(len(alignments), 3)
        self.assertEqual(alignments[0].source_paragraph_index, 1)
        self.assertEqual(alignments[0].target_paragraph_index, 1)
        self.assertEqual(alignments[1].source_paragraph_index, 2)
        self.assertEqual(alignments[1].target_paragraph_index, 1)
        self.assertEqual(alignments[2].source_paragraph_index, 3)
        self.assertEqual(alignments[2].target_paragraph_index, 2)


if __name__ == "__main__":
    unittest.main()
