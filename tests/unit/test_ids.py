from __future__ import annotations

import unittest

from szwejk.common.ids import (
    make_book_id,
    make_chapter_id,
    make_paragraph_id,
    make_sentence_id,
    slugify_fragment,
)


class IdHelpersTest(unittest.TestCase):
    def test_slugify_transliterates_and_compacts(self) -> None:
        self.assertEqual(slugify_fragment("Osudy dobrého vojáka Švejka"), "osudy-dobreho-vojaka-svejka")

    def test_book_and_nested_ids_are_stable(self) -> None:
        book_id = make_book_id("cs", "Osudy dobrého vojáka Švejka")
        chapter_id = make_chapter_id(book_id, 2, "Zasáhnutí dobrého vojáka Švejka do světové války")
        paragraph_id = make_paragraph_id(chapter_id, 7)
        sentence_id = make_sentence_id(paragraph_id, 3)

        self.assertEqual(book_id, "book-cs-osudy-dobreho-vojaka-svejka")
        self.assertTrue(chapter_id.endswith("ch-002-zasahnuti-dobreho-vojaka-svejka-do-svetove-valky"))
        self.assertEqual(paragraph_id, f"{chapter_id}-p-0007")
        self.assertEqual(sentence_id, f"{paragraph_id}-s-003")


if __name__ == "__main__":
    unittest.main()
