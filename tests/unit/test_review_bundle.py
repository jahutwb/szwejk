from __future__ import annotations

import unittest

from szwejk.align.diagnostics import augment_sentence_alignment
from szwejk.align.sentences import ChapterSentenceAlignment, SentenceAlignment
from szwejk.schemas import CanonicalChapter, CanonicalParagraph, CanonicalSentence, SourceFragmentRef
from szwejk.review_bundle import build_review_bundle, determine_safe_alignment_depth


class ReviewBundleUnitTest(unittest.TestCase):
    def test_determine_safe_alignment_depth_prefers_sentence_for_high_confidence(self) -> None:
        self.assertEqual(determine_safe_alignment_depth("high"), "sentence")
        self.assertEqual(determine_safe_alignment_depth("medium"), "sentence")
        self.assertEqual(determine_safe_alignment_depth("low"), "paragraph")

    def test_build_review_bundle_surfaces_low_confidence_items(self) -> None:
        source_chapter = CanonicalChapter(
            id="pl-ch1",
            index=1,
            title="Rozdział I",
            paragraphs=[
                CanonicalParagraph(
                    id="pl-p1",
                    index=1,
                    text="foo source paragraph",
                    sentences=[CanonicalSentence(id="pl-s1", index=1, text="foo")],
                    source_fragments=[SourceFragmentRef(source_path="OPS/pl.xhtml", element_tag="p", ordinal=1)],
                ),
                CanonicalParagraph(
                    id="pl-p2",
                    index=2,
                    text="alpha source paragraph",
                    sentences=[CanonicalSentence(id="pl-s2", index=1, text="alpha")],
                    source_fragments=[SourceFragmentRef(source_path="OPS/pl.xhtml", element_tag="p", ordinal=2)],
                ),
            ],
        )
        target_chapter = CanonicalChapter(
            id="cs-ch1",
            index=1,
            title="Zasáhnutí",
            paragraphs=[
                CanonicalParagraph(
                    id="cs-p1",
                    index=1,
                    text="bar target paragraph",
                    sentences=[CanonicalSentence(id="cs-s1", index=1, text="bar")],
                    source_fragments=[SourceFragmentRef(source_path="OPS/cs.xhtml", element_tag="p", ordinal=1)],
                ),
                CanonicalParagraph(
                    id="cs-p2",
                    index=2,
                    text="alfa target paragraph",
                    sentences=[CanonicalSentence(id="cs-s2", index=1, text="alfa")],
                    source_fragments=[SourceFragmentRef(source_path="OPS/cs.xhtml", element_tag="p", ordinal=2)],
                ),
            ],
        )

        low = ChapterSentenceAlignment(
            source_paragraph_id="pl-p1",
            target_paragraph_id="cs-p1",
            source_paragraph_index=1,
            target_paragraph_index=1,
            sentence_alignment=SentenceAlignment(
                source_index=1,
                target_index=1,
                source_text="foo",
                target_text="bar",
                score=0.3,
                signals={"length": 0.5, "char_trigram": 0.1, "punctuation": 0.0, "digit_pattern": 1.0, "total": 0.3},
            ),
        )
        high = ChapterSentenceAlignment(
            source_paragraph_id="pl-p2",
            target_paragraph_id="cs-p2",
            source_paragraph_index=2,
            target_paragraph_index=2,
            sentence_alignment=SentenceAlignment(
                source_index=1,
                target_index=1,
                source_text="alpha",
                target_text="alfa",
                score=0.7,
                signals={"length": 0.9, "char_trigram": 0.3, "punctuation": 1.0, "digit_pattern": 1.0, "total": 0.7},
            ),
        )

        bundle = build_review_bundle(
            source_chapter=source_chapter,
            target_chapter=target_chapter,
            source_title="Rozdział I",
            target_title="Zasáhnutí",
            chapter_alignments=[low, high],
        )

        self.assertEqual(bundle["summary"]["total_items"], 2)
        self.assertEqual(bundle["summary"]["adjudication_items"], 1)
        self.assertEqual(bundle["summary"]["safe_depth_counts"]["paragraph"], 1)
        self.assertEqual(bundle["summary"]["safe_depth_counts"]["sentence"], 1)
        self.assertEqual(bundle["items"][0]["source_chapter_id"], "pl-ch1")
        self.assertEqual(bundle["items"][0]["target_chapter_id"], "cs-ch1")
        self.assertEqual(bundle["items"][0]["source_paragraph_text"], "foo source paragraph")
        self.assertEqual(bundle["items"][0]["target_paragraph_preview"], "bar target paragraph")
        self.assertEqual(bundle["adjudication_items"][0]["safe_alignment_depth"], "paragraph")
        self.assertEqual(bundle["exception_queue"][0]["source_paragraph_preview"], "foo source paragraph")
        self.assertIn("rationale", bundle["adjudication_items"][0]["sentence_alignment"])


if __name__ == "__main__":
    unittest.main()
