from __future__ import annotations

import unittest

from szwejk.align.morphosyntax import analyze_sentence
from szwejk.align.subtree import align_subtrees, align_tree_units, extract_subtrees


class SubtreeAlignmentUnitTest(unittest.TestCase):
    def test_extract_subtrees_returns_root_and_major_branches(self) -> None:
        analysis = analyze_sentence("Pan Ferdynand siedzi spokojnie.", language="pl", mode="stanza")
        subtrees = extract_subtrees(analysis.tokens)

        self.assertTrue(subtrees)
        self.assertTrue(any(subtree.root_index == 3 for subtree in subtrees))

    def test_align_subtrees_finds_parallel_branches(self) -> None:
        source = analyze_sentence("Pan Ferdynand siedzi spokojnie.", language="pl", mode="stanza")
        target = analyze_sentence("Pan Ferdinand sedí pokojně.", language="cs", mode="stanza")

        alignments = align_subtrees(source.tokens, target.tokens)

        self.assertTrue(alignments)
        self.assertGreater(alignments[0]["score"], 0.35)

    def test_align_tree_units_derives_tokens_from_accepted_subtrees(self) -> None:
        source = analyze_sentence("Pan Ferdynand siedzi spokojnie.", language="pl", mode="stanza")
        target = analyze_sentence("Pan Ferdinand sedí pokojně.", language="cs", mode="stanza")

        alignment = align_tree_units(source.tokens, target.tokens)

        self.assertTrue(alignment["subtree_candidates"])
        self.assertTrue(alignment["token_pairs"])
        self.assertTrue(alignment["phrase_candidates"])
        self.assertTrue(
            all(
                any(
                    candidate["source_span"][0] <= pair["source_token_index"] <= candidate["source_span"][1]
                    and candidate["target_span"][0] <= pair["target_token_index"] <= candidate["target_span"][1]
                    for candidate in alignment["subtree_candidates"]
                )
                for pair in alignment["token_pairs"]
            )
        )


if __name__ == "__main__":
    unittest.main()
