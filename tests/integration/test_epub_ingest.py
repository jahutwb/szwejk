from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from szwejk.align import (
    DEFAULT_ALIGNMENT_BENCHMARK_PAIRS,
    HeuristicEmbeddingEncoder,
    build_alignment_candidates,
    build_chapter_paragraph_alignment,
)
from szwejk.align.diagnostics import build_benchmark_report
from szwejk.align.embedding_eval import build_embedding_benchmark_report
from szwejk.align.enrichment import build_sentence_enrichment_bundle
from szwejk.align.coverage import build_sentence_span_coverage_report
from szwejk.align.quality import build_alignment_quality_report
from szwejk.align.sentences import align_chapter_sentences
from szwejk.review_bundle import build_review_bundle_for_chapter_pair
from szwejk.normalize import build_canonical_book
from szwejk.review import render_book_summary, render_chapter_preview


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PL_EPUB = PROJECT_ROOT / "raw_texts/pl/Przygody_dobrego_wojaka_Szwejka.epub"
CS_EPUB = PROJECT_ROOT / "raw_texts/cs/Osudy_dobrého_vojáka_Švejka_za_světové_války.epub"
PL_CORPUS = PROJECT_ROOT / "data/corpus/pl_szwejk.json"
CS_CORPUS = PROJECT_ROOT / "data/corpus/cs_szwejk.json"
DEFAULT_POLICY = PROJECT_ROOT / "data/policies/progressive_czechization_v1.json"


class EpubIngestIntegrationTest(unittest.TestCase):
    def test_build_canonical_book_from_polish_epub(self) -> None:
        book = build_canonical_book(str(PL_EPUB))

        self.assertEqual(book.metadata.language, "pl")
        self.assertGreaterEqual(book.chapter_count, 30)
        self.assertGreaterEqual(len(book.chapters[0].paragraphs), 3)
        self.assertTrue(book.chapters[0].paragraphs[0].sentences)
        self.assertEqual(book.chapters[0].title, "WSTĘP")
        self.assertGreater(max(len(chapter.paragraphs) for chapter in book.chapters), 50)
        self.assertTrue(book.chapters[0].paragraphs[0].source_fragments[0].source_path.startswith("OPS/"))

    def test_build_canonical_book_from_czech_epub(self) -> None:
        book = build_canonical_book(str(CS_EPUB))

        self.assertEqual(book.metadata.language, "cs")
        self.assertGreaterEqual(book.chapter_count, 25)
        self.assertGreaterEqual(len(book.chapters[0].paragraphs), 3)
        self.assertTrue(book.chapters[0].paragraphs[0].sentences)
        self.assertEqual(book.chapters[0].title, "Úvod")
        self.assertGreater(max(len(chapter.paragraphs) for chapter in book.chapters), 50)

    def test_cli_writes_output_for_both_books(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pl_output = Path(tmp_dir) / "pl.json"
            cs_output = Path(tmp_dir) / "cs.json"

            for epub_path, output_path in ((PL_EPUB, pl_output), (CS_EPUB, cs_output)):
                result = subprocess.run(
                    [
                        "python3",
                        "-m",
                        "szwejk.cli",
                        "ingest",
                        "--epub",
                        str(epub_path),
                        "--output",
                        str(output_path),
                    ],
                    cwd=PROJECT_ROOT,
                    env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertIn(str(output_path), result.stdout)
                payload = json.loads(output_path.read_text(encoding="utf-8"))
                self.assertIn("metadata", payload)
                self.assertIn("chapters", payload)
                self.assertGreater(len(payload["chapters"]), 20)

    def test_policy_cli_generates_plan_and_review_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            plan_output = Path(tmp_dir) / "plan.json"
            review_output = Path(tmp_dir) / "review.json"

            subprocess.run(
                [
                    "python3",
                    "-m",
                    "szwejk.cli",
                    "plan-policy-chapter",
                    "--policy",
                    str(DEFAULT_POLICY),
                    "--source",
                    str(PL_CORPUS),
                    "--target",
                    str(CS_CORPUS),
                    "--source-chapter",
                    "2",
                    "--target-chapter",
                    "2",
                    "--level",
                    "l1-lexical-onboarding",
                    "--analysis-mode",
                    "heuristic",
                    "--output",
                    str(plan_output),
                ],
                cwd=PROJECT_ROOT,
                env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                [
                    "python3",
                    "-m",
                    "szwejk.cli",
                    "review-policy-plan",
                    "--input",
                    str(plan_output),
                    "--output",
                    str(review_output),
                ],
                cwd=PROJECT_ROOT,
                env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
                capture_output=True,
                text=True,
                check=True,
            )

            plan_payload = json.loads(plan_output.read_text(encoding="utf-8"))
            review_payload = json.loads(review_output.read_text(encoding="utf-8"))
            self.assertEqual(plan_payload["level_id"], "l1-lexical-onboarding")
            self.assertIn("selected_candidates", plan_payload)
            self.assertIn("summary", review_payload)
            self.assertIn("recommendations", review_payload)

    def test_policy_cli_generates_benchmark_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            benchmark_output = Path(tmp_dir) / "benchmark.json"

            subprocess.run(
                [
                    "python3",
                    "-m",
                    "szwejk.cli",
                    "benchmark-policy",
                    "--policy",
                    str(DEFAULT_POLICY),
                    "--source",
                    str(PL_CORPUS),
                    "--target",
                    str(CS_CORPUS),
                    "--level",
                    "l1-lexical-onboarding",
                    "--pairs",
                    "1:1,2:2",
                    "--analysis-mode",
                    "heuristic",
                    "--output",
                    str(benchmark_output),
                ],
                cwd=PROJECT_ROOT,
                env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
                capture_output=True,
                text=True,
                check=True,
            )

            payload = json.loads(benchmark_output.read_text(encoding="utf-8"))
            self.assertEqual(payload["chapter_pair_count"], 2)
            self.assertIn("global_summary", payload)
            self.assertIn("under_target_pairs", payload)

    def test_hybrid_cli_generates_materialized_chapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            plan_output = Path(tmp_dir) / "plan.json"
            hybrid_output = Path(tmp_dir) / "hybrid.json"

            subprocess.run(
                [
                    "python3",
                    "-m",
                    "szwejk.cli",
                    "plan-policy-chapter",
                    "--policy",
                    str(DEFAULT_POLICY),
                    "--source",
                    str(PL_CORPUS),
                    "--target",
                    str(CS_CORPUS),
                    "--source-chapter",
                    "2",
                    "--target-chapter",
                    "2",
                    "--level",
                    "l2-phrase-onboarding",
                    "--analysis-mode",
                    "heuristic",
                    "--output",
                    str(plan_output),
                ],
                cwd=PROJECT_ROOT,
                env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                [
                    "python3",
                    "-m",
                    "szwejk.cli",
                    "generate-hybrid-chapter",
                    "--source",
                    str(PL_CORPUS),
                    "--target",
                    str(CS_CORPUS),
                    "--plan",
                    str(plan_output),
                    "--source-chapter",
                    "2",
                    "--target-chapter",
                    "2",
                    "--analysis-mode",
                    "heuristic",
                    "--output",
                    str(hybrid_output),
                ],
                cwd=PROJECT_ROOT,
                env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
                capture_output=True,
                text=True,
                check=True,
            )

            payload = json.loads(hybrid_output.read_text(encoding="utf-8"))
            self.assertIn("paragraphs", payload)
            self.assertGreater(len(payload["paragraphs"]), 20)
            self.assertIn("hybrid_text", payload["paragraphs"][0])
            self.assertIn("sentences", payload["paragraphs"][0])

    def test_hybrid_book_cli_generates_gradual_progression_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "book.json"

            subprocess.run(
                [
                    "python3",
                    "-m",
                    "szwejk.cli",
                    "generate-hybrid-book",
                    "--policy",
                    str(DEFAULT_POLICY),
                    "--source",
                    str(PL_CORPUS),
                    "--target",
                    str(CS_CORPUS),
                    "--pairs",
                    "1:1,2:2,10:10,15:15",
                    "--analysis-mode",
                    "heuristic",
                    "--output",
                    str(output_path),
                ],
                cwd=PROJECT_ROOT,
                env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
                capture_output=True,
                text=True,
                check=True,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["matched_pair_count"], 4)
            self.assertEqual(payload["progression_mode"], "global-linear-3d")
            self.assertEqual(payload["chapters"][0]["level_id"], "global-linear-progression")
            self.assertEqual(payload["chapters"][-1]["level_id"], "global-linear-progression")

    def test_review_renderers_expose_readable_preview(self) -> None:
        book = build_canonical_book(str(PL_EPUB))

        summary = render_book_summary(book)
        preview = render_chapter_preview(book, 2, max_paragraphs=2)

        self.assertIn("Chapter overview:", summary)
        self.assertIn("Rozdział I", preview)
        self.assertIn("[001]", preview)

    def test_benchmark_subset_regression_counts(self) -> None:
        pl_book = build_canonical_book(str(PL_EPUB))
        cs_book = build_canonical_book(str(CS_EPUB))

        self.assertEqual(pl_book.chapters[0].title, "WSTĘP")
        self.assertEqual(pl_book.chapters[0].source_path, "OPS/c2_Przygody_dobrego_wojaka_Szwejka_Tom_I_Wstep.xhtml")
        self.assertEqual(pl_book.chapters[0].paragraphs[0].sentences[0].index, 1)
        self.assertEqual(len(pl_book.chapters[1].paragraphs), 86)
        self.assertEqual(len(pl_book.chapters[2].paragraphs), 101)

        self.assertEqual(cs_book.chapters[0].title, "Úvod")
        self.assertEqual(cs_book.chapters[0].source_path, "OPS/c1_Osudy_dobreho_vojaka_Svejka_za_svetove_valky_Uvod.xhtml")
        self.assertEqual(cs_book.chapters[0].paragraphs[0].sentences[0].index, 1)
        self.assertEqual(len(cs_book.chapters[1].paragraphs), 86)
        self.assertEqual(len(cs_book.chapters[2].paragraphs), 101)
        self.assertEqual(pl_book.chapters[9].title, "Rozdział IX")
        self.assertEqual(cs_book.chapters[9].title, "Švejk na garnizóně")
        self.assertEqual(len(pl_book.chapters[9].paragraphs), 153)
        self.assertEqual(len(cs_book.chapters[9].paragraphs), 143)
        self.assertEqual(pl_book.chapters[14].title, "XIV")
        self.assertEqual(cs_book.chapters[14].title, "Švejk vojenským sluhou u nadporučíka Lukáše")
        self.assertEqual(len(pl_book.chapters[14].paragraphs), 364)
        self.assertEqual(len(cs_book.chapters[14].paragraphs), 323)

    def test_default_benchmark_pairs_cover_multiple_regions(self) -> None:
        self.assertEqual(DEFAULT_ALIGNMENT_BENCHMARK_PAIRS, [(1, 1), (2, 2), (10, 10), (15, 15)])

    def test_alignment_candidate_generation_on_benchmark_subset(self) -> None:
        pl_book = build_canonical_book(str(PL_EPUB))
        cs_book = build_canonical_book(str(CS_EPUB))

        alignment = build_alignment_candidates(pl_book, cs_book, chapter_window=1, paragraph_window=3, top_k=3)

        self.assertEqual(alignment.chapter_candidates[pl_book.chapters[0].id][0].target_chapter_id, cs_book.chapters[0].id)
        self.assertEqual(alignment.chapter_candidates[pl_book.chapters[1].id][0].target_chapter_id, cs_book.chapters[1].id)
        self.assertEqual(alignment.chapter_candidates[pl_book.chapters[2].id][0].target_chapter_id, cs_book.chapters[2].id)

        narrative_pair = (pl_book.chapters[1].id, cs_book.chapters[1].id)
        first_paragraph_candidates = alignment.paragraph_candidates[narrative_pair][pl_book.chapters[1].paragraphs[0].id]
        self.assertTrue(first_paragraph_candidates)
        self.assertLessEqual(abs(first_paragraph_candidates[0].target_paragraph_index - 1), 1)

    def test_sentence_alignment_on_benchmark_chapter_pair(self) -> None:
        pl_book = build_canonical_book(str(PL_EPUB))
        cs_book = build_canonical_book(str(CS_EPUB))

        chapter_alignments = align_chapter_sentences(pl_book.chapters[1], cs_book.chapters[1], paragraph_window=3)

        self.assertTrue(chapter_alignments)
        self.assertGreaterEqual(len(chapter_alignments), 50)
        self.assertEqual(chapter_alignments[0].source_paragraph_id, pl_book.chapters[1].paragraphs[0].id)
        self.assertEqual(chapter_alignments[0].target_paragraph_id, cs_book.chapters[1].paragraphs[0].id)
        self.assertGreater(chapter_alignments[0].sentence_alignment.score, 0.4)

    def test_benchmark_report_contains_confidence_and_difficult_cases(self) -> None:
        pl_book = build_canonical_book(str(PL_EPUB))
        cs_book = build_canonical_book(str(CS_EPUB))

        report = build_benchmark_report(
            pl_book,
            cs_book,
            chapter_pairs=DEFAULT_ALIGNMENT_BENCHMARK_PAIRS,
            paragraph_window=3,
        )

        self.assertEqual(report["chapter_pair_count"], 4)
        self.assertIn("pair_reports", report)
        self.assertIn("global_score_distribution", report)
        self.assertIn("global_confidence_counts", report)
        self.assertIn("difficult_cases", report["pair_reports"][0])
        self.assertIn("confidence_label", report["pair_reports"][0]["sample_alignments"][0])

    def test_review_bundle_contains_adjudication_and_exception_queue(self) -> None:
        pl_book = build_canonical_book(str(PL_EPUB))
        cs_book = build_canonical_book(str(CS_EPUB))

        bundle = build_review_bundle_for_chapter_pair(
            pl_book.chapters[1],
            cs_book.chapters[1],
            paragraph_window=3,
        )

        self.assertIn("summary", bundle)
        self.assertIn("adjudication_items", bundle)
        self.assertIn("exception_queue", bundle)
        self.assertGreater(bundle["summary"]["total_items"], 50)
        self.assertIn("safe_alignment_depth", bundle["items"][0])

    def test_sentence_enrichment_bundle_is_parent_bounded(self) -> None:
        pl_book = build_canonical_book(str(PL_EPUB))
        cs_book = build_canonical_book(str(CS_EPUB))
        chapter_alignments = align_chapter_sentences(pl_book.chapters[1], cs_book.chapters[1], paragraph_window=3)

        bundle = build_sentence_enrichment_bundle(
            chapter_alignments=chapter_alignments,
            source_language=pl_book.metadata.language,
            target_language=cs_book.metadata.language,
        )

        self.assertIn("summary", bundle)
        self.assertGreater(bundle["summary"]["sentence_safe_items"], 0)
        self.assertGreater(bundle["summary"]["blocked_items"], 0)
        first_safe = next(item for item in bundle["items"] if item["enrichment_status"] == "sentence_safe")
        first_blocked = next(item for item in bundle["items"] if item["enrichment_status"] == "blocked_by_safe_depth")
        self.assertTrue(first_safe["token_pairs"])
        self.assertEqual(first_blocked["safe_alignment_depth"], "paragraph")
        self.assertFalse(first_blocked["token_pairs"])

    def test_sentence_enrichment_bundle_with_stanza_exposes_dependency_candidates(self) -> None:
        pl_book = build_canonical_book(str(PL_EPUB))
        cs_book = build_canonical_book(str(CS_EPUB))
        chapter_alignments = align_chapter_sentences(pl_book.chapters[1], cs_book.chapters[1], paragraph_window=3)

        bundle = build_sentence_enrichment_bundle(
            chapter_alignments=chapter_alignments,
            source_language=pl_book.metadata.language,
            target_language=cs_book.metadata.language,
            analysis_mode="stanza",
        )

        self.assertEqual(bundle["items"][0]["analysis_mode"], "stanza")
        first_safe = next(item for item in bundle["items"] if item["enrichment_status"] == "sentence_safe" and item["dependency_candidates"])
        self.assertTrue(first_safe["source_analysis"]["tokens"][0]["analysis_source"] in {"stanza", "heuristic"})
        self.assertTrue(first_safe["dependency_candidates"])

    def test_alignment_quality_report_exposes_threshold_verdicts(self) -> None:
        pl_book = build_canonical_book(str(PL_EPUB))
        cs_book = build_canonical_book(str(CS_EPUB))

        report = build_alignment_quality_report(
            pl_book,
            cs_book,
            chapter_pairs=DEFAULT_ALIGNMENT_BENCHMARK_PAIRS,
            paragraph_window=3,
        )

        self.assertEqual(report["chapter_pair_count"], 4)
        self.assertIn("pair_reports", report)
        self.assertIn("global_granularity", report)
        self.assertIn("global_threshold_verdict", report)
        self.assertEqual(report["global_threshold_verdict"]["overall_status"], "pass")
        self.assertIn("threshold_verdict", report["pair_reports"][0])

    def test_embedding_benchmark_report_builds_on_curated_pairs(self) -> None:
        pl_book = build_canonical_book(str(PL_EPUB))
        cs_book = build_canonical_book(str(CS_EPUB))

        report = build_embedding_benchmark_report(
            pl_book,
            cs_book,
            encoder=HeuristicEmbeddingEncoder(),
            model_name="heuristic",
            chapter_pairs=DEFAULT_ALIGNMENT_BENCHMARK_PAIRS,
            paragraph_window=3,
            distractor_window=1,
        )

        self.assertEqual(report["chapter_pair_count"], 4)
        self.assertEqual(report["model_name"], "heuristic")
        self.assertIn("global_metrics", report)
        self.assertIn("pair_reports", report)
        self.assertIn("top1_accuracy", report["global_metrics"])

    def test_sentence_span_coverage_report_exposes_split_merge_metrics(self) -> None:
        pl_book = build_canonical_book(str(PL_EPUB))
        cs_book = build_canonical_book(str(CS_EPUB))

        report = build_sentence_span_coverage_report(
            pl_book,
            cs_book,
            chapter_pairs=[(2, 2)],
            paragraph_window=3,
            analysis_mode="stanza",
        )

        self.assertEqual(report["chapter_pair_count"], 1)
        self.assertEqual(report["analysis_mode"], "stanza")
        self.assertIn("global_metrics", report)
        self.assertIn("split_merge_row_rate", report["global_metrics"])
        self.assertIn("dependency_row_rate", report["global_metrics"])
        self.assertIn("subtree_row_rate", report["global_metrics"])

    def test_chapter_paragraph_alignment_exposes_paragraph_coverage(self) -> None:
        pl_book = build_canonical_book(str(PL_EPUB))
        cs_book = build_canonical_book(str(CS_EPUB))

        report = build_chapter_paragraph_alignment(
            pl_book.chapters[1],
            cs_book.chapters[1],
            encoder=HeuristicEmbeddingEncoder(),
        )

        self.assertIn("source_paragraph_coverage", report["metrics"])
        self.assertIn("target_paragraph_coverage", report["metrics"])
        self.assertGreaterEqual(report["source_total"], 80)


if __name__ == "__main__":
    unittest.main()
