"""CLI entrypoint for corpus generation utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from szwejk.align import (
    DEFAULT_ALIGNMENT_BENCHMARK_PAIRS,
    HeuristicEmbeddingEncoder,
    SentenceTransformersEncoder,
    align_chapter_sentences,
    augment_sentence_alignment,
    build_alignment_candidates,
    build_benchmark_report,
    build_book_paragraph_alignment_report,
    build_book_chapter_span_report,
    build_hierarchical_alignment_report,
    build_replacement_safety_report,
    build_embedding_benchmark_report,
    build_alignment_quality_report,
    build_sentence_span_coverage_report,
    build_monotonic_chapter_alignment,
    build_sentence_enrichment_bundle,
    format_chapter_pairs,
)
from szwejk.generate.hybrid import build_hybrid_chapter_document
from szwejk.generate.diff import build_hybrid_diff_payload
from szwejk.generate.full_book_pdf import compile_full_book_pdf, write_full_book_artifacts
from szwejk.generate.lemma_timeline import build_lemma_timeline_report
from szwejk.generate.lemma_schedule import build_family_schedule_report
from szwejk.generate.notes import build_didactic_note_bundle, build_paragraph_note_bundle
from szwejk.generate.book import build_hybrid_book_package
from szwejk.generate.czechness_report import build_czechness_report
from szwejk.generate.paragraph_hybridization import build_paragraph_hybridization_plan, load_alignment_artifact
from szwejk.generate.progression import build_progression_report
from szwejk.generate.pdf_meta import build_meta_pdf_latex
from szwejk.normalize import build_canonical_book, write_canonical_book
from szwejk.policy import build_action_inventory, build_chapter_policy_plan, build_policy_benchmark_report, build_policy_review_report, load_policy_from_json
from szwejk.review import load_book_from_json, render_book_summary, render_chapter_preview
from szwejk.review_bundle import build_review_bundle_for_chapter_pair


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="szwejk")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build a canonical corpus JSON file from an EPUB.")
    ingest_parser.add_argument("--epub", required=True, help="Path to source EPUB file.")
    ingest_parser.add_argument("--language", help="Override language code from EPUB metadata.")
    ingest_parser.add_argument(
        "--output",
        help="Output JSON path. Defaults to data/corpus/<stem>.json",
    )
    ingest_parser.add_argument("--stdout", action="store_true", help="Print the canonical corpus JSON to stdout.")

    inspect_parser = subparsers.add_parser("inspect", help="Render a readable summary or chapter preview from canonical JSON.")
    inspect_parser.add_argument("--input", required=True, help="Path to canonical corpus JSON.")
    inspect_parser.add_argument("--chapter", type=int, help="Specific 1-based chapter index to preview.")
    inspect_parser.add_argument("--max-paragraphs", type=int, default=5, help="Max paragraphs to render in chapter preview.")
    inspect_parser.add_argument("--output", help="Optional output text path.")

    align_parser = subparsers.add_parser("align-candidates", help="Generate structural alignment candidates between canonical corpora.")
    align_parser.add_argument("--source", required=True, help="Path to source canonical corpus JSON.")
    align_parser.add_argument("--target", required=True, help="Path to target canonical corpus JSON.")
    align_parser.add_argument("--output", required=True, help="Output JSON path.")
    align_parser.add_argument("--chapter-window", type=int, default=1, help="Window radius around expected chapter index.")
    align_parser.add_argument("--paragraph-window", type=int, default=3, help="Window radius around expected paragraph index.")
    align_parser.add_argument("--top-k", type=int, default=3, help="Number of candidates kept per source unit.")

    sentence_parser = subparsers.add_parser("align-sentences", help="Generate sentence alignment backbone for a chapter pair.")
    sentence_parser.add_argument("--source", required=True, help="Path to source canonical corpus JSON.")
    sentence_parser.add_argument("--target", required=True, help="Path to target canonical corpus JSON.")
    sentence_parser.add_argument("--source-chapter", type=int, required=True, help="1-based source chapter index.")
    sentence_parser.add_argument("--target-chapter", type=int, required=True, help="1-based target chapter index.")
    sentence_parser.add_argument("--paragraph-window", type=int, default=3, help="Window radius around expected paragraph index.")
    sentence_parser.add_argument("--output", required=True, help="Output JSON path.")

    report_parser = subparsers.add_parser("benchmark-report", help="Generate confidence and diagnostics report for benchmark chapter pairs.")
    report_parser.add_argument("--source", required=True, help="Path to source canonical corpus JSON.")
    report_parser.add_argument("--target", required=True, help="Path to target canonical corpus JSON.")
    report_parser.add_argument("--pairs", required=True, help="Comma-separated chapter pairs in source:target form, e.g. 1:1,2:2")
    report_parser.add_argument("--paragraph-window", type=int, default=3, help="Window radius around expected paragraph index.")
    report_parser.add_argument("--output", required=True, help="Output JSON path.")

    review_parser = subparsers.add_parser("review-bundle", help="Generate adjudication-ready review bundle for a chapter pair.")
    review_parser.add_argument("--source", required=True, help="Path to source canonical corpus JSON.")
    review_parser.add_argument("--target", required=True, help="Path to target canonical corpus JSON.")
    review_parser.add_argument("--source-chapter", type=int, required=True, help="1-based source chapter index.")
    review_parser.add_argument("--target-chapter", type=int, required=True, help="1-based target chapter index.")
    review_parser.add_argument("--paragraph-window", type=int, default=3, help="Window radius around expected paragraph index.")
    review_parser.add_argument("--output", required=True, help="Output JSON path.")

    enrich_parser = subparsers.add_parser("enrich-chapter", help="Generate smaller-unit enrichment inside sentence-safe aligned regions.")
    enrich_parser.add_argument("--source", required=True, help="Path to source canonical corpus JSON.")
    enrich_parser.add_argument("--target", required=True, help="Path to target canonical corpus JSON.")
    enrich_parser.add_argument("--source-chapter", type=int, required=True, help="1-based source chapter index.")
    enrich_parser.add_argument("--target-chapter", type=int, required=True, help="1-based target chapter index.")
    enrich_parser.add_argument("--paragraph-window", type=int, default=3, help="Window radius around expected paragraph index.")
    enrich_parser.add_argument("--analysis-mode", choices=["heuristic", "stanza"], default="heuristic", help="Token analysis mode.")
    enrich_parser.add_argument("--output", required=True, help="Output JSON path.")

    quality_parser = subparsers.add_parser("quality-report", help="Generate QA report for safe-depth and smaller-unit coverage.")
    quality_parser.add_argument("--source", required=True, help="Path to source canonical corpus JSON.")
    quality_parser.add_argument("--target", required=True, help="Path to target canonical corpus JSON.")
    quality_parser.add_argument("--pairs", required=True, help="Comma-separated chapter pairs in source:target form, e.g. 1:1,2:2")
    quality_parser.add_argument("--paragraph-window", type=int, default=3, help="Window radius around expected paragraph index.")
    quality_parser.add_argument("--output", required=True, help="Output JSON path.")

    span_report_parser = subparsers.add_parser("span-coverage-report", help="Generate sentence-span coverage report with split/merge and deeper alignment metrics.")
    span_report_parser.add_argument("--source", required=True, help="Path to source canonical corpus JSON.")
    span_report_parser.add_argument("--target", required=True, help="Path to target canonical corpus JSON.")
    span_report_parser.add_argument("--pairs", default=format_chapter_pairs(DEFAULT_ALIGNMENT_BENCHMARK_PAIRS), help="Comma-separated chapter pairs in source:target form.")
    span_report_parser.add_argument("--paragraph-window", type=int, default=3, help="Window radius around expected paragraph index.")
    span_report_parser.add_argument("--analysis-mode", choices=["heuristic", "stanza"], default="stanza", help="Sentence analysis mode for deeper alignment signals.")
    span_report_parser.add_argument("--output", required=True, help="Output JSON path.")

    book_span_parser = subparsers.add_parser("book-span-report", help="Generate chapter-by-chapter whole-book span coverage report with monotonic chapter pairing.")
    book_span_parser.add_argument("--source", required=True, help="Path to source canonical corpus JSON.")
    book_span_parser.add_argument("--target", required=True, help="Path to target canonical corpus JSON.")
    book_span_parser.add_argument("--paragraph-window", type=int, default=3, help="Window radius around expected paragraph index.")
    book_span_parser.add_argument("--analysis-mode", choices=["heuristic", "stanza"], default="stanza", help="Sentence analysis mode for deeper alignment signals.")
    book_span_parser.add_argument("--min-match-score", type=float, default=0.72, help="Minimum chapter score accepted by monotonic chapter pairing.")
    book_span_parser.add_argument("--skip-penalty", type=float, default=0.2, help="Penalty used when chapter pairing skips a source or target chapter.")
    book_span_parser.add_argument("--output", required=True, help="Output JSON path.")

    book_paragraph_parser = subparsers.add_parser("book-paragraph-report", help="Generate whole-book monotonic many-to-many paragraph alignment coverage report.")
    book_paragraph_parser.add_argument("--source", required=True, help="Path to source canonical corpus JSON.")
    book_paragraph_parser.add_argument("--target", required=True, help="Path to target canonical corpus JSON.")
    book_paragraph_parser.add_argument("--model", default="intfloat/multilingual-e5-base", help="Embedding model name, or 'heuristic' for the local baseline encoder.")
    book_paragraph_parser.add_argument("--device", default="cpu", help="Encoder device, e.g. cpu or cuda.")
    book_paragraph_parser.add_argument("--min-match-score", type=float, default=0.72, help="Minimum chapter score for auto chapter pairing.")
    book_paragraph_parser.add_argument("--skip-penalty", type=float, default=0.2, help="Penalty used when auto pairing skips chapters.")
    book_paragraph_parser.add_argument("--max-source-span", type=int, default=4, help="Maximum source paragraph span size.")
    book_paragraph_parser.add_argument("--max-target-span", type=int, default=4, help="Maximum target paragraph span size.")
    book_paragraph_parser.add_argument("--skip-source-penalty", type=float, default=0.55, help="Penalty used when paragraph alignment skips source paragraphs.")
    book_paragraph_parser.add_argument("--skip-target-penalty", type=float, default=0.65, help="Penalty used when paragraph alignment skips target paragraphs.")
    book_paragraph_parser.add_argument("--pair-bonus", type=float, default=0.04, help="Small bonus for choosing a matched paragraph block over skips.")
    book_paragraph_parser.add_argument("--position-slack", type=float, default=0.08, help="Normalized diagonal band for paragraph block candidates.")
    book_paragraph_parser.add_argument("--output", required=True, help="Output JSON path.")

    hierarchical_parser = subparsers.add_parser("hierarchical-alignment-report", help="Generate paragraph->sentence->subtree hierarchical alignment report.")
    hierarchical_parser.add_argument("--source", required=True, help="Path to source canonical corpus JSON.")
    hierarchical_parser.add_argument("--target", required=True, help="Path to target canonical corpus JSON.")
    hierarchical_parser.add_argument("--pairs", help="Optional comma-separated chapter pairs in source:target form.")
    hierarchical_parser.add_argument("--analysis-mode", choices=["heuristic", "stanza"], default="stanza", help="Sentence analysis mode for deeper alignment signals.")
    hierarchical_parser.add_argument("--paragraph-model", default="heuristic", help="Paragraph alignment model name, or 'heuristic' for the local baseline encoder.")
    hierarchical_parser.add_argument("--sentence-model", default="heuristic", help="Sentence/subtree embedding model name, or 'heuristic' for the local baseline encoder.")
    hierarchical_parser.add_argument("--device", default="cpu", help="Encoder device, e.g. cpu or cuda.")
    hierarchical_parser.add_argument("--output", required=True, help="Output JSON path.")

    safety_parser = subparsers.add_parser("replacement-safety-report", help="Generate token coverage and confidence/source distribution for replacement candidates.")
    safety_parser.add_argument("--source", required=True, help="Path to source canonical corpus JSON.")
    safety_parser.add_argument("--target", required=True, help="Path to target canonical corpus JSON.")
    safety_parser.add_argument("--pairs", help="Optional comma-separated chapter pairs in source:target form.")
    safety_parser.add_argument("--analysis-mode", choices=["heuristic", "stanza"], default="stanza", help="Sentence analysis mode for deeper alignment signals.")
    safety_parser.add_argument("--paragraph-model", default="heuristic", help="Paragraph alignment model name, or 'heuristic' for the local baseline encoder.")
    safety_parser.add_argument("--sentence-model", default="heuristic", help="Sentence/subtree embedding model name, or 'heuristic' for the local baseline encoder.")
    safety_parser.add_argument("--device", default="cpu", help="Encoder device, e.g. cpu or cuda.")
    safety_parser.add_argument("--output", required=True, help="Output JSON path.")

    embed_parser = subparsers.add_parser("embedding-benchmark", help="Compare embedding-based sentence ranking on the curated alignment benchmark.")
    embed_parser.add_argument("--source", required=True, help="Path to source canonical corpus JSON.")
    embed_parser.add_argument("--target", required=True, help="Path to target canonical corpus JSON.")
    embed_parser.add_argument("--model", default="heuristic", help="Embedding model name, or 'heuristic' for the local baseline encoder.")
    embed_parser.add_argument("--device", default="cpu", help="Encoder device, e.g. cpu or cuda.")
    embed_parser.add_argument("--pairs", default=format_chapter_pairs(DEFAULT_ALIGNMENT_BENCHMARK_PAIRS), help="Comma-separated chapter pairs in source:target form.")
    embed_parser.add_argument("--paragraph-window", type=int, default=3, help="Window radius around expected paragraph index.")
    embed_parser.add_argument("--distractor-window", type=int, default=2, help="How many nearby distractor sentences to include on each side.")
    embed_parser.add_argument("--output", required=True, help="Output JSON path.")

    policy_parser = subparsers.add_parser("inspect-policy", help="Validate and summarize a hybridization policy JSON file.")
    policy_parser.add_argument("--input", required=True, help="Path to policy JSON.")
    policy_parser.add_argument("--output", help="Optional output JSON path for normalized policy payload.")

    plan_policy_parser = subparsers.add_parser("plan-policy-chapter", help="Build chapter-level hybridization decisions for a policy level.")
    plan_policy_parser.add_argument("--policy", required=True, help="Path to policy JSON.")
    plan_policy_parser.add_argument("--source", required=True, help="Path to source canonical corpus JSON.")
    plan_policy_parser.add_argument("--target", required=True, help="Path to target canonical corpus JSON.")
    plan_policy_parser.add_argument("--source-chapter", type=int, required=True, help="1-based source chapter index.")
    plan_policy_parser.add_argument("--target-chapter", type=int, required=True, help="1-based target chapter index.")
    plan_policy_parser.add_argument("--level", required=True, help="Policy level id.")
    plan_policy_parser.add_argument("--paragraph-window", type=int, default=3, help="Window radius around expected paragraph index.")
    plan_policy_parser.add_argument("--analysis-mode", choices=["heuristic", "stanza"], default="stanza", help="Analysis mode for token enrichment.")
    plan_policy_parser.add_argument("--output", required=True, help="Output JSON path.")

    review_policy_parser = subparsers.add_parser("review-policy-plan", help="Summarize a chapter-level policy plan for tuning and review.")
    review_policy_parser.add_argument("--input", required=True, help="Path to chapter policy plan JSON.")
    review_policy_parser.add_argument("--output", required=True, help="Output JSON path.")

    benchmark_policy_parser = subparsers.add_parser("benchmark-policy", help="Run a policy level across multiple chapter pairs and aggregate tuning results.")
    benchmark_policy_parser.add_argument("--policy", required=True, help="Path to policy JSON.")
    benchmark_policy_parser.add_argument("--source", required=True, help="Path to source canonical corpus JSON.")
    benchmark_policy_parser.add_argument("--target", required=True, help="Path to target canonical corpus JSON.")
    benchmark_policy_parser.add_argument("--level", required=True, help="Policy level id.")
    benchmark_policy_parser.add_argument("--pairs", default=format_chapter_pairs(DEFAULT_ALIGNMENT_BENCHMARK_PAIRS), help="Comma-separated chapter pairs in source:target form.")
    benchmark_policy_parser.add_argument("--paragraph-window", type=int, default=3, help="Window radius around expected paragraph index.")
    benchmark_policy_parser.add_argument("--analysis-mode", choices=["heuristic", "stanza"], default="stanza", help="Analysis mode for token enrichment.")
    benchmark_policy_parser.add_argument("--output", required=True, help="Output JSON path.")

    action_inventory_parser = subparsers.add_parser("policy-action-inventory", help="Build token/phrase/subtree action inventory for a chapter pair.")
    action_inventory_parser.add_argument("--source", required=True, help="Path to source canonical corpus JSON.")
    action_inventory_parser.add_argument("--target", required=True, help="Path to target canonical corpus JSON.")
    action_inventory_parser.add_argument("--source-chapter", type=int, required=True, help="1-based source chapter index.")
    action_inventory_parser.add_argument("--target-chapter", type=int, required=True, help="1-based target chapter index.")
    action_inventory_parser.add_argument("--paragraph-window", type=int, default=3, help="Window radius around expected paragraph index.")
    action_inventory_parser.add_argument("--analysis-mode", choices=["heuristic", "stanza"], default="stanza", help="Analysis mode for token enrichment.")
    action_inventory_parser.add_argument("--output", required=True, help="Output JSON path.")

    hybrid_parser = subparsers.add_parser("generate-hybrid-chapter", help="Materialize a hybrid chapter document from a policy plan.")
    hybrid_parser.add_argument("--source", required=True, help="Path to source canonical corpus JSON.")
    hybrid_parser.add_argument("--target", required=True, help="Path to target canonical corpus JSON.")
    hybrid_parser.add_argument("--plan", required=True, help="Path to chapter policy plan JSON.")
    hybrid_parser.add_argument("--source-chapter", type=int, required=True, help="1-based source chapter index.")
    hybrid_parser.add_argument("--target-chapter", type=int, required=True, help="1-based target chapter index.")
    hybrid_parser.add_argument("--paragraph-window", type=int, default=3, help="Window radius around expected paragraph index.")
    hybrid_parser.add_argument("--analysis-mode", choices=["heuristic", "stanza"], default="stanza", help="Analysis mode for token enrichment.")
    hybrid_parser.add_argument("--output", required=True, help="Output JSON path.")

    diff_parser = subparsers.add_parser("generate-hybrid-diff", help="Build diff payload between PL, hybrid, and CS chapter document states.")
    diff_parser.add_argument("--input", required=True, help="Path to hybrid chapter JSON.")
    diff_parser.add_argument("--output", required=True, help="Output JSON path.")

    notes_parser = subparsers.add_parser("generate-hybrid-notes", help="Build didactic note bundle from a hybrid chapter document.")
    notes_parser.add_argument("--input", required=True, help="Path to hybrid chapter JSON.")
    notes_parser.add_argument("--output", required=True, help="Output JSON path.")

    paragraph_notes_parser = subparsers.add_parser(
        "generate-paragraph-notes",
        help="Build didactic note bundle from a paragraph hybridization plan and full alignment artifact.",
    )
    paragraph_notes_parser.add_argument("--alignment", required=True, help="Path to full hierarchical alignment artifact JSON.")
    paragraph_notes_parser.add_argument("--plan", required=True, help="Path to paragraph hybridization plan JSON.")
    paragraph_notes_parser.add_argument(
        "--note-policy",
        choices=["conservative", "relaxed"],
        default="conservative",
        help="Note filtering policy.",
    )
    paragraph_notes_parser.add_argument("--output", required=True, help="Output JSON path.")

    book_parser = subparsers.add_parser("generate-hybrid-book", help="Build a gradual whole-book hybrid package across matched chapter pairs.")
    book_parser.add_argument("--policy", required=True, help="Path to policy JSON.")
    book_parser.add_argument("--source", required=True, help="Path to source canonical corpus JSON.")
    book_parser.add_argument("--target", required=True, help="Path to target canonical corpus JSON.")
    book_parser.add_argument("--pairs", help="Optional comma-separated chapter pairs in source:target form.")
    book_parser.add_argument("--paragraph-window", type=int, default=3, help="Window radius around expected paragraph index.")
    book_parser.add_argument("--analysis-mode", choices=["heuristic", "stanza"], default="stanza", help="Analysis mode for token enrichment.")
    book_parser.add_argument("--paragraph-model", default="heuristic", help="Paragraph alignment model name, or 'heuristic' for the local baseline encoder.")
    book_parser.add_argument("--paragraph-device", default="cpu", help="Paragraph encoder device, e.g. cpu or cuda.")
    book_parser.add_argument("--min-match-score", type=float, default=0.72, help="Minimum chapter score for auto chapter pairing.")
    book_parser.add_argument("--skip-penalty", type=float, default=0.2, help="Penalty used when auto pairing skips chapters.")
    book_parser.add_argument("--output", required=True, help="Output JSON path.")

    progression_parser = subparsers.add_parser("progression-report", help="Build a continuous global progression report over aligned chapter pairs.")
    progression_parser.add_argument("--source", required=True, help="Path to source canonical corpus JSON.")
    progression_parser.add_argument("--target", required=True, help="Path to target canonical corpus JSON.")
    progression_parser.add_argument("--pairs", required=True, help="Comma-separated chapter pairs in source:target form.")
    progression_parser.add_argument("--paragraph-window", type=int, default=3, help="Window radius around expected paragraph index.")
    progression_parser.add_argument("--analysis-mode", choices=["heuristic", "stanza"], default="stanza", help="Analysis mode for token enrichment.")
    progression_parser.add_argument("--output", required=True, help="Output JSON path.")

    paragraph_hybrid_parser = subparsers.add_parser(
        "plan-paragraph-hybridization",
        help="Build paragraph-by-paragraph hybridization plan from a full alignment artifact.",
    )
    paragraph_hybrid_parser.add_argument("--alignment", required=True, help="Path to full hierarchical alignment artifact JSON.")
    paragraph_hybrid_parser.add_argument("--target-power", type=float, default=1.0, help="Power applied to linear progress when building target future czechness.")
    paragraph_hybrid_parser.add_argument(
        "--blocked-standalone-upos",
        default="SCONJ,CCONJ,PART,ADP",
        help="Comma-separated UPOS tags that cannot be selected as standalone token substitutions.",
    )
    paragraph_hybrid_parser.add_argument(
        "--idiomaticity-penalty-weight",
        type=float,
        default=0.0,
        help="Soft penalty for early phrase/subtree candidates with strong whole-span score but weak internal token support.",
    )
    paragraph_hybrid_parser.add_argument(
        "--granularity-policy",
        choices=["staircase", "smooth", "visible-smooth", "reader-visible-late", "cumulative-simple"],
        default="staircase",
        help="How progression influences candidate size selection: staircase thresholds, smooth penalties, visible-smooth steering on visible local czechness, cumulative-simple steering on cumulative simple czechness, or staircase with a late visible-czechness bonus.",
    )
    paragraph_hybrid_parser.add_argument(
        "--chapter-simple-target-power",
        type=float,
        help="Optional second-pass chapter simple-czechness boost power.",
    )
    paragraph_hybrid_parser.add_argument(
        "--chapter-simple-target-max",
        type=float,
        default=0.85,
        help="Maximum target simple czechness for the late chapter boost.",
    )
    paragraph_hybrid_parser.add_argument(
        "--chapter-simple-boost-start",
        type=float,
        default=0.55,
        help="Book progress after which chapter simple-czechness boosting starts.",
    )
    paragraph_hybrid_parser.add_argument(
        "--lemma-schedule",
        help="Optional lemma-schedule JSON used as a global introduction-order prior for candidate ranking.",
    )
    paragraph_hybrid_parser.add_argument(
        "--wiktionary",
        default="data/reference/wiktionary_cs_pl.json",
        help="CS→PL dictionary JSON for bidirectional token alignment quality scoring "
             "(default: data/reference/wiktionary_cs_pl.json).",
    )
    paragraph_hybrid_parser.add_argument("--output", required=True, help="Output JSON path.")

    czechness_parser = subparsers.add_parser(
        "czechness-report",
        help="Build local czechness report per paragraph block and per chapter from alignment artifact and paragraph plan.",
    )
    czechness_parser.add_argument("--alignment", required=True, help="Path to full hierarchical alignment artifact JSON.")
    czechness_parser.add_argument("--plan", required=True, help="Path to paragraph hybridization plan JSON.")
    czechness_parser.add_argument("--output", required=True, help="Output JSON path.")

    meta_pdf_parser = subparsers.add_parser(
        "render-meta-pdf",
        help="Render a standalone meta PDF with hybridization metrics and charts for an experiment.",
    )
    meta_pdf_parser.add_argument("--alignment", required=True, help="Path to full hierarchical alignment artifact JSON.")
    meta_pdf_parser.add_argument("--plan", required=True, help="Path to paragraph hybridization plan JSON.")
    meta_pdf_parser.add_argument("--output-dir", required=True, help="Output directory for TeX, PDF, and chart assets.")
    meta_pdf_parser.add_argument("--label", required=True, help="Base filename label.")
    meta_pdf_parser.add_argument("--title", default="Meta eksperymentu hybrydyzacji", help="PDF title.")

    lemma_timeline_parser = subparsers.add_parser(
        "lemma-timeline-report",
        help="Build lemma-family occurrence timeline with remaining-count and urgency signals.",
    )
    lemma_timeline_parser.add_argument("--alignment", required=True, help="Path to full hierarchical alignment artifact JSON.")
    lemma_timeline_parser.add_argument(
        "--blocked-standalone-upos",
        default="SCONJ,CCONJ,PART,ADP",
        help="Comma-separated UPOS tags passed through unit building.",
    )
    lemma_timeline_parser.add_argument("--output", required=True, help="Output JSON path.")

    lemma_schedule_parser = subparsers.add_parser(
        "lemma-schedule-report",
        help="Build an experimental lemma-family introduction schedule from the lemma timeline artifact.",
    )
    lemma_schedule_parser.add_argument("--timeline", required=True, help="Path to lemma timeline JSON.")
    lemma_schedule_parser.add_argument("--target-power", type=float, default=1.4, help="Target future-czechness power curve.")
    lemma_schedule_parser.add_argument(
        "--early-weight",
        type=float,
        default=1.0,
        help="Extra prefix-error weight at the start of the book; actual weights decrease linearly to 1.0.",
    )
    lemma_schedule_parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum number of greedy residual steps.",
    )
    lemma_schedule_parser.add_argument(
        "--min-improvement",
        type=float,
        default=1e-9,
        help="Stop when the best available block improves weighted prefix error by at most this amount.",
    )
    lemma_schedule_parser.add_argument(
        "--late-fill-steps",
        type=int,
        default=0,
        help="Optional second-phase late-book fill steps after the prefix-disciplined phase.",
    )
    lemma_schedule_parser.add_argument(
        "--late-weight",
        type=float,
        default=0.0,
        help="Extra weight applied toward the end of the book during the optional late-fill phase.",
    )
    lemma_schedule_parser.add_argument(
        "--late-start-fraction",
        type=float,
        default=0.6,
        help="Book fraction after which late-fill weights ramp up toward the end.",
    )
    lemma_schedule_parser.add_argument("--output", required=True, help="Output JSON path.")

    render_pdf_parser = subparsers.add_parser(
        "render-full-book-pdf",
        help="Render and optionally compile a full-book hybrid PDF from corpus, paragraph plan, and notes.",
    )
    render_pdf_parser.add_argument("--corpus", required=True, help="Path to the Polish canonical corpus JSON.")
    render_pdf_parser.add_argument("--plan", required=True, help="Path to paragraph hybridization plan JSON.")
    render_pdf_parser.add_argument("--notes", required=True, help="Path to paragraph notes JSON.")
    render_pdf_parser.add_argument("--output-dir", required=True, help="Directory for generated TeX/PDF artifacts.")
    render_pdf_parser.add_argument("--label", default="hybrid_book", help="Basename for generated artifact files.")
    render_pdf_parser.add_argument("--template", default="template/template.tex", help="LaTeX template path.")
    render_pdf_parser.add_argument("--alignment", help="Optional alignment artifact for meta section generation.")
    render_pdf_parser.add_argument("--book-author", default="", help="Optional author text injected into the template.")
    render_pdf_parser.add_argument("--book-subtitle", default="", help="Optional subtitle text injected into the template.")
    render_pdf_parser.add_argument("--skip-compile", action="store_true", help="Only write TeX/JSON artifacts; skip PDF compilation.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        book = build_canonical_book(args.epub, language=args.language)
        output_path = Path(args.output) if args.output else Path("data/corpus") / f"{Path(args.epub).stem}.json"
        write_canonical_book(book, output_path)
        if args.stdout:
            print(json.dumps(book.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(output_path)
        return 0

    if args.command == "inspect":
        book = load_book_from_json(args.input)
        rendered = (
            render_chapter_preview(book, args.chapter, max_paragraphs=args.max_paragraphs)
            if args.chapter
            else render_book_summary(book)
        )
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered, encoding="utf-8")
            print(output_path)
        else:
            print(rendered)
        return 0

    if args.command == "align-candidates":
        source_book = load_book_from_json(args.source)
        target_book = load_book_from_json(args.target)
        alignment = build_alignment_candidates(
            source_book,
            target_book,
            chapter_window=args.chapter_window,
            paragraph_window=args.paragraph_window,
            top_k=args.top_k,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(alignment.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "align-sentences":
        source_book = load_book_from_json(args.source)
        target_book = load_book_from_json(args.target)
        alignments = align_chapter_sentences(
            source_book.chapters[args.source_chapter - 1],
            target_book.chapters[args.target_chapter - 1],
            paragraph_window=args.paragraph_window,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "source_paragraph_id": alignment.source_paragraph_id,
                "target_paragraph_id": alignment.target_paragraph_id,
                "source_paragraph_index": alignment.source_paragraph_index,
                "target_paragraph_index": alignment.target_paragraph_index,
                "sentence_alignment": augment_sentence_alignment(alignment.sentence_alignment),
            }
            for alignment in alignments
        ]
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "benchmark-report":
        source_book = load_book_from_json(args.source)
        target_book = load_book_from_json(args.target)
        chapter_pairs = []
        for pair in args.pairs.split(","):
            left, right = pair.split(":")
            chapter_pairs.append((int(left), int(right)))
        report = build_benchmark_report(
            source_book,
            target_book,
            chapter_pairs=chapter_pairs,
            paragraph_window=args.paragraph_window,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "review-bundle":
        source_book = load_book_from_json(args.source)
        target_book = load_book_from_json(args.target)
        bundle = build_review_bundle_for_chapter_pair(
            source_book.chapters[args.source_chapter - 1],
            target_book.chapters[args.target_chapter - 1],
            paragraph_window=args.paragraph_window,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "enrich-chapter":
        source_book = load_book_from_json(args.source)
        target_book = load_book_from_json(args.target)
        chapter_alignments = align_chapter_sentences(
            source_book.chapters[args.source_chapter - 1],
            target_book.chapters[args.target_chapter - 1],
            paragraph_window=args.paragraph_window,
        )
        bundle = build_sentence_enrichment_bundle(
            chapter_alignments=chapter_alignments,
            source_language=source_book.metadata.language,
            target_language=target_book.metadata.language,
            analysis_mode=args.analysis_mode,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "quality-report":
        source_book = load_book_from_json(args.source)
        target_book = load_book_from_json(args.target)
        chapter_pairs = []
        for pair in args.pairs.split(","):
            left, right = pair.split(":")
            chapter_pairs.append((int(left), int(right)))
        report = build_alignment_quality_report(
            source_book,
            target_book,
            chapter_pairs=chapter_pairs,
            paragraph_window=args.paragraph_window,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "span-coverage-report":
        source_book = load_book_from_json(args.source)
        target_book = load_book_from_json(args.target)
        chapter_pairs = []
        for pair in args.pairs.split(","):
            left, right = pair.split(":")
            chapter_pairs.append((int(left), int(right)))
        report = build_sentence_span_coverage_report(
            source_book,
            target_book,
            chapter_pairs=chapter_pairs,
            paragraph_window=args.paragraph_window,
            analysis_mode=args.analysis_mode,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "book-span-report":
        source_book = load_book_from_json(args.source)
        target_book = load_book_from_json(args.target)
        report = build_book_chapter_span_report(
            source_book,
            target_book,
            paragraph_window=args.paragraph_window,
            analysis_mode=args.analysis_mode,
            min_match_score=args.min_match_score,
            skip_penalty=args.skip_penalty,
            paragraph_model=args.paragraph_model,
            paragraph_device=args.paragraph_device,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "book-paragraph-report":
        from szwejk.align.paragraphs import ParagraphAlignmentConfig

        source_book = load_book_from_json(args.source)
        target_book = load_book_from_json(args.target)
        encoder = HeuristicEmbeddingEncoder() if args.model == "heuristic" else SentenceTransformersEncoder(args.model, device=args.device)
        report = build_book_paragraph_alignment_report(
            source_book,
            target_book,
            encoder=encoder,
            model_name=args.model,
            min_match_score=args.min_match_score,
            skip_penalty=args.skip_penalty,
            config=ParagraphAlignmentConfig(
                max_source_span=args.max_source_span,
                max_target_span=args.max_target_span,
                skip_source_penalty=args.skip_source_penalty,
                skip_target_penalty=args.skip_target_penalty,
                pair_bonus=args.pair_bonus,
                position_slack=args.position_slack,
            ),
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "hierarchical-alignment-report":
        source_book = load_book_from_json(args.source)
        target_book = load_book_from_json(args.target)
        chapter_pairs = None
        if args.pairs:
            chapter_pairs = []
            for pair in args.pairs.split(","):
                left, right = pair.split(":")
                chapter_pairs.append((int(left), int(right)))
        paragraph_encoder = HeuristicEmbeddingEncoder() if args.paragraph_model == "heuristic" else SentenceTransformersEncoder(args.paragraph_model, device=args.device)
        sentence_encoder = HeuristicEmbeddingEncoder() if args.sentence_model == "heuristic" else SentenceTransformersEncoder(args.sentence_model, device=args.device)
        report = build_hierarchical_alignment_report(
            source_book,
            target_book,
            chapter_pairs=chapter_pairs,
            paragraph_encoder=paragraph_encoder,
            sentence_encoder=sentence_encoder,
            analysis_mode=args.analysis_mode,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "replacement-safety-report":
        source_book = load_book_from_json(args.source)
        target_book = load_book_from_json(args.target)
        chapter_pairs = None
        if args.pairs:
            chapter_pairs = []
            for pair in args.pairs.split(","):
                left, right = pair.split(":")
                chapter_pairs.append((int(left), int(right)))
        paragraph_encoder = HeuristicEmbeddingEncoder() if args.paragraph_model == "heuristic" else SentenceTransformersEncoder(args.paragraph_model, device=args.device)
        sentence_encoder = HeuristicEmbeddingEncoder() if args.sentence_model == "heuristic" else SentenceTransformersEncoder(args.sentence_model, device=args.device)
        report = build_replacement_safety_report(
            source_book,
            target_book,
            chapter_pairs=chapter_pairs,
            paragraph_encoder=paragraph_encoder,
            sentence_encoder=sentence_encoder,
            analysis_mode=args.analysis_mode,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "embedding-benchmark":
        source_book = load_book_from_json(args.source)
        target_book = load_book_from_json(args.target)
        chapter_pairs = []
        for pair in args.pairs.split(","):
            left, right = pair.split(":")
            chapter_pairs.append((int(left), int(right)))
        encoder = HeuristicEmbeddingEncoder() if args.model == "heuristic" else SentenceTransformersEncoder(args.model, device=args.device)
        report = build_embedding_benchmark_report(
            source_book,
            target_book,
            encoder=encoder,
            model_name=args.model,
            chapter_pairs=chapter_pairs,
            paragraph_window=args.paragraph_window,
            distractor_window=args.distractor_window,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "inspect-policy":
        result = load_policy_from_json(args.input)
        normalized = result.policy.to_dict()
        summary = {
            "id": result.policy.id,
            "version": result.policy.version,
            "label": result.policy.label,
            "level_count": len(result.policy.levels),
            "levels": [
                {
                    "id": level.id,
                    "label": level.label,
                    "base_mode": level.base_mode,
                    "allowed_unit_classes": level.allowed_unit_classes,
                    "minimum_safe_depth": level.gates.minimum_safe_depth,
                    "target_surface_czechness": level.budget.target_surface_czechness,
                }
                for level in result.policy.levels
            ],
        }
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
            print(output_path)
        else:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "plan-policy-chapter":
        policy = load_policy_from_json(args.policy).policy
        source_book = load_book_from_json(args.source)
        target_book = load_book_from_json(args.target)
        plan = build_chapter_policy_plan(
            source_book,
            target_book,
            source_chapter_index=args.source_chapter,
            target_chapter_index=args.target_chapter,
            policy=policy,
            level_id=args.level,
            paragraph_window=args.paragraph_window,
            analysis_mode=args.analysis_mode,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "review-policy-plan":
        input_path = Path(args.input)
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        report = build_policy_review_report(payload)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "benchmark-policy":
        policy = load_policy_from_json(args.policy).policy
        source_book = load_book_from_json(args.source)
        target_book = load_book_from_json(args.target)
        chapter_pairs = []
        for pair in args.pairs.split(","):
            left, right = pair.split(":")
            chapter_pairs.append((int(left), int(right)))
        report = build_policy_benchmark_report(
            source_book,
            target_book,
            policy=policy,
            level_id=args.level,
            chapter_pairs=chapter_pairs,
            paragraph_window=args.paragraph_window,
            analysis_mode=args.analysis_mode,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "policy-action-inventory":
        source_book = load_book_from_json(args.source)
        target_book = load_book_from_json(args.target)
        chapter_alignments = align_chapter_sentences(
            source_book.chapters[args.source_chapter - 1],
            target_book.chapters[args.target_chapter - 1],
            paragraph_window=args.paragraph_window,
        )
        enrichment_bundle = build_sentence_enrichment_bundle(
            chapter_alignments=chapter_alignments,
            source_language=source_book.metadata.language,
            target_language=target_book.metadata.language,
            analysis_mode=args.analysis_mode,
        )
        inventory = build_action_inventory(enrichment_bundle)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "generate-hybrid-chapter":
        source_book = load_book_from_json(args.source)
        target_book = load_book_from_json(args.target)
        plan_payload = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        document = build_hybrid_chapter_document(
            source_book,
            target_book,
            source_chapter_index=args.source_chapter,
            target_chapter_index=args.target_chapter,
            plan_payload=plan_payload,
            paragraph_window=args.paragraph_window,
            analysis_mode=args.analysis_mode,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(document.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "generate-hybrid-diff":
        document = json.loads(Path(args.input).read_text(encoding="utf-8"))
        payload = build_hybrid_diff_payload(document)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "generate-hybrid-notes":
        document = json.loads(Path(args.input).read_text(encoding="utf-8"))
        payload = build_didactic_note_bundle(document)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "generate-paragraph-notes":
        alignment = load_alignment_artifact(args.alignment)
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        payload = build_paragraph_note_bundle(alignment, plan, note_policy=args.note_policy)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "generate-hybrid-book":
        policy = load_policy_from_json(args.policy).policy
        source_book = load_book_from_json(args.source)
        target_book = load_book_from_json(args.target)
        chapter_pairs = None
        if args.pairs:
            chapter_pairs = []
            for pair in args.pairs.split(","):
                left, right = pair.split(":")
                chapter_pairs.append((int(left), int(right)))
        package = build_hybrid_book_package(
            source_book,
            target_book,
            policy=policy,
            chapter_pairs=chapter_pairs,
            paragraph_window=args.paragraph_window,
            analysis_mode=args.analysis_mode,
            min_match_score=args.min_match_score,
            skip_penalty=args.skip_penalty,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(package.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "progression-report":
        source_book = load_book_from_json(args.source)
        target_book = load_book_from_json(args.target)
        chapter_pairs = []
        for pair in args.pairs.split(","):
            left, right = pair.split(":")
            chapter_pairs.append((int(left), int(right)))
        payload = build_progression_report(
            source_book,
            target_book,
            chapter_pairs=chapter_pairs,
            paragraph_window=args.paragraph_window,
            analysis_mode=args.analysis_mode,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "plan-paragraph-hybridization":
        artifact = load_alignment_artifact(args.alignment)
        blocked_upos = frozenset(
            item.strip().upper()
            for item in str(args.blocked_standalone_upos).split(",")
            if item.strip()
        )
        from szwejk.generate.alignment_quality import build_wiktionary_lookup
        cs_to_pl, pl_to_cs = build_wiktionary_lookup(getattr(args, "wiktionary", None) or "")
        payload = build_paragraph_hybridization_plan(
            artifact,
            target_power=args.target_power,
            blocked_standalone_upos=blocked_upos,
            idiomaticity_penalty_weight=args.idiomaticity_penalty_weight,
            granularity_policy=args.granularity_policy,
            chapter_simple_target_power=args.chapter_simple_target_power,
            chapter_simple_target_max=args.chapter_simple_target_max,
            chapter_simple_boost_start=args.chapter_simple_boost_start,
            lemma_schedule_payload=None if not args.lemma_schedule else json.loads(Path(args.lemma_schedule).read_text(encoding="utf-8")),
            wiktionary_lookup=cs_to_pl or None,
            pl_to_cs_lookup=pl_to_cs or None,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "czechness-report":
        alignment = load_alignment_artifact(args.alignment)
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        payload = build_czechness_report(alignment, plan)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "render-meta-pdf":
        alignment = load_alignment_artifact(args.alignment)
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        latex = build_meta_pdf_latex(
            root=Path.cwd(),
            plan_payload=plan,
            alignment_artifact=alignment,
            output_dir=output_dir / "meta",
            label=args.label,
            title=args.title,
        )
        tex_path = output_dir / f"{args.label}_meta.tex"
        tex_path.write_text(latex, encoding="utf-8")
        pdf_path = compile_full_book_pdf(tex_path=tex_path, output_dir=output_dir, cwd=Path.cwd())
        print(pdf_path)
        return 0

    if args.command == "lemma-timeline-report":
        alignment = load_alignment_artifact(args.alignment)
        blocked_upos = frozenset(
            item.strip().upper()
            for item in str(args.blocked_standalone_upos).split(",")
            if item.strip()
        )
        payload = build_lemma_timeline_report(alignment, blocked_standalone_upos=blocked_upos)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "lemma-schedule-report":
        timeline = json.loads(Path(args.timeline).read_text(encoding="utf-8"))
        payload = build_family_schedule_report(
            timeline,
            target_power=args.target_power,
            early_weight=args.early_weight,
            max_steps=args.max_steps,
            min_improvement=args.min_improvement,
            late_fill_steps=args.late_fill_steps,
            late_weight=args.late_weight,
            late_start_fraction=args.late_start_fraction,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return 0

    if args.command == "render-full-book-pdf":
        corpus_payload = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
        plan_payload = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        notes_payload = json.loads(Path(args.notes).read_text(encoding="utf-8"))
        template_path = Path(args.template)
        template_text = template_path.read_text(encoding="utf-8")
        output_dir = Path(args.output_dir)
        alignment_artifact = None if not args.alignment else load_alignment_artifact(args.alignment)
        payload = write_full_book_artifacts(
            corpus_payload=corpus_payload,
            plan_payload=plan_payload,
            notes_payload=notes_payload,
            template_text=template_text,
            output_dir=output_dir,
            label=args.label,
            alignment_artifact=alignment_artifact,
            root=Path.cwd(),
            book_author=args.book_author,
            book_subtitle=args.book_subtitle,
        )
        dictionary_entries = payload.get("dictionary_entries", [])
        print(f"dictionary_entries={len(dictionary_entries)}")
        for label, subset in (
            ("first_20", dictionary_entries[:20]),
            ("last_20", dictionary_entries[-20:]),
        ):
            print(label)
            for entry in subset:
                print(f"{entry.czech} --- {entry.polish}")
        print(payload["tex_path"])
        print(payload["summary_path"])
        print(payload["dictionary_entries_path"])
        if not args.skip_compile:
            pdf_path = compile_full_book_pdf(tex_path=payload["tex_path"], output_dir=output_dir, cwd=Path.cwd())
            print(pdf_path)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
