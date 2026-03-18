"""Alignment helpers and candidate generation."""

from .benchmark import DEFAULT_ALIGNMENT_BENCHMARK_PAIRS, format_chapter_pairs
from .candidates import (
    AlignmentCandidates,
    ChapterCandidate,
    ParagraphCandidate,
    build_alignment_candidates,
    build_monotonic_chapter_alignment,
    generate_chapter_candidates,
    generate_paragraph_candidates,
)
from .coverage import build_book_chapter_span_report, build_sentence_span_coverage_report
from .diagnostics import augment_sentence_alignment, build_benchmark_report, classify_confidence, summarize_sentence_alignments
from .embedding_eval import (
    HeuristicEmbeddingEncoder,
    SentenceTransformersEncoder,
    build_embedding_benchmark_report,
    build_embedding_eval_samples,
    evaluate_encoder_on_samples,
)
from .enrichment import build_sentence_enrichment, build_sentence_enrichment_bundle, tokenize_text
from .hierarchical import build_hierarchical_alignment_report
from .morphosyntax import analyze_sentence, stanza_available
from .paragraphs import build_book_paragraph_alignment_report, build_chapter_paragraph_alignment
from .quality import build_alignment_quality_report, build_pair_quality_report, evaluate_quality_thresholds
from .replacement_safety import build_replacement_safety_report
from .sentences import (
    ChapterSentenceAlignment,
    ChapterSentenceSpanAlignment,
    SentenceAlignment,
    SentenceSpanAlignment,
    align_chapter_sentence_spans,
    align_chapter_sentences,
    align_chapter_sentences_with_paragraph_blocks,
    align_sentence_sequences,
    align_sentence_spans,
)

__all__ = [
    "AlignmentCandidates",
    "ChapterCandidate",
    "ChapterSentenceAlignment",
    "ChapterSentenceSpanAlignment",
    "DEFAULT_ALIGNMENT_BENCHMARK_PAIRS",
    "HeuristicEmbeddingEncoder",
    "analyze_sentence",
    "ParagraphCandidate",
    "SentenceAlignment",
    "SentenceSpanAlignment",
    "SentenceTransformersEncoder",
    "augment_sentence_alignment",
    "align_chapter_sentences",
    "align_chapter_sentences_with_paragraph_blocks",
    "align_chapter_sentence_spans",
    "build_alignment_candidates",
    "build_book_chapter_span_report",
    "build_book_paragraph_alignment_report",
    "build_chapter_paragraph_alignment",
    "build_monotonic_chapter_alignment",
    "build_alignment_quality_report",
    "build_sentence_span_coverage_report",
    "build_pair_quality_report",
    "build_replacement_safety_report",
    "build_embedding_benchmark_report",
    "build_embedding_eval_samples",
    "build_sentence_enrichment",
    "build_sentence_enrichment_bundle",
    "build_hierarchical_alignment_report",
    "align_sentence_sequences",
    "align_sentence_spans",
    "build_benchmark_report",
    "classify_confidence",
    "evaluate_quality_thresholds",
    "evaluate_encoder_on_samples",
    "format_chapter_pairs",
    "stanza_available",
    "generate_chapter_candidates",
    "generate_paragraph_candidates",
    "summarize_sentence_alignments",
    "tokenize_text",
]
