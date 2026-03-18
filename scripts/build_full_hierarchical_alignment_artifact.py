from __future__ import annotations

import argparse
import json
from pathlib import Path

from szwejk.align.embedding_eval import HeuristicEmbeddingEncoder, SentenceTransformersEncoder
from szwejk.align.enrichment import build_sentence_enrichment_bundle
from szwejk.align.paragraphs import build_book_paragraph_alignment_report
from szwejk.align.sentence_blocks import align_chapter_sentence_spans_with_paragraph_blocks
from szwejk.align.sentences import ChapterSentenceAlignment, SentenceAlignment
from szwejk.generate.corpus_scope import apply_default_tail_chapter_overrides, scoped_source_chapter
from szwejk.schemas import CanonicalBook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a full hierarchical alignment artifact for the comparable corpus.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--analysis-mode", default="stanza")
    parser.add_argument("--paragraph-model", default="heuristic")
    parser.add_argument("--sentence-model", default="heuristic")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_book(path: str) -> CanonicalBook:
    return CanonicalBook.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def make_encoder(model_name: str, device: str):
    if model_name == "heuristic":
        return HeuristicEmbeddingEncoder()
    return SentenceTransformersEncoder(model_name, device=device)


def to_chapter_sentence_alignment(row) -> ChapterSentenceAlignment:
    span = row.sentence_span_alignment
    return ChapterSentenceAlignment(
        source_paragraph_id=row.source_paragraph_id,
        target_paragraph_id=row.target_paragraph_id,
        source_paragraph_index=row.source_paragraph_index,
        target_paragraph_index=row.target_paragraph_index,
        sentence_alignment=SentenceAlignment(
            source_index=span.source_span[0],
            target_index=span.target_span[0],
            source_text=span.source_text,
            target_text=span.target_text,
            score=span.score,
            signals=span.signals,
        ),
    )


def main() -> None:
    args = parse_args()
    source_book = load_book(args.source)
    target_book = load_book(args.target)
    normalized_pairs = apply_default_tail_chapter_overrides(
        [(index, index) for index in range(1, min(source_book.chapter_count, target_book.chapter_count) + 1)],
        source_chapter_count=source_book.chapter_count,
        target_chapter_count=target_book.chapter_count,
    )
    paragraph_encoder = make_encoder(args.paragraph_model, args.device)
    sentence_encoder = make_encoder(args.sentence_model, args.device)
    paragraph_report = build_book_paragraph_alignment_report(
        source_book,
        target_book,
        chapter_pairs=normalized_pairs,
        encoder=paragraph_encoder,
        model_name=args.paragraph_model,
    )
    pair_lookup = {
        (int(item["source_chapter_index"]), int(item["target_chapter_index"])): item
        for item in paragraph_report["pair_reports"]
    }

    pair_reports: list[dict[str, object]] = []
    for source_index, target_index in normalized_pairs:
        source_chapter = scoped_source_chapter(source_book.chapters[source_index - 1])
        target_chapter = target_book.chapters[target_index - 1]
        paragraph_pair_report = pair_lookup[(source_index, target_index)]
        span_rows = align_chapter_sentence_spans_with_paragraph_blocks(
            source_chapter,
            target_chapter,
            paragraph_blocks=list(paragraph_pair_report["matched_blocks"]),
            source_language=source_book.metadata.language,
            target_language=target_book.metadata.language,
            analysis_mode=args.analysis_mode,
            encoder=sentence_encoder,
        )
        enrichment_bundle = build_sentence_enrichment_bundle(
            chapter_alignments=[to_chapter_sentence_alignment(row) for row in span_rows],
            source_language=source_book.metadata.language,
            target_language=target_book.metadata.language,
            analysis_mode=args.analysis_mode,
            subtree_encoder=sentence_encoder,
        )
        pair_reports.append(
            {
                "source_chapter_index": source_index,
                "target_chapter_index": target_index,
                "source_title": source_chapter.title,
                "target_title": target_chapter.title,
                "paragraph_alignment": paragraph_pair_report,
                "sentence_rows": [row.to_dict() for row in span_rows],
                "enrichment_bundle": enrichment_bundle,
            }
        )
        print(f"done {source_index}:{target_index} {source_chapter.title} -> {target_chapter.title}", flush=True)

    payload = {
        "analysis_mode": args.analysis_mode,
        "paragraph_model": args.paragraph_model,
        "sentence_model": args.sentence_model,
        "chapter_pair_count": len(pair_reports),
        "paragraph_global_metrics": paragraph_report["global_metrics"],
        "pair_reports": pair_reports,
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
