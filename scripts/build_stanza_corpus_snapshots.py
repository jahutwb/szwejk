from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from szwejk.align.morphosyntax import _analyze_with_stanza
from szwejk.generate.corpus_scope import scoped_source_chapter
from szwejk.review import load_book_from_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build persistent stanza snapshots for comparable PL/CS corpora.")
    parser.add_argument("--source", required=True, help="Path to Polish canonical corpus JSON.")
    parser.add_argument("--target", required=True, help="Path to Czech canonical corpus JSON.")
    parser.add_argument("--output-dir", default="data/cache/stanza", help="Output directory for snapshot JSON files.")
    parser.add_argument("--source-max-chapter", type=int, default=29, help="Highest Polish chapter included in comparable corpus.")
    parser.add_argument("--target-max-chapter", type=int, default=29, help="Highest Czech chapter included in comparable corpus.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_book = load_book_from_json(args.source)
    target_book = load_book_from_json(args.target)

    source_payload = _build_snapshot_payload(
        source_book,
        language="pl",
        scope_source=True,
        max_chapter_index=args.source_max_chapter,
    )
    target_payload = _build_snapshot_payload(
        target_book,
        language="cs",
        scope_source=False,
        max_chapter_index=args.target_max_chapter,
    )

    source_path = output_dir / "pl_comparable_snapshot.json"
    target_path = output_dir / "cs_comparable_snapshot.json"
    source_path.write_text(json.dumps(source_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    target_path.write_text(json.dumps(target_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(source_path)
    print(target_path)
    return 0


def _build_snapshot_payload(book, *, language: str, scope_source: bool, max_chapter_index: int) -> dict[str, object]:
    started = time.perf_counter()
    chapters = []
    unique_texts: dict[str, list[dict[str, object]]] = {}
    sentence_count = 0
    for chapter in book.chapters:
        if chapter.index > max_chapter_index:
            continue
        current_chapter = scoped_source_chapter(chapter) if scope_source else chapter
        chapters.append(
            {
                "chapter_index": current_chapter.index,
                "title": current_chapter.title,
                "paragraph_count": len(current_chapter.paragraphs),
            }
        )
        for paragraph in current_chapter.paragraphs:
            for sentence in paragraph.sentences:
                text = sentence.text
                if text in unique_texts:
                    sentence_count += 1
                    continue
                unique_texts[text] = _analyze_with_stanza(text, language=language)
                sentence_count += 1
    elapsed = time.perf_counter() - started
    return {
        "language": language,
        "scope": "comparable_source_trimmed" if scope_source else "comparable_target_full",
        "sentence_count": sentence_count,
        "unique_sentence_count": len(unique_texts),
        "elapsed_seconds": round(elapsed, 3),
        "chapters": chapters,
        "entries": [
            {
                "text": text,
                "tokens": tokens,
            }
            for text, tokens in unique_texts.items()
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
