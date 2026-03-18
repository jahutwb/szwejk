"""Render canonical corpora into review-friendly text formats."""

from __future__ import annotations

import json
from pathlib import Path

from szwejk.schemas import CanonicalBook


def load_book_from_json(path: str | Path) -> CanonicalBook:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CanonicalBook.from_dict(payload)


def render_book_summary(book: CanonicalBook) -> str:
    lines = [
        f"Title: {book.metadata.title}",
        f"Language: {book.metadata.language}",
        f"Source: {book.metadata.source_path}",
        f"Chapters: {book.chapter_count}",
        "",
        "Chapter overview:",
    ]
    for chapter in book.chapters:
        lines.append(
            f"- {chapter.index:02d}. {chapter.title} | paragraphs={len(chapter.paragraphs)} | source={chapter.source_path}"
        )
    return "\n".join(lines)


def render_chapter_preview(book: CanonicalBook, chapter_index: int, max_paragraphs: int = 5) -> str:
    chapter = book.chapters[chapter_index - 1]
    lines = [
        f"Chapter {chapter.index}: {chapter.title}",
        f"Source: {chapter.source_path}",
        f"Paragraphs: {len(chapter.paragraphs)}",
        "",
    ]
    for paragraph in chapter.paragraphs[:max_paragraphs]:
        lines.append(f"[{paragraph.index:03d}] {paragraph.text}")
        for sentence in paragraph.sentences:
            lines.append(f"  - ({sentence.index:02d}) {sentence.text}")
        lines.append("")
    return "\n".join(lines).rstrip()
