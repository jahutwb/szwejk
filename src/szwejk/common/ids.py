"""Stable identifier helpers for canonical corpus artifacts."""

from __future__ import annotations

import re
import unicodedata


def slugify_fragment(value: str) -> str:
    """Create an ASCII-safe slug fragment stable enough for artifact IDs."""
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    lowered = normalized.lower()
    compact = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return compact or "item"


def make_book_id(language: str, title: str) -> str:
    return f"book-{slugify_fragment(language)}-{slugify_fragment(title)}"


def make_chapter_id(book_id: str, chapter_index: int, label: str | None = None) -> str:
    suffix = slugify_fragment(label or f"chapter-{chapter_index:03d}")
    return f"{book_id}-ch-{chapter_index:03d}-{suffix}"


def make_paragraph_id(chapter_id: str, paragraph_index: int) -> str:
    return f"{chapter_id}-p-{paragraph_index:04d}"


def make_sentence_id(paragraph_id: str, sentence_index: int) -> str:
    return f"{paragraph_id}-s-{sentence_index:03d}"
