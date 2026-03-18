"""Text cleanup and baseline sentence segmentation."""

from __future__ import annotations

import re


SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?…])\s+(?=[\"'“”«»„A-Z0-9\-])")


def clean_text(value: str) -> str:
    text = value.replace("\xa0", " ").replace("\u2009", " ").replace("\u200a", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    parts = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(cleaned) if part.strip()]
    return parts or [cleaned]
