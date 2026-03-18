"""Curated benchmark chapter pairs for alignment evaluation."""

from __future__ import annotations


DEFAULT_ALIGNMENT_BENCHMARK_PAIRS: list[tuple[int, int]] = [
    (1, 1),
    (2, 2),
    (10, 10),
    (15, 15),
]


def format_chapter_pairs(pairs: list[tuple[int, int]]) -> str:
    return ",".join(f"{source}:{target}" for source, target in pairs)
