"""Monotonic many-to-many paragraph alignment for volume IV tail review."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np

from szwejk.schemas import CanonicalBook


GERMAN_HINTS = {
    "ein",
    "zwei",
    "drei",
    "vier",
    "fuenf",
    "funf",
    "achtung",
    "jawohl",
    "bitte",
    "herr",
    "frau",
    "kamerad",
    "marsch",
    "bahnhof",
    "schwein",
    "los",
    "raus",
    "komm",
    "weiter",
    "halt",
    "nicht",
    "ja",
    "nein",
    "gut",
    "wurst",
    "bier",
    "gott",
    "mit",
    "uns",
    "der",
    "die",
    "das",
    "schla",
    "befehl",
    "fertig",
}
LATIN_HINTS = {"etc", "sic", "ergo", "nota", "bene", "corpus", "delicti", "in", "vino", "veritas"}
WORD_RE = re.compile(r"[A-Za-zÀ-ž']+")


@dataclass(slots=True)
class ParagraphItem:
    seq: int
    chapter_index: int
    chapter_title: str
    paragraph_index: int
    paragraph_id: str
    text: str
    char_len: int
    sentence_count: int
    foreign_ratio: float
    normalized_pos: float
    embedding: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "seq": self.seq,
            "chapter_index": self.chapter_index,
            "chapter_title": self.chapter_title,
            "paragraph_index": self.paragraph_index,
            "paragraph_id": self.paragraph_id,
            "text": self.text,
            "char_len": self.char_len,
            "sentence_count": self.sentence_count,
            "foreign_ratio": round(self.foreign_ratio, 6),
            "normalized_pos": round(self.normalized_pos, 6),
        }


@dataclass(slots=True)
class _PrefixStats:
    embedding: np.ndarray
    char_len: np.ndarray
    sentence_count: np.ndarray
    foreign_ratio: np.ndarray
    normalized_pos: np.ndarray


def collect_chapter_paragraphs(
    book: CanonicalBook,
    chapter_index: int,
    *,
    start_paragraph: int = 1,
    end_paragraph: int | None = None,
) -> list[ParagraphItem]:
    chapter = book.chapters[chapter_index - 1]
    paragraphs = chapter.paragraphs[start_paragraph - 1 : end_paragraph]
    total = len(paragraphs)
    items: list[ParagraphItem] = []
    for seq, paragraph in enumerate(paragraphs, start=1):
        text = paragraph.text.strip()
        items.append(
            ParagraphItem(
                seq=seq,
                chapter_index=chapter_index,
                chapter_title=chapter.title,
                paragraph_index=paragraph.index,
                paragraph_id=paragraph.id,
                text=text,
                char_len=len(text),
                sentence_count=len(paragraph.sentences),
                foreign_ratio=_foreign_ratio(text),
                normalized_pos=(seq - 1) / max(total - 1, 1),
                embedding=np.zeros(1, dtype=float),
            )
        )
    return items


def attach_embeddings(items: list[ParagraphItem], embeddings: np.ndarray) -> None:
    for item, embedding in zip(items, embeddings, strict=True):
        item.embedding = np.asarray(embedding, dtype=float)


def resolve_anchor_paragraph(chapter, needle: str) -> int:
    lowered = needle.lower()
    for paragraph in chapter.paragraphs:
        if lowered in paragraph.text.lower():
            return paragraph.index
    raise ValueError(f"anchor not found: chapter={chapter.index} needle={needle!r}")


def align_monotonic_many_to_many(
    source_items: list[ParagraphItem],
    target_items: list[ParagraphItem],
    *,
    max_source_span: int = 4,
    max_target_span: int = 4,
    skip_source_penalty: float = 0.55,
    skip_target_penalty: float = 0.65,
    pair_bonus: float = 0.04,
    position_slack: float = 0.08,
) -> dict[str, object]:
    dp: dict[tuple[int, int], float] = {(0, 0): 0.0}
    backpointers: dict[tuple[int, int], tuple[tuple[int, int], dict[str, object]]] = {}
    source_total = len(source_items)
    target_total = len(target_items)
    state_slack = position_slack + max(max_source_span / max(source_total, 1), max_target_span / max(target_total, 1)) + 0.02
    source_stats = _build_prefix_stats(source_items)
    target_stats = _build_prefix_stats(target_items)
    for source_done in range(0, source_total + 1):
        for target_done in range(0, target_total + 1):
            state = (source_done, target_done)
            if state not in dp:
                continue
            if not _within_band(source_done, target_done, source_total, target_total, state_slack):
                continue
            score = dp[state]
            if source_done < source_total:
                next_state = (source_done + 1, target_done)
                if _within_band(next_state[0], next_state[1], source_total, target_total, state_slack):
                    _update_state(
                        dp,
                        backpointers,
                        next_state,
                        score - skip_source_penalty,
                        state,
                        {
                            "kind": "skip_source",
                            "source_range": [source_done + 1, source_done + 1],
                            "target_range": None,
                            "score": round(-skip_source_penalty, 6),
                        },
                    )
            if target_done < target_total:
                next_state = (source_done, target_done + 1)
                if _within_band(next_state[0], next_state[1], source_total, target_total, state_slack):
                    _update_state(
                        dp,
                        backpointers,
                        next_state,
                        score - skip_target_penalty,
                        state,
                        {
                            "kind": "skip_target",
                            "source_range": None,
                            "target_range": [target_done + 1, target_done + 1],
                            "score": round(-skip_target_penalty, 6),
                        },
                    )
            for source_span in range(1, max_source_span + 1):
                if source_done + source_span > source_total:
                    break
                for target_span in range(1, max_target_span + 1):
                    if target_done + target_span > target_total:
                        break
                    next_source = source_done + source_span
                    next_target = target_done + target_span
                    if not _within_band(next_source, next_target, source_total, target_total, state_slack):
                        continue
                    source_progress = (source_done + source_span) / max(source_total, 1)
                    target_progress = (target_done + target_span) / max(target_total, 1)
                    if abs(source_progress - target_progress) > position_slack:
                        continue
                    block_score, signals = _score_block_pair_fast(
                        source_stats,
                        target_stats,
                        source_done,
                        source_done + source_span,
                        target_done,
                        target_done + target_span,
                    )
                    candidate_score = score + block_score + pair_bonus
                    _update_state(
                        dp,
                        backpointers,
                        (next_source, next_target),
                        candidate_score,
                        state,
                        {
                            "kind": "match",
                            "source_range": [source_done + 1, source_done + source_span],
                            "target_range": [target_done + 1, target_done + target_span],
                            "score": round(block_score + pair_bonus, 6),
                            "signals": {key: round(value, 6) for key, value in signals.items()},
                        },
                    )

    best_score = dp[(source_total, target_total)]
    best_path = _reconstruct_path(backpointers, (source_total, target_total))
    matched_blocks = [item for item in best_path if item["kind"] == "match"]
    unmatched_source = [item for item in best_path if item["kind"] == "skip_source"]
    unmatched_target = [item for item in best_path if item["kind"] == "skip_target"]
    materialized_blocks = _materialize_blocks(matched_blocks, source_items, target_items)
    return {
        "best_score": round(best_score, 6),
        "matched_blocks": materialized_blocks,
        "unmatched_source": _materialize_skips(unmatched_source, source_items, target_items, side="source"),
        "unmatched_target": _materialize_skips(unmatched_target, source_items, target_items, side="target"),
        "source_coverage": round(
            sum(block["source_span"] for block in materialized_blocks) / max(source_total, 1),
            6,
        ),
        "target_coverage": round(
            sum(block["target_span"] for block in materialized_blocks) / max(target_total, 1),
            6,
        ),
    }


def score_block_pair(source_block: list[ParagraphItem], target_block: list[ParagraphItem]) -> tuple[float, dict[str, float]]:
    source_stats = _build_prefix_stats(source_block)
    target_stats = _build_prefix_stats(target_block)
    return _score_block_pair_fast(
        source_stats,
        target_stats,
        0,
        len(source_block),
        0,
        len(target_block),
    )


def _score_block_pair_fast(
    source_stats: _PrefixStats,
    target_stats: _PrefixStats,
    source_start: int,
    source_end: int,
    target_start: int,
    target_end: int,
) -> tuple[float, dict[str, float]]:
    source_embedding = _aggregate_embedding_from_prefix(source_stats.embedding, source_start, source_end)
    target_embedding = _aggregate_embedding_from_prefix(target_stats.embedding, target_start, target_end)
    similarity = float(np.dot(source_embedding, target_embedding))
    source_chars = int(_sum_range(source_stats.char_len, source_start, source_end))
    target_chars = int(_sum_range(target_stats.char_len, target_start, target_end))
    length_similarity = _count_similarity(source_chars, target_chars)
    source_sentences = int(_sum_range(source_stats.sentence_count, source_start, source_end))
    target_sentences = int(_sum_range(target_stats.sentence_count, target_start, target_end))
    sentence_similarity = _count_similarity(source_sentences, target_sentences)
    source_position = float(_mean_range(source_stats.normalized_pos, source_start, source_end))
    target_position = float(_mean_range(target_stats.normalized_pos, target_start, target_end))
    position_similarity = 1.0 - abs(source_position - target_position)
    source_span = source_end - source_start
    target_span = target_end - target_start
    span_similarity = 1.0 - (abs(source_span - target_span) / max(source_span, target_span, 1))
    foreign_penalty = 0.10 * max(
        float(_mean_range(source_stats.foreign_ratio, source_start, source_end)),
        float(_mean_range(target_stats.foreign_ratio, target_start, target_end)),
    )
    score = (
        (0.50 * similarity)
        + (0.17 * length_similarity)
        + (0.12 * sentence_similarity)
        + (0.11 * position_similarity)
        + (0.10 * span_similarity)
        - foreign_penalty
    )
    return score, {
        "embedding": similarity,
        "length": length_similarity,
        "sentences": sentence_similarity,
        "position": position_similarity,
        "span": span_similarity,
        "foreign_penalty": foreign_penalty,
    }


def summarize_volume4_alignment(chapter_alignments: list[dict[str, object]]) -> dict[str, object]:
    total_source = sum(int(item["source_total"]) for item in chapter_alignments)
    total_target = sum(int(item["target_total"]) for item in chapter_alignments)
    matched_source = sum(sum(int(block["source_span"]) for block in item["matched_blocks"]) for item in chapter_alignments)
    matched_target = sum(sum(int(block["target_span"]) for block in item["matched_blocks"]) for item in chapter_alignments)
    return {
        "source_paragraph_coverage": round(matched_source / max(total_source, 1), 6),
        "target_paragraph_coverage": round(matched_target / max(total_target, 1), 6),
        "source_total": total_source,
        "target_total": total_target,
        "matched_source": matched_source,
        "matched_target": matched_target,
    }


def _materialize_blocks(
    blocks: list[dict[str, object]],
    source_items: list[ParagraphItem],
    target_items: list[ParagraphItem],
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for block in blocks:
        source_start, source_end = block["source_range"]
        target_start, target_end = block["target_range"]
        source_block = source_items[source_start - 1 : source_end]
        target_block = target_items[target_start - 1 : target_end]
        payload.append(
            {
                "source_range": block["source_range"],
                "target_range": block["target_range"],
                "source_span": len(source_block),
                "target_span": len(target_block),
                "score": block["score"],
                "signals": dict(block.get("signals", {})),
                "source_refs": [_item_ref(item) for item in source_block],
                "target_refs": [_item_ref(item) for item in target_block],
                "source_preview": " ".join(item.text for item in source_block)[:280],
                "target_preview": " ".join(item.text for item in target_block)[:280],
            }
        )
    return payload


def _materialize_skips(
    skips: list[dict[str, object]],
    source_items: list[ParagraphItem],
    target_items: list[ParagraphItem],
    *,
    side: str,
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for item in skips:
        if side == "source":
            start, end = item["source_range"]
            block = source_items[start - 1 : end]
        else:
            start, end = item["target_range"]
            block = target_items[start - 1 : end]
        payload.append(
            {
                "range": [start, end],
                "refs": [_item_ref(ref) for ref in block],
                "preview": " ".join(ref.text for ref in block)[:280],
            }
        )
    return payload


def _item_ref(item: ParagraphItem) -> dict[str, object]:
    return {
        "seq": item.seq,
        "chapter_index": item.chapter_index,
        "paragraph_index": item.paragraph_index,
        "paragraph_id": item.paragraph_id,
    }


def _aggregate_embedding(items: Iterable[ParagraphItem]) -> np.ndarray:
    vectors = [item.embedding for item in items]
    matrix = np.vstack(vectors)
    mean = np.mean(matrix, axis=0)
    norm = np.linalg.norm(mean)
    if norm == 0:
        return mean
    return mean / norm


def _build_prefix_stats(items: list[ParagraphItem]) -> _PrefixStats:
    embedding_matrix = np.vstack([item.embedding for item in items])
    embedding_prefix = np.vstack([np.zeros((1, embedding_matrix.shape[1]), dtype=float), np.cumsum(embedding_matrix, axis=0)])
    char_prefix = np.concatenate([[0.0], np.cumsum([item.char_len for item in items], dtype=float)])
    sentence_prefix = np.concatenate([[0.0], np.cumsum([item.sentence_count for item in items], dtype=float)])
    foreign_prefix = np.concatenate([[0.0], np.cumsum([item.foreign_ratio for item in items], dtype=float)])
    pos_prefix = np.concatenate([[0.0], np.cumsum([item.normalized_pos for item in items], dtype=float)])
    return _PrefixStats(
        embedding=embedding_prefix,
        char_len=char_prefix,
        sentence_count=sentence_prefix,
        foreign_ratio=foreign_prefix,
        normalized_pos=pos_prefix,
    )


def _aggregate_embedding_from_prefix(prefix: np.ndarray, start: int, end: int) -> np.ndarray:
    summed = prefix[end] - prefix[start]
    norm = np.linalg.norm(summed)
    if norm == 0:
        return summed
    return summed / norm


def _sum_range(prefix: np.ndarray, start: int, end: int) -> float:
    return float(prefix[end] - prefix[start])


def _mean_range(prefix: np.ndarray, start: int, end: int) -> float:
    span = max(end - start, 1)
    return _sum_range(prefix, start, end) / span


def _count_similarity(left: int, right: int) -> float:
    if left <= 0 and right <= 0:
        return 1.0
    return min(left, right) / max(left, right, 1)


def _foreign_ratio(text: str) -> float:
    tokens = [token.lower() for token in WORD_RE.findall(text)]
    if not tokens:
        return 0.0
    german_hits = sum(token in GERMAN_HINTS for token in tokens)
    latin_hits = sum(token in LATIN_HINTS for token in tokens)
    return (german_hits + latin_hits) / len(tokens)


def _update_state(
    dp: dict[tuple[int, int], float],
    backpointers: dict[tuple[int, int], tuple[tuple[int, int], dict[str, object]]],
    state: tuple[int, int],
    score: float,
    previous_state: tuple[int, int],
    action: dict[str, object],
) -> None:
    previous = dp.get(state)
    if previous is None or score > previous:
        dp[state] = score
        backpointers[state] = (previous_state, action)


def _reconstruct_path(
    backpointers: dict[tuple[int, int], tuple[tuple[int, int], dict[str, object]]],
    state: tuple[int, int],
) -> list[dict[str, object]]:
    path: list[dict[str, object]] = []
    current = state
    while current != (0, 0):
        previous, action = backpointers[current]
        path.append(action)
        current = previous
    path.reverse()
    return path


def _within_band(
    source_index: int,
    target_index: int,
    source_total: int,
    target_total: int,
    slack: float,
) -> bool:
    if source_total <= 0 or target_total <= 0:
        return True
    source_progress = source_index / max(source_total, 1)
    target_progress = target_index / max(target_total, 1)
    return abs(source_progress - target_progress) <= slack
