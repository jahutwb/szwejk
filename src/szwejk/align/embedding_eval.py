"""Embedding-based comparison harness for sentence alignment."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

from szwejk.align.benchmark import DEFAULT_ALIGNMENT_BENCHMARK_PAIRS
from szwejk.align.diagnostics import classify_confidence
from szwejk.align.sentences import ChapterSentenceAlignment, align_chapter_sentences
from szwejk.common.ids import slugify_fragment
from szwejk.schemas import CanonicalBook, CanonicalChapter


class SentenceEncoder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass(slots=True)
class HeuristicEmbeddingEncoder:
    dim: int = 128

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [_hashed_char_vector(text, dim=self.dim) for text in texts]


@dataclass(slots=True)
class SentenceTransformersEncoder:
    model_name: str
    device: str = "cpu"
    batch_size: int = 32
    _model: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_model", _load_sentence_transformer(self.model_name, self.device))

    def encode(self, texts: list[str]) -> list[list[float]]:
        matrix = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [row.tolist() for row in matrix]


def build_candidate_set(target_sentences: list[str], *, positive_index: int, distractor_window: int = 2) -> list[str]:
    positive = target_sentences[positive_index - 1]
    candidates = [positive]
    for offset in range(1, distractor_window + 1):
        left = positive_index - 1 - offset
        right = positive_index - 1 + offset
        if left >= 0:
            candidates.append(target_sentences[left])
        if right < len(target_sentences):
            candidates.append(target_sentences[right])
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        unique.append(candidate)
        seen.add(candidate)
    return unique


def build_embedding_eval_samples(
    source_chapter: CanonicalChapter,
    target_chapter: CanonicalChapter,
    *,
    paragraph_window: int = 3,
    distractor_window: int = 2,
) -> list[dict[str, object]]:
    target_paragraphs = {paragraph.id: paragraph for paragraph in target_chapter.paragraphs}
    chapter_alignments = align_chapter_sentences(
        source_chapter,
        target_chapter,
        paragraph_window=paragraph_window,
    )
    samples: list[dict[str, object]] = []
    for row in chapter_alignments:
        target_paragraph = target_paragraphs[row.target_paragraph_id]
        candidate_texts = build_candidate_set(
            [sentence.text for sentence in target_paragraph.sentences],
            positive_index=row.sentence_alignment.target_index,
            distractor_window=distractor_window,
        )
        samples.append(
            {
                "source_text": row.sentence_alignment.source_text,
                "positive_text": row.sentence_alignment.target_text,
                "candidate_texts": candidate_texts,
                "baseline_score": row.sentence_alignment.score,
                "baseline_confidence_label": classify_confidence(row.sentence_alignment.score),
                "source_paragraph_index": row.source_paragraph_index,
                "target_paragraph_index": row.target_paragraph_index,
            }
        )
    return samples


def evaluate_encoder_on_samples(
    *,
    encoder: SentenceEncoder,
    model_name: str,
    samples: list[dict[str, object]],
) -> dict[str, object]:
    if not samples:
        return {
            "model_name": model_name,
            "sample_count": 0,
            "top1_accuracy": 0.0,
            "mrr": 0.0,
            "hard_case_count": 0,
            "hard_case_top1_accuracy": 0.0,
            "sample_results": [],
        }

    sample_results: list[dict[str, object]] = []
    top1_hits = 0
    reciprocal_ranks: list[float] = []
    hard_case_hits = 0
    hard_case_total = 0

    for sample in samples:
        source_vec = encoder.encode([str(sample["source_text"])])[0]
        candidate_texts = [str(text) for text in sample["candidate_texts"]]
        candidate_vecs = encoder.encode(candidate_texts)
        scored = sorted(
            (
                {
                    "text": text,
                    "score": round(_cosine_similarity(source_vec, vector), 6),
                }
                for text, vector in zip(candidate_texts, candidate_vecs, strict=True)
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        positive_text = str(sample["positive_text"])
        baseline_confidence_label = str(sample.get("baseline_confidence_label", classify_confidence(float(sample.get("baseline_score", 0.0)))))
        rank = next(index for index, item in enumerate(scored, start=1) if item["text"] == positive_text)
        top1 = rank == 1
        if top1:
            top1_hits += 1
        reciprocal_ranks.append(1.0 / rank)
        if baseline_confidence_label == "low":
            hard_case_total += 1
            if top1:
                hard_case_hits += 1
        sample_results.append(
            {
                "source_text": sample["source_text"],
                "positive_text": positive_text,
                "baseline_score": round(float(sample["baseline_score"]), 6),
                "baseline_confidence_label": baseline_confidence_label,
                "rank": rank,
                "top1": top1,
                "top_candidate": scored[0],
                "candidates": scored,
            }
        )

    return {
        "model_name": model_name,
        "sample_count": len(samples),
        "top1_accuracy": round(top1_hits / len(samples), 6),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 6),
        "hard_case_count": hard_case_total,
        "hard_case_top1_accuracy": round((hard_case_hits / hard_case_total), 6) if hard_case_total else 0.0,
        "sample_results": sample_results[:20],
    }


def build_embedding_benchmark_report(
    source_book: CanonicalBook,
    target_book: CanonicalBook,
    *,
    encoder: SentenceEncoder,
    model_name: str,
    chapter_pairs: list[tuple[int, int]] | None = None,
    paragraph_window: int = 3,
    distractor_window: int = 2,
) -> dict[str, object]:
    selected_pairs = chapter_pairs or DEFAULT_ALIGNMENT_BENCHMARK_PAIRS
    pair_reports: list[dict[str, object]] = []
    aggregate_top1: list[float] = []
    aggregate_mrr: list[float] = []
    aggregate_hard_acc: list[float] = []

    for source_index, target_index in selected_pairs:
        samples = build_embedding_eval_samples(
            source_book.chapters[source_index - 1],
            target_book.chapters[target_index - 1],
            paragraph_window=paragraph_window,
            distractor_window=distractor_window,
        )
        report = evaluate_encoder_on_samples(
            encoder=encoder,
            model_name=model_name,
            samples=samples,
        )
        report.update(
            {
                "source_chapter_index": source_index,
                "target_chapter_index": target_index,
                "source_title": source_book.chapters[source_index - 1].title,
                "target_title": target_book.chapters[target_index - 1].title,
            }
        )
        pair_reports.append(report)
        aggregate_top1.append(report["top1_accuracy"])
        aggregate_mrr.append(report["mrr"])
        if report["hard_case_count"] > 0:
            aggregate_hard_acc.append(report["hard_case_top1_accuracy"])

    return {
        "model_name": model_name,
        "chapter_pair_count": len(selected_pairs),
        "pair_reports": pair_reports,
        "global_metrics": {
            "top1_accuracy": round(sum(aggregate_top1) / len(aggregate_top1), 6) if aggregate_top1 else 0.0,
            "mrr": round(sum(aggregate_mrr) / len(aggregate_mrr), 6) if aggregate_mrr else 0.0,
            "hard_case_top1_accuracy": round(sum(aggregate_hard_acc) / len(aggregate_hard_acc), 6) if aggregate_hard_acc else 0.0,
        },
    }


def _hashed_char_vector(text: str, *, dim: int) -> list[float]:
    normalized = slugify_fragment(text).replace("-", "")
    vector = [0.0] * dim
    if not normalized:
        return vector
    grams = _char_ngrams(normalized)
    for gram in grams:
        bucket = hash(gram) % dim
        vector[bucket] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _char_ngrams(text: str, n: int = 3) -> list[str]:
    if len(text) < n:
        return [text] if text else []
    return [text[index : index + n] for index in range(0, len(text) - n + 1)]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True))


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)
