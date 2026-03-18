"""Whole-book hybrid package driven by global linear progression."""

from __future__ import annotations

from dataclasses import dataclass, field

from szwejk.align import (
    HeuristicEmbeddingEncoder,
    SentenceTransformersEncoder,
    build_book_paragraph_alignment_report,
    build_monotonic_chapter_alignment,
)
from szwejk.schemas import CanonicalBook
from szwejk.policy import HybridizationPolicy

from .corpus_scope import apply_default_tail_chapter_overrides
from .hybrid import HybridChapterDocument, build_hybrid_chapter_document
from .notes import build_didactic_note_bundle
from .progression import build_progression_report


GLOBAL_LINEAR_LEVEL_ID = "global-linear-progression"


@dataclass(slots=True)
class HybridBookChapter:
    sequence_index: int
    source_chapter_index: int
    target_chapter_index: int
    level_id: str
    introduced_family_count: int
    document: HybridChapterDocument
    notes: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence_index": self.sequence_index,
            "source_chapter_index": self.source_chapter_index,
            "target_chapter_index": self.target_chapter_index,
            "level_id": self.level_id,
            "introduced_family_count": self.introduced_family_count,
            "document": self.document.to_dict(),
            "notes": self.notes,
        }


@dataclass(slots=True)
class HybridBookPackage:
    policy_id: str
    analysis_mode: str
    matched_pair_count: int
    progression_mode: str
    paragraph_skeleton_model: str
    chapters: list[HybridBookChapter] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "analysis_mode": self.analysis_mode,
            "matched_pair_count": self.matched_pair_count,
            "progression_mode": self.progression_mode,
            "paragraph_skeleton_model": self.paragraph_skeleton_model,
            "chapters": [item.to_dict() for item in self.chapters],
        }


def build_hybrid_book_package(
    source_book: CanonicalBook,
    target_book: CanonicalBook,
    *,
    policy: HybridizationPolicy,
    chapter_pairs: list[tuple[int, int]] | None = None,
    paragraph_window: int = 3,
    analysis_mode: str = "stanza",
    min_match_score: float = 0.72,
    skip_penalty: float = 0.2,
    paragraph_model: str = "heuristic",
    paragraph_device: str = "cpu",
) -> HybridBookPackage:
    selected_pairs = chapter_pairs or _pairs_from_monotonic_alignment(
        source_book,
        target_book,
        min_match_score=min_match_score,
        skip_penalty=skip_penalty,
    )
    paragraph_encoder = (
        HeuristicEmbeddingEncoder()
        if paragraph_model == "heuristic"
        else SentenceTransformersEncoder(paragraph_model, device=paragraph_device)
    )
    paragraph_report = build_book_paragraph_alignment_report(
        source_book,
        target_book,
        encoder=paragraph_encoder,
        model_name=paragraph_model,
        chapter_pairs=selected_pairs,
        min_match_score=min_match_score,
        skip_penalty=skip_penalty,
    )
    paragraph_pair_reports = {
        (int(item["source_chapter_index"]), int(item["target_chapter_index"])): item
        for item in paragraph_report["pair_reports"]
    }
    progression = build_progression_report(
        source_book,
        target_book,
        chapter_pairs=selected_pairs,
        paragraph_window=paragraph_window,
        analysis_mode=analysis_mode,
        paragraph_pair_reports=paragraph_pair_reports,
    )

    steps_by_pair: dict[tuple[int, int], list[dict[str, object]]] = {}
    introduced_families: set[str] = set()
    introduced_count_by_pair: dict[tuple[int, int], int] = {}
    for step in progression["steps"]:
        chapter_pair = (int(step["chapter_pair"][0]), int(step["chapter_pair"][1]))
        steps_by_pair.setdefault(chapter_pair, []).append(step)
        introduced_families.update(str(item) for item in step.get("new_families", []))
        introduced_count_by_pair[chapter_pair] = len(introduced_families)

    seen_note_keys: set[tuple[str, str, str]] = set()
    chapters: list[HybridBookChapter] = []
    for offset, (source_chapter_index, target_chapter_index) in enumerate(selected_pairs, start=1):
        chapter_pair = (source_chapter_index, target_chapter_index)
        chapter_steps = steps_by_pair.get(chapter_pair, [])
        plan_payload = _plan_payload_from_progression_steps(
            source_chapter_index=source_chapter_index,
            target_chapter_index=target_chapter_index,
            source_title=source_book.chapters[source_chapter_index - 1].title,
            target_title=target_book.chapters[target_chapter_index - 1].title,
            chapter_steps=chapter_steps,
        )
        document = build_hybrid_chapter_document(
            source_book,
            target_book,
            source_chapter_index=source_chapter_index,
            target_chapter_index=target_chapter_index,
            plan_payload=plan_payload,
            paragraph_window=paragraph_window,
            analysis_mode=analysis_mode,
            paragraph_pair_report=paragraph_pair_reports.get(chapter_pair),
        )
        note_bundle = build_didactic_note_bundle(document.to_dict())
        chapter_notes = []
        for note in note_bundle["notes"]:
            key = (
                str(note["granularity"]),
                str(note["source_text"]).strip().lower(),
                str(note["target_text"]).strip().lower(),
            )
            if key in seen_note_keys:
                continue
            seen_note_keys.add(key)
            chapter_notes.append(note)
        chapters.append(
            HybridBookChapter(
                sequence_index=offset,
                source_chapter_index=source_chapter_index,
                target_chapter_index=target_chapter_index,
                level_id=GLOBAL_LINEAR_LEVEL_ID,
                introduced_family_count=introduced_count_by_pair.get(chapter_pair, len(introduced_families)),
                document=document,
                notes=chapter_notes,
            )
        )

    return HybridBookPackage(
        policy_id=policy.id,
        analysis_mode=analysis_mode,
        matched_pair_count=len(selected_pairs),
        progression_mode="paragraph-skeleton-global-linear-3d",
        paragraph_skeleton_model=paragraph_model,
        chapters=chapters,
    )


def _plan_payload_from_progression_steps(
    *,
    source_chapter_index: int,
    target_chapter_index: int,
    source_title: str,
    target_title: str,
    chapter_steps: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "policy_id": GLOBAL_LINEAR_LEVEL_ID,
        "level_id": GLOBAL_LINEAR_LEVEL_ID,
        "source_chapter_index": source_chapter_index,
        "target_chapter_index": target_chapter_index,
        "source_title": source_title,
        "target_title": target_title,
        "selected_candidates": [
            {
                "granularity": step["granularity"],
                "sentence_index": int(step["sentence_index"]),
                "source_span": list(step["source_span"]),
                "target_span": list(step["target_span"]),
                "source_text": step["source_text"],
                "target_text": step["target_text"],
                "family_id": "::".join(str(item) for item in step.get("new_families", [])) if step.get("new_families") else None,
            }
            for step in chapter_steps
        ],
        "blocked_candidates": [],
    }


def _pairs_from_monotonic_alignment(
    source_book: CanonicalBook,
    target_book: CanonicalBook,
    *,
    min_match_score: float,
    skip_penalty: float,
) -> list[tuple[int, int]]:
    alignment = build_monotonic_chapter_alignment(
        source_book,
        target_book,
        min_match_score=min_match_score,
        skip_penalty=skip_penalty,
    )
    pairs = [
        (int(item["source_chapter_index"]), int(item["target_chapter_index"]))
        for item in alignment["matches"]
    ]
    return apply_default_tail_chapter_overrides(
        pairs,
        source_chapter_count=source_book.chapter_count,
        target_chapter_count=target_book.chapter_count,
    )
