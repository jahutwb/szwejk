"""Diff payloads for PL / CS / hybrid chapter documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher


@dataclass(slots=True)
class DiffSegment:
    op: str
    source_text: str
    target_text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "op": self.op,
            "source_text": self.source_text,
            "target_text": self.target_text,
        }


@dataclass(slots=True)
class SentenceDiffRow:
    sentence_index: int
    source_text: str
    hybrid_text: str
    target_text: str
    pl_to_hybrid: list[DiffSegment] = field(default_factory=list)
    hybrid_to_cs: list[DiffSegment] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "sentence_index": self.sentence_index,
            "source_text": self.source_text,
            "hybrid_text": self.hybrid_text,
            "target_text": self.target_text,
            "pl_to_hybrid": [item.to_dict() for item in self.pl_to_hybrid],
            "hybrid_to_cs": [item.to_dict() for item in self.hybrid_to_cs],
        }


def build_hybrid_diff_payload(document: dict[str, object]) -> dict[str, object]:
    paragraph_rows: list[dict[str, object]] = []
    total_changed_sentences = 0

    for paragraph in document.get("paragraphs", []):
        sentence_rows: list[SentenceDiffRow] = []
        changed_sentences = 0
        for sentence in paragraph.get("sentences", []):
            pl_to_hybrid = _build_diff_segments(str(sentence["source_text"]), str(sentence["hybrid_text"]))
            hybrid_to_cs = _build_diff_segments(str(sentence["hybrid_text"]), str(sentence["target_text"]))
            if _has_change(pl_to_hybrid) or _has_change(hybrid_to_cs):
                changed_sentences += 1
            sentence_rows.append(
                SentenceDiffRow(
                    sentence_index=int(sentence["sentence_index"]),
                    source_text=str(sentence["source_text"]),
                    hybrid_text=str(sentence["hybrid_text"]),
                    target_text=str(sentence["target_text"]),
                    pl_to_hybrid=pl_to_hybrid,
                    hybrid_to_cs=hybrid_to_cs,
                )
            )
        total_changed_sentences += changed_sentences
        paragraph_rows.append(
            {
                "source_paragraph_id": paragraph["source_paragraph_id"],
                "target_paragraph_id": paragraph["target_paragraph_id"],
                "source_text": paragraph["source_text"],
                "hybrid_text": paragraph["hybrid_text"],
                "target_text": paragraph["target_text"],
                "changed_sentence_count": changed_sentences,
                "sentences": [row.to_dict() for row in sentence_rows],
            }
        )

    return {
        "source_chapter_index": document["source_chapter_index"],
        "target_chapter_index": document["target_chapter_index"],
        "level_id": document["level_id"],
        "paragraphs": paragraph_rows,
        "summary": {
            "paragraph_count": len(paragraph_rows),
            "changed_sentence_count": total_changed_sentences,
        },
    }


def _build_diff_segments(source_text: str, target_text: str) -> list[DiffSegment]:
    source_tokens = source_text.split()
    target_tokens = target_text.split()
    matcher = SequenceMatcher(a=source_tokens, b=target_tokens)
    segments: list[DiffSegment] = []
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        segments.append(
            DiffSegment(
                op=op,
                source_text=" ".join(source_tokens[a0:a1]),
                target_text=" ".join(target_tokens[b0:b1]),
            )
        )
    return segments


def _has_change(segments: list[DiffSegment]) -> bool:
    return any(segment.op != "equal" for segment in segments)
