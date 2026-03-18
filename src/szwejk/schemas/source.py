"""Source EPUB provenance models."""

from __future__ import annotations

from dataclasses import dataclass

from .base import SchemaModel


@dataclass(slots=True)
class SourceFragmentRef(SchemaModel):
    source_path: str
    element_tag: str
    ordinal: int
    text_offset_start: int = 0
    text_offset_end: int | None = None

    def __post_init__(self) -> None:
        if not self.source_path:
            raise ValueError("source_path must not be empty")
        if not self.element_tag:
            raise ValueError("element_tag must not be empty")
        if self.ordinal < 0:
            raise ValueError("ordinal must be >= 0")
        if self.text_offset_start < 0:
            raise ValueError("text_offset_start must be >= 0")
        if self.text_offset_end is not None and self.text_offset_end < self.text_offset_start:
            raise ValueError("text_offset_end must be >= text_offset_start")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SourceFragmentRef":
        return cls(
            source_path=str(data["source_path"]),
            element_tag=str(data["element_tag"]),
            ordinal=int(data["ordinal"]),
            text_offset_start=int(data.get("text_offset_start", 0)),
            text_offset_end=None if data.get("text_offset_end") is None else int(data["text_offset_end"]),
        )
