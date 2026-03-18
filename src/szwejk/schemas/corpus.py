"""Canonical corpus models shared by intake and alignment stages."""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import SchemaModel
from .source import SourceFragmentRef


@dataclass(slots=True)
class CanonicalSentence(SchemaModel):
    id: str
    index: int
    text: str

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("sentence id must not be empty")
        if self.index < 0:
            raise ValueError("sentence index must be >= 0")
        if not self.text.strip():
            raise ValueError("sentence text must not be blank")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CanonicalSentence":
        return cls(id=str(data["id"]), index=int(data["index"]), text=str(data["text"]))


@dataclass(slots=True)
class CanonicalParagraph(SchemaModel):
    id: str
    index: int
    text: str
    sentences: list[CanonicalSentence] = field(default_factory=list)
    source_fragments: list[SourceFragmentRef] = field(default_factory=list)
    kind: str = "paragraph"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("paragraph id must not be empty")
        if self.index < 0:
            raise ValueError("paragraph index must be >= 0")
        if not self.text.strip():
            raise ValueError("paragraph text must not be blank")
        if not self.sentences:
            raise ValueError("paragraph must contain at least one sentence")
        if not self.source_fragments:
            raise ValueError("paragraph must reference at least one source fragment")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CanonicalParagraph":
        return cls(
            id=str(data["id"]),
            index=int(data["index"]),
            text=str(data["text"]),
            sentences=[CanonicalSentence.from_dict(item) for item in data.get("sentences", [])],
            source_fragments=[SourceFragmentRef.from_dict(item) for item in data.get("source_fragments", [])],
            kind=str(data.get("kind", "paragraph")),
        )


@dataclass(slots=True)
class CanonicalChapter(SchemaModel):
    id: str
    index: int
    title: str
    paragraphs: list[CanonicalParagraph] = field(default_factory=list)
    label: str | None = None
    source_path: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("chapter id must not be empty")
        if self.index < 0:
            raise ValueError("chapter index must be >= 0")
        if not self.title.strip():
            raise ValueError("chapter title must not be blank")
        if not self.paragraphs:
            raise ValueError("chapter must contain at least one paragraph")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CanonicalChapter":
        return cls(
            id=str(data["id"]),
            index=int(data["index"]),
            title=str(data["title"]),
            paragraphs=[CanonicalParagraph.from_dict(item) for item in data.get("paragraphs", [])],
            label=None if data.get("label") is None else str(data["label"]),
            source_path=None if data.get("source_path") is None else str(data["source_path"]),
        )


@dataclass(slots=True)
class BookMetadata(SchemaModel):
    id: str
    language: str
    title: str
    source_path: str
    identifier: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("book id must not be empty")
        if not self.language:
            raise ValueError("language must not be empty")
        if not self.title.strip():
            raise ValueError("title must not be blank")
        if not self.source_path:
            raise ValueError("source_path must not be empty")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "BookMetadata":
        return cls(
            id=str(data["id"]),
            language=str(data["language"]),
            title=str(data["title"]),
            source_path=str(data["source_path"]),
            identifier=None if data.get("identifier") is None else str(data["identifier"]),
        )


@dataclass(slots=True)
class CanonicalBook(SchemaModel):
    metadata: BookMetadata
    chapters: list[CanonicalChapter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.chapters:
            raise ValueError("book must contain at least one chapter")

    @property
    def chapter_count(self) -> int:
        return len(self.chapters)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CanonicalBook":
        return cls(
            metadata=BookMetadata.from_dict(data["metadata"]),
            chapters=[CanonicalChapter.from_dict(item) for item in data.get("chapters", [])],
        )
