"""Canonical corpus builder over EPUB spine documents."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from szwejk.common.ids import make_book_id, make_chapter_id, make_paragraph_id, make_sentence_id, slugify_fragment
from szwejk.ingest import load_epub_package, parse_xhtml_document
from szwejk.schemas import BookMetadata, CanonicalBook, CanonicalChapter, CanonicalParagraph, CanonicalSentence, SourceFragmentRef

from .text import clean_text, split_sentences


def build_canonical_book(epub_path: str, language: str | None = None) -> CanonicalBook:
    package = load_epub_package(epub_path)
    book_language = language or package.language
    metadata = BookMetadata(
        id=make_book_id(book_language, package.title),
        language=book_language,
        title=package.title,
        source_path=epub_path,
        identifier=package.identifier,
    )

    chapters: list[CanonicalChapter] = []
    for raw_chapter_index, spine_document in enumerate(package.spine_documents, start=1):
        root = parse_xhtml_document(spine_document.body_xml)
        paragraphs = _extract_paragraphs(root, metadata.id, raw_chapter_index, spine_document.href)
        if not paragraphs:
            continue
        if _is_front_matter_document(spine_document, package.title, paragraphs):
            continue
        chapter_index = len(chapters) + 1
        chapter_title = _select_chapter_title(root, spine_document.title, chapter_index)
        chapter_id = make_chapter_id(metadata.id, chapter_index, chapter_title)
        rekeyed_paragraphs = _rekey_paragraphs(paragraphs, chapter_id)
        chapters.append(
            CanonicalChapter(
                id=chapter_id,
                index=chapter_index,
                title=chapter_title,
                label=chapter_title,
                paragraphs=rekeyed_paragraphs,
                source_path=spine_document.href,
            )
        )

    return CanonicalBook(metadata=metadata, chapters=chapters)


def write_canonical_book(book: CanonicalBook, output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(book.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _extract_paragraphs(root: ET.Element, book_id: str, chapter_index: int, source_path: str) -> list[CanonicalParagraph]:
    body_prefix = f"{book_id}-tmp-ch-{chapter_index:03d}"
    paragraphs: list[CanonicalParagraph] = []
    paragraph_index = 0

    for element in root.iter():
        tag = _local_name(element.tag)
        if tag != "p":
            continue
        if _has_ancestor_table(root, element):
            continue
        for segment_ordinal, segment_text in enumerate(_extract_segments_from_p(element)):
            cleaned = clean_text(segment_text)
            if not cleaned or _looks_like_noise(cleaned):
                continue
            paragraph_index += 1
            paragraph_id = make_paragraph_id(body_prefix, paragraph_index)
            sentence_models = [
                CanonicalSentence(
                    id=make_sentence_id(paragraph_id, sentence_index),
                    index=sentence_index,
                    text=sentence_text,
                )
                for sentence_index, sentence_text in enumerate(split_sentences(cleaned), start=1)
            ]
            paragraphs.append(
                CanonicalParagraph(
                    id=paragraph_id,
                    index=paragraph_index,
                    text=cleaned,
                    sentences=sentence_models,
                    source_fragments=[
                        SourceFragmentRef(
                            source_path=source_path,
                            element_tag="p",
                            ordinal=paragraph_index - 1,
                            text_offset_start=0,
                            text_offset_end=len(cleaned),
                        )
                    ],
                )
            )
    return paragraphs


def _select_chapter_title(root: ET.Element, fallback: str, chapter_index: int) -> str:
    headings: list[str] = []
    for element in root.iter():
        tag = _local_name(element.tag)
        if tag not in {"h1", "h2", "h3", "div"}:
            continue
        text = clean_text("".join(element.itertext()))
        if not text:
            continue
        if tag.startswith("h"):
            headings.append(text)
        elif tag == "div" and len(text) < 160:
            if text.isupper() or text[:1].isdigit() or text.startswith("Rozdział") or text.startswith("Rozdzial"):
                headings.append(text)
    for heading in headings:
        if len(heading) > 2:
            return heading
    return clean_text(fallback) or f"Chapter {chapter_index}"


def _extract_segments_from_p(element: ET.Element) -> list[str]:
    segments: list[str] = []
    current: list[str] = []

    if element.text:
        current.append(element.text)

    for child in list(element):
        tag = _local_name(child.tag)
        if tag == "br":
            text = clean_text("".join(current))
            if text:
                segments.append(text)
            current = []
        else:
            child_text = "".join(child.itertext())
            if child_text:
                current.append(child_text)
        if child.tail:
            current.append(child.tail)

    final_text = clean_text("".join(current))
    if final_text:
        segments.append(final_text)

    return segments or [clean_text("".join(element.itertext()))]


def _rekey_paragraphs(paragraphs: list[CanonicalParagraph], chapter_id: str) -> list[CanonicalParagraph]:
    rebuilt: list[CanonicalParagraph] = []
    for paragraph_index, paragraph in enumerate(paragraphs, start=1):
        paragraph_id = make_paragraph_id(chapter_id, paragraph_index)
        rebuilt_sentences = [
            CanonicalSentence(
                id=make_sentence_id(paragraph_id, sentence_index),
                index=sentence_index,
                text=sentence.text,
            )
            for sentence_index, sentence in enumerate(paragraph.sentences, start=1)
        ]
        rebuilt.append(
            CanonicalParagraph(
                id=paragraph_id,
                index=paragraph_index,
                text=paragraph.text,
                sentences=rebuilt_sentences,
                source_fragments=paragraph.source_fragments,
                kind=paragraph.kind,
            )
        )
    return rebuilt


def _has_ancestor_table(root: ET.Element, target: ET.Element) -> bool:
    parent_map = {child: parent for parent in root.iter() for child in parent}
    current = parent_map.get(target)
    while current is not None:
        if _local_name(current.tag) == "table":
            return True
        current = parent_map.get(current)
    return False


def _looks_like_noise(text: str) -> bool:
    lowered = text.lower()
    if "tekst jest własnością publiczną" in lowered or "text je vlastnictvim" in lowered:
        return True
    if len(slugify_fragment(text)) <= 1:
        return True
    return False


def _is_front_matter_document(spine_document: object, book_title: str, paragraphs: list[CanonicalParagraph]) -> bool:
    if not paragraphs:
        return True

    first_text = paragraphs[0].text.lower()
    title_slug = slugify_fragment(getattr(spine_document, "title"))
    book_slug = slugify_fragment(book_title)
    href = getattr(spine_document, "href", "").lower()

    if "c0_" in href and len(paragraphs) <= 1:
        return True
    if len(paragraphs) <= 1 and title_slug == book_slug:
        return True
    if any(token in first_text for token in ("беларуская", "polski", "čeština", "cestina")):
        return True
    return False


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
