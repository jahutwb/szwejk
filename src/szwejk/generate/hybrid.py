"""Materialize hybrid chapter text from a policy chapter plan."""

from __future__ import annotations

from dataclasses import dataclass, field

from szwejk.align.enrichment import build_sentence_enrichment_bundle
from szwejk.align.sentences import align_chapter_sentences, align_chapter_sentences_with_paragraph_blocks
from szwejk.schemas import CanonicalBook

from .corpus_scope import scoped_source_chapter


@dataclass(slots=True)
class HybridSentence:
    sentence_index: int
    source_paragraph_id: str
    target_paragraph_id: str
    source_text: str
    target_text: str
    hybrid_text: str
    actions_applied: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "sentence_index": self.sentence_index,
            "source_paragraph_id": self.source_paragraph_id,
            "target_paragraph_id": self.target_paragraph_id,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "hybrid_text": self.hybrid_text,
            "actions_applied": self.actions_applied,
        }


@dataclass(slots=True)
class HybridParagraph:
    source_paragraph_id: str
    target_paragraph_id: str
    source_paragraph_index: int
    target_paragraph_index: int
    source_text: str
    target_text: str
    hybrid_text: str
    sentences: list[HybridSentence] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_paragraph_id": self.source_paragraph_id,
            "target_paragraph_id": self.target_paragraph_id,
            "source_paragraph_index": self.source_paragraph_index,
            "target_paragraph_index": self.target_paragraph_index,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "hybrid_text": self.hybrid_text,
            "sentences": [item.to_dict() for item in self.sentences],
        }


@dataclass(slots=True)
class HybridChapterDocument:
    source_chapter_index: int
    target_chapter_index: int
    source_title: str
    target_title: str
    level_id: str
    analysis_mode: str
    paragraphs: list[HybridParagraph]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_chapter_index": self.source_chapter_index,
            "target_chapter_index": self.target_chapter_index,
            "source_title": self.source_title,
            "target_title": self.target_title,
            "level_id": self.level_id,
            "analysis_mode": self.analysis_mode,
            "paragraphs": [item.to_dict() for item in self.paragraphs],
        }


def build_hybrid_chapter_document(
    source_book: CanonicalBook,
    target_book: CanonicalBook,
    *,
    source_chapter_index: int,
    target_chapter_index: int,
    plan_payload: dict[str, object],
    paragraph_window: int = 3,
    analysis_mode: str = "stanza",
    paragraph_pair_report: dict[str, object] | None = None,
) -> HybridChapterDocument:
    source_chapter = scoped_source_chapter(source_book.chapters[source_chapter_index - 1])
    target_chapter = target_book.chapters[target_chapter_index - 1]
    if paragraph_pair_report is not None:
        chapter_alignments = align_chapter_sentences_with_paragraph_blocks(
            source_chapter,
            target_chapter,
            paragraph_blocks=list(paragraph_pair_report.get("matched_blocks", [])),
        )
    else:
        chapter_alignments = align_chapter_sentences(
            source_chapter,
            target_chapter,
            paragraph_window=paragraph_window,
        )
    enrichment_bundle = build_sentence_enrichment_bundle(
        chapter_alignments=chapter_alignments,
        source_language=source_book.metadata.language,
        target_language=target_book.metadata.language,
        analysis_mode=analysis_mode,
    )
    actions_by_sentence: dict[int, list[dict[str, object]]] = {}
    for item in plan_payload.get("selected_candidates", []):
        actions_by_sentence.setdefault(int(item["sentence_index"]), []).append(item)

    safe_items = [item for item in enrichment_bundle["items"] if item["enrichment_status"] == "sentence_safe"]
    rendered_rows = _rendered_rows_by_source_sentence(safe_items, actions_by_sentence)

    hybrid_paragraphs: list[HybridParagraph] = []
    target_paragraph_lookup = _target_paragraph_lookup(safe_items)
    for paragraph in source_chapter.paragraphs:
        source_paragraph_id = paragraph.id
        target_info = target_paragraph_lookup.get(source_paragraph_id)
        target_paragraph_id = "" if target_info is None else str(target_info["target_paragraph_id"])
        target_paragraph_index = 0 if target_info is None else int(target_info["target_paragraph_index"])
        target_text = "" if target_info is None else str(_paragraph_text(target_book, target_chapter_index, target_paragraph_index))
        sentences: list[HybridSentence] = []
        for sentence in paragraph.sentences:
            row = rendered_rows.get((source_paragraph_id, sentence.text))
            if row is None:
                sentences.append(
                    HybridSentence(
                        sentence_index=sentence.index,
                        source_paragraph_id=source_paragraph_id,
                        target_paragraph_id=target_paragraph_id,
                        source_text=sentence.text,
                        target_text="",
                        hybrid_text=sentence.text,
                        actions_applied=[],
                    )
                )
                continue
            sentences.append(
                HybridSentence(
                    sentence_index=sentence.index,
                    source_paragraph_id=source_paragraph_id,
                    target_paragraph_id=target_paragraph_id,
                    source_text=str(row["source_text"]),
                    target_text=str(row["target_text"]),
                    hybrid_text=str(row["hybrid_text"]),
                    actions_applied=list(row["actions_applied"]),
                )
            )
        hybrid_paragraphs.append(
            HybridParagraph(
                source_paragraph_id=source_paragraph_id,
                target_paragraph_id=target_paragraph_id,
                source_paragraph_index=paragraph.index,
                target_paragraph_index=target_paragraph_index,
                source_text=paragraph.text,
                target_text=target_text,
                hybrid_text=" ".join(sentence.hybrid_text for sentence in sentences),
                sentences=sentences,
            )
        )

    return HybridChapterDocument(
        source_chapter_index=source_chapter_index,
        target_chapter_index=target_chapter_index,
        source_title=source_chapter.title,
        target_title=target_chapter.title,
        level_id=str(plan_payload["level_id"]),
        analysis_mode=analysis_mode,
        paragraphs=hybrid_paragraphs,
    )


def _apply_actions_to_tokens(tokens: list[dict[str, object]], actions: list[dict[str, object]]) -> str:
    replacements: dict[tuple[int, int], str] = {}
    for action in actions:
        start = int(action["source_span"][0])
        end = int(action["source_span"][1])
        replacements[(start, end)] = str(action["target_text"])

    rendered_parts: list[str] = []
    skip_until = 0
    for token in tokens:
        if token.get("kind") != "word":
            rendered_parts.append(str(token["text"]))
            continue
        token_index = int(token["index"])
        if token_index <= skip_until:
            continue
        matching = [(span, text) for span, text in replacements.items() if span[0] == token_index]
        if matching:
            (start, end), replacement = max(matching, key=lambda item: item[0][1] - item[0][0])
            rendered_parts.append(replacement)
            skip_until = end
            continue
        text = str(token["text"])
        if _should_attach_left(token) and rendered_parts:
            rendered_parts[-1] = f"{rendered_parts[-1]}{text}"
        else:
            rendered_parts.append(text)
    return _normalize_rendered_text(rendered_parts)


def _normalize_rendered_text(parts: list[str]) -> str:
    text = " ".join(part for part in parts if part)
    text = text.replace(" ,", ",").replace(" .", ".").replace(" ;", ";").replace(" :", ":")
    text = text.replace(" !", "!").replace(" ?", "?").replace(" )", ")").replace("( ", "(")
    text = text.replace("” ", "”").replace("“ ", "“")
    return " ".join(text.split())


def _should_attach_left(token: dict[str, object]) -> bool:
    if token.get("kind") != "word":
        return False
    feats = str(token.get("feats") or "")
    deprel = str(token.get("deprel") or "")
    upos = str(token.get("upos") or "")
    if "Variant=Short" in feats and upos in {"AUX", "PART"}:
        return True
    if deprel == "aux:clitic":
        return True
    if upos in {"PART", "AUX"} and len(str(token.get("text", ""))) <= 2:
        return True
    return False


def _paragraph_text(book: CanonicalBook, chapter_index: int, paragraph_index: int) -> str:
    return book.chapters[chapter_index - 1].paragraphs[paragraph_index - 1].text


def _rendered_rows_by_source_sentence(
    safe_items: list[dict[str, object]],
    actions_by_sentence: dict[int, list[dict[str, object]]],
) -> dict[tuple[str, str], dict[str, object]]:
    rendered: dict[tuple[str, str], dict[str, object]] = {}
    for sentence_index, item in enumerate(safe_items, start=1):
        actions = sorted(actions_by_sentence.get(sentence_index, []), key=lambda row: int(row["source_span"][0]), reverse=True)
        hybrid_text = _apply_actions_to_tokens(item["source_tokens"], actions)
        rendered[(str(item["source_paragraph_id"]), str(item["sentence_alignment"]["source_text"]))] = {
            "source_text": str(item["sentence_alignment"]["source_text"]),
            "target_text": str(item["sentence_alignment"]["target_text"]),
            "hybrid_text": hybrid_text,
            "actions_applied": [
                {
                    "granularity": action["granularity"],
                    "source_span": action["source_span"],
                    "target_span": action["target_span"],
                    "source_text": action["source_text"],
                    "target_text": action["target_text"],
                }
                for action in actions
            ],
            "target_paragraph_id": str(item["target_paragraph_id"]),
            "target_paragraph_index": int(item["target_paragraph_index"]),
        }
    return rendered


def _target_paragraph_lookup(safe_items: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    lookup: dict[str, dict[str, object]] = {}
    for item in safe_items:
        source_paragraph_id = str(item["source_paragraph_id"])
        lookup.setdefault(
            source_paragraph_id,
            {
                "target_paragraph_id": str(item["target_paragraph_id"]),
                "target_paragraph_index": int(item["target_paragraph_index"]),
            },
        )
    return lookup
