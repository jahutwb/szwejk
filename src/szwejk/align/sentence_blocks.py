"""Sentence-span alignment inside paragraph blocks with optional embedding support."""

from __future__ import annotations

from dataclasses import dataclass

from szwejk.align.embedding_eval import SentenceEncoder
from szwejk.align.morphosyntax import analyze_sentence
from szwejk.align.sentences import (
    ChapterSentenceSpanAlignment,
    SentenceSpanAlignment,
    _char_trigram_jaccard,
    _digit_pattern_similarity,
    _length_similarity,
    punctuation_shape_similarity,
)
from szwejk.schemas import CanonicalChapter, CanonicalParagraph


@dataclass(slots=True)
class SentenceBlockItem:
    seq: int
    paragraph_id: str
    paragraph_index: int
    sentence_id: str
    sentence_index: int
    text: str


def align_chapter_sentence_spans_with_paragraph_blocks(
    source_chapter: CanonicalChapter,
    target_chapter: CanonicalChapter,
    *,
    paragraph_blocks: list[dict[str, object]],
    source_language: str,
    target_language: str,
    analysis_mode: str = "stanza",
    encoder: SentenceEncoder | None = None,
    max_source_span: int = 2,
    max_target_span: int = 2,
    skip_source_penalty: float = 0.42,
    skip_target_penalty: float = 0.42,
) -> list[ChapterSentenceSpanAlignment]:
    source_lookup = {paragraph.id: paragraph for paragraph in source_chapter.paragraphs}
    target_lookup = {paragraph.id: paragraph for paragraph in target_chapter.paragraphs}
    rows: list[ChapterSentenceSpanAlignment] = []

    for block in paragraph_blocks:
        source_paragraphs = _paragraphs_from_refs(source_lookup, block.get("source_refs", []))
        target_paragraphs = _paragraphs_from_refs(target_lookup, block.get("target_refs", []))
        if not source_paragraphs or not target_paragraphs:
            continue
        source_items = _flatten_sentence_items(source_paragraphs)
        target_items = _flatten_sentence_items(target_paragraphs)
        span_alignments = align_sentence_block_spans(
            source_items,
            target_items,
            source_language=source_language,
            target_language=target_language,
            analysis_mode=analysis_mode,
            encoder=encoder,
            max_source_span=max_source_span,
            max_target_span=max_target_span,
            skip_source_penalty=skip_source_penalty,
            skip_target_penalty=skip_target_penalty,
        )
        for alignment in span_alignments:
            source_first = source_items[alignment.source_span[0] - 1]
            target_first = target_items[alignment.target_span[0] - 1]
            source_sentence_ids = [
                source_items[index - 1].sentence_id
                for index in range(alignment.source_span[0], alignment.source_span[1] + 1)
            ]
            target_sentence_ids = [
                target_items[index - 1].sentence_id
                for index in range(alignment.target_span[0], alignment.target_span[1] + 1)
            ]
            rows.append(
                ChapterSentenceSpanAlignment(
                    source_paragraph_id=source_first.paragraph_id,
                    target_paragraph_id=target_first.paragraph_id,
                    source_paragraph_index=source_first.paragraph_index,
                    target_paragraph_index=target_first.paragraph_index,
                    source_sentence_ids=source_sentence_ids,
                    target_sentence_ids=target_sentence_ids,
                    sentence_span_alignment=alignment,
                )
            )
    return rows


def align_sentence_block_spans(
    source_items: list[SentenceBlockItem],
    target_items: list[SentenceBlockItem],
    *,
    source_language: str,
    target_language: str,
    analysis_mode: str,
    encoder: SentenceEncoder | None,
    max_source_span: int,
    max_target_span: int,
    skip_source_penalty: float,
    skip_target_penalty: float,
) -> list[SentenceSpanAlignment]:
    if not source_items or not target_items:
        return []

    embedding_lookup = _sentence_embeddings(source_items, target_items, encoder)
    source_analysis_lookup = {
        item.sentence_id: analyze_sentence(item.text, language=source_language, mode=analysis_mode)
        for item in source_items
    }
    target_analysis_lookup = {
        item.sentence_id: analyze_sentence(item.text, language=target_language, mode=analysis_mode)
        for item in target_items
    }
    dp: dict[tuple[int, int], tuple[float, list[SentenceSpanAlignment]]] = {(0, 0): (0.0, [])}

    for source_done in range(0, len(source_items) + 1):
        for target_done in range(0, len(target_items) + 1):
            state = (source_done, target_done)
            if state not in dp:
                continue
            current_score, current_path = dp[state]
            if source_done < len(source_items):
                next_state = (source_done + 1, target_done)
                candidate = (current_score - skip_source_penalty, current_path)
                previous = dp.get(next_state)
                if previous is None or candidate[0] > previous[0]:
                    dp[next_state] = candidate
            if target_done < len(target_items):
                next_state = (source_done, target_done + 1)
                candidate = (current_score - skip_target_penalty, current_path)
                previous = dp.get(next_state)
                if previous is None or candidate[0] > previous[0]:
                    dp[next_state] = candidate

            for source_span_size in range(1, max_source_span + 1):
                if source_done + source_span_size > len(source_items):
                    break
                for target_span_size in range(1, max_target_span + 1):
                    if target_done + target_span_size > len(target_items):
                        break
                    source_span_items = source_items[source_done : source_done + source_span_size]
                    target_span_items = target_items[target_done : target_done + target_span_size]
                    signals = sentence_block_similarity_signals(
                        source_span_items,
                        target_span_items,
                        source_language=source_language,
                        target_language=target_language,
                        analysis_mode=analysis_mode,
                        embedding_lookup=embedding_lookup,
                        source_analysis_lookup=source_analysis_lookup,
                        target_analysis_lookup=target_analysis_lookup,
                    )
                    alignment = SentenceSpanAlignment(
                        source_span=(source_done + 1, source_done + source_span_size),
                        target_span=(target_done + 1, target_done + target_span_size),
                        source_text=" ".join(item.text for item in source_span_items),
                        target_text=" ".join(item.text for item in target_span_items),
                        score=signals["total"],
                        signals=signals,
                        alignment_type=f"{source_span_size}:{target_span_size}",
                    )
                    next_state = (source_done + source_span_size, target_done + target_span_size)
                    next_score = current_score + (alignment.score * max(source_span_size, target_span_size))
                    previous = dp.get(next_state)
                    if previous is None or next_score > previous[0]:
                        dp[next_state] = (next_score, current_path + [alignment])

    return dp.get((len(source_items), len(target_items)), (0.0, []))[1]


def sentence_block_similarity_signals(
    source_items: list[SentenceBlockItem],
    target_items: list[SentenceBlockItem],
    *,
    source_language: str,
    target_language: str,
    analysis_mode: str,
    embedding_lookup: dict[str, list[float]] | None,
    source_analysis_lookup: dict[str, object],
    target_analysis_lookup: dict[str, object],
) -> dict[str, float]:
    source_text = " ".join(item.text for item in source_items)
    target_text = " ".join(item.text for item in target_items)
    length = _length_similarity(source_text, target_text)
    char_trigram = _char_trigram_jaccard(source_text, target_text)
    punctuation = punctuation_shape_similarity(source_text, target_text)
    digit = _digit_pattern_similarity(source_text, target_text)
    source_tokens = [
        token
        for item in source_items
        for token in source_analysis_lookup[item.sentence_id].tokens
        if token["kind"] == "word"
    ]
    target_tokens = [
        token
        for item in target_items
        for token in target_analysis_lookup[item.sentence_id].tokens
        if token["kind"] == "word"
    ]
    source_lemmas = {str(token["lemma"]) for token in source_tokens}
    target_lemmas = {str(token["lemma"]) for token in target_tokens}
    source_upos = {str(token["upos"]) for token in source_tokens if token.get("upos")}
    target_upos = {str(token["upos"]) for token in target_tokens if token.get("upos")}
    source_deprel = {str(token["deprel"]) for token in source_tokens if token.get("deprel")}
    target_deprel = {str(token["deprel"]) for token in target_tokens if token.get("deprel")}
    lemma_overlap = _set_overlap(source_lemmas, target_lemmas)
    upos_overlap = _set_overlap(source_upos, target_upos)
    dependency_overlap = _set_overlap(source_deprel, target_deprel)
    embedding = _embedding_similarity(source_items, target_items, embedding_lookup)
    total = (
        (0.26 * length)
        + (0.18 * char_trigram)
        + (0.08 * punctuation)
        + (0.03 * digit)
        + (0.17 * lemma_overlap)
        + (0.07 * upos_overlap)
        + (0.07 * dependency_overlap)
        + (0.14 * embedding)
    )
    return {
        "length": length,
        "char_trigram": char_trigram,
        "punctuation": punctuation,
        "digit_pattern": digit,
        "lemma_overlap": lemma_overlap,
        "upos_overlap": upos_overlap,
        "dependency_overlap": dependency_overlap,
        "embedding": embedding,
        "total": total,
    }


def _sentence_embeddings(
    source_items: list[SentenceBlockItem],
    target_items: list[SentenceBlockItem],
    encoder: SentenceEncoder | None,
) -> dict[str, list[float]] | None:
    if encoder is None:
        return None
    texts = ["passage: " + item.text for item in source_items + target_items]
    vectors = encoder.encode(texts)
    lookup: dict[str, list[float]] = {}
    for item, vector in zip(source_items + target_items, vectors, strict=True):
        lookup[item.sentence_id] = [float(value) for value in vector]
    return lookup


def _embedding_similarity(
    source_items: list[SentenceBlockItem],
    target_items: list[SentenceBlockItem],
    lookup: dict[str, list[float]] | None,
) -> float:
    if lookup is None:
        return 0.0
    source_vectors = [lookup[item.sentence_id] for item in source_items if item.sentence_id in lookup]
    target_vectors = [lookup[item.sentence_id] for item in target_items if item.sentence_id in lookup]
    if not source_vectors or not target_vectors:
        return 0.0
    source_mean = _mean_vector(source_vectors)
    target_mean = _mean_vector(target_vectors)
    return sum(left * right for left, right in zip(source_mean, target_mean, strict=True))


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    dim = len(vectors[0])
    summed = [0.0] * dim
    for vector in vectors:
        for index, value in enumerate(vector):
            summed[index] += value
    normalized = [value / len(vectors) for value in summed]
    norm = sum(value * value for value in normalized) ** 0.5
    if norm == 0.0:
        return normalized
    return [value / norm for value in normalized]


def _paragraphs_from_refs(
    lookup: dict[str, CanonicalParagraph],
    refs: list[dict[str, object]],
) -> list[CanonicalParagraph]:
    paragraphs: list[CanonicalParagraph] = []
    for ref in refs:
        paragraph_id = str(ref["paragraph_id"])
        paragraph = lookup.get(paragraph_id)
        if paragraph is not None:
            paragraphs.append(paragraph)
    return paragraphs


def _flatten_sentence_items(paragraphs: list[CanonicalParagraph]) -> list[SentenceBlockItem]:
    items: list[SentenceBlockItem] = []
    for seq, paragraph in enumerate(paragraphs, start=1):
        for sentence in paragraph.sentences:
            items.append(
                SentenceBlockItem(
                    seq=len(items) + 1,
                    paragraph_id=paragraph.id,
                    paragraph_index=paragraph.index,
                    sentence_id=sentence.id,
                    sentence_index=sentence.index,
                    text=sentence.text,
                )
            )
    return items


def _set_overlap(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)

