"""Deterministic sentence-level alignment inside bounded chapter regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from szwejk.align.candidates import generate_paragraph_candidates
from szwejk.align.morphosyntax import analyze_sentence
from szwejk.common.ids import slugify_fragment
from szwejk.schemas import CanonicalChapter, CanonicalParagraph, CanonicalSentence


@dataclass(slots=True)
class SentenceAlignment:
    source_index: int
    target_index: int
    source_text: str
    target_text: str
    score: float
    signals: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_index": self.source_index,
            "target_index": self.target_index,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "score": round(self.score, 6),
            "signals": {key: round(value, 6) for key, value in self.signals.items()},
        }


@dataclass(slots=True)
class ChapterSentenceAlignment:
    source_paragraph_id: str
    target_paragraph_id: str
    source_paragraph_index: int
    target_paragraph_index: int
    sentence_alignment: SentenceAlignment

    def to_dict(self) -> dict[str, object]:
        return {
            "source_paragraph_id": self.source_paragraph_id,
            "target_paragraph_id": self.target_paragraph_id,
            "source_paragraph_index": self.source_paragraph_index,
            "target_paragraph_index": self.target_paragraph_index,
            "sentence_alignment": self.sentence_alignment.to_dict(),
        }


@dataclass(slots=True)
class ChapterSentenceSpanAlignment:
    source_paragraph_id: str
    target_paragraph_id: str
    source_paragraph_index: int
    target_paragraph_index: int
    source_sentence_ids: list[str]
    target_sentence_ids: list[str]
    sentence_span_alignment: SentenceSpanAlignment

    def to_dict(self) -> dict[str, object]:
        return {
            "source_paragraph_id": self.source_paragraph_id,
            "target_paragraph_id": self.target_paragraph_id,
            "source_paragraph_index": self.source_paragraph_index,
            "target_paragraph_index": self.target_paragraph_index,
            "source_sentence_ids": self.source_sentence_ids,
            "target_sentence_ids": self.target_sentence_ids,
            "sentence_span_alignment": self.sentence_span_alignment.to_dict(),
        }


@dataclass(slots=True)
class SentenceSpanAlignment:
    source_span: tuple[int, int]
    target_span: tuple[int, int]
    source_text: str
    target_text: str
    score: float
    signals: dict[str, float]
    alignment_type: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_span": list(self.source_span),
            "target_span": list(self.target_span),
            "source_text": self.source_text,
            "target_text": self.target_text,
            "score": round(self.score, 6),
            "signals": {key: round(value, 6) for key, value in self.signals.items()},
            "alignment_type": self.alignment_type,
        }


def sentence_similarity_signals(source_text: str, target_text: str) -> dict[str, float]:
    length = _length_similarity(source_text, target_text)
    char_trigram = _char_trigram_jaccard(source_text, target_text)
    punctuation = punctuation_shape_similarity(source_text, target_text)
    digit = _digit_pattern_similarity(source_text, target_text)
    total = (0.35 * length) + (0.40 * char_trigram) + (0.20 * punctuation) + (0.05 * digit)
    return {
        "length": length,
        "char_trigram": char_trigram,
        "punctuation": punctuation,
        "digit_pattern": digit,
        "total": total,
    }


def punctuation_shape_similarity(source_text: str, target_text: str) -> float:
    source_pattern = _punctuation_shape(source_text)
    target_pattern = _punctuation_shape(target_text)
    if not source_pattern and not target_pattern:
        return 1.0
    if not source_pattern or not target_pattern:
        return 0.0
    source_set = set(source_pattern)
    target_set = set(target_pattern)
    return len(source_set & target_set) / len(source_set | target_set)


def align_sentence_sequences(source_sentences: list[str], target_sentences: list[str]) -> list[SentenceAlignment]:
    if not source_sentences or not target_sentences:
        return []

    target_start = 1
    alignments: list[SentenceAlignment] = []

    for source_index, source_text in enumerate(source_sentences, start=1):
        best_alignment: SentenceAlignment | None = None
        remaining_target_slots = len(source_sentences) - source_index
        max_target_index = len(target_sentences) - remaining_target_slots
        for target_index in range(target_start, max_target_index + 1):
            target_text = target_sentences[target_index - 1]
            signals = sentence_similarity_signals(source_text, target_text)
            score = signals["total"]
            candidate = SentenceAlignment(
                source_index=source_index,
                target_index=target_index,
                source_text=source_text,
                target_text=target_text,
                score=score,
                signals=signals,
            )
            if best_alignment is None or candidate.score > best_alignment.score:
                best_alignment = candidate
        if best_alignment is None:
            break
        alignments.append(best_alignment)
        target_start = best_alignment.target_index + 1
        if target_start > len(target_sentences):
            break

    return alignments


def align_sentence_spans(
    source_sentences: list[str],
    target_sentences: list[str],
    *,
    source_language: str = "pl",
    target_language: str = "cs",
    analysis_mode: str = "heuristic",
) -> list[SentenceSpanAlignment]:
    if not source_sentences or not target_sentences:
        return []

    operations = ((1, 1), (1, 2), (2, 1))
    dp: dict[tuple[int, int], tuple[float, list[SentenceSpanAlignment]]] = {(0, 0): (0.0, [])}

    for source_done in range(0, len(source_sentences) + 1):
        for target_done in range(0, len(target_sentences) + 1):
            state = (source_done, target_done)
            if state not in dp:
                continue
            current_score, current_path = dp[state]
            for source_count, target_count in operations:
                next_source = source_done + source_count
                next_target = target_done + target_count
                if next_source > len(source_sentences) or next_target > len(target_sentences):
                    continue
                source_span = source_sentences[source_done:next_source]
                target_span = target_sentences[target_done:next_target]
                signals = sentence_span_similarity_signals(
                    source_span,
                    target_span,
                    source_language=source_language,
                    target_language=target_language,
                    analysis_mode=analysis_mode,
                )
                coverage_weight = max(source_count, target_count)
                candidate = SentenceSpanAlignment(
                    source_span=(source_done + 1, next_source),
                    target_span=(target_done + 1, next_target),
                    source_text=" ".join(source_span),
                    target_text=" ".join(target_span),
                    score=signals["total"],
                    signals=signals,
                    alignment_type=f"{source_count}:{target_count}",
                )
                next_state = (next_source, next_target)
                next_score = current_score + (candidate.score * coverage_weight)
                previous = dp.get(next_state)
                if previous is None or next_score > previous[0]:
                    dp[next_state] = (next_score, current_path + [candidate])

    return dp.get((len(source_sentences), len(target_sentences)), (0.0, []))[1]


def align_chapter_sentences(
    source_chapter: CanonicalChapter,
    target_chapter: CanonicalChapter,
    *,
    paragraph_window: int = 3,
) -> list[ChapterSentenceAlignment]:
    paragraph_candidates = generate_paragraph_candidates(source_chapter, target_chapter, window=paragraph_window, top_k=1)
    target_lookup = {paragraph.id: paragraph for paragraph in target_chapter.paragraphs}

    chapter_alignments: list[ChapterSentenceAlignment] = []
    for source_paragraph in source_chapter.paragraphs:
        top_candidates = paragraph_candidates.get(source_paragraph.id, [])
        if not top_candidates:
            continue
        target_paragraph = target_lookup[top_candidates[0].target_paragraph_id]
        sentence_alignments = align_sentence_sequences(
            [sentence.text for sentence in source_paragraph.sentences],
            [sentence.text for sentence in target_paragraph.sentences],
        )
        for sentence_alignment in sentence_alignments:
            chapter_alignments.append(
                ChapterSentenceAlignment(
                    source_paragraph_id=source_paragraph.id,
                    target_paragraph_id=target_paragraph.id,
                    source_paragraph_index=source_paragraph.index,
                    target_paragraph_index=target_paragraph.index,
                    sentence_alignment=sentence_alignment,
                )
            )
    return chapter_alignments


def align_chapter_sentences_with_paragraph_blocks(
    source_chapter: CanonicalChapter,
    target_chapter: CanonicalChapter,
    *,
    paragraph_blocks: list[dict[str, object]],
) -> list[ChapterSentenceAlignment]:
    source_lookup = {paragraph.id: paragraph for paragraph in source_chapter.paragraphs}
    target_lookup = {paragraph.id: paragraph for paragraph in target_chapter.paragraphs}
    chapter_alignments: list[ChapterSentenceAlignment] = []

    for block in paragraph_blocks:
        source_paragraphs = _paragraphs_from_refs(source_lookup, block.get("source_refs", []))
        target_paragraphs = _paragraphs_from_refs(target_lookup, block.get("target_refs", []))
        if not source_paragraphs or not target_paragraphs:
            continue
        for source_paragraph in source_paragraphs:
            target_paragraph = _best_target_paragraph_within_block(source_paragraph, target_paragraphs)
            sentence_alignments = align_sentence_sequences(
                [sentence.text for sentence in source_paragraph.sentences],
                [sentence.text for sentence in target_paragraph.sentences],
            )
            for sentence_alignment in sentence_alignments:
                chapter_alignments.append(
                    ChapterSentenceAlignment(
                        source_paragraph_id=source_paragraph.id,
                        target_paragraph_id=target_paragraph.id,
                        source_paragraph_index=source_paragraph.index,
                        target_paragraph_index=target_paragraph.index,
                        sentence_alignment=sentence_alignment,
                    )
                )
    return chapter_alignments


def align_chapter_sentence_spans(
    source_chapter: CanonicalChapter,
    target_chapter: CanonicalChapter,
    *,
    paragraph_window: int = 3,
    source_language: str = "pl",
    target_language: str = "cs",
    analysis_mode: str = "heuristic",
) -> list[ChapterSentenceSpanAlignment]:
    paragraph_candidates = generate_paragraph_candidates(source_chapter, target_chapter, window=paragraph_window, top_k=1)
    target_lookup = {paragraph.id: paragraph for paragraph in target_chapter.paragraphs}

    chapter_alignments: list[ChapterSentenceSpanAlignment] = []
    for source_paragraph in source_chapter.paragraphs:
        top_candidates = paragraph_candidates.get(source_paragraph.id, [])
        if not top_candidates:
            continue
        target_paragraph = target_lookup[top_candidates[0].target_paragraph_id]
        sentence_alignments = align_sentence_spans(
            [sentence.text for sentence in source_paragraph.sentences],
            [sentence.text for sentence in target_paragraph.sentences],
            source_language=source_language,
            target_language=target_language,
            analysis_mode=analysis_mode,
        )
        for sentence_alignment in sentence_alignments:
            source_sentence_ids = [
                source_paragraph.sentences[index - 1].id
                for index in range(sentence_alignment.source_span[0], sentence_alignment.source_span[1] + 1)
            ]
            target_sentence_ids = [
                target_paragraph.sentences[index - 1].id
                for index in range(sentence_alignment.target_span[0], sentence_alignment.target_span[1] + 1)
            ]
            chapter_alignments.append(
                ChapterSentenceSpanAlignment(
                    source_paragraph_id=source_paragraph.id,
                    target_paragraph_id=target_paragraph.id,
                    source_paragraph_index=source_paragraph.index,
                    target_paragraph_index=target_paragraph.index,
                    source_sentence_ids=source_sentence_ids,
                    target_sentence_ids=target_sentence_ids,
                    sentence_span_alignment=sentence_alignment,
                )
            )
    return chapter_alignments


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


def _flatten_sentences(paragraphs: list[CanonicalParagraph]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for paragraph in paragraphs:
        for sentence in paragraph.sentences:
            payload.append(
                {
                    "paragraph_id": paragraph.id,
                    "paragraph_index": paragraph.index,
                    "sentence": sentence,
                }
            )
    return payload


def _best_target_paragraph_within_block(
    source_paragraph: CanonicalParagraph,
    target_paragraphs: list[CanonicalParagraph],
) -> CanonicalParagraph:
    best = target_paragraphs[0]
    best_score = -1.0
    target_total = len(target_paragraphs)
    for offset, target_paragraph in enumerate(target_paragraphs, start=1):
        position_signal = 1.0 - abs(((offset - 1) / max(target_total - 1, 1)) - 0.0)
        length_signal = _length_similarity(source_paragraph.text, target_paragraph.text)
        trigram_signal = _char_trigram_jaccard(source_paragraph.text, target_paragraph.text)
        score = (0.15 * position_signal) + (0.35 * length_signal) + (0.50 * trigram_signal)
        if score > best_score:
            best = target_paragraph
            best_score = score
    return best


def sentence_span_similarity_signals(
    source_span: list[str],
    target_span: list[str],
    *,
    source_language: str,
    target_language: str,
    analysis_mode: str,
) -> dict[str, float]:
    source_text = " ".join(source_span)
    target_text = " ".join(target_span)
    base = sentence_similarity_signals(source_text, target_text)
    source_analysis = analyze_sentence(source_text, language=source_language, mode=analysis_mode)
    target_analysis = analyze_sentence(target_text, language=target_language, mode=analysis_mode)
    source_lemmas = {str(token["lemma"]) for token in source_analysis.tokens if token["kind"] == "word"}
    target_lemmas = {str(token["lemma"]) for token in target_analysis.tokens if token["kind"] == "word"}
    source_upos = {str(token["upos"]) for token in source_analysis.tokens if token.get("upos")}
    target_upos = {str(token["upos"]) for token in target_analysis.tokens if token.get("upos")}
    source_deprel = {str(token["deprel"]) for token in source_analysis.tokens if token.get("deprel")}
    target_deprel = {str(token["deprel"]) for token in target_analysis.tokens if token.get("deprel")}
    lemma_overlap = _set_overlap(source_lemmas, target_lemmas)
    upos_overlap = _set_overlap(source_upos, target_upos)
    dependency_overlap = _set_overlap(source_deprel, target_deprel)
    total = (
        (0.28 * base["length"])
        + (0.28 * base["char_trigram"])
        + (0.14 * base["punctuation"])
        + (0.05 * base["digit_pattern"])
        + (0.15 * lemma_overlap)
        + (0.05 * upos_overlap)
        + (0.05 * dependency_overlap)
    )
    return {
        **base,
        "lemma_overlap": lemma_overlap,
        "upos_overlap": upos_overlap,
        "dependency_overlap": dependency_overlap,
        "total": total,
    }


def _length_similarity(source_text: str, target_text: str) -> float:
    source_len = len(source_text)
    target_len = len(target_text)
    if source_len <= 0 or target_len <= 0:
        return 0.0
    return min(source_len, target_len) / max(source_len, target_len)


def _digit_pattern_similarity(source_text: str, target_text: str) -> float:
    source_digits = "".join(ch for ch in source_text if ch.isdigit())
    target_digits = "".join(ch for ch in target_text if ch.isdigit())
    if not source_digits and not target_digits:
        return 1.0
    return 1.0 if source_digits == target_digits else 0.0


def _punctuation_shape(text: str) -> str:
    classes = []
    mapping = {
        "„": "Q",
        "”": "Q",
        "\"": "Q",
        "'": "Q",
        "—": "D",
        "-": "D",
        ",": "C",
        ".": "P",
        "!": "E",
        "?": "QMARK",
        ";": "S",
        ":": "COL",
        "(": "B",
        ")": "B",
    }
    for ch in text:
        token = mapping.get(ch)
        if token:
            classes.append(token)
    return "|".join(classes)


def _char_trigram_jaccard(left: str, right: str) -> float:
    left_grams = set(_char_ngrams(left))
    right_grams = set(_char_ngrams(right))
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def _char_ngrams(text: str, n: int = 3) -> Iterable[str]:
    slug = slugify_fragment(text).replace("-", "")
    if len(slug) < n:
        if slug:
            yield slug
        return
    for index in range(0, len(slug) - n + 1):
        yield slug[index : index + n]


def _set_overlap(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
