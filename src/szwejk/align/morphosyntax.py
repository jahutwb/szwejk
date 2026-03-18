"""Morphological and dependency analysis adapters for PL/CS alignment."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from szwejk.common.ids import slugify_fragment


TOKEN_PATTERN = re.compile(r"\d+|[^\W\d_]+|[^\w\s]", flags=re.UNICODE)
STOPWORDS = {
    "pl": {"a", "ale", "bo", "by", "do", "i", "jak", "jest", "na", "nie", "o", "od", "po", "się", "to", "w", "z", "że"},
    "cs": {"a", "ale", "by", "do", "i", "jak", "je", "na", "ne", "o", "od", "po", "se", "si", "to", "v", "z", "že", "už"},
}
HEURISTIC_SUFFIXES = {
    "pl": ("ami", "ach", "ego", "emu", "owie", "owi", "owa", "owe", "iem", "ych", "om", "em", "ie", "y", "a", "ę", "ą"),
    "cs": ("ami", "ech", "ich", "ého", "emu", "ové", "ovi", "ova", "ove", "em", "ou", "ím", "ý", "á", "é", "y", "a", "u"),
}
DEFAULT_STANZA_CACHE_DIR = Path("data/cache/stanza")
DEFAULT_STANZA_CACHE_FILES = {
    "pl": "pl_comparable_snapshot.json",
    "cs": "cs_comparable_snapshot.json",
}


@dataclass(slots=True)
class SentenceAnalysis:
    mode: str
    language: str
    tokens: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {"mode": self.mode, "language": self.language, "tokens": self.tokens}


def analyze_sentence(text: str, *, language: str, mode: str = "heuristic") -> SentenceAnalysis:
    return _analyze_sentence_cached(text, language, mode)


@lru_cache(maxsize=20000)
def _analyze_sentence_cached(text: str, language: str, mode: str) -> SentenceAnalysis:
    if mode == "stanza":
        persisted_tokens = _lookup_persisted_stanza_analysis(text, language)
        if persisted_tokens is not None:
            return SentenceAnalysis(mode="stanza", language=language, tokens=persisted_tokens)
        analyzer = _get_stanza_pipeline(language)
        if analyzer is not None:
            return SentenceAnalysis(mode="stanza", language=language, tokens=_analyze_with_stanza(text, language=language))
    return SentenceAnalysis(mode="heuristic", language=language, tokens=_analyze_heuristically(text, language=language))


def tokenize_text(text: str, *, language: str, mode: str = "heuristic") -> list[dict[str, object]]:
    return analyze_sentence(text, language=language, mode=mode).tokens


def stanza_available(language: str) -> bool:
    return _get_stanza_pipeline(language) is not None


def _analyze_heuristically(text: str, *, language: str) -> list[dict[str, object]]:
    tokens: list[dict[str, object]] = []
    for index, match in enumerate(TOKEN_PATTERN.finditer(text), start=1):
        raw = match.group(0)
        kind = _classify_token(raw)
        normalized = _normalize_token(raw, kind=kind)
        lemma = _heuristic_lemma(raw, language=language, kind=kind)
        tokens.append(
            {
                "index": index,
                "text": raw,
                "kind": kind,
                "normalized": normalized,
                "lemma": lemma,
                "lemma_source": "heuristic",
                "is_stopword": normalized in STOPWORDS.get(language, set()) if kind == "word" else False,
                "upos": None,
                "feats": None,
                "head": None,
                "deprel": None,
                "analysis_source": "heuristic",
            }
        )
    return tokens


def _analyze_with_stanza(text: str, *, language: str) -> list[dict[str, object]]:
    pipeline = _get_stanza_pipeline(language)
    if pipeline is None:
        return _analyze_heuristically(text, language=language)

    doc = pipeline(text)
    tokens: list[dict[str, object]] = []
    running_index = 1
    for sentence in doc.sentences:
        for word in sentence.words:
            normalized = _normalize_token(word.text, kind=_classify_token(word.text))
            kind = _classify_token(word.text)
            tokens.append(
                {
                    "index": running_index,
                    "text": word.text,
                    "kind": kind,
                    "normalized": normalized,
                    "lemma": _normalize_token(word.lemma or word.text, kind="word" if kind == "word" else kind),
                    "lemma_source": "stanza",
                    "is_stopword": normalized in STOPWORDS.get(language, set()) if kind == "word" else False,
                    "upos": word.upos,
                    "feats": word.feats,
                    "head": int(word.head) if word.head is not None else None,
                    "deprel": word.deprel,
                    "analysis_source": "stanza",
                }
            )
            running_index += 1
    return tokens


def _classify_token(token: str) -> str:
    if token.isdigit():
        return "number"
    if any(char.isalpha() for char in token):
        return "word"
    return "punct"


def _normalize_token(token: str, *, kind: str) -> str:
    if kind == "punct":
        return token
    return slugify_fragment(token).replace("-", "")


def _heuristic_lemma(token: str, *, language: str, kind: str) -> str:
    if kind != "word":
        return _normalize_token(token, kind=kind)
    normalized = _normalize_token(token, kind=kind)
    for suffix in HEURISTIC_SUFFIXES.get(language, ()):
        if normalized.endswith(suffix) and len(normalized) - len(suffix) >= 3:
            return normalized[: -len(suffix)]
    return normalized


@lru_cache(maxsize=4)
def _get_stanza_pipeline(language: str):
    try:
        import stanza
    except Exception:
        return None

    try:
        return stanza.Pipeline(
            lang=language,
            processors="tokenize,pos,lemma,depparse",
            tokenize_no_ssplit=True,
            download_method=None,
            verbose=False,
        )
    except Exception:
        return None


def _lookup_persisted_stanza_analysis(text: str, language: str) -> list[dict[str, object]] | None:
    snapshot = _load_persisted_stanza_snapshot(language)
    if snapshot is None:
        return None
    return snapshot.get(text)


@lru_cache(maxsize=4)
def _load_persisted_stanza_snapshot(language: str) -> dict[str, list[dict[str, object]]] | None:
    cache_path = _stanza_cache_path(language)
    if cache_path is None or not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    lookup: dict[str, list[dict[str, object]]] = {}
    for entry in entries:
        text = str(entry["text"])
        lookup[text] = list(entry["tokens"])
    return lookup


def _stanza_cache_path(language: str) -> Path | None:
    filename = DEFAULT_STANZA_CACHE_FILES.get(language)
    if filename is None:
        return None
    cache_dir = Path(os.environ.get("SZWJK_STANZA_CACHE_DIR", str(DEFAULT_STANZA_CACHE_DIR)))
    return cache_dir / filename
