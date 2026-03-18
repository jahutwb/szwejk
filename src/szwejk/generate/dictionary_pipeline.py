"""Dictionary extraction and enrichment for the hybrid Polish-Czech book."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from szwejk.generate.debug_subset import DebugSubstitution
from szwejk.generate.notes import detect_foreign_explanation


DEFAULT_TRANSLATION_CACHE_PATH = Path("data/cache/cs_pl_translations.json")
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_PUNCTUATION_RE = re.compile(r"[.!?;:]{1,}")
_NON_LETTER_RE = re.compile(r"[\W_]+", re.UNICODE)
_CZECH_ALPHABET_ORDER = [
    "a", "á", "b", "c", "č", "d", "ď", "e", "é", "ě", "f", "g", "h", "ch", "i", "í",
    "j", "k", "l", "m", "n", "ň", "o", "ó", "p", "q", "r", "ř", "s", "š", "t", "ť",
    "u", "ú", "ů", "v", "w", "x", "y", "ý", "z", "ž",
]
_CZECH_ORDER_MAP = {item: index for index, item in enumerate(_CZECH_ALPHABET_ORDER)}


@dataclass(slots=True)
class UsedDictionaryEntry:
    czech: str
    polish_fallback: str
    unit_index: int
    granularity: str
    candidate_id: str
    note_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "czech": self.czech,
            "polish_fallback": self.polish_fallback,
            "unit_index": self.unit_index,
            "granularity": self.granularity,
            "candidate_id": self.candidate_id,
            "note_id": self.note_id,
        }


@dataclass(slots=True)
class DictionaryEntry:
    czech: str
    polish: str
    source: str
    unit_index: int
    granularity: str
    polish_fallback: str
    note_id: str | None = None
    explanation: str | None = None
    sort_czech: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "czech": self.czech,
            "polish": self.polish,
            "source": self.source,
            "unit_index": self.unit_index,
            "granularity": self.granularity,
            "polish_fallback": self.polish_fallback,
            "note_id": self.note_id,
            "explanation": self.explanation,
            "sort_czech": self.sort_czech,
        }


def extract_used_entries(
    used_substitutions: list[DebugSubstitution],
    *,
    notes_payload: dict[str, object] | None = None,
) -> list[UsedDictionaryEntry]:
    notes_by_candidate = _notes_by_candidate(notes_payload or {})
    notes_by_id = _notes_by_id(notes_payload or {})
    unique: dict[tuple[str, str], UsedDictionaryEntry] = {}

    for substitution in used_substitutions:
        note = None
        if substitution.note_id:
            note = notes_by_id.get(str(substitution.note_id))
        if note is None:
            note = notes_by_candidate.get(str(substitution.candidate_id))
        czech = normalize_whitespace(substitution.target_text)
        if note is not None:
            anchor = normalize_whitespace(str(note.get("target_anchor_text", "")))
            if _is_cleaner_czech_anchor(anchor, czech):
                czech = anchor
        polish_fallback = normalize_whitespace(substitution.source_text)
        if not czech or not polish_fallback:
            continue
        if not _used_entry_eligible_for_dictionary(substitution=substitution, note=note, czech=czech):
            continue
        key = (normalize_for_compare(czech), normalize_for_compare(polish_fallback))
        if key not in unique:
            unique[key] = UsedDictionaryEntry(
                czech=czech,
                polish_fallback=polish_fallback,
                unit_index=substitution.unit_index,
                granularity=substitution.granularity,
                candidate_id=substitution.candidate_id,
                note_id=substitution.note_id,
            )

    return list(unique.values())


def enrich_dictionary_entries(
    used_entries: list[UsedDictionaryEntry],
    *,
    notes_payload: dict[str, object],
    cache_path: str | Path = DEFAULT_TRANSLATION_CACHE_PATH,
    deepl_api_key: str | None = None,
    libre_url: str | None = None,
) -> list[DictionaryEntry]:
    notes_by_candidate = _notes_by_candidate(notes_payload)
    note_matches = _build_note_matches(notes_payload)
    cache = _load_translation_cache(cache_path)
    translators = _build_translators(deepl_api_key=deepl_api_key, libre_url=libre_url)

    entries: list[DictionaryEntry] = []
    for used_entry in used_entries:
        note = notes_by_candidate.get(used_entry.candidate_id) or note_matches.get(normalize_for_compare(used_entry.czech))
        czech = used_entry.czech
        polish_best = None
        source = "alignment"
        explanation = None
        sort_czech = None

        if note is not None:
            note_czech = normalize_whitespace(
                str(note.get("target_anchor_text", "")) or str(note.get("target_candidate_text", ""))
            )
            if _is_cleaner_czech_anchor(note_czech, czech):
                czech = note_czech
            czech = _dictionary_headword_from_note(note, czech)
            polish_from_note = _best_polish_from_note(note)
            if polish_from_note:
                polish_best = polish_from_note
                source = "note"
                explanation = _dictionary_explanation_from_note(note, czech=czech, polish=polish_best)
                sort_czech = _sort_czech_from_note(note)

        if (not polish_best or _is_weak_polish(polish_best, czech=czech)) and _eligible_for_machine_translation(czech):
            cache_key = normalize_for_compare(czech)
            cached = cache.get(cache_key)
            if cached:
                polish_best = str(cached.get("polish", "")).strip() or polish_best
                source = str(cached.get("source", "alignment"))
            else:
                for translator_name, translator in translators:
                    translated = translator(czech)
                    if translated:
                        polish_best = translated
                        source = translator_name
                        cache[cache_key] = {"polish": translated, "source": translator_name}
                        break

        if not polish_best:
            polish_best = used_entry.polish_fallback
            source = "alignment"

        entries.append(
            DictionaryEntry(
                czech=czech,
                polish=normalize_whitespace(polish_best),
                source=source,
                unit_index=used_entry.unit_index,
                granularity=used_entry.granularity,
                polish_fallback=used_entry.polish_fallback,
                note_id=used_entry.note_id,
                explanation=explanation,
                sort_czech=sort_czech,
            )
        )

    _write_translation_cache(cache_path, cache)
    deduped = _dedupe_dictionary_entries(entries)
    return sorted(deduped, key=lambda item: czech_sort_key(item.sort_czech or item.czech))


def build_dictionary_entries(
    *,
    used_substitutions: list[DebugSubstitution],
    notes_payload: dict[str, object],
    cache_path: str | Path = DEFAULT_TRANSLATION_CACHE_PATH,
    deepl_api_key: str | None = None,
    libre_url: str | None = None,
) -> tuple[list[UsedDictionaryEntry], list[DictionaryEntry]]:
    used_entries = extract_used_entries(used_substitutions, notes_payload=notes_payload)
    final_entries = enrich_dictionary_entries(
        used_entries,
        notes_payload=notes_payload,
        cache_path=cache_path,
        deepl_api_key=deepl_api_key,
        libre_url=libre_url,
    )
    return used_entries, final_entries


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(text).strip())


def normalize_for_compare(text: str) -> str:
    return normalize_whitespace(text).casefold()


def normalize_for_similarity(text: str) -> str:
    lowered = normalize_for_compare(text)
    decomposed = unicodedata.normalize("NFD", lowered)
    without_marks = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return _NON_LETTER_RE.sub("", without_marks)


def czech_sort_key(text: str) -> tuple:
    lowered = unicodedata.normalize("NFC", normalize_whitespace(text).lower())
    chunks: list[int] = []
    index = 0
    while index < len(lowered):
        if lowered[index:index + 2] == "ch":
            chunks.append(_CZECH_ORDER_MAP["ch"])
            index += 2
            continue
        char = lowered[index]
        chunks.append(_CZECH_ORDER_MAP.get(char, 100 + ord(char)))
        index += 1
    return tuple(chunks), lowered


def _notes_by_candidate(notes_payload: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(note.get("candidate_id")): note
        for note in notes_payload.get("notes", [])
        if str(note.get("candidate_id", "")).strip()
    }


def _notes_by_id(notes_payload: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(note.get("note_id")): note
        for note in notes_payload.get("notes", [])
        if str(note.get("note_id", "")).strip()
    }


def _build_note_matches(notes_payload: dict[str, object]) -> dict[str, dict[str, object]]:
    matches: dict[str, dict[str, object]] = {}
    for note in notes_payload.get("notes", []):
        for field in ("target_anchor_text", "target_candidate_text"):
            value = normalize_whitespace(str(note.get(field, "")))
            if value:
                matches.setdefault(normalize_for_compare(value), note)
    return matches


def _is_cleaner_czech_anchor(anchor: str, fallback: str) -> bool:
    if not anchor:
        return False
    if normalize_for_compare(anchor) == normalize_for_compare(fallback):
        return True
    if _SENTENCE_PUNCTUATION_RE.search(anchor):
        return False
    anchor_words = len(anchor.split())
    fallback_words = len(fallback.split())
    return anchor_words <= fallback_words and len(anchor) <= len(fallback)


def _used_entry_eligible_for_dictionary(
    *,
    substitution: DebugSubstitution,
    note: dict[str, object] | None,
    czech: str,
) -> bool:
    if not czech:
        return False
    if detect_foreign_explanation(
        str((note or {}).get("source_anchor_text", "")),
        str((note or {}).get("target_anchor_text", "")),
        str((note or {}).get("note_text", "")),
    ) is not None:
        return False
    word_count = len(czech.split())
    granularity = str(substitution.granularity or "").strip().lower()
    note_type = str((note or {}).get("note_type", "")).strip().lower()
    note_text = normalize_whitespace(str((note or {}).get("note_text", "")))
    is_plain_equality = bool(note_text) and note_text == f"{czech} = {normalize_whitespace(substitution.source_text)}"
    if normalize_for_similarity(czech) == normalize_for_similarity(substitution.source_text):
        return False

    if granularity == "token":
        return True
    if note_type == "idiomatic":
        if _SENTENCE_PUNCTUATION_RE.search(czech) is not None:
            return False
        return word_count <= 6 or _sort_czech_from_note(note or {}) is not None
    if note_type == "lexical":
        return word_count <= 4 and _SENTENCE_PUNCTUATION_RE.search(czech) is None
    if granularity in {"phrase", "subtree"}:
        if word_count > 3 or _SENTENCE_PUNCTUATION_RE.search(czech):
            return False
        return not is_plain_equality
    return False


def _best_polish_from_note(note: dict[str, object]) -> str | None:
    if detect_foreign_explanation(
        str(note.get("source_anchor_text", "")),
        str(note.get("target_anchor_text", "")),
        str(note.get("note_text", "")),
    ) is not None:
        return None
    source_anchor = normalize_whitespace(str(note.get("source_anchor_text", "")))
    if source_anchor:
        return source_anchor
    note_text = normalize_whitespace(str(note.get("note_text", "")))
    if not note_text:
        return None
    if " = " in note_text:
        return normalize_whitespace(note_text.split(" = ", 1)[1])
    return note_text


def _dictionary_explanation_from_note(note: dict[str, object], *, czech: str, polish: str) -> str | None:
    note_text = normalize_whitespace(str(note.get("note_text", "")))
    if not note_text:
        return None
    if note_text == f"{czech} = {polish}":
        return None
    return note_text


def _sort_czech_from_note(note: dict[str, object]) -> str | None:
    family_ids = note.get("new_family_ids", []) or []
    if len(family_ids) != 1:
        return None
    parsed = _parse_family_id(str(family_ids[0]))
    if parsed is None:
        return None
    return parsed[1]


def _dictionary_headword_from_note(note: dict[str, object], czech: str) -> str:
    note_type = str(note.get("note_type", "")).strip().lower()
    if len(czech.split()) <= 4 and _SENTENCE_PUNCTUATION_RE.search(czech) is None:
        return czech
    if note_type not in {"idiomatic", "contextual", "lexical"}:
        return czech
    lemma_headword = _sort_czech_from_note(note)
    if lemma_headword:
        surface = _surface_form_for_lemma(czech, lemma_headword)
        return surface or lemma_headword
    return czech


def _surface_form_for_lemma(czech: str, lemma_headword: str) -> str | None:
    wanted = normalize_for_similarity(lemma_headword)
    if not wanted:
        return None
    for token in re.findall(r"\b[\wÀ-ž-]+\b", czech, flags=re.UNICODE):
        if normalize_for_similarity(token) == wanted:
            return token
    return None


def _parse_family_id(family_id: str) -> tuple[str, str] | None:
    if "::" not in family_id:
        return None
    source_lemma, target_lemma = (part.strip() for part in family_id.split("::", 1))
    if not source_lemma or not target_lemma:
        return None
    return source_lemma, target_lemma


def _eligible_for_machine_translation(czech: str) -> bool:
    word_count = len(czech.split())
    if word_count == 0:
        return False
    if word_count > 8:
        return False
    return _SENTENCE_PUNCTUATION_RE.search(czech) is None


def _is_weak_polish(polish: str, *, czech: str) -> bool:
    normalized_polish = normalize_for_compare(polish)
    normalized_czech = normalize_for_compare(czech)
    if not normalized_polish:
        return True
    if normalized_polish == normalized_czech:
        return True
    return False


def _load_translation_cache(cache_path: str | Path) -> dict[str, dict[str, str]]:
    path = Path(cache_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): value
        for key, value in payload.items()
        if isinstance(value, dict)
    }


def _write_translation_cache(cache_path: str | Path, cache: dict[str, dict[str, str]]) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_translators(
    *,
    deepl_api_key: str | None,
    libre_url: str | None,
) -> list[tuple[str, callable]]:
    translators: list[tuple[str, callable]] = []
    api_key = deepl_api_key or os.environ.get("DEEPL_API_KEY", "")
    if api_key:
        translators.append(("deepl", lambda text: _translate_with_deepl(text, api_key)))
    libre_endpoint = libre_url or os.environ.get("LIBRETRANSLATE_URL", "http://localhost:5000/translate")
    if libre_endpoint:
        translators.append(("libre", lambda text: _translate_with_libre(text, libre_endpoint)))
    return translators


def _translate_with_deepl(text: str, api_key: str) -> str | None:
    payload = json.dumps(
        {
            "text": [text],
            "source_lang": "CS",
            "target_lang": "PL",
        }
    ).encode("utf-8")
    request = Request(
        "https://api-free.deepl.com/v2/translate",
        data=payload,
        headers={
            "Authorization": f"DeepL-Auth-Key {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=5.0) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None
    translations = data.get("translations", [])
    if not translations:
        return None
    return normalize_whitespace(str(translations[0].get("text", ""))) or None


def _translate_with_libre(text: str, endpoint: str) -> str | None:
    payload = json.dumps(
        {
            "q": text,
            "source": "cs",
            "target": "pl",
            "format": "text",
        }
    ).encode("utf-8")
    request = Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=5.0) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None
    return normalize_whitespace(str(data.get("translatedText", ""))) or None


def _dedupe_dictionary_entries(entries: list[DictionaryEntry]) -> list[DictionaryEntry]:
    deduped: dict[tuple[str, str], DictionaryEntry] = {}
    priority = {"note": 0, "deepl": 1, "libre": 2, "alignment": 3}
    for entry in entries:
        if normalize_for_similarity(entry.czech) == normalize_for_similarity(entry.polish):
            continue
        if entry.source == "alignment" and len(entry.czech.split()) > 3:
            continue
        key = (normalize_for_compare(entry.czech), normalize_for_compare(entry.polish))
        current = deduped.get(key)
        if current is None or priority.get(entry.source, 9) < priority.get(current.source, 9):
            deduped[key] = entry
    return list(deduped.values())
