"""Didactic note bundles for hybrid outputs and paragraph-level plans."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Iterable

from szwejk.generate.paragraph_hybridization import (
    DEFAULT_BLOCKED_STANDALONE_UPOS,
    ParagraphHybridCandidate,
    _build_units,
)


FOREIGN_EXPLANATIONS: dict[str, str] = {
    "dominus vobiscum et cum spiritu tuo": "lac. \"Pan z wami - i z duchem twoim\".",
    "dominus vobiscum": "lac. \"Pan z wami\".",
    "et cum spiritu tuo": "lac. \"i z duchem twoim\".",
    "himmelherrgott": "niem. okrzyk w rodzaju \"dobry Boze!\" / \"do licha!\".",
    "himlhergot": "niem. okrzyk w rodzaju \"dobry Boze!\" / \"do licha!\".",
    "herr wachtmeister": "niem. \"panie wachmistrzu\".",
    "ich gratuliere ihnen": "niem. \"gratuluje panu\".",
    "herr": "niem. \"pan\".",
}


@dataclass(slots=True)
class DidacticNote:
    note_id: str
    sentence_index: int
    source_paragraph_id: str
    granularity: str
    source_text: str
    target_text: str
    kind: str
    prompt: str

    def to_dict(self) -> dict[str, object]:
        return {
            "note_id": self.note_id,
            "sentence_index": self.sentence_index,
            "source_paragraph_id": self.source_paragraph_id,
            "granularity": self.granularity,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "kind": self.kind,
            "prompt": self.prompt,
        }


def build_didactic_note_bundle(document: dict[str, object]) -> dict[str, object]:
    notes: list[DidacticNote] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for paragraph in document.get("paragraphs", []):
        for sentence in paragraph.get("sentences", []):
            for action in sentence.get("actions_applied", []):
                key = (
                    str(action["granularity"]),
                    str(action["source_text"]).strip().lower(),
                    str(action["target_text"]).strip().lower(),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                notes.append(
                    DidacticNote(
                        note_id=f"note-{len(notes) + 1:04d}",
                        sentence_index=int(sentence["sentence_index"]),
                        source_paragraph_id=str(sentence["source_paragraph_id"]),
                        granularity=str(action["granularity"]),
                        source_text=str(action["source_text"]),
                        target_text=str(action["target_text"]),
                        kind=_note_kind(action),
                        prompt=_note_prompt(action),
                    )
                )

    return {
        "source_chapter_index": document["source_chapter_index"],
        "target_chapter_index": document["target_chapter_index"],
        "level_id": document["level_id"],
        "summary": {
            "note_count": len(notes),
            "deduplicated_action_count": len(seen_keys),
        },
        "notes": [item.to_dict() for item in notes],
    }


def _note_kind(action: dict[str, object]) -> str:
    if action["granularity"] == "token":
        return "lexical"
    if action["granularity"] in {"phrase", "subtree", "span"}:
        return "phrase"
    return "construction"


def _note_prompt(action: dict[str, object]) -> str:
    source_text = str(action["source_text"])
    target_text = str(action["target_text"])
    granularity = str(action["granularity"])
    if granularity == "token":
        return f"Nowa forma czeska: '{target_text}' zamiast polskiego '{source_text}'."
    if granularity in {"phrase", "subtree", "span"}:
        return f"Nowa fraza czeska: '{target_text}' odpowiada polskiemu '{source_text}'."
    return f"Nowa konstrukcja czeska: '{target_text}' zastępuje fragment '{source_text}'."


@dataclass(slots=True)
class ParagraphPlanNote:
    note_id: str
    unit_index: int
    chapter_pair: tuple[int, int]
    candidate_id: str
    candidate_granularity: str
    anchor_granularity: str
    note_type: str
    source_anchor_text: str
    target_anchor_text: str
    source_candidate_text: str
    target_candidate_text: str
    new_family_ids: list[str]
    alignment_score: float
    idiomaticity_score: float
    note_text: str
    editorial_attention: bool
    editorial_comment: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "note_id": self.note_id,
            "unit_index": self.unit_index,
            "chapter_pair": list(self.chapter_pair),
            "candidate_id": self.candidate_id,
            "candidate_granularity": self.candidate_granularity,
            "anchor_granularity": self.anchor_granularity,
            "note_type": self.note_type,
            "source_anchor_text": self.source_anchor_text,
            "target_anchor_text": self.target_anchor_text,
            "source_candidate_text": self.source_candidate_text,
            "target_candidate_text": self.target_candidate_text,
            "new_family_ids": list(self.new_family_ids),
            "alignment_score": round(self.alignment_score, 6),
            "idiomaticity_score": round(self.idiomaticity_score, 6),
            "note_text": self.note_text,
            "editorial_attention": self.editorial_attention,
            "editorial_comment": self.editorial_comment,
        }


def build_paragraph_note_bundle(
    alignment_artifact: dict[str, object],
    plan_payload: dict[str, object],
    *,
    note_policy: str = "conservative",
) -> dict[str, object]:
    blocked_upos = frozenset(
        item.strip().upper()
        for item in plan_payload.get("blocked_standalone_upos", [])
        if str(item).strip()
    ) or DEFAULT_BLOCKED_STANDALONE_UPOS
    units = _build_units(alignment_artifact, blocked_standalone_upos=blocked_upos)
    unit_lookup = {int(unit["unit_index"]): unit for unit in units}

    introduced_families: set[str] = set()
    introduced_source_lemmas: set[str] = set()
    notes: list[ParagraphPlanNote] = []

    for plan_unit in plan_payload.get("units", []):
        unit_index = int(plan_unit["unit_index"])
        unit = unit_lookup.get(unit_index)
        if unit is None:
            continue
        inventory = {
            candidate.candidate_id: candidate
            for candidate in unit.get("candidates", [])
            if not candidate.is_fallback
        }
        for selected in plan_unit.get("selected_candidates", []):
            candidate_id = str(selected["candidate_id"])
            selected_families = [str(item) for item in selected.get("family_ids", [])]
            # Deduplicate by SOURCE LEMMA (left side of "::", split by "+").
            # The same Czech lemma can appear via different family_ids when the
            # Polish source has multiple inflected forms (e.g. "ferdynand::ferdinand"
            # vs a phrase family like "ferdynand+ten::ferdinand").  We emit a note
            # only the first time any source lemma in the family is seen.
            def _source_lemmas_of(family_id: str) -> list[str]:
                left = family_id.split("::")[0] if "::" in family_id else family_id
                return [part.strip() for part in left.split("+") if part.strip()]

            new_family_ids = [
                family_id for family_id in selected_families
                if family_id not in introduced_families
                and not any(lemma in introduced_source_lemmas for lemma in _source_lemmas_of(family_id))
            ]
            introduced_families.update(selected_families)
            for family_id in selected_families:
                introduced_source_lemmas.update(_source_lemmas_of(family_id))
            if not new_family_ids:
                continue
            candidate = inventory.get(candidate_id)
            if candidate is None:
                candidate = ParagraphHybridCandidate(
                    candidate_id=candidate_id,
                    chapter_pair=(int(selected["chapter_pair"][0]), int(selected["chapter_pair"][1])),
                    unit_index=int(selected["unit_index"]),
                    granularity=str(selected["granularity"]),
                    scope_id=str(selected["scope_id"]),
                    source_span=(int(selected["source_span"][0]), int(selected["source_span"][1])),
                    target_span=(int(selected["target_span"][0]), int(selected["target_span"][1])),
                    source_text=str(selected["source_text"]),
                    target_text=str(selected["target_text"]),
                    family_ids=selected_families,
                    score=float(selected["score"]),
                    relation=str(selected["relation"]),
                    metadata=dict(selected.get("metadata", {})),
                )

            anchor = _pick_note_anchor(candidate, unit.get("candidates", []), new_family_ids)
            idiomaticity = _idiomaticity_score(anchor)
            note_type = _note_type(anchor, idiomaticity)
            if not _should_emit_note(anchor, note_type=note_type, idiomaticity_score=idiomaticity, note_policy=note_policy):
                continue
            editorial_attention = note_type == "idiomatic"
            editorial_comment = (
                "Zwrot mniej dosłowny; warto dopisać krótkie objaśnienie sensu."
                if editorial_attention
                else None
            )
            note_text = _render_note_text(anchor, note_type)
            notes.append(
                ParagraphPlanNote(
                    note_id=f"note-{len(notes) + 1:05d}",
                    unit_index=unit_index,
                    chapter_pair=tuple(int(item) for item in selected["chapter_pair"]),
                    candidate_id=candidate.candidate_id,
                    candidate_granularity=candidate.granularity,
                    anchor_granularity=anchor.granularity,
                    note_type=note_type,
                    source_anchor_text=anchor.source_text,
                    target_anchor_text=anchor.target_text,
                    source_candidate_text=candidate.source_text,
                    target_candidate_text=candidate.target_text,
                    new_family_ids=new_family_ids,
                    alignment_score=anchor.score,
                    idiomaticity_score=idiomaticity,
                    note_text=note_text,
                    editorial_attention=editorial_attention,
                    editorial_comment=editorial_comment,
                )
            )

    return {
        "mode": "paragraph-note-bundle-v1",
        "note_policy": note_policy,
        "target_power": plan_payload.get("target_power"),
        "blocked_standalone_upos": list(plan_payload.get("blocked_standalone_upos", [])),
        "summary": {
            "note_count": len(notes),
            "idiomatic_note_count": sum(1 for note in notes if note.note_type == "idiomatic"),
            "contextual_note_count": sum(1 for note in notes if note.note_type == "contextual"),
            "lexical_note_count": sum(1 for note in notes if note.note_type == "lexical"),
            "editorial_attention_count": sum(1 for note in notes if note.editorial_attention),
        },
        "notes": [note.to_dict() for note in notes],
    }


def _pick_note_anchor(
    selected_candidate: ParagraphHybridCandidate,
    inventory: Iterable[ParagraphHybridCandidate],
    new_family_ids: list[str],
) -> ParagraphHybridCandidate:
    covering = []
    for candidate in inventory:
        if candidate.is_fallback:
            continue
        if not set(candidate.family_ids) & set(new_family_ids):
            continue
        if not _contains_candidate_span(selected_candidate, candidate):
            continue
        if candidate.granularity == "paragraph":
            continue
        covering.append(candidate)

    preferred = [
        candidate
        for candidate in covering
        if candidate.granularity in {"phrase", "subtree", "span"}
        and _word_count(candidate.source_text) >= 2
        and candidate.score >= 0.6
        and _anchor_is_compact(candidate)
    ]
    if preferred:
        return sorted(
            preferred,
            key=lambda item: (
                _word_count(item.source_text),
                _word_count(item.target_text),
                -item.score,
                item.candidate_id,
            ),
        )[0]

    token_candidates = [candidate for candidate in covering if candidate.granularity == "token" and candidate.score >= 0.6]
    if token_candidates:
        return sorted(token_candidates, key=lambda item: (-item.score, item.candidate_id))[0]

    non_token = [candidate for candidate in covering if candidate.granularity != "token" and candidate.score >= 0.6]
    if non_token:
        return sorted(
            non_token,
            key=lambda item: (
                _word_count(item.source_text),
                _word_count(item.target_text),
                -item.score,
                item.candidate_id,
            ),
        )[0]
    return selected_candidate


def _contains_span(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def _candidate_token_keys(candidate: ParagraphHybridCandidate, *, side: str) -> set[tuple[str, int]]:
    metadata_key = "coverage_source_token_keys" if side == "source" else "coverage_target_token_keys"
    explicit_keys = candidate.metadata.get(metadata_key)
    if explicit_keys:
        normalized: set[tuple[str, int]] = set()
        for item in explicit_keys:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                normalized.add((str(item[0]), int(item[1])))
        if normalized:
            return normalized
    span = candidate.source_span if side == "source" else candidate.target_span
    return {(candidate.scope_id, index) for index in range(int(span[0]), int(span[1]) + 1)}


def _contains_candidate_span(outer: ParagraphHybridCandidate, inner: ParagraphHybridCandidate) -> bool:
    outer_source_keys = _candidate_token_keys(outer, side="source")
    inner_source_keys = _candidate_token_keys(inner, side="source")
    outer_target_keys = _candidate_token_keys(outer, side="target")
    inner_target_keys = _candidate_token_keys(inner, side="target")
    if outer_source_keys and inner_source_keys and outer_target_keys and inner_target_keys:
        return inner_source_keys.issubset(outer_source_keys) and inner_target_keys.issubset(outer_target_keys)
    return _contains_span(outer.source_span, inner.source_span) and _contains_span(outer.target_span, inner.target_span)


def _word_count(text: str) -> int:
    return len([part for part in text.split() if part.strip()])


def _anchor_is_compact(candidate: ParagraphHybridCandidate) -> bool:
    source_words = _word_count(candidate.source_text)
    target_words = _word_count(candidate.target_text)
    width = max(source_words, target_words, 1)
    family_density = len(set(candidate.family_ids)) / width
    similarity = _text_similarity(candidate.source_text, candidate.target_text)
    return family_density >= 0.4 or similarity >= 0.65


def _idiomaticity_score(candidate: ParagraphHybridCandidate) -> float:
    if candidate.granularity not in {"phrase", "subtree", "span", "sentence"}:
        return 0.0
    support = float(candidate.metadata.get("token_support_ratio", 1.0))
    signals = dict(candidate.metadata.get("signals", {}))
    lemma = float(signals.get("lemma", 0.0))
    embedding = float(signals.get("embedding", candidate.score))
    semantic = max(float(candidate.score), embedding)
    return max(0.0, semantic - max(support, lemma))


def _note_type(candidate: ParagraphHybridCandidate, idiomaticity_score: float) -> str:
    if idiomaticity_score >= 0.5:
        return "idiomatic"
    if candidate.granularity == "token":
        return "lexical"
    return "contextual"


def _render_note_text(candidate: ParagraphHybridCandidate, note_type: str) -> str:
    source_text = candidate.source_text.strip()
    target_text = candidate.target_text.strip()
    foreign_explanation = detect_foreign_explanation(source_text, target_text)
    if foreign_explanation is not None:
        foreign_text, explanation = foreign_explanation
        return f"{foreign_text} - {explanation}"
    if note_type == "idiomatic":
        return f"{target_text}: {source_text}."
    return f"{target_text} = {source_text}"


def detect_foreign_explanation(*texts: str) -> tuple[str, str] | None:
    best_match: tuple[str, str] | None = None
    best_len = -1
    for text in texts:
        matched = _match_foreign_phrase(text)
        if matched is None:
            continue
        phrase, explanation = matched
        if len(phrase) > best_len:
            best_match = (phrase, explanation)
            best_len = len(phrase)
    return best_match


def _match_foreign_phrase(text: str) -> tuple[str, str] | None:
    if not text.strip():
        return None
    normalized = _normalize_foreign_text(text)
    for key in sorted(FOREIGN_EXPLANATIONS, key=len, reverse=True):
        if key in normalized:
            original = _recover_original_phrase(text, key)
            return original, FOREIGN_EXPLANATIONS[key]
    return None


def _normalize_foreign_text(text: str) -> str:
    lowered = text.lower()
    lowered = lowered.replace("„", " ").replace("”", " ").replace("\"", " ").replace("‚", " ").replace("‘", " ")
    lowered = re.sub(r"[^\w\s-]+", " ", lowered, flags=re.UNICODE)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _recover_original_phrase(text: str, normalized_key: str) -> str:
    lowered = text.lower()
    idx = lowered.find(normalized_key)
    if idx != -1:
        return text[idx : idx + len(normalized_key)].strip(" \"„”‚‘.,:;!?-")
    token_pattern = re.escape(normalized_key).replace("\\ ", r"\s+")
    match = re.search(token_pattern, _normalize_foreign_text(text), flags=re.UNICODE)
    if match is not None:
        return normalized_key
    return normalized_key


def _should_emit_note(
    candidate: ParagraphHybridCandidate,
    *,
    note_type: str,
    idiomaticity_score: float,
    note_policy: str,
) -> bool:
    if note_policy == "conservative":
        return True
    if note_policy != "relaxed":
        raise ValueError(f"Unsupported note policy: {note_policy}")

    transparency = _text_similarity(candidate.source_text, candidate.target_text)

    if note_type == "idiomatic":
        return True

    if _looks_suspicious_for_note(candidate):
        return False

    if candidate.granularity == "paragraph":
        return idiomaticity_score >= 0.5 or candidate.score < 0.58

    if note_type == "lexical":
        if transparency >= 0.72 and candidate.score >= 0.75:
            return False
        return candidate.score < 0.9 or transparency < 0.85

    if note_type == "contextual":
        if transparency >= 0.78 and candidate.score >= 0.8:
            return False
        return True

    return True


def _text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(a=left.strip().lower(), b=right.strip().lower()).ratio()


def _looks_suspicious_for_note(candidate: ParagraphHybridCandidate) -> bool:
    similarity = _text_similarity(candidate.source_text, candidate.target_text)
    family_trust = float(candidate.metadata.get("family_trust", 0.0))
    relation = str(candidate.relation)
    if candidate.granularity == "token":
        if relation in {"upos_cognate", "cognate"} and similarity < 0.42:
            return True
        if family_trust and family_trust < 0.45 and candidate.score < 0.72:
            return True
    return False
