"""ParagraphHybridCandidate dataclass and shared constants."""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Shared constants (imported by all sub-modules to avoid circular deps)
# ---------------------------------------------------------------------------

FALLBACK_RELATIONS: frozenset[str] = frozenset({"embedding_assisted", "dictionary", "manual_forced"})
TOKEN_CONFIDENCE_PENALTY = 0.025
STRUCTURAL_CONFIDENCE_PENALTY = 0.008
FUNCTIONAL_UPOS = frozenset({"SCONJ", "CCONJ", "PART", "ADP", "PRON", "DET", "AUX"})
DEFAULT_BLOCKED_STANDALONE_UPOS = FUNCTIONAL_UPOS
FALLBACK_STOPWORDS: dict[str, set[str]] = {
    "pl": {
        "a", "ale", "bo", "by", "co", "czy", "do", "go", "i", "ich", "jego", "jej",
        "jak", "już", "na", "nie", "o", "od", "po", "przed", "przy", "się", "swoim",
        "ten", "to", "w", "za", "z", "że",
    },
    "cs": {
        "a", "ale", "bo", "by", "co", "do", "ho", "i", "jak", "je", "jeho", "její",
        "již", "na", "ne", "o", "od", "po", "před", "pri", "se", "svým", "ten", "to",
        "u", "už", "v", "za", "z", "že",
    },
}
SURFACE_FUNCTION_WORDS: set[str] = FALLBACK_STOPWORDS["pl"] | FALLBACK_STOPWORDS["cs"] | {
    "mu", "mi", "mnie", "mně", "tě", "ci", "pana", "pan", "pani", "ją", "jąż", "go", "ho",
    "me", "mě", "jsem", "jsi", "jest", "je", "było", "byl", "była", "bylo", "byli",
}
IDIOMATICITY_PENALTY_WEIGHT = 0.06
VISIBLE_FUTURE_BLEND = 0.18


# ---------------------------------------------------------------------------
# Helpers used by ParagraphHybridCandidate.to_dict()
# ---------------------------------------------------------------------------

def _is_span_granularity(granularity: str) -> bool:
    return granularity in {"phrase", "subtree", "span"}


def _display_granularity(granularity: str) -> str:
    return "span" if granularity in {"phrase", "subtree"} else granularity


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ParagraphHybridCandidate:
    candidate_id: str
    chapter_pair: tuple[int, int]
    unit_index: int
    granularity: str
    scope_id: str
    source_span: tuple[int, int]
    target_span: tuple[int, int]
    source_text: str
    target_text: str
    family_ids: list[str]
    score: float
    relation: str
    metadata: dict[str, object]

    @property
    def is_fallback(self) -> bool:
        return self.relation in FALLBACK_RELATIONS

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "chapter_pair": list(self.chapter_pair),
            "unit_index": self.unit_index,
            "granularity": _display_granularity(self.granularity),
            "scope_id": self.scope_id,
            "source_span": list(self.source_span),
            "target_span": list(self.target_span),
            "source_text": self.source_text,
            "target_text": self.target_text,
            "family_ids": list(self.family_ids),
            "score": round(self.score, 6),
            "relation": self.relation,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Deserialisation helper
# ---------------------------------------------------------------------------

def candidate_from_dict(payload: dict[str, object]) -> ParagraphHybridCandidate:
    return ParagraphHybridCandidate(
        candidate_id=str(payload["candidate_id"]),
        chapter_pair=(int(payload["chapter_pair"][0]), int(payload["chapter_pair"][1])),
        unit_index=int(payload["unit_index"]),
        granularity=str(payload["granularity"]),
        scope_id=str(payload["scope_id"]),
        source_span=(int(payload["source_span"][0]), int(payload["source_span"][1])),
        target_span=(int(payload["target_span"][0]), int(payload["target_span"][1])),
        source_text=str(payload["source_text"]),
        target_text=str(payload["target_text"]),
        family_ids=[str(item) for item in payload.get("family_ids", [])],
        score=float(payload.get("score", 0.0)),
        relation=str(payload.get("relation", "")),
        metadata=dict(payload.get("metadata", {})),
    )

