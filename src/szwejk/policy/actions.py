"""Mixed-granularity policy action inventory built from enriched alignments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PolicyActionCandidate:
    granularity: str
    sentence_index: int
    family_id: str | None
    source_span: tuple[int, int]
    target_span: tuple[int, int]
    source_text: str
    target_text: str
    score: float
    support_depth: str
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "granularity": self.granularity,
            "sentence_index": self.sentence_index,
            "family_id": self.family_id,
            "source_span": list(self.source_span),
            "target_span": list(self.target_span),
            "source_text": self.source_text,
            "target_text": self.target_text,
            "score": round(self.score, 6),
            "support_depth": self.support_depth,
            "metadata": self.metadata,
        }


def build_action_inventory(enrichment_bundle: dict[str, object]) -> dict[str, object]:
    token_actions: list[PolicyActionCandidate] = []
    phrase_actions: list[PolicyActionCandidate] = []
    subtree_actions: list[PolicyActionCandidate] = []

    sentence_safe_items = [item for item in enrichment_bundle["items"] if item["enrichment_status"] == "sentence_safe"]
    for sentence_index, item in enumerate(sentence_safe_items, start=1):
        source_tokens = item["source_tokens"]
        target_tokens = item["target_tokens"]
        token_actions_for_sentence: list[PolicyActionCandidate] = []

        for pair in item.get("token_pairs", []):
            source_index = int(pair["source_token_index"])
            target_index = int(pair["target_token_index"])
            candidate = PolicyActionCandidate(
                granularity="token",
                sentence_index=sentence_index,
                family_id=f"{pair['source_lemma']}::{pair['target_lemma']}",
                source_span=(source_index, source_index),
                target_span=(target_index, target_index),
                source_text=str(pair["source_text"]),
                target_text=str(pair["target_text"]),
                score=float(pair["score"]),
                support_depth="token",
                metadata={
                    "relation": pair["relation"],
                    "source_upos": pair.get("source_upos"),
                    "target_upos": pair.get("target_upos"),
                    "family_ids": [f"{pair['source_lemma']}::{pair['target_lemma']}"],
                },
            )
            token_actions.append(candidate)
            token_actions_for_sentence.append(candidate)

        for candidate in item.get("phrase_candidates", []):
            family_ids = _family_ids_for_span(token_actions_for_sentence, int(candidate["source_span"][0]), int(candidate["source_span"][1]))
            phrase_actions.append(
                PolicyActionCandidate(
                    granularity="phrase",
                    sentence_index=sentence_index,
                    family_id=_structural_family_id(
                        "phrase",
                        _span_text(source_tokens, int(candidate["source_span"][0]), int(candidate["source_span"][1])),
                        _span_text(target_tokens, int(candidate["target_span"][0]), int(candidate["target_span"][1])),
                    ),
                    source_span=(int(candidate["source_span"][0]), int(candidate["source_span"][1])),
                    target_span=(int(candidate["target_span"][0]), int(candidate["target_span"][1])),
                    source_text=_span_text(source_tokens, int(candidate["source_span"][0]), int(candidate["source_span"][1])),
                    target_text=_span_text(target_tokens, int(candidate["target_span"][0]), int(candidate["target_span"][1])),
                    score=float(candidate["score"]),
                    support_depth="phrase",
                    metadata={
                        "relation": candidate["relation"],
                        "token_count": int(candidate["token_count"]),
                        "family_ids": family_ids,
                    },
                )
            )

        for candidate in item.get("subtree_candidates", []):
            family_ids = _family_ids_for_span(token_actions_for_sentence, int(candidate["source_span"][0]), int(candidate["source_span"][1]))
            subtree_actions.append(
                PolicyActionCandidate(
                    granularity="subtree",
                    sentence_index=sentence_index,
                    family_id=_structural_family_id("subtree", str(candidate["source_text"]), str(candidate["target_text"])),
                    source_span=(int(candidate["source_span"][0]), int(candidate["source_span"][1])),
                    target_span=(int(candidate["target_span"][0]), int(candidate["target_span"][1])),
                    source_text=str(candidate["source_text"]),
                    target_text=str(candidate["target_text"]),
                    score=float(candidate["score"]),
                    support_depth="subtree",
                    metadata={
                        "source_root_index": int(candidate["source_root_index"]),
                        "target_root_index": int(candidate["target_root_index"]),
                        "family_ids": family_ids,
                    },
                )
            )

    return {
        "summary": {
            "sentence_safe_items": len(sentence_safe_items),
            "token_action_count": len(token_actions),
            "phrase_action_count": len(phrase_actions),
            "subtree_action_count": len(subtree_actions),
        },
        "token_actions": [item.to_dict() for item in token_actions],
        "phrase_actions": [item.to_dict() for item in phrase_actions],
        "subtree_actions": [item.to_dict() for item in subtree_actions],
    }


def _span_text(tokens: list[dict[str, object]], start: int, end: int) -> str:
    selected = [str(token["text"]) for token in tokens if start <= int(token["index"]) <= end and token["kind"] == "word"]
    return " ".join(selected)


def _structural_family_id(granularity: str, source_text: str, target_text: str) -> str:
    return f"{granularity}::{source_text.lower()}::{target_text.lower()}"


def _family_ids_for_span(actions: list[PolicyActionCandidate], start: int, end: int) -> list[str]:
    family_ids = []
    for action in actions:
        action_start, action_end = action.source_span
        if start <= action_start and action_end <= end and action.family_id:
            family_ids.append(action.family_id)
    return sorted(set(family_ids))
