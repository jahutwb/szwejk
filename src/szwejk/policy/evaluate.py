"""Policy evaluation and chapter-level candidate selection."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from szwejk.align.enrichment import build_sentence_enrichment_bundle
from szwejk.align.sentences import align_chapter_sentences
from szwejk.schemas import CanonicalBook

from .actions import build_action_inventory
from .schema import HybridizationLevel, HybridizationPolicy


CONTENT_UPOS_WEIGHTS = {
    "NOUN": 1.0,
    "VERB": 1.0,
    "PROPN": 1.0,
    "ADJ": 0.7,
    "ADV": 0.7,
    "NUM": 0.65,
    "PRON": 0.45,
    "ADP": 0.25,
    "CCONJ": 0.15,
    "SCONJ": 0.15,
    "PART": 0.1,
    "AUX": 0.35,
    "DET": 0.2,
    "INTJ": 0.2,
}


@dataclass(slots=True)
class PolicyCandidate:
    granularity: str
    family_id: str
    unit_class: str
    source_text: str
    target_text: str
    source_lemma: str
    target_lemma: str
    source_upos: str | None
    target_upos: str | None
    source_token_index: int
    target_token_index: int
    sentence_index: int
    sentence_alignment_score: float
    token_pair_score: float
    content_weight: float
    didactic_value: float
    cohesion_bonus: float
    unnaturalness_penalty: float
    inflection_risk_penalty: float
    ambiguity_risk_penalty: float
    target_fit_gain: float
    total_score: float
    state_preferred: bool
    selection_reason: str
    support_depth: str
    source_span: tuple[int, int]
    target_span: tuple[int, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "granularity": self.granularity,
            "family_id": self.family_id,
            "unit_class": self.unit_class,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "source_lemma": self.source_lemma,
            "target_lemma": self.target_lemma,
            "source_upos": self.source_upos,
            "target_upos": self.target_upos,
            "source_token_index": self.source_token_index,
            "target_token_index": self.target_token_index,
            "sentence_index": self.sentence_index,
            "sentence_alignment_score": round(self.sentence_alignment_score, 6),
            "token_pair_score": round(self.token_pair_score, 6),
            "content_weight": round(self.content_weight, 6),
            "didactic_value": round(self.didactic_value, 6),
            "cohesion_bonus": round(self.cohesion_bonus, 6),
            "unnaturalness_penalty": round(self.unnaturalness_penalty, 6),
            "inflection_risk_penalty": round(self.inflection_risk_penalty, 6),
            "ambiguity_risk_penalty": round(self.ambiguity_risk_penalty, 6),
            "target_fit_gain": round(self.target_fit_gain, 6),
            "total_score": round(self.total_score, 6),
            "state_preferred": self.state_preferred,
            "selection_reason": self.selection_reason,
            "support_depth": self.support_depth,
            "source_span": list(self.source_span),
            "target_span": list(self.target_span),
        }


@dataclass(slots=True)
class PolicyChapterPlan:
    policy_id: str
    level_id: str
    source_chapter_index: int
    target_chapter_index: int
    source_title: str
    target_title: str
    actual_surface_czechness: float
    target_surface_czechness: float
    selected_candidates: list[PolicyCandidate] = field(default_factory=list)
    blocked_candidates: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "level_id": self.level_id,
            "source_chapter_index": self.source_chapter_index,
            "target_chapter_index": self.target_chapter_index,
            "source_title": self.source_title,
            "target_title": self.target_title,
            "actual_surface_czechness": round(self.actual_surface_czechness, 6),
            "target_surface_czechness": round(self.target_surface_czechness, 6),
            "selected_candidates": [item.to_dict() for item in self.selected_candidates],
            "blocked_candidates": self.blocked_candidates,
        }


def build_chapter_policy_plan(
    source_book: CanonicalBook,
    target_book: CanonicalBook,
    *,
    source_chapter_index: int,
    target_chapter_index: int,
    policy: HybridizationPolicy,
    level_id: str,
    paragraph_window: int = 3,
    analysis_mode: str = "stanza",
    introduced_families: set[str] | None = None,
) -> PolicyChapterPlan:
    level = _resolve_level(policy, level_id)
    chapter_alignments = align_chapter_sentences(
        source_book.chapters[source_chapter_index - 1],
        target_book.chapters[target_chapter_index - 1],
        paragraph_window=paragraph_window,
    )
    enrichment_bundle = build_sentence_enrichment_bundle(
        chapter_alignments=chapter_alignments,
        source_language=source_book.metadata.language,
        target_language=target_book.metadata.language,
        analysis_mode=analysis_mode,
    )
    return evaluate_policy_on_enrichment_bundle(
        enrichment_bundle,
        policy=policy,
        level=level,
        source_chapter_index=source_chapter_index,
        target_chapter_index=target_chapter_index,
        source_title=source_book.chapters[source_chapter_index - 1].title,
        target_title=target_book.chapters[target_chapter_index - 1].title,
        introduced_families=introduced_families or set(),
    )


def evaluate_policy_on_enrichment_bundle(
    enrichment_bundle: dict[str, object],
    *,
    policy: HybridizationPolicy,
    level: HybridizationLevel,
    source_chapter_index: int,
    target_chapter_index: int,
    source_title: str,
    target_title: str,
    introduced_families: set[str],
) -> PolicyChapterPlan:
    sentence_safe_items = [item for item in enrichment_bundle["items"] if item["enrichment_status"] == "sentence_safe"]
    total_content_weight = sum(_token_weight(token) for item in sentence_safe_items for token in item["source_tokens"] if token["kind"] == "word")
    action_inventory = build_action_inventory(enrichment_bundle)
    if level.gates.minimum_safe_depth in {"phrase", "subtree"}:
        return _evaluate_structural_actions(
            enrichment_bundle,
            action_inventory=action_inventory,
            policy=policy,
            level=level,
            source_chapter_index=source_chapter_index,
            target_chapter_index=target_chapter_index,
            source_title=source_title,
            target_title=target_title,
            total_content_weight=total_content_weight,
        )

    family_occurrences = Counter()
    token_keys_seen = []
    token_meta = {}
    for sentence_index, item in enumerate(sentence_safe_items, start=1):
        for pair in item["token_pairs"]:
            token = _find_source_token(item, int(pair["source_token_index"]))
            if token is None:
                continue
            family_id = _family_id(pair)
            token_key = (sentence_index, int(pair["source_token_index"]))
            token_keys_seen.append(token_key)
            token_meta[token_key] = (item, pair, token, family_id)
            family_occurrences[family_id] += 1

    cumulative_weight = 0.0
    selected_weight = 0.0
    selected_candidates: list[PolicyCandidate] = []
    blocked_candidates: list[dict[str, object]] = []
    selected_token_keys: set[tuple[int, int]] = set()
    active_families = set(introduced_families)

    for token_key in token_keys_seen:
        item, pair, token, family_id = token_meta[token_key]
        token_weight = _token_weight(token)
        cumulative_weight += token_weight
        support_depth = _candidate_support_depth(item, pair)
        gate_status = _check_gates(item, pair, level, support_depth=support_depth)
        if not gate_status["eligible"]:
            blocked_candidates.append(gate_status)
            continue
        if token_weight < _minimum_content_weight(level):
            blocked_candidates.append(
                {
                    "family_id": family_id,
                    "source_text": pair["source_text"],
                    "target_text": pair["target_text"],
                    "reason": f"content_weight_below_{_minimum_content_weight(level):.2f}",
                }
            )
            continue

        unit_class = _classify_unit_class(pair)
        if unit_class not in level.allowed_unit_classes:
            blocked_candidates.append(
                {
                    "family_id": family_id,
                    "source_text": pair["source_text"],
                    "target_text": pair["target_text"],
                    "reason": f"unit_class_{unit_class}_not_allowed",
                }
            )
            continue

        if token_key in selected_token_keys:
            continue

        progress = cumulative_weight / total_content_weight if total_content_weight else 0.0
        target_surface = level.budget.target_surface_czechness * progress
        current_surface = selected_weight / total_content_weight if total_content_weight else 0.0
        target_fit_gain = abs(target_surface - current_surface) - abs(target_surface - ((selected_weight + token_weight) / total_content_weight if total_content_weight else 0.0))
        didactic_value = family_occurrences[family_id] / max(sum(family_occurrences.values()), 1)
        cohesion_bonus = _cohesion_bonus(item, pair)
        unnaturalness_penalty = _unnaturalness_penalty(pair)
        inflection_risk_penalty = _inflection_risk_penalty(pair)
        ambiguity_risk_penalty = _ambiguity_risk_penalty(pair)
        state_preferred = family_id in active_families
        if state_preferred:
            didactic_value = min(didactic_value + 0.1, 1.0)
        total_score = (
            (policy.weights.target_fit * max(target_fit_gain, -0.05))
            + (policy.weights.didactic_value * didactic_value)
            + (policy.weights.cohesion_bonus * cohesion_bonus)
            - (policy.weights.unnaturalness_penalty * unnaturalness_penalty)
            - (policy.weights.inflection_risk_penalty * inflection_risk_penalty)
            - (policy.weights.ambiguity_risk_penalty * ambiguity_risk_penalty)
        )
        should_select = total_score > 0.0
        selection_reason = "best_positive_candidate"
        if state_preferred and total_score > -0.1:
            should_select = True
            selection_reason = "family_propagation"
        if not should_select:
            blocked_candidates.append(
                {
                    "family_id": family_id,
                    "source_text": pair["source_text"],
                    "target_text": pair["target_text"],
                    "reason": "non_positive_total_score",
                    "total_score": round(total_score, 6),
                }
            )
            continue

        selected_candidates.append(
            PolicyCandidate(
                granularity="token",
                family_id=family_id,
                unit_class=unit_class,
                source_text=str(pair["source_text"]),
                target_text=str(pair["target_text"]),
                source_lemma=str(pair["source_lemma"]),
                target_lemma=str(pair["target_lemma"]),
                source_upos=pair.get("source_upos"),
                target_upos=pair.get("target_upos"),
                source_token_index=int(pair["source_token_index"]),
                target_token_index=int(pair["target_token_index"]),
                sentence_index=token_key[0],
                sentence_alignment_score=float(item["sentence_alignment"]["score"]),
                token_pair_score=float(pair["score"]),
                content_weight=token_weight,
                didactic_value=didactic_value,
                cohesion_bonus=cohesion_bonus,
                unnaturalness_penalty=unnaturalness_penalty,
                inflection_risk_penalty=inflection_risk_penalty,
                ambiguity_risk_penalty=ambiguity_risk_penalty,
                target_fit_gain=target_fit_gain,
                total_score=total_score,
                state_preferred=state_preferred,
                selection_reason=selection_reason,
                support_depth=support_depth,
                source_span=(int(pair["source_token_index"]), int(pair["source_token_index"])),
                target_span=(int(pair["target_token_index"]), int(pair["target_token_index"])),
            )
        )
        selected_token_keys.add(token_key)
        selected_weight += token_weight
        active_families.add(family_id)

    actual_surface = selected_weight / total_content_weight if total_content_weight else 0.0
    return PolicyChapterPlan(
        policy_id=policy.id,
        level_id=level.id,
        source_chapter_index=source_chapter_index,
        target_chapter_index=target_chapter_index,
        source_title=source_title,
        target_title=target_title,
        actual_surface_czechness=actual_surface,
        target_surface_czechness=level.budget.target_surface_czechness,
        selected_candidates=selected_candidates,
        blocked_candidates=blocked_candidates,
    )


def _resolve_level(policy: HybridizationPolicy, level_id: str) -> HybridizationLevel:
    for level in policy.levels:
        if level.id == level_id:
            return level
    raise ValueError(f"unknown policy level: {level_id}")


def _evaluate_structural_actions(
    enrichment_bundle: dict[str, object],
    *,
    action_inventory: dict[str, object],
    policy: HybridizationPolicy,
    level: HybridizationLevel,
    source_chapter_index: int,
    target_chapter_index: int,
    source_title: str,
    target_title: str,
    total_content_weight: float,
) -> PolicyChapterPlan:
    selected_candidates: list[PolicyCandidate] = []
    blocked_candidates: list[dict[str, object]] = []
    selected_weight = 0.0
    occupied_source_indices: dict[int, set[int]] = defaultdict(set)
    structural_actions = _structural_action_rows(action_inventory, minimum_safe_depth=level.gates.minimum_safe_depth)
    if not structural_actions:
        for action in action_inventory["token_actions"]:
            blocked_candidates.append(
                {
                    "family_id": action["family_id"],
                    "source_text": action["source_text"],
                    "target_text": action["target_text"],
                    "reason": f"support_depth_below_{level.gates.minimum_safe_depth}",
                    "support_depth": "token",
                }
            )
        return PolicyChapterPlan(
            policy_id=policy.id,
            level_id=level.id,
            source_chapter_index=source_chapter_index,
            target_chapter_index=target_chapter_index,
            source_title=source_title,
            target_title=target_title,
            actual_surface_czechness=0.0,
            target_surface_czechness=level.budget.target_surface_czechness,
            selected_candidates=[],
            blocked_candidates=blocked_candidates,
        )

    for action in structural_actions:
        sentence_index = int(action["sentence_index"])
        source_span = (int(action["source_span"][0]), int(action["source_span"][1]))
        if _span_overlaps(occupied_source_indices[sentence_index], source_span):
            blocked_candidates.append(
                {
                    "family_id": action["family_id"],
                    "source_text": action["source_text"],
                    "target_text": action["target_text"],
                    "reason": "span_overlap_with_selected_action",
                }
            )
            continue

        source_weight = _span_content_weight(enrichment_bundle, sentence_index=sentence_index, source_span=source_span)
        if source_weight <= 0.0:
            blocked_candidates.append(
                {
                    "family_id": action["family_id"],
                    "source_text": action["source_text"],
                    "target_text": action["target_text"],
                    "reason": "empty_structural_span",
                }
            )
            continue

        unit_class = _classify_structural_unit(action)
        if unit_class not in level.allowed_unit_classes:
            blocked_candidates.append(
                {
                    "family_id": action["family_id"],
                    "source_text": action["source_text"],
                    "target_text": action["target_text"],
                    "reason": f"unit_class_{unit_class}_not_allowed",
                }
            )
            continue

        score = float(action["score"])
        if score < level.gates.minimum_alignment_confidence:
            blocked_candidates.append(
                {
                    "family_id": action["family_id"],
                    "source_text": action["source_text"],
                    "target_text": action["target_text"],
                    "reason": "alignment_confidence_below_threshold",
                }
            )
            continue

        progress = (selected_weight + source_weight) / total_content_weight if total_content_weight else 0.0
        current_surface = selected_weight / total_content_weight if total_content_weight else 0.0
        target_surface = level.budget.target_surface_czechness * progress
        target_fit_gain = abs(target_surface - current_surface) - abs(target_surface - progress)
        token_count = max((source_span[1] - source_span[0]) + 1, 1)
        didactic_value = min(max(source_weight / 2.0, token_count / 4.0), 1.0)
        cohesion_bonus = 0.35 if action["granularity"] == "subtree" else 0.25
        unnaturalness_penalty = 0.08 if action["granularity"] == "phrase" else 0.04
        inflection_risk_penalty = 0.1 if action["granularity"] == "phrase" else 0.15
        ambiguity_risk_penalty = 0.0 if score >= 0.9 else (0.08 if score >= 0.8 else 0.2)
        total_score = (
            (0.6 * score)
            + 
            (policy.weights.target_fit * max(target_fit_gain, -0.05))
            + (policy.weights.didactic_value * didactic_value)
            + (policy.weights.cohesion_bonus * cohesion_bonus)
            - (policy.weights.unnaturalness_penalty * unnaturalness_penalty)
            - (policy.weights.inflection_risk_penalty * inflection_risk_penalty)
            - (policy.weights.ambiguity_risk_penalty * ambiguity_risk_penalty)
        )
        if total_score <= 0.2:
            blocked_candidates.append(
                {
                    "family_id": action["family_id"],
                    "source_text": action["source_text"],
                    "target_text": action["target_text"],
                    "reason": "non_positive_total_score",
                    "total_score": round(total_score, 6),
                }
            )
            continue

        selected_candidates.append(
            PolicyCandidate(
                granularity=str(action["granularity"]),
                family_id=str(action["family_id"]),
                unit_class=unit_class,
                source_text=str(action["source_text"]),
                target_text=str(action["target_text"]),
                source_lemma=str(action["family_id"]),
                target_lemma=str(action["family_id"]),
                source_upos=None,
                target_upos=None,
                source_token_index=source_span[0],
                target_token_index=int(action["target_span"][0]),
                sentence_index=sentence_index,
                sentence_alignment_score=score,
                token_pair_score=score,
                content_weight=source_weight,
                didactic_value=didactic_value,
                cohesion_bonus=cohesion_bonus,
                unnaturalness_penalty=unnaturalness_penalty,
                inflection_risk_penalty=inflection_risk_penalty,
                ambiguity_risk_penalty=ambiguity_risk_penalty,
                target_fit_gain=target_fit_gain,
                total_score=total_score,
                state_preferred=False,
                selection_reason="structural_action",
                support_depth=str(action["support_depth"]),
                source_span=source_span,
                target_span=(int(action["target_span"][0]), int(action["target_span"][1])),
            )
        )
        occupied_source_indices[sentence_index].update(range(source_span[0], source_span[1] + 1))
        selected_weight += source_weight

    actual_surface = selected_weight / total_content_weight if total_content_weight else 0.0
    return PolicyChapterPlan(
        policy_id=policy.id,
        level_id=level.id,
        source_chapter_index=source_chapter_index,
        target_chapter_index=target_chapter_index,
        source_title=source_title,
        target_title=target_title,
        actual_surface_czechness=actual_surface,
        target_surface_czechness=level.budget.target_surface_czechness,
        selected_candidates=selected_candidates,
        blocked_candidates=blocked_candidates,
    )


def _check_gates(
    item: dict[str, object],
    pair: dict[str, object],
    level: HybridizationLevel,
    *,
    support_depth: str,
) -> dict[str, object]:
    safe_depth = str(item["safe_alignment_depth"])
    confidence = float(item["sentence_alignment"]["score"])
    if safe_depth == "paragraph":
        return {
            "family_id": _family_id(pair),
            "source_text": pair["source_text"],
            "target_text": pair["target_text"],
            "reason": "parent_alignment_not_sentence_safe",
            "eligible": False,
        }
    if not _support_depth_satisfies(level.gates.minimum_safe_depth, support_depth):
        return {
            "family_id": _family_id(pair),
            "source_text": pair["source_text"],
            "target_text": pair["target_text"],
            "reason": f"support_depth_below_{level.gates.minimum_safe_depth}",
            "support_depth": support_depth,
            "eligible": False,
        }
    if confidence < level.gates.minimum_alignment_confidence:
        return {
            "family_id": _family_id(pair),
            "source_text": pair["source_text"],
            "target_text": pair["target_text"],
            "reason": "alignment_confidence_below_threshold",
            "eligible": False,
        }
    if level.gates.require_low_false_friend_risk and _ambiguity_risk_penalty(pair) >= 0.4:
        return {
            "family_id": _family_id(pair),
            "source_text": pair["source_text"],
            "target_text": pair["target_text"],
            "reason": "false_friend_or_ambiguity_risk",
            "eligible": False,
        }
    if level.gates.require_low_inflection_risk and _inflection_risk_penalty(pair) >= 0.4:
        return {
            "family_id": _family_id(pair),
            "source_text": pair["source_text"],
            "target_text": pair["target_text"],
            "reason": "inflection_risk_too_high",
            "eligible": False,
        }
    return {"eligible": True}


def _classify_unit_class(pair: dict[str, object]) -> str:
    relation = str(pair["relation"])
    score = float(pair["score"])
    if relation in {"exact", "lemma_like"} or score >= 0.82:
        return "A"
    if relation in {"upos_cognate", "cognate"} and score >= 0.7:
        return "B"
    if pair.get("source_deprel") and pair.get("target_deprel"):
        return "C"
    return "D"


def _family_id(pair: dict[str, object]) -> str:
    return f"{pair['source_lemma']}::{pair['target_lemma']}"


def _structural_action_rows(action_inventory: dict[str, object], *, minimum_safe_depth: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if minimum_safe_depth == "phrase":
        rows.extend(action_inventory["phrase_actions"])
    if minimum_safe_depth == "subtree":
        rows.extend(action_inventory["subtree_actions"])
    return sorted(rows, key=lambda item: (int(item["sentence_index"]), -float(item["score"]), item["source_span"][0]))


def _classify_structural_unit(action: dict[str, object]) -> str:
    score = float(action["score"])
    granularity = str(action["granularity"])
    if granularity == "phrase":
        if score >= 0.84:
            return "A"
        if score >= 0.68:
            return "B"
        return "D"
    if granularity == "subtree":
        if score >= 0.82:
            return "B"
        if score >= 0.6:
            return "C"
        return "D"
    return "D"


def _span_overlaps(occupied_indices: set[int], source_span: tuple[int, int]) -> bool:
    return any(index in occupied_indices for index in range(source_span[0], source_span[1] + 1))


def _span_content_weight(
    enrichment_bundle: dict[str, object],
    *,
    sentence_index: int,
    source_span: tuple[int, int],
) -> float:
    item = [entry for entry in enrichment_bundle["items"] if entry["enrichment_status"] == "sentence_safe"][sentence_index - 1]
    total = 0.0
    for token in item["source_tokens"]:
        index = int(token["index"])
        if source_span[0] <= index <= source_span[1]:
            total += _token_weight(token)
    return total


def _candidate_support_depth(item: dict[str, object], pair: dict[str, object]) -> str:
    source_index = int(pair["source_token_index"])
    if _token_in_any_span(item.get("subtree_candidates", []), source_index):
        return "subtree"
    if _token_in_any_span(item.get("dependency_candidates", []), source_index):
        return "subtree"
    if _token_in_any_span(item.get("phrase_candidates", []), source_index):
        return "phrase"
    return "token"


def _support_depth_satisfies(required_depth: str, support_depth: str) -> bool:
    if required_depth == "sentence":
        return True
    if required_depth == "token":
        return support_depth in {"token", "phrase", "subtree"}
    if required_depth == "phrase":
        return support_depth in {"phrase", "subtree"}
    if required_depth == "subtree":
        return support_depth == "subtree"
    if required_depth == "paragraph":
        return True
    return False


def _token_weight(token: dict[str, object]) -> float:
    if token["kind"] != "word":
        return 0.0
    return CONTENT_UPOS_WEIGHTS.get(token.get("upos") or "", 0.5)


def _find_source_token(item: dict[str, object], source_token_index: int) -> dict[str, object] | None:
    for token in item["source_tokens"]:
        if int(token["index"]) == source_token_index:
            return token
    return None


def _cohesion_bonus(item: dict[str, object], pair: dict[str, object]) -> float:
    source_index = int(pair["source_token_index"])
    if _token_in_any_span(item.get("subtree_candidates", []), source_index):
        return 0.3
    if _token_in_any_span(item.get("dependency_candidates", []), source_index):
        return 0.25
    if _token_in_any_span(item.get("phrase_candidates", []), source_index):
        return 0.15
    return 0.0


def _token_in_any_span(candidates: list[dict[str, object]], source_index: int) -> bool:
    for candidate in candidates:
        start, end = candidate["source_span"]
        if int(start) <= source_index <= int(end):
            return True
    return False


def _unnaturalness_penalty(pair: dict[str, object]) -> float:
    relation = str(pair["relation"])
    if relation == "exact":
        return 0.0
    if relation == "lemma_like":
        return 0.05
    if relation == "upos_cognate":
        return 0.15
    if relation == "cognate":
        return 0.2
    return 0.35


def _inflection_risk_penalty(pair: dict[str, object]) -> float:
    source_upos = pair.get("source_upos")
    target_upos = pair.get("target_upos")
    if source_upos and target_upos and source_upos != target_upos:
        return 0.5
    if str(pair["relation"]) == "exact":
        return 0.0
    if str(pair["relation"]) == "lemma_like":
        return 0.1
    return 0.25


def _ambiguity_risk_penalty(pair: dict[str, object]) -> float:
    score = float(pair["score"])
    if score >= 0.95:
        return 0.0
    if score >= 0.82:
        return 0.1
    if score >= 0.72:
        return 0.25
    return 0.45


def _minimum_content_weight(level: HybridizationLevel) -> float:
    if level.id == "l1-lexical-onboarding":
        return 0.45
    return 0.0
