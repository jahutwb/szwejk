"""Schemas for staged hybridization policy."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from szwejk.schemas.base import SchemaModel


PolicyUnitClass = str
DefaultSelectionPolicy = str
GlobalStatePolicy = str

ALLOWED_UNIT_CLASSES = {"A", "B", "C", "D"}
ALLOWED_SAFE_DEPTHS = {"paragraph", "sentence", "phrase", "subtree", "token"}
ALLOWED_SELECTION_POLICIES = {"target_fit_first", "balanced", "didactic_first"}
ALLOWED_GLOBAL_STATE_POLICIES = {"prefer_czech_after_intro", "strict_lock", "review_required"}


@dataclass(slots=True)
class PolicyGateConfig(SchemaModel):
    minimum_safe_depth: str
    allow_foreign_span: bool = False
    allow_residual_unmatched: bool = False
    require_low_false_friend_risk: bool = True
    require_low_inflection_risk: bool = True
    require_subtree_cohesion: bool = False
    minimum_alignment_confidence: float = 0.7

    def __post_init__(self) -> None:
        if self.minimum_safe_depth not in ALLOWED_SAFE_DEPTHS:
            raise ValueError(f"unsupported safe depth: {self.minimum_safe_depth}")
        if not 0.0 <= self.minimum_alignment_confidence <= 1.0:
            raise ValueError("minimum_alignment_confidence must be within [0.0, 1.0]")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PolicyGateConfig":
        return cls(
            minimum_safe_depth=str(data["minimum_safe_depth"]),
            allow_foreign_span=bool(data.get("allow_foreign_span", False)),
            allow_residual_unmatched=bool(data.get("allow_residual_unmatched", False)),
            require_low_false_friend_risk=bool(data.get("require_low_false_friend_risk", True)),
            require_low_inflection_risk=bool(data.get("require_low_inflection_risk", True)),
            require_subtree_cohesion=bool(data.get("require_subtree_cohesion", False)),
            minimum_alignment_confidence=float(data.get("minimum_alignment_confidence", 0.7)),
        )


@dataclass(slots=True)
class PolicySelectionWeights(SchemaModel):
    target_fit: float
    didactic_value: float
    cohesion_bonus: float
    unnaturalness_penalty: float
    inflection_risk_penalty: float
    ambiguity_risk_penalty: float

    def __post_init__(self) -> None:
        for field_name, value in self.to_dict().items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be numeric")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PolicySelectionWeights":
        return cls(
            target_fit=float(data["target_fit"]),
            didactic_value=float(data["didactic_value"]),
            cohesion_bonus=float(data["cohesion_bonus"]),
            unnaturalness_penalty=float(data["unnaturalness_penalty"]),
            inflection_risk_penalty=float(data["inflection_risk_penalty"]),
            ambiguity_risk_penalty=float(data["ambiguity_risk_penalty"]),
        )


@dataclass(slots=True)
class PolicyLevelBudget(SchemaModel):
    target_exposure: float
    target_surface_czechness: float
    max_readability_load: float
    max_new_families_per_1000_tokens: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.target_exposure <= 1.0:
            raise ValueError("target_exposure must be within [0.0, 1.0]")
        if not 0.0 <= self.target_surface_czechness <= 1.0:
            raise ValueError("target_surface_czechness must be within [0.0, 1.0]")
        if self.max_readability_load < 0.0:
            raise ValueError("max_readability_load must be >= 0.0")
        if self.max_new_families_per_1000_tokens < 0:
            raise ValueError("max_new_families_per_1000_tokens must be >= 0")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PolicyLevelBudget":
        return cls(
            target_exposure=float(data["target_exposure"]),
            target_surface_czechness=float(data["target_surface_czechness"]),
            max_readability_load=float(data["max_readability_load"]),
            max_new_families_per_1000_tokens=int(data["max_new_families_per_1000_tokens"]),
        )


@dataclass(slots=True)
class HybridizationLevel(SchemaModel):
    id: str
    label: str
    description: str
    allowed_unit_classes: list[PolicyUnitClass] = field(default_factory=list)
    base_mode: str = "pl_dominant"
    selection_policy: DefaultSelectionPolicy = "balanced"
    global_state_policy: GlobalStatePolicy = "prefer_czech_after_intro"
    budget: PolicyLevelBudget = field(default_factory=lambda: PolicyLevelBudget(0.0, 0.0, 0.0, 0))
    gates: PolicyGateConfig = field(default_factory=lambda: PolicyGateConfig(minimum_safe_depth="sentence"))

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("level id must not be empty")
        if not self.label.strip():
            raise ValueError("level label must not be blank")
        if not self.description.strip():
            raise ValueError("level description must not be blank")
        if self.base_mode not in {"pl_dominant", "mixed", "cs_dominant"}:
            raise ValueError(f"unsupported base mode: {self.base_mode}")
        if self.selection_policy not in ALLOWED_SELECTION_POLICIES:
            raise ValueError(f"unsupported selection policy: {self.selection_policy}")
        if self.global_state_policy not in ALLOWED_GLOBAL_STATE_POLICIES:
            raise ValueError(f"unsupported global state policy: {self.global_state_policy}")
        if not self.allowed_unit_classes:
            raise ValueError("level must allow at least one unit class")
        invalid_classes = [item for item in self.allowed_unit_classes if item not in ALLOWED_UNIT_CLASSES]
        if invalid_classes:
            raise ValueError(f"unsupported unit classes: {invalid_classes}")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "HybridizationLevel":
        return cls(
            id=str(data["id"]),
            label=str(data["label"]),
            description=str(data["description"]),
            allowed_unit_classes=[str(item) for item in data.get("allowed_unit_classes", [])],
            base_mode=str(data.get("base_mode", "pl_dominant")),
            selection_policy=str(data.get("selection_policy", "balanced")),
            global_state_policy=str(data.get("global_state_policy", "prefer_czech_after_intro")),
            budget=PolicyLevelBudget.from_dict(dict(data["budget"])),
            gates=PolicyGateConfig.from_dict(dict(data["gates"])),
        )


@dataclass(slots=True)
class HybridizationPolicy(SchemaModel):
    id: str
    version: str
    label: str
    description: str
    unit_class_definitions: dict[str, str]
    readability_axes: list[str]
    weights: PolicySelectionWeights
    levels: list[HybridizationLevel] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("policy id must not be empty")
        if not self.version:
            raise ValueError("policy version must not be empty")
        if not self.label.strip():
            raise ValueError("policy label must not be blank")
        if not self.description.strip():
            raise ValueError("policy description must not be blank")
        if sorted(self.unit_class_definitions.keys()) != ["A", "B", "C", "D"]:
            raise ValueError("unit_class_definitions must define exactly A, B, C, D")
        if self.readability_axes != ["didactic_exposure", "surface_czechness", "readability_load"]:
            raise ValueError("readability_axes must follow the canonical 3-axis model")
        if not self.levels:
            raise ValueError("policy must define at least one level")
        level_ids = [level.id for level in self.levels]
        if len(level_ids) != len(set(level_ids)):
            raise ValueError("policy level ids must be unique")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "HybridizationPolicy":
        return cls(
            id=str(data["id"]),
            version=str(data["version"]),
            label=str(data["label"]),
            description=str(data["description"]),
            unit_class_definitions={str(key): str(value) for key, value in dict(data["unit_class_definitions"]).items()},
            readability_axes=[str(item) for item in data.get("readability_axes", [])],
            weights=PolicySelectionWeights.from_dict(dict(data["weights"])),
            levels=[HybridizationLevel.from_dict(dict(item)) for item in data.get("levels", [])],
        )


@dataclass(slots=True)
class LoadPolicyResult(SchemaModel):
    path: str
    policy: HybridizationPolicy


def load_policy_from_json(path: str | Path) -> LoadPolicyResult:
    policy_path = Path(path)
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    return LoadPolicyResult(path=str(policy_path), policy=HybridizationPolicy.from_dict(payload))
