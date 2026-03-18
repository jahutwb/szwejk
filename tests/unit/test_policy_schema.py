from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from szwejk.policy import HybridizationPolicy, load_policy_from_json


class PolicySchemaUnitTest(unittest.TestCase):
    def test_load_policy_from_json_validates_default_contract(self) -> None:
        result = load_policy_from_json("data/policies/progressive_czechization_v1.json")

        self.assertEqual(result.policy.id, "progressive-czechization-v1")
        self.assertEqual(len(result.policy.levels), 4)
        self.assertEqual(result.policy.levels[0].allowed_unit_classes, ["A"])
        self.assertEqual(result.policy.levels[-1].base_mode, "cs_dominant")

    def test_policy_rejects_non_canonical_axes(self) -> None:
        payload = {
            "id": "broken",
            "version": "1",
            "label": "Broken",
            "description": "Broken policy",
            "unit_class_definitions": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "readability_axes": ["surface_czechness"],
            "weights": {
                "target_fit": 1.0,
                "didactic_value": 1.0,
                "cohesion_bonus": 1.0,
                "unnaturalness_penalty": 1.0,
                "inflection_risk_penalty": 1.0,
                "ambiguity_risk_penalty": 1.0,
            },
            "levels": [
                {
                    "id": "l1",
                    "label": "L1",
                    "description": "desc",
                    "allowed_unit_classes": ["A"],
                    "budget": {
                        "target_exposure": 0.1,
                        "target_surface_czechness": 0.1,
                        "max_readability_load": 0.2,
                        "max_new_families_per_1000_tokens": 5,
                    },
                    "gates": {
                        "minimum_safe_depth": "sentence",
                    },
                }
            ],
        }
        with self.assertRaises(ValueError):
            HybridizationPolicy.from_dict(payload)

    def test_policy_loader_reads_temp_file(self) -> None:
        payload = {
            "id": "temp-policy",
            "version": "1",
            "label": "Temp",
            "description": "Temporary",
            "unit_class_definitions": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "readability_axes": ["didactic_exposure", "surface_czechness", "readability_load"],
            "weights": {
                "target_fit": 1.0,
                "didactic_value": 1.0,
                "cohesion_bonus": 1.0,
                "unnaturalness_penalty": 1.0,
                "inflection_risk_penalty": 1.0,
                "ambiguity_risk_penalty": 1.0,
            },
            "levels": [
                {
                    "id": "l1",
                    "label": "L1",
                    "description": "desc",
                    "allowed_unit_classes": ["A"],
                    "budget": {
                        "target_exposure": 0.1,
                        "target_surface_czechness": 0.1,
                        "max_readability_load": 0.2,
                        "max_new_families_per_1000_tokens": 5,
                    },
                    "gates": {
                        "minimum_safe_depth": "sentence",
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "policy.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_policy_from_json(path)

        self.assertEqual(loaded.policy.id, "temp-policy")


if __name__ == "__main__":
    unittest.main()
