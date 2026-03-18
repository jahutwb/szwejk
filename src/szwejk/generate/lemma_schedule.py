"""Global residual-greedy scheduler for lemma-family introduction blocks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy


@dataclass(slots=True)
class BlockOption:
    family_index: int
    family_id: str
    intro_unit_index: int
    last_unit_index: int
    total_count: int
    gain: float
    gain_index: int


def load_lemma_timeline(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_family_schedule_report(
    timeline_payload: dict[str, object],
    *,
    target_power: float = 1.4,
    early_weight: float = 1.0,
    max_steps: int = 1000,
    min_improvement: float = 1e-9,
    late_fill_steps: int = 0,
    late_weight: float = 0.0,
    late_start_fraction: float = 0.6,
) -> dict[str, object]:
    problem = _build_problem(timeline_payload)
    target_curve = _target_curve(problem.unit_count, target_power=target_power, max_value=problem.reachable_max)
    weights = _prefix_weights(problem.unit_count, early_weight=early_weight)
    actual_curve = numpy.zeros(problem.unit_count, dtype=numpy.float64)
    available_options = problem.gains > 0.0

    debug_steps: list[dict[str, object]] = []
    selected_families: list[dict[str, object]] = []

    prefix_error = compute_prefix_error(actual_curve, target_curve, weights)
    step_counter = 0
    prefix_error, step_counter = _schedule_phase(
        phase_name="prefix",
        problem=problem,
        target_curve=target_curve,
        phase_weights=weights,
        available_options=available_options,
        actual_curve=actual_curve,
        selected_families=selected_families,
        debug_steps=debug_steps,
        min_improvement=min_improvement,
        max_steps=max_steps,
        starting_step=step_counter,
        min_intro_unit_index=1,
    )

    if late_fill_steps > 0 and late_weight > 0.0:
        late_weights = _late_weights(
            problem.unit_count,
            late_weight=late_weight,
            late_start_fraction=late_start_fraction,
        )
        prefix_error, step_counter = _schedule_phase(
            phase_name="late-fill",
            problem=problem,
            target_curve=target_curve,
            phase_weights=late_weights,
            available_options=available_options,
            actual_curve=actual_curve,
            selected_families=selected_families,
            debug_steps=debug_steps,
            min_improvement=min_improvement,
            max_steps=late_fill_steps,
            starting_step=step_counter,
            min_intro_unit_index=max(1, int(problem.unit_count * late_start_fraction)),
        )

    residual = target_curve - actual_curve
    checkpoints = _curve_checkpoints(actual_curve, target_curve)
    inspection = _inspection_payload(problem, target_curve)
    total_prefix_error = compute_prefix_error(actual_curve, target_curve, weights)

    return {
        "mode": "lemma-family-geometry-v3-residual-greedy",
        "target_power": target_power,
        "early_weight": early_weight,
        "max_steps": max_steps,
        "min_improvement": min_improvement,
        "late_fill_steps": late_fill_steps,
        "late_weight": late_weight,
        "late_start_fraction": late_start_fraction,
        "unit_count": problem.unit_count,
        "family_count": problem.family_count,
        "option_count": problem.option_count,
        "scheduled_family_count": len(selected_families),
        "reachable_max": round(problem.reachable_max, 8),
        "total_prefix_l1_error": round(total_prefix_error, 8),
        "early_section_error_20": round(_section_prefix_error(actual_curve, target_curve, weights, 0.2), 8),
        "max_overshoot": round(max(0.0, float(numpy.max(-residual))), 8),
        "max_undershoot": round(max(0.0, float(numpy.max(residual))), 8),
        "final_czechness": round(float(actual_curve[-1]) if len(actual_curve) else 0.0, 8),
        "target_final_czechness": round(float(target_curve[-1]) if len(target_curve) else 0.0, 8),
        "target_curve": [round(float(item), 8) for item in target_curve.tolist()],
        "actual_curve": [round(float(item), 8) for item in actual_curve.tolist()],
        "checkpoints": checkpoints,
        "inspection": inspection,
        "selected_families": selected_families,
        "debug_steps": debug_steps,
    }


def simulate_block(
    family_id: str,
    intro_unit_index: int,
    H: list[float] | numpy.ndarray,
    *,
    gain: float,
) -> list[float]:
    curve = numpy.array(H, dtype=numpy.float64, copy=True)
    if gain <= 0.0 or len(curve) == 0:
        return curve.tolist()
    start_index = min(len(curve), max(1, int(intro_unit_index))) - 1
    curve[start_index:] += gain
    return curve.tolist()


def compute_prefix_error(
    H: list[float] | numpy.ndarray,
    G: list[float] | numpy.ndarray,
    weights: list[float] | numpy.ndarray | None = None,
) -> float:
    actual = numpy.array(H, dtype=numpy.float64, copy=False)
    target = numpy.array(G, dtype=numpy.float64, copy=False)
    if weights is None:
        return float(numpy.abs(actual - target).sum())
    weight_array = numpy.array(weights, dtype=numpy.float64, copy=False)
    return float((weight_array * numpy.abs(actual - target)).sum())


@dataclass(slots=True)
class SchedulingProblem:
    unit_count: int
    family_count: int
    option_count: int
    reachable_max: float
    options: list[BlockOption]
    gains: numpy.ndarray
    gain_indices: numpy.ndarray
    start_indices: numpy.ndarray
    options_by_family: list[list[int]]
    unique_gains: numpy.ndarray
    family_ids: list[str]
    families_preview: list[dict[str, object]]


def _build_problem(timeline_payload: dict[str, object]) -> SchedulingProblem:
    families_payload = timeline_payload.get("families", [])
    unit_count = _max_unit_index(timeline_payload)
    unique_gain_values: set[float] = set()
    options: list[BlockOption] = []
    options_by_family: list[list[int]] = []
    family_ids: list[str] = []
    families_preview: list[dict[str, object]] = []

    for family_index, family in enumerate(families_payload):
        family_id = str(family.get("family_id", ""))
        if not family_id:
            continue
        family_ids.append(family_id)
        occurrences = family.get("occurrences", [])
        preview_occurrences = []
        family_option_indices: list[int] = []
        total_count = int(family.get("total_count", len(occurrences)))
        last_unit_index = int(family.get("last_unit_index", 0) or 0)
        for occurrence in occurrences:
            intro_unit_index = int(occurrence.get("unit_index", 0) or 0)
            if intro_unit_index <= 0:
                continue
            gain = float(occurrence.get("future_gain_if_introduced_here", 0.0) or 0.0)
            unique_gain_values.add(gain)
            option_index = len(options)
            options.append(
                BlockOption(
                    family_index=family_index,
                    family_id=family_id,
                    intro_unit_index=intro_unit_index,
                    last_unit_index=last_unit_index,
                    total_count=total_count,
                    gain=gain,
                    gain_index=-1,
                )
            )
            family_option_indices.append(option_index)
            preview_occurrences.append(
                {
                    "unit_index": intro_unit_index,
                    "future_gain_if_introduced_here": round(gain, 8),
                    "remaining_after_occurrence": occurrence.get("remaining_after_occurrence"),
                    "urgency": occurrence.get("urgency"),
                }
            )
        options_by_family.append(family_option_indices)
        if family_index < 10:
            families_preview.append(
                {
                    "family_id": family_id,
                    "total_count": total_count,
                    "last_unit_index": last_unit_index,
                    "occurrences": preview_occurrences,
                }
            )

    unique_gains = numpy.array(sorted(unique_gain_values), dtype=numpy.float64)
    gain_to_index = {float(value): index for index, value in enumerate(unique_gains.tolist())}
    for option_index, option in enumerate(options):
        options[option_index] = BlockOption(
            family_index=option.family_index,
            family_id=option.family_id,
            intro_unit_index=option.intro_unit_index,
            last_unit_index=option.last_unit_index,
            total_count=option.total_count,
            gain=option.gain,
            gain_index=gain_to_index[float(option.gain)],
        )

    gains = numpy.array([option.gain for option in options], dtype=numpy.float64)
    gain_indices = numpy.array([option.gain_index for option in options], dtype=numpy.int32)
    start_indices = numpy.array([option.intro_unit_index - 1 for option in options], dtype=numpy.int32)
    reachable_max = float(sum(max((options[index].gain for index in option_indices), default=0.0) for option_indices in options_by_family))

    return SchedulingProblem(
        unit_count=unit_count,
        family_count=len(options_by_family),
        option_count=len(options),
        reachable_max=reachable_max,
        options=options,
        gains=gains,
        gain_indices=gain_indices,
        start_indices=start_indices,
        options_by_family=options_by_family,
        unique_gains=unique_gains,
        family_ids=family_ids,
        families_preview=families_preview,
    )


def _compute_option_improvements(
    problem: SchedulingProblem,
    residual: numpy.ndarray,
    weights: numpy.ndarray,
    available_options: numpy.ndarray,
) -> numpy.ndarray | None:
    if not available_options.any():
        return None
    improvement_by_gain_and_start = _gain_suffix_improvements(problem.unique_gains, residual, weights)
    option_improvements = improvement_by_gain_and_start[problem.gain_indices, problem.start_indices]
    option_improvements = numpy.where(available_options, option_improvements, -numpy.inf)
    return option_improvements


def _schedule_phase(
    *,
    phase_name: str,
    problem: SchedulingProblem,
    target_curve: numpy.ndarray,
    phase_weights: numpy.ndarray,
    available_options: numpy.ndarray,
    actual_curve: numpy.ndarray,
    selected_families: list[dict[str, object]],
    debug_steps: list[dict[str, object]],
    min_improvement: float,
    max_steps: int,
    starting_step: int,
    min_intro_unit_index: int,
) -> tuple[float, int]:
    prefix_error = compute_prefix_error(actual_curve, target_curve, phase_weights)
    step_counter = starting_step
    for _ in range(max_steps):
        residual = target_curve - actual_curve
        option_improvements = _compute_option_improvements(problem, residual, phase_weights, available_options)
        if option_improvements is None:
            break
        if min_intro_unit_index > 1:
            option_improvements = numpy.where(
                problem.start_indices >= (min_intro_unit_index - 1),
                option_improvements,
                -numpy.inf,
            )
        best_option_index = int(numpy.argmax(option_improvements))
        best_improvement = float(option_improvements[best_option_index])
        if best_improvement <= min_improvement:
            break

        option = problem.options[best_option_index]
        start_index = option.intro_unit_index - 1
        actual_curve[start_index:] += option.gain
        prefix_error = compute_prefix_error(actual_curve, target_curve, phase_weights)

        family_index = option.family_index
        for option_index in problem.options_by_family[family_index]:
            available_options[option_index] = False

        overshoot = actual_curve - target_curve
        step_counter += 1
        selected_families.append(
            {
                "step": step_counter,
                "phase": phase_name,
                "family_id": option.family_id,
                "intro_unit_index": option.intro_unit_index,
                "total_count": option.total_count,
                "last_unit_index": option.last_unit_index,
                "future_gain_if_introduced_here": round(option.gain, 8),
                "improvement": round(best_improvement, 10),
                "prefix_error_after": round(prefix_error, 10),
            }
        )
        debug_steps.append(
            {
                "step": step_counter,
                "phase": phase_name,
                "family_id": option.family_id,
                "intro_unit_index": option.intro_unit_index,
                "gain": round(option.gain, 8),
                "improvement": round(best_improvement, 10),
                "prefix_error_after": round(prefix_error, 10),
                "overshoot_max_after": round(max(0.0, float(numpy.max(overshoot))), 10),
                "undershoot_max_after": round(max(0.0, float(numpy.max(-overshoot))), 10),
            }
        )
    return prefix_error, step_counter


def _gain_suffix_improvements(
    unique_gains: numpy.ndarray,
    residual: numpy.ndarray,
    weights: numpy.ndarray,
) -> numpy.ndarray:
    if len(unique_gains) == 0 or len(residual) == 0:
        return numpy.zeros((0, len(residual)), dtype=numpy.float64)
    rows = numpy.empty((len(unique_gains), len(residual)), dtype=numpy.float64)
    abs_residual = numpy.abs(residual)
    for gain_index, gain in enumerate(unique_gains):
        delta = weights * (abs_residual - numpy.abs(residual - gain))
        rows[gain_index] = numpy.cumsum(delta[::-1])[::-1]
    return rows


def _target_curve(unit_count: int, *, target_power: float, max_value: float) -> numpy.ndarray:
    if unit_count <= 0:
        return numpy.zeros(0, dtype=numpy.float64)
    progress = numpy.arange(1, unit_count + 1, dtype=numpy.float64) / float(unit_count)
    return max_value * numpy.power(progress, target_power)


def _prefix_weights(unit_count: int, *, early_weight: float) -> numpy.ndarray:
    if unit_count <= 0:
        return numpy.zeros(0, dtype=numpy.float64)
    progress = numpy.arange(1, unit_count + 1, dtype=numpy.float64) / float(unit_count)
    return 1.0 + early_weight * (1.0 - progress)


def _late_weights(
    unit_count: int,
    *,
    late_weight: float,
    late_start_fraction: float,
) -> numpy.ndarray:
    if unit_count <= 0:
        return numpy.zeros(0, dtype=numpy.float64)
    progress = numpy.arange(1, unit_count + 1, dtype=numpy.float64) / float(unit_count)
    start = min(0.999999, max(0.0, late_start_fraction))
    tail_progress = numpy.clip((progress - start) / max(1e-9, 1.0 - start), 0.0, 1.0)
    return 1.0 + late_weight * tail_progress


def _max_unit_index(timeline_payload: dict[str, object]) -> int:
    max_index = 0
    for family in timeline_payload.get("families", []):
        max_index = max(max_index, int(family.get("last_unit_index", 0) or 0))
        for occurrence in family.get("occurrences", []):
            max_index = max(max_index, int(occurrence.get("unit_index", 0) or 0))
    return max_index


def _section_prefix_error(
    actual_curve: numpy.ndarray,
    target_curve: numpy.ndarray,
    weights: numpy.ndarray,
    fraction: float,
) -> float:
    if len(actual_curve) == 0:
        return 0.0
    end = min(len(actual_curve), max(1, int(round(len(actual_curve) * fraction))))
    return float((weights[:end] * numpy.abs(actual_curve[:end] - target_curve[:end])).sum())


def _curve_checkpoints(actual: numpy.ndarray, target: numpy.ndarray) -> list[dict[str, object]]:
    checkpoints: list[dict[str, object]] = []
    if len(actual) == 0:
        return checkpoints
    for fraction in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        index = min(len(actual) - 1, max(0, int(round((len(actual) - 1) * fraction))))
        checkpoints.append(
            {
                "fraction": fraction,
                "unit_index": index + 1,
                "target": round(float(target[index]), 8),
                "actual": round(float(actual[index]), 8),
            }
        )
    return checkpoints


def _inspection_payload(problem: SchedulingProblem, target_curve: numpy.ndarray) -> dict[str, object]:
    preview_count = min(20, len(target_curve))
    return {
        "target_curve_head": [round(float(item), 8) for item in target_curve[:preview_count].tolist()],
        "first_10_families": problem.families_preview,
    }
