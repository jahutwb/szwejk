"""Candidate selection algorithms for paragraph hybridization."""
from __future__ import annotations

from collections import Counter, defaultdict

from szwejk.generate.alignment_quality import (
    alignment_quality_score,
    alignment_quality_threshold,
    linguistic_similarity_score,
)
from szwejk.generate.candidate_model import (
    ParagraphHybridCandidate,
    candidate_from_dict as _candidate_from_dict,
    _is_span_granularity,
    _display_granularity,
    VISIBLE_FUTURE_BLEND,
)
from szwejk.generate.plan_metrics import (
    _word_count,
    _target_token_keys,
    _source_token_keys,
    _covered_target_tokens,
    _candidate_simple_mass,
    _target_span_word_count,
    _candidate_marginal_target_simple_mass,
    _selected_simple_mass,
    _selected_marginal_target_simple_mass,
    _visible_czechness,
    _future_czechness,
    _target_future_czechness,
    _visible_target,
    _simple_czechness,
    _mean,
)
from szwejk.generate.family_tracking import (
    _cost_family_ids,
    _candidate_cost_target_lemmas,
    _build_family_weight_lookup,
)
from szwejk.generate.candidate_scoring import (
    _text_similarity,
    _candidate_visible_mass,
    _selected_visible_mass,
    _candidate_idiomaticity,
    _semantic_confidence,
    _structural_support,
    _candidate_allowed_by_safety,
    _confidence_penalty,
    _late_visible_bonus,
    _effective_new_family_count,
    _new_family_penalty,
    _known_context_mass,
    _known_context_credit,
    _early_carrier_penalty,
    _local_scope_penalty,
    _uncovered_content_penalty,
    _lemma_urgency_bonus,
    _lemma_deferrability_penalty,
    _target_lemma_penalty,
    _family_schedule_penalty,
    _family_schedule_bonus,
    _idiomaticity_penalty,
    _granularity_progress_penalty,
    _smooth_acceptance_slack,
)
from szwejk.generate.candidate_filters import (
    _conflicts,
    _spans_overlap,
    _span_contains,
    _candidate_contains,
    _candidate_allowed_as_surface_carrier,
    _candidate_has_better_narrower_carrier,
    _token_can_stand_alone,
    _candidate_allowed_by_progress,
    _candidate_allowed_by_promotion_safety,
    _candidate_allowed_by_promotion_shape,
    _candidate_blocked_by_family_schedule,
    _promotion_score_floor,
)


# ---------------------------------------------------------------------------
# Extracted selection functions (verbatim from paragraph_hybridization.py)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Three-phase selection helpers
# ---------------------------------------------------------------------------

GRATIS_FAMILIARITY_THRESHOLD = 0.89


def _source_lemmas_from_fid(fid: str) -> list[str]:
    left = fid.split("::")[0] if "::" in fid else fid
    return [p.strip().lower() for p in left.split("+") if p.strip()]


def _familiarity_ratio(
    candidate: ParagraphHybridCandidate,
    introduced_families: set[str],
    introduced_source_lemmas: set[str],
) -> float:
    fids = _cost_family_ids(candidate)
    if not fids:
        return 0.0
    known = sum(
        1
        for fid in fids
        if fid in introduced_families
        or any(lemma in introduced_source_lemmas for lemma in _source_lemmas_from_fid(fid))
    )
    return known / len(fids)


def _approaches_curve(
    current_mass: float,
    marginal_mass: float,
    target: float,
    seen_word_count: int,
    visible_delta: float,
) -> bool:
    """True when taking a candidate with the given marginal mass moves cumulative
    czechness closer to the target (with a small tolerance for cognate-like candidates
    that have low visible mass and thus low reader impact)."""
    if seen_word_count <= 0 or marginal_mass <= 0.0:
        return False
    current = current_mass / seen_word_count
    after = (current_mass + marginal_mass) / seen_word_count
    effective_visible = visible_delta / seen_word_count
    tolerance = max(0.0, 0.02 - effective_visible)
    return abs(after - target) < abs(current - target) + tolerance


def _introduce_locally(
    candidate: ParagraphHybridCandidate,
    local_families: set[str],
    local_source_lemmas: set[str],
    local_target_lemmas: set[str],
) -> None:
    """Update local within-unit tracking sets after selecting a candidate."""
    local_families.update(candidate.family_ids)
    for fid in _cost_family_ids(candidate):
        local_source_lemmas.update(_source_lemmas_from_fid(fid))
    local_target_lemmas.update(_candidate_cost_target_lemmas(candidate))


def _select_candidates_three_phase(
    candidates: list[ParagraphHybridCandidate],
    *,
    introduced_families: set[str],
    introduced_target_lemmas: set[str],
    cumulative_simple_mass: float,
    seen_word_count_after: int,
    target_cumulative_after: float,
    progress: float,
    wiktionary_lookup: dict[str, list[str]] | None = None,
    pl_to_cs_lookup: dict[str, list[str]] | None = None,
) -> list[ParagraphHybridCandidate]:
    """Three-phase candidate selector for the cumulative-simple policy.

    Phase 1 — GRATIS
        Candidates where ≥89 % of cost lemmas are already familiar.
        Taken without checking the curve; any unfamiliar lemmas are introduced
        via the normal path (they count as newly introduced for phases 2-3).

    Phase 2 — FILTER  (no czechness update between candidates)
        Remaining candidates sorted by alignment_quality_score.
        Stop when quality drops below per-granularity threshold.
        Skip candidates that would move czechness away from the curve.
        Survivors form the pool.

    Phase 3 — GREEDY from pool  (czechness updated after each selection)
        Pool re-sorted by (alignment_quality_score DESC, linguistic_similarity DESC).
        Take each candidate that approaches the curve; skip those that don't
        (continue checking the rest of the pool — never stop early).
    """
    # ── derive introduced source lemmas from introduced_families ────────────
    intro_source: set[str] = set()
    for fid in introduced_families:
        intro_source.update(_source_lemmas_from_fid(fid))

    local_fam = set(introduced_families)
    local_src = set(intro_source)
    local_tgt = set(introduced_target_lemmas)
    covered: set[tuple[str, int]] = set()
    running_mass = cumulative_simple_mass

    # ── PHASE 1 ─────────────────────────────────────────────────────────────
    gratis_pool = [
        c for c in candidates
        if _familiarity_ratio(c, local_fam, local_src) >= GRATIS_FAMILIARITY_THRESHOLD
        and _candidate_allowed_as_surface_carrier(c)
    ]
    _aq = lambda c: alignment_quality_score(c, wiktionary_lookup, pl_to_cs_lookup)  # noqa: E731
    gratis_pool.sort(key=lambda c: (
        -float(_target_span_word_count(c)),
        -_aq(c),
        c.candidate_id,
    ))
    selected_gratis: list[ParagraphHybridCandidate] = []
    for cand in gratis_pool:
        if any(_conflicts(cand, other) for other in selected_gratis):
            continue
        selected_gratis.append(cand)
        keys = _target_token_keys(cand)
        running_mass += float(len(keys - covered))
        covered.update(keys)
        _introduce_locally(cand, local_fam, local_src, local_tgt)

    gratis_ids = {c.candidate_id for c in selected_gratis}

    # ── PHASE 2 ─────────────────────────────────────────────────────────────
    remaining = [
        c for c in candidates
        if c.candidate_id not in gratis_ids
        and _candidate_allowed_as_surface_carrier(c)
        and not _candidate_blocked_by_family_schedule(c, paragraph_families=local_fam)
    ]
    remaining.sort(key=lambda c: (-_aq(c), c.candidate_id))

    pool: list[ParagraphHybridCandidate] = []
    for cand in remaining:
        aq = _aq(cand)
        if aq < alignment_quality_threshold(cand.granularity):
            break  # list is sorted — tail is all below threshold
        keys = _target_token_keys(cand)
        marginal = float(len(keys - covered))
        if marginal <= 0.0:
            continue
        if _approaches_curve(
            running_mass, marginal, target_cumulative_after,
            seen_word_count_after, _candidate_visible_mass(cand),
        ):
            pool.append(cand)
        # else: skip but keep scanning

    # ── PHASE 3 ─────────────────────────────────────────────────────────────
    pool.sort(key=lambda c: (
        -_aq(c),
        -linguistic_similarity_score(c, wiktionary_lookup),
        c.candidate_id,
    ))
    selected_new: list[ParagraphHybridCandidate] = []
    p3_covered = set(covered)
    p3_mass = running_mass
    p3_fam = set(local_fam)
    p3_src = set(local_src)
    p3_tgt = set(local_tgt)

    for cand in pool:
        if any(_conflicts(cand, other) for other in selected_new):
            continue
        keys = _target_token_keys(cand)
        marginal = float(len(keys - p3_covered))
        if marginal <= 0.0:
            continue
        if not _approaches_curve(
            p3_mass, marginal, target_cumulative_after,
            seen_word_count_after, _candidate_visible_mass(cand),
        ):
            continue  # skip, check next — never stop early
        selected_new.append(cand)
        p3_covered.update(keys)
        p3_mass += marginal
        _introduce_locally(cand, p3_fam, p3_src, p3_tgt)

    return selected_gratis + selected_new


def _promotion_known_lemma_threshold(candidate: ParagraphHybridCandidate, progress: float) -> float:
    _ = progress
    if candidate.granularity == "paragraph":
        return 0.95
    return 0.85



def _select_known_candidates(
    candidates: list[ParagraphHybridCandidate],
    introduced_families: set[str],
    *,
    introduced_target_lemmas: set[str] | None = None,
    granularity_policy: str = "staircase",
    target_after: float = 0.0,
    current_simple_mass: float = 0.0,
    seen_word_count_after: int = 0,
) -> tuple[list[ParagraphHybridCandidate], set[str]]:
    introduced_target_lemmas = introduced_target_lemmas or set()
    selected: list[ParagraphHybridCandidate] = []
    blocked_ids: set[str] = set()
    known_candidates = [
        item
        for item in candidates
        if (
            _candidate_cost_target_lemmas(item)
            and all(lemma in introduced_target_lemmas for lemma in _candidate_cost_target_lemmas(item))
        )
    ]
    def _known_sort_key(item: ParagraphHybridCandidate) -> tuple[float, float, str]:
        if granularity_policy == "cumulative-simple":
            return (
                -float(_target_span_word_count(item)),
                -item.score,
                item.candidate_id,
            )
        return (
            -item.score,
            0.0,
            item.candidate_id,
        )

    for candidate in sorted(known_candidates, key=_known_sort_key):
        if not _candidate_allowed_as_surface_carrier(candidate):
            blocked_ids.add(candidate.candidate_id)
            continue
        if any(_conflicts(candidate, other) for other in selected):
            blocked_ids.add(candidate.candidate_id)
            continue
        selected.append(candidate)
    return selected, {item.candidate_id for item in selected} | blocked_ids



def _select_introducing_candidates(
    candidates: list[ParagraphHybridCandidate],
    *,
    selected_candidates: list[ParagraphHybridCandidate],
    introduced_families: set[str],
    introduced_target_lemmas: set[str],
    remaining_after: Counter[str],
    remaining_target_after: Counter[str],
    family_weight_lookup: dict[str, float],
    target_after: float,
    progress: float,
    idiomaticity_penalty_weight: float,
    granularity_policy: str,
    current_simple_mass: float,
    current_visible_mass: float,
    seen_word_count_after: int,
) -> tuple[list[ParagraphHybridCandidate], set[str]]:
    chosen: list[ParagraphHybridCandidate] = []
    removed_ids: set[str] = set()
    paragraph_target_lemmas = set(introduced_target_lemmas)
    paragraph_families = set(introduced_families)
    if granularity_policy == "visible-smooth":
        current_distance = abs(_visible_czechness(current_visible_mass, seen_word_count_after) - target_after)
    elif granularity_policy == "cumulative-simple":
        current_distance = abs(_simple_czechness(current_simple_mass, seen_word_count_after) - target_after)
    else:
        current_distance = abs(_target_future_czechness(remaining_target_after, paragraph_target_lemmas) - target_after)
    current_target_lemma_distance = abs(_target_future_czechness(remaining_target_after, paragraph_target_lemmas) - target_after)
    simple_mass = current_simple_mass
    available = list(candidates)
    visible_mass = current_visible_mass
    current_selected = list(selected_candidates)
    initially_selected_ids = {candidate.candidate_id for candidate in selected_candidates}

    while available:
        ranked = []
        for candidate in available:
            replaceable_conflicts: list[ParagraphHybridCandidate] = []
            hard_conflict = False
            for other in current_selected:
                if not _conflicts(candidate, other):
                    continue
                if granularity_policy == "cumulative-simple" and _candidate_contains(candidate, other):
                    replaceable_conflicts.append(other)
                    continue
                hard_conflict = True
                break
            if hard_conflict:
                continue
            if not _candidate_allowed_as_surface_carrier(candidate):
                continue
            if _candidate_has_better_narrower_carrier(
                candidate,
                candidates=available,
                paragraph_target_lemmas=paragraph_target_lemmas,
            ):
                continue
            if not _candidate_allowed_by_safety(candidate):
                continue
            if _candidate_blocked_by_family_schedule(candidate, paragraph_families=paragraph_families):
                continue
            new_target_lemmas = [lemma for lemma in _candidate_cost_target_lemmas(candidate) if lemma not in paragraph_target_lemmas]
            if not new_target_lemmas:
                continue
            after_target_lemmas = paragraph_target_lemmas | set(new_target_lemmas)
            target_lemma_distance = abs(_target_future_czechness(remaining_target_after, after_target_lemmas) - target_after)
            effective_new_family_count = float(len(set(new_target_lemmas)))
            known_context_mass = _known_context_mass(candidate, paragraph_target_lemmas=paragraph_target_lemmas)
            contextual_credit = _known_context_credit(
                candidate,
                known_context_mass=known_context_mass,
                effective_new_family_count=effective_new_family_count,
                progress=progress,
            )
            early_carrier_penalty = _early_carrier_penalty(
                candidate,
                progress=progress,
                effective_new_family_count=effective_new_family_count,
                known_context_mass=known_context_mass,
            )
            deferrability_penalty = _lemma_deferrability_penalty(candidate, progress=progress)
            weighted_distance = target_lemma_distance + _confidence_penalty(candidate) + _idiomaticity_penalty(
                candidate,
                progress=progress,
                idiomaticity_penalty_weight=idiomaticity_penalty_weight,
            ) + early_carrier_penalty + deferrability_penalty - _lemma_urgency_bonus(candidate, progress=progress) - contextual_credit
            if granularity_policy == "smooth":
                weighted_distance += _granularity_progress_penalty(candidate, progress=progress)
            elif granularity_policy == "visible-smooth":
                visible_after = _visible_czechness(visible_mass + _candidate_visible_mass(candidate), seen_word_count_after)
                visible_distance = abs(visible_after - target_after)
                weighted_distance = (
                    visible_distance
                    + (0.35 * target_lemma_distance)
                    + _confidence_penalty(candidate)
                    + _idiomaticity_penalty(
                        candidate,
                        progress=progress,
                        idiomaticity_penalty_weight=idiomaticity_penalty_weight,
                    )
                    + early_carrier_penalty
                    + deferrability_penalty
                    + _granularity_progress_penalty(candidate, progress=progress)
                    - _lemma_urgency_bonus(candidate, progress=progress)
                    - contextual_credit
                )
            elif granularity_policy == "cumulative-simple":
                covered_target_tokens = _covered_target_tokens(
                    [item for item in current_selected if item not in replaceable_conflicts]
                )
                marginal_simple_gain = _candidate_marginal_target_simple_mass(
                    candidate,
                    covered_target_tokens=covered_target_tokens,
                )
                if marginal_simple_gain <= 0.0:
                    continue
                simple_after = _simple_czechness(simple_mass + marginal_simple_gain, seen_word_count_after)
                simple_distance = abs(simple_after - target_after)
                # Rank cumulative-simple candidates purely by fit to the
                # cumulative target; novelty stays only as a secondary sort key.
                weighted_distance = simple_distance
            elif granularity_policy == "reader-visible-late":
                weighted_distance = (
                    target_lemma_distance
                    + _confidence_penalty(candidate)
                    + _idiomaticity_penalty(
                        candidate,
                        progress=progress,
                        idiomaticity_penalty_weight=idiomaticity_penalty_weight,
                    )
                    + early_carrier_penalty
                    + deferrability_penalty
                    - _late_visible_bonus(candidate, progress=progress)
                    - _lemma_urgency_bonus(candidate, progress=progress)
                    - contextual_credit
                )
            ranked.append(
                (
                    weighted_distance,
                    target_lemma_distance,
                    effective_new_family_count,
                    -known_context_mass,
                    -candidate.score,
                    tuple(item.candidate_id for item in replaceable_conflicts),
                    candidate,
                )
            )
        if not ranked:
            break
        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5], item[6].candidate_id))
        best_weighted_distance, best_target_lemma_distance, _eff_new, _known_mass, _score, replaceable_ids, best_candidate = ranked[0]
        if granularity_policy in {"smooth", "visible-smooth", "reader-visible-late", "cumulative-simple"}:
            slack = _smooth_acceptance_slack(best_candidate, progress=progress)
            if granularity_policy == "visible-smooth":
                slack = max(slack, 0.02 + ((1.0 - progress) * 0.035))
            elif granularity_policy == "cumulative-simple":
                slack = 0.0
            elif granularity_policy == "reader-visible-late":
                slack = max(slack, 0.004 + ((progress**2.1) * 0.028))
            current_acceptance_distance = current_distance if granularity_policy in {"visible-smooth", "cumulative-simple"} else current_target_lemma_distance
            if granularity_policy == "cumulative-simple":
                if best_weighted_distance >= current_acceptance_distance:
                    break
            elif best_weighted_distance > current_acceptance_distance + slack:
                break
        elif best_target_lemma_distance > current_distance:
                break
        if granularity_policy == "cumulative-simple" and replaceable_ids:
            removed = [candidate for candidate in current_selected if candidate.candidate_id in set(replaceable_ids)]
            removed_ids.update(item.candidate_id for item in removed if item.candidate_id in initially_selected_ids)
            current_selected = [candidate for candidate in current_selected if candidate.candidate_id not in set(replaceable_ids)]
        chosen.append(best_candidate)
        current_selected.append(best_candidate)
        paragraph_families.update(best_candidate.family_ids)
        paragraph_target_lemmas.update(_candidate_cost_target_lemmas(best_candidate))
        if granularity_policy == "visible-smooth":
            visible_mass += _candidate_visible_mass(best_candidate)
            current_distance = abs(_visible_czechness(visible_mass, seen_word_count_after) - target_after)
        elif granularity_policy == "cumulative-simple":
            covered_target_tokens = _covered_target_tokens(
                [candidate for candidate in current_selected if candidate.candidate_id != best_candidate.candidate_id]
            )
            simple_mass += _candidate_marginal_target_simple_mass(
                best_candidate,
                covered_target_tokens=covered_target_tokens,
            )
            current_distance = abs(_simple_czechness(simple_mass, seen_word_count_after) - target_after)
        else:
            current_distance = best_target_lemma_distance
        current_target_lemma_distance = best_target_lemma_distance
        available = [candidate for candidate in available if candidate.candidate_id != best_candidate.candidate_id]

    return chosen, removed_ids



def _promote_large_carriers(
    *,
    selected_candidates: list[ParagraphHybridCandidate],
    candidates: list[ParagraphHybridCandidate],
    introduced_target_lemmas: set[str],
    progress: float,
) -> list[ParagraphHybridCandidate]:
    current = list(selected_candidates)
    by_id = {candidate.candidate_id: candidate for candidate in current}
    sentence_pool = sorted(
        [
            candidate
            for candidate in candidates
            if candidate.granularity == "sentence"
            and candidate.candidate_id not in by_id
            and _candidate_allowed_as_surface_carrier(candidate)
            and _candidate_allowed_by_promotion_safety(candidate)
        ],
        key=lambda item: (
            item.scope_id,
            -float(_target_span_word_count(item)),
            -item.score,
            item.candidate_id,
        ),
    )
    for candidate in sentence_pool:
        scope_selected = [item for item in current if item.scope_id == candidate.scope_id]
        if len(scope_selected) < 1:
            continue
        conflicts = [item for item in scope_selected if _conflicts(candidate, item)]
        if len(conflicts) < 1:
            continue
        candidate_lemmas = set(_candidate_cost_target_lemmas(candidate))
        if not candidate_lemmas:
            continue
        base_lemmas = set(introduced_target_lemmas)
        for item in conflicts:
            base_lemmas.update(_candidate_cost_target_lemmas(item))
        known_ratio = len(candidate_lemmas & base_lemmas) / max(len(candidate_lemmas), 1)
        if known_ratio < _promotion_known_lemma_threshold(candidate, progress):
            continue
        conflict_ids = {item.candidate_id for item in conflicts}
        current = [item for item in current if item.candidate_id not in conflict_ids]
        current.append(candidate)

    by_id = {candidate.candidate_id: candidate for candidate in current}
    candidate_pool = sorted(
        [
            candidate
            for candidate in candidates
            if candidate.granularity == "paragraph"
            and candidate.candidate_id not in by_id
            and _candidate_allowed_as_surface_carrier(candidate)
            and _candidate_allowed_by_promotion_safety(candidate)
        ],
        key=lambda item: (
            item.scope_id,
            -float(_target_span_word_count(item)),
            -item.score,
            item.candidate_id,
        ),
    )

    for candidate in candidate_pool:
        if candidate.granularity == "paragraph":
            pool_selected = list(current)
        else:
            pool_selected = [item for item in current if item.scope_id == candidate.scope_id]
        if len(pool_selected) < 1:
            continue
        conflicts = [item for item in pool_selected if _conflicts(candidate, item)]
        if len(conflicts) < 1:
            continue
        candidate_lemmas = set(_candidate_cost_target_lemmas(candidate))
        if not candidate_lemmas:
            continue
        base_lemmas = set(introduced_target_lemmas)
        for item in conflicts:
            base_lemmas.update(_candidate_cost_target_lemmas(item))
        known_ratio = len(candidate_lemmas & base_lemmas) / max(len(candidate_lemmas), 1)
        if known_ratio < _promotion_known_lemma_threshold(candidate, progress):
            continue
        if not _candidate_allowed_by_promotion_shape(candidate, conflicts=conflicts, base_lemmas=base_lemmas):
            continue
        conflict_ids = {item.candidate_id for item in conflicts}
        current = [item for item in current if item.candidate_id not in conflict_ids]
        current.append(candidate)
    current.sort(key=lambda item: item.candidate_id)
    return current





def _apply_chapter_simple_boost(
    *,
    plan_units: list[dict[str, object]],
    units: list[dict[str, object]],
    target_power: float,
    target_max: float,
    boost_start: float,
) -> None:
    unit_lookup = {int(unit["unit_index"]): unit for unit in units}
    chapter_groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for plan_unit in plan_units:
        chapter_groups[int(plan_unit["chapter_pair"][0])].append(plan_unit)

    chapter_indices = sorted(chapter_groups)
    total_chapters = max(len(chapter_indices), 1)

    for ordinal, chapter_index in enumerate(chapter_indices, start=1):
        progress = ordinal / total_chapters
        if progress < boost_start:
            continue
        target_simple = min(1.0, target_max * (progress**target_power))
        chapter_units = chapter_groups[chapter_index]
        chapter_word_count = sum(int(unit.get("source_word_count", 0)) for unit in chapter_units)
        if chapter_word_count <= 0:
            continue

        current_simple_mass = sum(
            _candidate_simple_mass(_candidate_from_dict(candidate))
            for unit in chapter_units
            for candidate in unit.get("selected_candidates", [])
        )
        current_simple = current_simple_mass / chapter_word_count
        if current_simple >= target_simple:
            continue

        while current_simple < target_simple:
            best_pick: tuple[float, dict[str, object], ParagraphHybridCandidate, list[ParagraphHybridCandidate]] | None = None
            for plan_unit in chapter_units:
                unit_data = unit_lookup.get(int(plan_unit["unit_index"]))
                if unit_data is None:
                    continue
                selected = [_candidate_from_dict(item) for item in plan_unit.get("selected_candidates", [])]
                selected_ids = {candidate.candidate_id for candidate in selected}
                selected_target_lemmas = {
                    lemma
                    for selected_candidate in selected
                    for lemma in _candidate_cost_target_lemmas(selected_candidate)
                }
                for candidate in unit_data.get("candidates", []):
                    if candidate.is_fallback or not candidate.family_ids or candidate.candidate_id in selected_ids:
                        continue
                    conflicting = [other for other in selected if _conflicts(candidate, other)]
                    gain = _candidate_simple_mass(candidate) - sum(_candidate_simple_mass(other) for other in conflicting)
                    visible_delta = _candidate_visible_mass(candidate) - sum(_candidate_visible_mass(other) for other in conflicting)
                    if gain <= 0 and visible_delta <= 0:
                        continue
                    penalty = (
                        _confidence_penalty(candidate)
                        + _uncovered_content_penalty(
                            candidate,
                            paragraph_target_lemmas=selected_target_lemmas,
                            progress=progress,
                        )
                    )
                    value = gain + (0.18 * visible_delta) - (9.0 * penalty)
                    if best_pick is None or value > best_pick[0]:
                        best_pick = (value, plan_unit, candidate, conflicting)
            if best_pick is None or best_pick[0] <= 0.0:
                break
            _value, plan_unit, candidate, conflicting = best_pick
            conflicting_ids = {item.candidate_id for item in conflicting}
            selected_payloads = [
                item for item in plan_unit.get("selected_candidates", [])
                if str(item["candidate_id"]) not in conflicting_ids
            ]
            selected_payloads.append(candidate.to_dict())
            plan_unit["selected_candidates"] = selected_payloads
            plan_unit["selected_count"] = len(plan_unit["selected_candidates"])
            plan_unit["selected_granularity_counts"] = dict(
                Counter(_display_granularity(str(item["granularity"])) for item in plan_unit["selected_candidates"])
            )
            current_simple_mass += _candidate_simple_mass(candidate) - sum(_candidate_simple_mass(other) for other in conflicting)
            current_simple = current_simple_mass / chapter_word_count



def _refresh_plan_metrics(
    *,
    plan_units: list[dict[str, object]],
    units: list[dict[str, object]],
    total_occurrence_count: int,
    total_word_count: int,
    family_weight_lookup: dict[str, float],
) -> None:
    unit_lookup = {int(unit["unit_index"]): unit for unit in units}
    remaining_occurrences = Counter()
    for unit in units:
        remaining_occurrences.update(unit["family_occurrences"])

    introduced_families: set[str] = set()
    cumulative_occurrence_count = 0
    cumulative_word_count = 0
    cumulative_visible_mass = 0.0

    for plan_unit in plan_units:
        unit_data = unit_lookup[int(plan_unit["unit_index"])]
        cumulative_occurrence_count += int(unit_data["source_occurrence_count"])
        cumulative_word_count += int(unit_data["source_word_count"])
        remaining_after = remaining_occurrences.copy()
        remaining_after.subtract(unit_data["family_occurrences"])
        remaining_after = Counter({key: value for key, value in remaining_after.items() if value > 0})
        actual_before = _future_czechness(remaining_after, introduced_families, family_weight_lookup=family_weight_lookup)
        visible_before = _visible_czechness(cumulative_visible_mass, cumulative_word_count)
        selected_candidates = [_candidate_from_dict(item) for item in plan_unit.get("selected_candidates", [])]
        for candidate in selected_candidates:
            introduced_families.update(candidate.family_ids)
        cumulative_visible_mass += _selected_visible_mass(selected_candidates)
        actual_after = _future_czechness(remaining_after, introduced_families, family_weight_lookup=family_weight_lookup)
        visible_after = _visible_czechness(cumulative_visible_mass, cumulative_word_count)
        plan_unit["selected_count"] = len(selected_candidates)
        plan_unit["selected_granularity_counts"] = dict(
            Counter(_display_granularity(candidate.granularity) for candidate in selected_candidates)
        )
        plan_unit["actual_future_czechness_before"] = round(actual_before, 6)
        plan_unit["actual_future_czechness_after"] = round(actual_after, 6)
        plan_unit["actual_visible_czechness_before"] = round(visible_before, 6)
        plan_unit["actual_visible_czechness_after"] = round(visible_after, 6)
        plan_unit["visible_mass_after"] = round(cumulative_visible_mass, 6)
        plan_unit["distance_after"] = round(abs(actual_after - float(plan_unit["target_future_czechness_after"])), 6)
        plan_unit["introduced_family_count"] = len(introduced_families)
        remaining_occurrences = remaining_after

