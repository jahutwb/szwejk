# Lemma Timeline Artifact

This document defines the `lemma timeline` artifact introduced during hybridization tuning on 2026-03-17.

## Purpose

The hybridization planner originally optimized mainly for:

- closeness to target `future czechness`
- confidence / safety of the aligned candidate

That objective misses an important distinction:

- some lemma families still have many future opportunities to be introduced later
- others are close to disappearing from the book

The `lemma timeline` artifact makes that distinction explicit.

## Source of Truth

The artifact is built from the full alignment backbone:

- [full_hierarchical_alignment_e5_stanza.json](/home/jahu/PycharmProjects/szwejk/data/reports/full_hierarchical_alignment_e5_stanza.json)

It reuses the same paragraph-block unitization as the paragraph hybridization planner.

## File Shape

Top-level fields:

- `family_count`
- `total_occurrence_count`
- `families`

Each family entry:

- `family_id`
- `total_count`
- `first_unit_index`
- `last_unit_index`
- `occurrences`

Each occurrence entry:

- `unit_index`
- `chapter_pair`
- `source_paragraph_range`
- `target_paragraph_range`
- `local_occurrence_index`
- `occurrence_index`
- `remaining_after_occurrence`
- `urgency`
- `future_gain_if_introduced_here`
- `source_preview`

## Meaning of the Signals

### `remaining_after_occurrence`

How many aligned occurrences of that family remain later in the book after this point.

Interpretation:

- large value: the family is deferrable
- small value: the family is running out of future chances

### `urgency`

A simple monotonic urgency score:

- `urgency = 1 / (1 + remaining_after_occurrence)`

Interpretation:

- close to `1.0`: near last chance
- small value: still many later opportunities

### `future_gain_if_introduced_here`

A coarse estimate of how much raw future-support mass this family still has if introduced at this point.

Interpretation:

- large value: introducing it now strongly affects `future czechness`
- small value: it may still be urgent editorially, but it does not move the future metric much

## Planner Integration

The current planner does **not** replace its main objective with the lemma timeline.

Current integration is deliberately conservative:

- `future czechness` remains the main objective
- `lemma urgency` is an additional ranking bonus
- no hard accept/reject thresholds are driven by the timeline yet

Candidate metadata now includes:

- `family_remaining_after_unit`
- `family_urgency`
- `family_future_gain_if_introduced_here`

## Why This Matters

Without the timeline, the planner can behave badly in two opposite ways:

1. introduce a very common family too early, because it gives a large future-gain jump
2. miss the last good chance to introduce a rare family, because it barely moves the future metric

The timeline is meant to help balance those two pressures.

## CLI

Build the artifact with:

```bash
PYTHONPATH=src python3 -m szwejk.cli lemma-timeline-report \
  --alignment data/reports/full_hierarchical_alignment_e5_stanza.json \
  --blocked-standalone-upos SCONJ,CCONJ,PART,ADP,PRON,DET,AUX \
  --output data/reports/lemma_timeline_e5_stanza.json
```

## Intended Next Uses

- refine candidate ranking inside the future-czechness baseline
- support later chapter-level czechness correction
- inspect which families are being introduced too early or too late
- eventually guide note policy for first useful introductions
