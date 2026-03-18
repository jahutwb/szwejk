# Hybridization Notes Contract

This document defines the current `v1` note bundle contract for paragraph-first hybridization.

## Scope

The note generator works on:

- the full alignment artifact
- the selected paragraph hybridization plan

It does **not** generate notes for all available candidates. It generates notes only for candidates that were actually selected into the final substitution plan.

## Primary Rule

A note is considered only when a selected candidate introduces at least one `new_lemma`, meaning a lemma family that is not yet present in `introduced_lemmas` at that point in the book.

If a selected candidate contains only already introduced families:

- no note

## Why Notes Are a Separate Bundle

For now notes live in a separate file, not inline in the hybridized text document.

Reason:

- note generation depends on both the final plan and the alignment inventory
- notes need review and deduplication independently of text rendering
- the reader can later join text and notes through stable identifiers such as `unit_index` and `candidate_id`

The reader layer can still inline or tooltip these notes later.

## Input

The generator takes:

- `alignment_artifact`
- `plan_payload`

For each selected candidate it uses:

- `candidate_id`
- `unit_index`
- `chapter_pair`
- `granularity`
- `source_text`
- `target_text`
- `family_ids`
- `score`
- `relation`
- `metadata`

## Note Eligibility

For each selected candidate:

1. Compute `new_family_ids = candidate.family_ids - introduced_lemmas`.
2. If `new_family_ids` is empty:
   - skip note
3. Otherwise:
   - generate note
   - update `introduced_lemmas`

## Anchor Selection

Notes should be anchored in the smallest well-aligned context that explains the new lemma naturally.

Rules:

1. Search inside the current alignment unit.
2. Consider only non-fallback aligned candidates.
3. Anchor must:
   - intersect `new_family_ids`
   - stay inside the selected candidate span
4. Prefer:
   - `phrase`
   - then `subtree`
   - then another non-token structural unit
   - finally the selected candidate itself if nothing better exists

Practical consequence:

- not `rameno = ramię`
- prefer `za rameno = za ramię`

## Note Types

### `lexical`

Use when:

- anchor is token-sized
- or the change is small and compositional

Template:

- `{CS_anchor} = {PL_anchor}`

### `contextual`

Use when:

- a larger aligned phrase or subtree better explains the new lemma than a naked token

Template:

- `{CS_anchor} = {PL_anchor}`

### `idiomatic`

Use when:

- whole-span similarity is good
- but internal token support is weak
- and the phrase behaves more like a non-literal equivalent than a direct compositional mapping

Current compact template:

- `{CS_anchor}: {PL_anchor}.`

Idiomatic notes are also flagged for editorial review so a human or a later assisted pass can add a short gloss such as:

- `sens: ...`

## Idiomaticity Signal

`idiomaticity_score` is diagnostic. It is currently used for notes, not for hybridization policy.

Heuristic:

- higher whole-span semantic score
- lower token support
- lower lemma-level internal support

This points to:

- a subtree or phrase that aligns well as a whole
- but does not decompose cleanly into token-to-token mapping

## Output Contract

Each note record contains:

- `note_id`
- `unit_index`
- `chapter_pair`
- `candidate_id`
- `candidate_granularity`
- `anchor_granularity`
- `note_type`
- `source_anchor_text`
- `target_anchor_text`
- `source_candidate_text`
- `target_candidate_text`
- `new_family_ids`
- `alignment_score`
- `idiomaticity_score`
- `note_text`
- `editorial_attention`
- `editorial_comment`

## Editorial Principle

Notes should be:

- sparse
- local
- useful for reading

They should explain:

- the new lemma
- in the smallest meaningful aligned context

And for idioms:

- the whole turn of phrase
- not isolated token glosses
