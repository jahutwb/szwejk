# Hybridization Policy v1

The project uses a staged, stateful hybridization policy rather than linear token replacement.

## Core model

The policy is controlled along three axes:

- `didactic_exposure`
- `surface_czechness`
- `readability_load`

This means the system does not optimize only for the raw share of Czech forms. It also controls how much novelty is introduced and how much cognitive load is added at a given point in the book.

## Unit classes

- `A`: transparent cognates and near-identical families
- `B`: safe lexical shifts readable from context
- `C`: syntax-sensitive, inflection-dependent units
- `D`: opaque, idiomatic, or false-friend-prone units

The first production path starts with `A`, then expands toward `B`, then `C`, and only later considers `D`.

## Safety gates

Hybridization decisions must not cross below the safe alignment depth available for a region.

Policy gates explicitly control:

- minimum safe depth
- whether `foreign-span` regions are allowed
- whether residual unmatched regions are allowed
- false-friend risk
- inflection risk
- subtree cohesion
- minimum alignment confidence

If a gate fails, the unit is not eligible for substitution even if it would improve surface czechness.

## Levels

The default policy defines four levels:

1. `L1 Lexical Onboarding`
2. `L2 Phrase Onboarding`
3. `L3 Mixed Subtree Shift`
4. `L4 Czech-Dominant Finish`

The final level is intentionally Czech-dominant, because the long-term project goal is to let the reader end up reading effectively in Czech rather than staying in a perpetual mixed register.

## Default policy artifact

Default policy JSON:

- [progressive_czechization_v1.json](/home/jahu/PycharmProjects/szwejk/data/policies/progressive_czechization_v1.json)

This artifact is the source of truth for Phase `04-01`. Later phases should consume this policy contract rather than re-encoding the same rules ad hoc.
