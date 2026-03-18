# Hybridization Decisions and Experiment Log

This document records the current working decisions for hybridization, the most important experiments, and the practical conclusions drawn from them.

It is intentionally more operational than [hybridization-policy.md](/home/jahu/PycharmProjects/szwejk/docs/hybridization-policy.md). That policy document defines the long-term editorial model; this document captures what the current implementation actually does and what we learned from testing it.

## Current Baseline

The current hybridization workflow is driven by the full alignment artifact:

- [full_hierarchical_alignment_e5_stanza.json](/home/jahu/PycharmProjects/szwejk/data/reports/full_hierarchical_alignment_e5_stanza.json)

This artifact is the source of truth for:

- paragraph skeleton
- sentence alignment inside paragraph blocks
- subtree / phrase / token candidates
- candidate scores and signals

## Confirmed Decisions

### 1. Paragraph-first hybridization

Hybridization is no longer driven by a global chapter-level level switch.

The active model is:

- iterate over aligned paragraph blocks in reading order
- evaluate candidates inside the current block
- update global state after the block

Reason:

- chapters are editorial containers, not the right unit for decision-making
- paragraph blocks preserve continuity without dropping unresolved material
- the final system must not lose text simply because sentence-level alignment is incomplete

### 2. Candidate hierarchy and conflict model

Candidates are collected across multiple granularities:

- paragraph
- sentence
- subtree
- phrase
- token

They conflict hierarchically:

- overlapping `source_span`
- or overlapping `target_span`

If a larger unit is chosen, overlapping smaller ones cannot also be chosen.

### 3. Alignment-only v1, no fallback candidates

The first real hybridization selector uses only candidates from proper alignment.

Excluded from selection in v1:

- `embedding_assisted`
- `dictionary`
- `manual_forced`

Reason:

- fallbacks are useful diagnostic information
- but they should not drive first-pass editorial substitutions

### 4. Global state is `introduced_lemmas`

The central state variable is the set of already introduced lemma families.

If a candidate contains only already introduced families:

- it is effectively free from the didactic-progress perspective
- and may be selected automatically if it does not conflict with another chosen unit

Reason:

- this matches the intended pedagogical model
- later in the book larger units naturally become cheap because their contents are already familiar

### 5. Objective is closeness to target future czechness

The active selector does not optimize “largest gain”.

It selects candidates so that after the current paragraph block the system is as close as possible to the target `future czechness`.

Distance is measured as absolute distance:

- `abs(actual_after - target_after)`

Overshoot is allowed.

This was an explicit decision and replaced earlier logic that still had soft biases toward larger units.

### 6. Larger units are no longer preferred by default

Earlier selectors still had a tie-break preference for larger units.

That preference was removed.

Current rule:

- closeness to the target is primary
- score/confidence is secondary
- unit size is not rewarded by default

Reason:

- the user explicitly wanted the target-tracking objective to dominate
- early hybridization should not jump to whole paragraphs just because they are available

### 7. Best target curve so far is delayed, not linear

Experiments show that linear target growth is too fast for this metric.

Variants tested:

- `1.0` linear baseline
- `1.4`
- `1.8`

Current preferred target shape:

- `progress ^ 1.8`

Reason:

- it delays lemma introduction
- improves average distance to target
- makes the progression less front-loaded

### 8. `future czechness` stays, but it is not the only useful reading metric

We keep `future czechness` as a control metric because it captures:

- how much of the future text is already covered by introduced lemma families

But it has known limitations:

- the denominator shrinks over time
- late jumps can become very sharp
- it is not a direct measure of surface readability

Therefore we also track:

- local czechness per paragraph block
- local czechness per chapter

### 8a. Chapter-level czechness targets must be length-weighted

Chapter-level target tracking should not treat all chapters as equally large editorial steps.

The intended interpretation is:

- chapter progress is weighted by text mass
- the relevant mass is chapter length in words and aligned lemma occurrences
- later chapter-level boosts should therefore target cumulative progress through the whole book, not chapter ordinal alone

Reason:

- a short chapter should not advance the same target fraction as a long chapter
- otherwise chapter-level “correction” logic overreacts in some places and underreacts in others

### 8b. Lemma introduction should respect future opportunity, not only future gain

We now treat “how many future occurrences this lemma still has” as a separate planning signal.

New artifact:

- `lemma timeline`

For each lemma family, it stores:

- all aligned occurrences in reading order
- remaining occurrence count after each occurrence
- urgency of introducing the family at that point
- future-gain estimate if introduced there

Operational rule:

- families that still return many times later are more deferrable
- families that are close to disappearing from the book are more urgent

This does **not** replace the existing `future czechness` objective.
It refines it.

Planner effect:

- `future czechness` remains the main objective
- candidate ranking now also sees a small `lemma urgency` bonus
- this is meant to reduce wasting early substitutions on families that still have many later opportunities

Current implementation status:

- `lemma timeline` artifact exists
- candidate metadata now includes:
  - `family_remaining_after_unit`
  - `family_urgency`
  - `family_future_gain_if_introduced_here`
- the urgency signal is currently a ranking bonus, not a hard accept/reject rule

Reason:

- a hard rule would destabilize the baseline too much
- a ranking signal gives us the information we need without breaking the existing selection model

Next refinement under discussion:

- split the current notion of `urgency` into:
  - `urgency`: this family may disappear soon, so introduce it now
  - `deferrability`: this family is frequent and still has many future opportunities, so do not burn it too early

Intended behavior:

- rare or near-disappearing families get an early bonus
- common families with many future occurrences get an early penalty
- that penalty fades later in the book

Why:

- the planner should not waste early substitutions on very common lemmas that can still be introduced later in better, larger contexts
- the cost of a candidate should depend on whether its new families are urgent or safely deferrable, not only on how many of them it introduces

Likely implementation direction:

- keep `lemma_timeline` as the precomputed source of truth
- add per-family signals such as:
  - `global_frequency`
  - `remaining_count`
  - `distance_to_last_occurrence`
  - later possibly `best_future_context_score`
- use:
  - `urgency_bonus`
  - minus `deferrability_penalty`
  in the candidate ranking over new families only

### 8d. Candidate cost is moving toward new-family economics, but early over-expansion is still a live risk

The current experimental direction is:

- candidate cost should depend primarily on newly introduced families
- larger phrases should become cheap naturally when they contain mostly already introduced material
- functional-word families inside larger phrases should be treated as near-free context, not as real didactic novelty
- high semantic confidence with weak internal structural support should be treated as a possible idiomatic whole, not automatically downgraded

Latest experiment:

- [paragraph_hybridization_plan_p14_reader_econ_v3.json](/home/jahu/PycharmProjects/szwejk/data/reports/paragraph_hybridization_plan_p14_reader_econ_v3.json)
- [czechness_report_p14_reader_econ_v3.json](/home/jahu/PycharmProjects/szwejk/data/reports/czechness_report_p14_reader_econ_v3.json)

Observed effect:

- clear improvement over `econ_v1` in:
  - weighted simple czechness
  - weighted visible czechness
  - number of phrase and sentence substitutions
- but also a regression at the beginning of the book:

### 8e. Separate experiment: geometric lemma-introduction planning

This is a separate planning experiment, not the current production planner.

Core intuition:

- treat each lemma family as a geometric block
- introducing a family at a given occurrence creates a block-shaped contribution to future familiarity
- the task is to place those blocks so that their sum approximates the desired `future czechness` curve

Current artifact:

- [lemma_timeline_e5_stanza.json](/home/jahu/PycharmProjects/szwejk/data/reports/lemma_timeline_e5_stanza.json)

Current prototype report:

- [lemma_schedule_report_p14_v1.json](/home/jahu/PycharmProjects/szwejk/data/reports/lemma_schedule_report_p14_v1.json)
- [lemma_schedule_debug.json](/home/jahu/PycharmProjects/szwejk/output/lemma_schedule_debug.json)

`v1` approximation:

- each family is scheduled independently
- for each family we consider a few earliest occurrences
- each occurrence defines a rectangular block:
  - height = `future_gain_if_introduced_here`
  - start = `intro_unit_index`
  - end = end of book
- for each family we choose the intro point whose block best fits that family's share of the global target curve

Important limitation of `v1`:

- this approximation is fast and clean conceptually
- but it fits families independently, so families do not coordinate with each other
- result:
  - the combined curve overshoots early
  - many common families still get introduced too soon in aggregate

Observed metrics for `p14_v1`:

- `family_count = 31098`
- `scheduled_family_count = 31098`
- `reachable_max = 0.55842838`
- `mean_absolute_error = 0.19550354`

Checkpoint behavior:

- 10% of book:
  - target `0.02228531`
  - actual `0.29593203`
- 50%:
  - target `0.21166175`
  - actual `0.44913253`
- 100%:
  - target `0.55842838`
  - actual `0.49731279`

Conclusion:

- the geometric framing is useful and matches the user's intended reasoning
- but independent per-family fitting is not enough
- next serious version should coordinate families globally, for example by:
  - beam search over lemma families
  - residual-fitting by batches
  - or scheduling families into target mass buckets before materializing candidates

`v3` residual-greedy prototype now exists and is much closer to the intended formulation:

- it considers all occurrences
- each occurrence defines a suffix rectangle
- one family can be used only once
- the scheduler iteratively adds the block that most reduces weighted prefix `L1` error
- objective:
  - minimize `sum_t w(t) * |H(t) - G(t)|`
  - with larger weights at the beginning of the book

Important interaction discovered after integration with the paragraph planner:

- a good global lemma schedule can still produce a bad opening if the text planner materializes a "due" family through an overly wide phrase candidate
- this creates a mismatch where `future czechness` stays low, but early simple/surface czechness jumps because one weakly aligned phrase inserts a lot of Czech text with very little future gain
- mitigation now in the planner:
  - early wide carriers are penalized when they introduce too few new families relative to their width
  - early source/target span asymmetry is penalized
  - high embedding score alone no longer grants an "idiomatic" exemption; the candidate must also be cheap or structurally compact

### Candidate-level lemma cost

Planner cost can no longer depend only on aligned token-pair families inside a larger phrase.

Reason:

- phrases like `jego losy czasu wojny światowej -> jeho osudy za světové války` may look cheap if token-pair alignment exposes only one family such as `los::osud`
- but the actual Czech fragment introduces more lexical material than that

Current direction:

- every candidate now carries its own target-side content lemmas derived from the already analyzed Stanza tokens inside its span
- selection cost can therefore see Czech lexical novelty that is not explicitly present in `family_ids`
- this is intended to prevent larger phrases from looking artificially cheap just because deep token alignment is sparse

Current `v3` full-book debug run:

- [lemma_schedule_debug.json](/home/jahu/PycharmProjects/szwejk/output/lemma_schedule_debug.json)

Metrics for the `1000`-step run:

- `scheduled_family_count = 1000`
- `total_prefix_l1_error = 319.07165146`
- `early_section_error_20 = 12.71360866`
- `max_overshoot = 0.03052086`
- `max_undershoot = 0.25156155`
- `final_czechness = 0.30686683`
- `target_final_czechness = 0.55842838`

Checkpoint behavior:

- 10% of book:
  - target `0.02228531`
  - actual `0.01898474`
- 25%:
  - target `0.08024798`
  - actual `0.09596025`
- 50%:
  - target `0.21166175`
  - actual `0.23642189`
- 100%:
  - target `0.55842838`
  - actual `0.30686683`

Comparison with `v1`:

- `v1` had catastrophic early overshoot
  - early 20% error about `525.27`
  - max overshoot about `0.306`
- `v3` reduces early overshoot dramatically
  - early 20% error about `12.71`
  - max overshoot about `0.031`

Current conclusion:

- residual-greedy is the right formal direction
- it clearly outperforms `v1` on prefix tracking
- but the current stopping / gain economics still leave too much target unmet by the end
- next step should focus on:
  - deeper runs,
  - stronger late-book filling,
  - or a batched / beam version that can keep the same early discipline while pushing the tail higher
  - too many sentence / paragraph substitutions appear already in the first 10% of units

Operational conclusion:

- the direction is correct
- the current credit for large cheap contexts is too strong early
- the next tuning step must keep the new-family economics, but dampen early sentence/paragraph expansion without reintroducing the old staircase gating

### 8c. The end-of-book dictionary is compact, Czech -> Polish, and built from actually used substitutions

The dictionary appendix is not a dump of all notes and not a list of all raw substitutions.

Current rule:

- it is Czech -> Polish
- it contains only entries that were actually used in the rendered book
- the source of truth is the actual rendered substitution list, not the notes bundle
- it extracts Czech headwords from the inserted target fragments, preferring a cleaner anchor when available
- it enriches Polish equivalents with this priority:
  - note/editorial explanation
  - cached DeepL translation
  - cached LibreTranslate translation
  - alignment fallback
- it excludes foreign-language explanatory notes
- it excludes long sentence-like material and plain equality-style contextual sentence fragments
- it is sorted by Czech headword

Operational effect:

- footnotes remain the place for local explanation, idioms, and third-language fragments
- the dictionary is a compact reference section at the end of the book
- Czech display text preserves real diacritics and is rendered in the same Czech typographic style as the book body, with bold headwords

Implementation:

- extraction and enrichment live in [dictionary_pipeline.py](/home/jahu/PycharmProjects/szwejk/src/szwejk/generate/dictionary_pipeline.py)
- translation cache lives at:
  - [cs_pl_translations.json](/home/jahu/PycharmProjects/szwejk/data/cache/cs_pl_translations.json)
- debug output for actually included entries is written to:
  - [debug_dictionary_entries.json](/home/jahu/PycharmProjects/szwejk/output/debug_dictionary_entries.json)

Reason:

- the user explicitly rejected feeding all notes into the end-of-book dictionary
- the appendix should read like a real dictionary, not like a note dump
- the dictionary may use external translation support for Polish equivalents, but only as fallback and never over a better human note

## Alignment and Candidate Quality Lessons

### 9. Good alignment is necessary but not sufficient

The project now has a strong alignment backbone, but hybridization quality still depends on candidate quality.

Observed problem:

- some token pairs are alignment-consistent enough to exist
- but still bad as standalone substitutions

Examples observed during review:

- `stycznia -> Silvestra`
- `następstwem -> žízně`
- `jeśli -> že`
- `z ciebie -> do něho`

These are not alignment failures at every level. They are failures of using too small a substitution unit.

### 10. Deep alignment should be subtree-first

The active deep alignment model is:

- subtree
- phrase
- token

not the other way around.

Reason:

- idiomatic or freer translations often fail at token-to-token matching
- but still make sense as aligned subtrees or spans
- token alignment should only be trusted where the bigger structure remains coherent

### 11. `e5` helps the deeper layers

From benchmark comparisons:

- paragraph coverage is complete with `e5`
- sentence backbone changes little
- phrase and subtree coverage improve with `e5 + stanza`

Working stack:

- `e5` for paragraph skeleton
- `stanza` for morpho-syntactic analysis
- `e5 + stanza` for deeper phrase/subtree support

## Experiment Results

### Experiment A: target curve power

Plans:

- [paragraph_hybridization_plan_v1.json](/home/jahu/PycharmProjects/szwejk/data/reports/paragraph_hybridization_plan_v1.json) for `1.0`
- [paragraph_hybridization_plan_p14.json](/home/jahu/PycharmProjects/szwejk/data/reports/paragraph_hybridization_plan_p14.json) for `1.4`
- [paragraph_hybridization_plan_p18.json](/home/jahu/PycharmProjects/szwejk/data/reports/paragraph_hybridization_plan_p18.json) for `1.8`

Results:

`1.0`

- `introduced_family_count: 21916`
- `avg_distance: 0.096457`
- `avg_selected: 6.943665`

Granularity mix:

- `token: 31974`
- `phrase: 1219`
- `sentence: 2720`
- `paragraph: 146`
- `subtree: 55`

`1.4`

- `introduced_family_count: 20812`
- `avg_distance: 0.070229`
- `avg_selected: 6.380119`

Granularity mix:

- `token: 29641`
- `phrase: 1144`
- `sentence: 2207`
- `paragraph: 140`
- `subtree: 51`

`1.8`

- `introduced_family_count: 19960`
- `avg_distance: 0.057512`
- `avg_selected: 5.883868`

Granularity mix:

- `token: 27516`
- `phrase: 1036`
- `sentence: 1874`
- `paragraph: 123`
- `subtree: 53`

Conclusion:

- delayed target curves fit the metric better than linear growth
- `1.8` is the best current control curve

### Experiment B: hard token filter

We introduced a strong standalone-token filter to block weak token pairs.

Observed result:

- some very bad token pairs disappeared
- but the whole system became too conservative

Effects:

- `introduced_family_count` collapsed to `5666`
- weighted local czechness dropped to `0.351622`
- final chapter local czechness dropped heavily

Conclusion:

- the direction was correct
- the filter was too strong

### Experiment C: softened token filter plus confidence weighting

We then softened the token filter and added a confidence penalty in candidate ranking.

Current `1.8` after this calibration:

- [paragraph_hybridization_plan_p18.json](/home/jahu/PycharmProjects/szwejk/data/reports/paragraph_hybridization_plan_p18.json)
- [czechness_report_p18.json](/home/jahu/PycharmProjects/szwejk/data/reports/czechness_report_p18.json)

Results:

- `introduced_family_count: 8899`
- `avg_distance: 0.082486`
- `avg_selected: 3.850413`

Granularity mix:

- `token: 15555`
- `phrase: 1759`
- `sentence: 2206`
- `paragraph: 350`
- `subtree: 156`

Weighted local czechness:

- `0.412723`

Final chapter local czechness:

- `0.662232`

Conclusion:

- better than the overly hard filter
- but still too restrictive overall
- some bad standalone functional tokens still survive

## Reading-Level Assessment

Qualitative review of real fragments shows:

- early lexical substitutions can be sensible and helpful
- full sentence substitutions sometimes work well
- but poor standalone token substitutions still damage naturalness

Examples of problematic substitutions seen during review:

- `stycznia -> Silvestra`
- `następstwem -> žízně`
- `jeśli -> že`
- `z ciebie -> do něho`

This means:

- target-curve tuning alone is not enough
- candidate quality must improve further

## Overnight Experiment: Function-Word Standalone Filters on `p1.8`

We then tested whether standalone token substitutions for function words should be blocked by UPOS class.

The comparison summary is also saved in:

- [hybridization_experiment_summary.json](/home/jahu/PycharmProjects/szwejk/data/reports/hybridization_experiment_summary.json)

Plans and reports:

- baseline softened `p1.8`:
  - [paragraph_hybridization_plan_p18.json](/home/jahu/PycharmProjects/szwejk/data/reports/paragraph_hybridization_plan_p18.json)
  - [czechness_report_p18.json](/home/jahu/PycharmProjects/szwejk/data/reports/czechness_report_p18.json)
- `fw_basic` (`SCONJ, CCONJ, PART, ADP`):
  - [paragraph_hybridization_plan_p18_fw_basic.json](/home/jahu/PycharmProjects/szwejk/data/reports/paragraph_hybridization_plan_p18_fw_basic.json)
  - [czechness_report_p18_fw_basic.json](/home/jahu/PycharmProjects/szwejk/data/reports/czechness_report_p18_fw_basic.json)
- `fw_pron_det` (`SCONJ, CCONJ, PART, ADP, PRON, DET`):
  - [paragraph_hybridization_plan_p18_fw_pron_det.json](/home/jahu/PycharmProjects/szwejk/data/reports/paragraph_hybridization_plan_p18_fw_pron_det.json)
  - [czechness_report_p18_fw_pron_det.json](/home/jahu/PycharmProjects/szwejk/data/reports/czechness_report_p18_fw_pron_det.json)
- `fw_pron_det_aux` (`SCONJ, CCONJ, PART, ADP, PRON, DET, AUX`):
  - [paragraph_hybridization_plan_p18_fw_pron_det_aux.json](/home/jahu/PycharmProjects/szwejk/data/reports/paragraph_hybridization_plan_p18_fw_pron_det_aux.json)
  - [czechness_report_p18_fw_pron_det_aux.json](/home/jahu/PycharmProjects/szwejk/data/reports/czechness_report_p18_fw_pron_det_aux.json)

### Numeric comparison

`baseline_soft_p18`

- `introduced_family_count: 8899`
- `avg_distance_after: 0.082486`
- `avg_selected_per_block: 3.850413`
- `weighted_local_czechness_after: 0.412723`
- `last_chapter_local_czechness_after: 0.662232`
- selected `jeśli/zanim/že`-like standalone tokens: `341`

`fw_basic`

- `introduced_family_count: 9586`
- `avg_distance_after: 0.084702`
- `avg_selected_per_block: 3.631994`
- `weighted_local_czechness_after: 0.421938`
- `last_chapter_local_czechness_after: 0.655242`
- selected `jeśli/zanim/že`-like standalone tokens: `0`

`fw_pron_det`

- `introduced_family_count: 10397`
- `avg_distance_after: 0.084085`
- `avg_selected_per_block: 3.419919`
- `weighted_local_czechness_after: 0.435953`
- `last_chapter_local_czechness_after: 0.653890`
- selected `jeśli/zanim/že`-like standalone tokens: `0`

`fw_pron_det_aux`

- `introduced_family_count: 10665`
- `avg_distance_after: 0.082288`
- `avg_selected_per_block: 3.351279`
- `weighted_local_czechness_after: 0.441136`
- `last_chapter_local_czechness_after: 0.659301`
- selected `jeśli/zanim/že`-like standalone tokens: `0`

### Qualitative conclusion

This experiment produced a stronger result than expected.

Blocking standalone function words did **not** choke the planner. Instead, it:

- removed the obvious `jeśli -> že` class of errors,
- shifted substitutions from naked tokens toward phrases and sentences,
- and slightly improved weighted local czechness.

The best current variant is:

- `p1.8 + blocked standalone UPOS = SCONJ, CCONJ, PART, ADP, PRON, DET, AUX`

Reason:

- it keeps `final_future_czechness = 1.0`
- it has the best weighted local czechness among the tested filtered variants
- it also has the lowest average distance among those filtered variants
- and it removes the most obvious bad standalone functional substitutions

Important caveat:

- this does **not** solve all bad token choices
- there are still questionable standalone lexical substitutions with low surface similarity
- but the remaining problem is now much more lexical/semantic than purely functional-word based
- one renderer/tokenization issue also remains visible around enclitic-like source forms
- observed sample: `boś` became `bo ś` after a standalone verb substitution

### Reading notes from sampled blocks

Reading samples from the beginning, middle, and end of the book suggests:

- early sections remain conservative, which is acceptable
- the new filters encourage more phrase-level substitutions such as:
  - `na ścianie -> na stěně`
  - `za ramię -> za rameno`
- later sections still rely on some large paragraph/sentence substitutions when token-level options are weak
- this is preferable to standalone junk tokens, but still needs later editorial calibration

## Current Charts

Useful charts for current work:

- [paragraph_future_czechness_raw_compare.svg](/home/jahu/PycharmProjects/szwejk/data/reports/charts/paragraph_future_czechness_raw_compare.svg)
- [paragraph_local_czechness_smoothed_21.svg](/home/jahu/PycharmProjects/szwejk/data/reports/charts/paragraph_local_czechness_smoothed_21.svg)
- [paragraph_local_czechness_smoothed_51_compare.svg](/home/jahu/PycharmProjects/szwejk/data/reports/charts/paragraph_local_czechness_smoothed_51_compare.svg)
- [chapter_local_czechness_compare.svg](/home/jahu/PycharmProjects/szwejk/data/reports/charts/chapter_local_czechness_compare.svg)

## Next Recommended Step

The next adjustment should be narrower than another global rewrite.

Recommended focus:

- restrict which function words may be standalone token candidates
- especially `SCONJ`, `CCONJ`, `PART`, `ADP`, and selected pronouns/determiners
- allow them mainly inside:
  - phrase candidates
  - subtree candidates
  - sentence candidates

Reason:

- this directly targets the remaining bad examples
- without throwing away the entire token layer

## 2026-03 candidate-level lemma costing and `reader_visible_late_v6`

We corrected a real planner bug:

- wide phrase cost was previously derived too much from deep token families
- if internal token alignment was sparse, a wide phrase could look artificially cheap
- candidate cost must include the candidate's own target-side content lemmas, not only the deepest aligned families

Implementation direction:

- candidate metadata now collects target-side content and functional lemmas from:
  - explicit token-pair lemmas when available
  - fallback span-level extraction for uncovered tokens
- this lets small expansions like:
  - `przed swoim pogrzebem -> před svým pohřbem`
  stay cheap
- while wide carriers like:
  - `jego losy czasu wojny światowej -> jeho osudy za světové války`
  correctly pay for uncovered content like `světové`, `války`

We also removed the broad early phrase-carrier penalty and replaced it with a narrow local-scope rule:

- if a wide candidate carries essentially the same novelty as a tighter candidate in the same scope, the wide one is penalized locally
- redundant sentence-vs-paragraph duplicates in the same scope are also penalized locally

### Result of full-book run `reader_visible_late_v6`

Qualitative outcome:

- unit tests improve in the intended direction
- small cheap phrase expansion works
- bad early wide single-novelty carrier is blocked

But the full-book plan is worse than the older baseline:

- old `reader_visible_late`
  - weighted local: `0.464962`
  - weighted visible: `0.219791`
  - last chapter local: `0.615107`
  - last chapter visible: `0.228760`
- `reader_visible_late_v6`
  - weighted local: `0.273818`
  - weighted simple: `0.211882`
  - weighted visible: `0.110038`
  - last chapter local: `0.448478`
  - last chapter simple: `0.296464`
  - last chapter visible: `0.152993`

Conclusion:

- candidate-level lemma costing fixed a genuine modeling hole
- removing broad phrase penalties did **not** by itself improve the final plan
- `reader_visible_late_v6` should not become the PDF baseline

## 2026-03 target-lemma future steering (`reader_visible_late_v8`)

Next experiment:

- keep candidate-level target lemmas
- make the main selection distance depend on target-side future czechness
- use family-based future czechness only as a secondary term
- keep urgency / deferrability / local carrier choice as tie-break layers

Result:

- this still does **not** beat the older `reader_visible_late` baseline
- it also introduces many weak early token substitutions such as:
  - `ten -> ten`
  - `się -> se`
  - `to -> to`

Metrics:

- old `reader_visible_late`
  - weighted local: `0.464962`
  - weighted visible: `0.219791`
  - last chapter local: `0.615107`
  - last chapter visible: `0.228760`
- `reader_visible_late_v8`
  - weighted local: `0.356653`
  - weighted simple: `0.199695`
  - weighted visible: `0.101749`
  - last chapter local: `0.457948`
  - last chapter simple: `0.203626`
  - last chapter visible: `0.101753`

Additional report signal:

- last chapter idiomatic selected count: `6`
- last chapter idiomatic visible share: `0.070397`

Conclusion:

- steering directly by target-lemma future czechness is still too weak as a standalone policy
- it encourages cheap lemma-level drift without enough reader-facing czechization
- this should remain an experiment, not the production planner
