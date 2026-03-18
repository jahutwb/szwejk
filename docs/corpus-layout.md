# Corpus Layout and Benchmark Subset

## Canonical Artifact Layout

Phase 1 writes canonical corpus JSON artifacts to `data/corpus/`.

Each artifact contains:

- `metadata`: book identity, language, source EPUB path, optional upstream identifier
- `chapters[]`: ordered canonical chapters derived from EPUB spine documents after front-matter filtering
- `paragraphs[]`: ordered paragraph units with stable IDs and source provenance
- `sentences[]`: baseline sentence segmentation for each paragraph
- `source_fragments[]`: path back to the originating XHTML file and block ordinal

## Current Intake Behavior

- `title.xhtml`, `about.xhtml`, and similar non-content documents are skipped.
- Main title / interwiki front matter documents are filtered out from canonical chapters.
- Introductory chapters such as `WSTĘP` and `Úvod` are preserved as content.
- Polish XHTML blocks containing many `<br/>` separators inside one `<p>` are split into multiple canonical paragraphs.

## Benchmark Subset for Alignment Work

The benchmark should cover both intro-style prose and later narrative chapters from different parts of the book. The opening chapters remain important, but they are no longer sufficient on their own.

### Polish

1. `WSTĘP` — 4 paragraphs — `OPS/c2_Przygody_dobrego_wojaka_Szwejka_Tom_I_Wstep.xhtml`
2. `Rozdział I` — 86 paragraphs — `OPS/c3_Przygody_dobrego_wojaka_Szwejka_Tom_I_I.xhtml`
3. `Rozdział IX` — 153 paragraphs — `OPS/c11_Przygody_dobrego_wojaka_Szwejka_Tom_I_IX.xhtml`
4. `XIV` — 364 paragraphs — `OPS/c16_Przygody_dobrego_wojaka_Szwejka_Tom_I_XIV.xhtml`

### Czech

1. `Úvod` — 3 paragraphs — `OPS/c1_Osudy_dobreho_vojaka_Svejka_za_svetove_valky_Uvod.xhtml`
2. `Zasáhnutí dobrého vojáka Švejka do světové války` — 86 paragraphs — `OPS/c2_Osudy_dobreho_vojaka_Svejka_za_svetove_valky_Zasahnuti_dobreho_vojaka_Svejka_do_svetove_valky.xhtml`
3. `Švejk na garnizóně` — 143 paragraphs — `OPS/c10_Osudy_dobreho_vojaka_Svejka_za_svetove_valky_Svejk_na_garnizone.xhtml`
4. `Švejk vojenským sluhou u nadporučíka Lukáše` — 323 paragraphs — `OPS/c15_Osudy_dobreho_vojaka_Svejka_za_svetove_valky_Svejk_vojenskym_sluhou_u_nadporucika_Lukase.xhtml`

## Default Curated Pairs

The default chapter-pair benchmark is:

- `1:1` — `WSTĘP` -> `Úvod`
- `2:2` — `Rozdział I` -> `Zasáhnutí dobrého vojáka Švejka do světové války`
- `10:10` — `Rozdział IX` -> `Švejk na garnizóně`
- `15:15` — `XIV` -> `Švejk vojenským sluhou u nadporučíka Lukáše`

## Review Workflow

1. Generate canonical JSON:
   `PYTHONPATH=src python3 -m szwejk.cli ingest --epub raw_texts/pl/Przygody_dobrego_wojaka_Szwejka.epub`
2. Render a chapter preview:
   `PYTHONPATH=src python3 -m szwejk.cli inspect --input data/corpus/Przygody_dobrego_wojaka_Szwejka.json --chapter 2`
3. Compare preview output against source XHTML when formatting anomalies appear.

This benchmark set is the default handoff surface for advanced alignment experiments, regression checks, and future embedding-model comparisons.
