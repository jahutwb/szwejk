"""
Script to generate a hybridized markdown manuscript with footnotes/glosses from project JSON artifacts.

- Input: hybridization plan, notes bundle, source corpus
- Output: markdown file with [[cz]]...[[/cz]] and [[vocab]]...[[/vocab]] tags, and footnotes/glosses

Usage:
    python build_hybrid_manuscript.py --plan <plan.json> --notes <notes.json> --corpus <pl_corpus.json> --output <manuscript.md>
"""
import argparse
import json
from pathlib import Path


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    plan = load_json(args.plan)
    notes = load_json(args.notes)
    corpus = load_json(args.corpus)

    # Build a lookup for notes by unit_index and anchor text
    notes_by_unit = {}
    for note in notes.get("notes", []):
        key = (note["unit_index"], note["anchor_granularity"], note["source_anchor_text"].strip())
        notes_by_unit.setdefault(key, []).append(note)

    # Track vocab/footnote numbering globally
    vocab_counter = 1
    vocab_map = {}  # (unit_index, word) -> number
    footnotes = []  # (number, note_text)
    footnote_refs = {}  # (unit_index, anchor) -> footnote label

    def mark_vocab(text, unit_index, anchor_granularity):
        # For each note for this unit, mark new vocab in the anchor text
        for key, note_list in notes_by_unit.items():
            if key[0] == unit_index and key[1] == anchor_granularity:
                for note in note_list:
                    anchor = note["source_anchor_text"].strip()
                    # Only mark if anchor is in text
                    if anchor and anchor in text:
                        nonlocal vocab_counter
                        if (unit_index, anchor) not in vocab_map:
                            label = str(vocab_counter)
                            vocab_map[(unit_index, anchor)] = label
                            footnotes.append((label, note["note_text"]))
                            footnote_refs[(unit_index, anchor)] = label
                            vocab_counter += 1
                        label = vocab_map[(unit_index, anchor)]
                        # Replace anchor with **anchor**[^label]
                        text = text.replace(anchor, f"**{anchor}**[^{label}]")
        return text

    # Render manuscript
    manuscript_lines = []
    for unit in plan.get("units", []):
        # Use source_preview as base text
        text = unit.get("source_preview", "")
        unit_index = unit.get("unit_index")
        selected_candidates = unit.get("selected_candidates", [])
        # Insert Czech fragments for selected candidates (if any)
        if selected_candidates:
            for cand in selected_candidates:
                cz_text = cand.get("target_text", "")
                anchor = cand.get("source_text", "")
                if cz_text and anchor and anchor in text:
                    # Replace anchor with [[cz]]cz_text[[/cz]]
                    text = text.replace(anchor, f"[[cz]]{cz_text}[[/cz]")
            granularity = selected_candidates[0].get("granularity", "paragraph")
        else:
            granularity = "paragraph"
        # Mark new vocab (footnoted) in text
        text = mark_vocab(text, unit_index, granularity)
        manuscript_lines.append(text)
        manuscript_lines.append("")  # Paragraph break

    # Add Pandoc/LaTeX-style footnotes at the end
    if footnotes:
        for label, note in footnotes:
            manuscript_lines.append(f"[^{label}]: {note}")

    # Write to output
    Path(args.output).write_text("\n".join(manuscript_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
