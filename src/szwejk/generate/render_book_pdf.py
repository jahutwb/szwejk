"""Render a full-book LaTeX/PDF from corpus + paragraph plan + notes."""

from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess

from szwejk.generate.debug_subset import (
    DebugNote,
    DebugSubstitution,
    _candidate_span,
    _paragraph_scoped_candidates,
    apply_selected_candidates,
    escape_latex,
)
from szwejk.generate.pdf_meta import build_pdf_meta_section


def _glossary_latex(notes: list[dict[str, object]]) -> str:
    seen: set[tuple[str, str]] = set()
    rows: list[tuple[str, str]] = []
    for note in notes:
        target = str(note.get("target_anchor_text", "")).strip()
        text = str(note.get("note_text", "")).strip()
        if not target or not text:
            continue
        key = (target, text)
        if key in seen:
            continue
        seen.add(key)
        rows.append(key)
    if not rows:
        return ""
    lines = ["\\begin{description}"]
    for target, text in rows:
        lines.append(f"  \\item[\\textbf{{{escape_latex(target)}}}] {escape_latex(text)}")
    lines.append("\\end{description}")
    return "\n".join(lines)


def _render_safe_candidates(paragraph: dict[str, object], candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    accepted: list[dict[str, object]] = []
    occupied: list[tuple[int, int]] = []
    for candidate in candidates:
        try:
            span = _candidate_span(paragraph, candidate, occupied)
        except Exception:
            continue
        occupied.append(span)
        accepted.append(candidate)
    return accepted


def build_book_pdf_payload(
    root: Path,
    *,
    plan_path: Path,
    notes_path: Path,
    alignment_path: Path,
) -> dict[str, object]:
    pl = json.loads((root / "data/corpus/pl_szwejk.json").read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    notes = json.loads(notes_path.read_text(encoding="utf-8"))
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    template = (root / "template/template.tex").read_text(encoding="utf-8")

    units_by_pair: dict[tuple[int, int], list[dict[str, object]]] = {}
    for unit in plan.get("units", []):
        pair = (int(unit["chapter_pair"][0]), int(unit["chapter_pair"][1]))
        units_by_pair.setdefault(pair, []).append(unit)

    notes_by_candidate: dict[str, dict[str, object]] = {}
    for note in notes.get("notes", []):
        candidate_id = str(note["candidate_id"])
        if candidate_id not in notes_by_candidate:
            notes_by_candidate[candidate_id] = note

    introduced_families: set[str] = set()
    used_substitutions: list[DebugSubstitution] = []
    used_notes: list[DebugNote] = []
    body_lines: list[str] = []

    for chapter in pl["chapters"]:
        chapter_index = int(chapter["index"])
        chapter_title = str(chapter.get("title", ""))
        if chapter_index == 1 and chapter_title.strip().upper() == "WSTĘP":
            body_lines.append(f"\\chapter*{{{escape_latex(chapter_title)}}}")
            body_lines.append(f"\\addcontentsline{{toc}}{{chapter}}{{{escape_latex(chapter_title)}}}")
        else:
            body_lines.append(f"\\chapter{{{escape_latex(chapter_title)}}}")
        body_lines.append("")

        chapter_units = units_by_pair.get((chapter_index, chapter_index), [])
        for paragraph in chapter.get("paragraphs", []):
            paragraph_index = int(paragraph["index"])
            matches = [
                unit
                for unit in chapter_units
                if int(unit["source_paragraph_range"][0]) <= paragraph_index <= int(unit["source_paragraph_range"][1])
            ]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous unit mapping for chapter {chapter_index} paragraph {paragraph_index}")
            candidates = [] if not matches else _render_safe_candidates(
                paragraph,
                _paragraph_scoped_candidates(paragraph, list(matches[0].get("selected_candidates", []))),
            )
            if not candidates:
                body_lines.append(escape_latex(str(paragraph["text"])))
                body_lines.append("")
                continue

            new_candidate_ids: set[str] = set()
            for candidate in candidates:
                family_ids = [str(item) for item in candidate.get("family_ids", [])]
                if any(family_id not in introduced_families for family_id in family_ids):
                    new_candidate_ids.add(str(candidate["candidate_id"]))
                introduced_families.update(family_ids)

            rendered, used_here, notes_here = apply_selected_candidates(
                paragraph,
                candidates,
                notes_by_candidate,
                new_candidate_ids,
            )
            for candidate, note, is_new in used_here:
                used_substitutions.append(
                    DebugSubstitution(
                        unit_index=int(candidate["unit_index"]),
                        chapter_pair=(int(candidate["chapter_pair"][0]), int(candidate["chapter_pair"][1])),
                        paragraph_index=paragraph_index,
                        candidate_id=str(candidate["candidate_id"]),
                        source_text=str(candidate["source_text"]),
                        target_text=str(candidate["target_text"]),
                        granularity=str(candidate["granularity"]),
                        note_id=None if note is None else str(note["note_id"]),
                        is_new_lemma=is_new,
                    )
                )
            for note in notes_here:
                used_notes.append(
                    DebugNote(
                        note_id=str(note["note_id"]),
                        unit_index=int(note["unit_index"]),
                        chapter_pair=(int(note["chapter_pair"][0]), int(note["chapter_pair"][1])),
                        candidate_id=str(note["candidate_id"]),
                        source_anchor_text=str(note["source_anchor_text"]),
                        target_anchor_text=str(note["target_anchor_text"]),
                        note_text=str(note["note_text"]),
                        note_type=str(note["note_type"]),
                    )
                )
            body_lines.append(rendered)
            body_lines.append("")

    meta_dir = root / "output" / "meta"
    meta_latex = build_pdf_meta_section(
        root=root,
        plan_payload=plan,
        alignment_artifact=alignment,
        output_dir=meta_dir,
        label=plan_path.stem,
    )
    latex = (
        template.replace("%%BOOK_TITLE%%", escape_latex(str(pl["metadata"].get("title", ""))))
        .replace("%%BOOK_AUTHOR%%", "")
        .replace("%%BOOK_SUBTITLE%%", "")
        .replace("%%BOOK_CONTENT%%", "\n".join(body_lines).strip())
        .replace("%%GLOSSARY_CONTENT%%", _glossary_latex(notes.get("notes", [])))
        .replace("%%META_CONTENT%%", meta_latex)
    )
    return {
        "latex": latex,
        "used_substitutions": used_substitutions,
        "used_notes": used_notes,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    plan_path = Path(os.environ.get("SZW_PLAN_PATH", root / "data/reports/paragraph_hybridization_plan_p14_reader.json"))
    notes_path = Path(os.environ.get("SZW_NOTES_PATH", root / "data/reports/paragraph_notes_p14_reader_relaxed.json"))
    alignment_path = Path(os.environ.get("SZW_ALIGNMENT_PATH", root / "data/reports/full_hierarchical_alignment_e5_stanza.json"))
    basename = os.environ.get("SZW_OUTPUT_BASENAME", plan_path.stem)

    payload = build_book_pdf_payload(
        root,
        plan_path=plan_path,
        notes_path=notes_path,
        alignment_path=alignment_path,
    )

    output_dir = root / "output"
    output_dir.mkdir(exist_ok=True)
    tex_path = output_dir / f"{basename}.tex"
    pdf_path = output_dir / f"{basename}.pdf"
    tex_path.write_text(str(payload["latex"]), encoding="utf-8")

    (output_dir / f"{basename}_substitutions.json").write_text(
        json.dumps([item.to_dict() for item in payload["used_substitutions"]], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"{basename}_notes.json").write_text(
        json.dumps([item.to_dict() for item in payload["used_notes"]], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    subprocess.run(
        [
            "latexmk",
            "-lualatex",
            "-interaction=nonstopmode",
            "-file-line-error",
            "-output-directory=output",
            str(tex_path),
        ],
        cwd=root,
        check=True,
    )
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
