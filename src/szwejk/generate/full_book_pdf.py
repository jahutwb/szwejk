"""Reusable full-book PDF renderer for the hybrid Polish-Czech edition."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from szwejk.generate.debug_subset import (
    DebugNote,
    DebugSubstitution,
    _candidate_span,
    _merge_adjacent_noted_hits,
    _paragraph_scoped_candidates,
    escape_latex,
)
from szwejk.generate.dictionary_pipeline import (
    DEFAULT_TRANSLATION_CACHE_PATH,
    DictionaryEntry,
    build_dictionary_entries,
)
from szwejk.generate.pdf_meta import build_pdf_meta_section


@dataclass(slots=True)
class FullBookRenderSummary:
    chapter_count: int
    paragraph_count: int
    substitution_count: int
    note_count: int
    glossary_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "chapter_count": self.chapter_count,
            "paragraph_count": self.paragraph_count,
            "substitution_count": self.substitution_count,
            "note_count": self.note_count,
            "glossary_count": self.glossary_count,
        }


@dataclass(frozen=True, slots=True)
class ChapterHeading:
    part_title: str | None
    chapter_title: str


def build_full_book_latex(
    *,
    corpus_payload: dict[str, object],
    plan_payload: dict[str, object],
    notes_payload: dict[str, object],
    template_text: str,
    alignment_artifact: dict[str, object] | None = None,
    root: Path | None = None,
    meta_output_dir: Path | None = None,
    meta_label: str = "hybrid_book",
    book_author: str = "",
    book_subtitle: str = "",
    translation_cache_path: Path | None = None,
) -> dict[str, object]:
    chapters = list(corpus_payload.get("chapters", []))
    chapter_headings = _chapter_headings_for_corpus(corpus_payload=corpus_payload, chapter_count=len(chapters), root=root)
    units_by_source_chapter = _units_by_source_chapter(plan_payload)
    notes_by_candidate = _notes_by_candidate(notes_payload)
    new_candidate_ids = _introduced_candidate_ids(plan_payload)

    used_substitutions: list[DebugSubstitution] = []
    used_notes: list[DebugNote] = []
    body_lines: list[str] = []
    paragraph_count = 0

    for chapter, heading in zip(chapters, chapter_headings):
        chapter_index = int(chapter["index"])
        body_lines.extend(_render_chapter_heading(chapter_index, heading.chapter_title, part_title=heading.part_title))
        chapter_units = units_by_source_chapter.get(chapter_index, [])

        for paragraph in chapter.get("paragraphs", []):
            paragraph_count += 1
            paragraph_index = int(paragraph["index"])
            matched_unit = _match_unit_for_paragraph(chapter_units, paragraph_index)
            candidates = (
                []
                if matched_unit is None
                else _paragraph_scoped_candidates(paragraph, list(matched_unit.get("selected_candidates", [])))
            )
            rendered = escape_latex(str(paragraph["text"]))
            if candidates:
                rendered, used_here, notes_here = _apply_renderable_candidates(
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
                            note_type=str(note.get("note_type", "")),
                        )
                    )
            body_lines.append(rendered)
            body_lines.append("")

    used_dictionary_entries, dictionary_entries = build_dictionary_entries(
        used_substitutions=used_substitutions,
        notes_payload=notes_payload,
        cache_path=translation_cache_path or DEFAULT_TRANSLATION_CACHE_PATH,
    )
    glossary = _build_dictionary_content(dictionary_entries)
    meta = ""
    if alignment_artifact is not None and root is not None and meta_output_dir is not None:
        meta = build_pdf_meta_section(
            root=root,
            plan_payload=plan_payload,
            alignment_artifact=alignment_artifact,
            output_dir=meta_output_dir,
            label=meta_label,
        )

    latex = (
        template_text.replace("%%BOOK_TITLE%%", escape_latex(str(corpus_payload.get("metadata", {}).get("title", ""))))
        .replace("%%BOOK_AUTHOR%%", escape_latex(book_author))
        .replace("%%BOOK_SUBTITLE%%", escape_latex(book_subtitle))
        .replace("%%BOOK_CONTENT%%", "\n".join(body_lines).strip())
        .replace("%%GLOSSARY_CONTENT%%", glossary)
        .replace("%%META_CONTENT%%", meta)
    )
    summary = FullBookRenderSummary(
        chapter_count=len(chapters),
        paragraph_count=paragraph_count,
        substitution_count=len(used_substitutions),
        note_count=len(used_notes),
        glossary_count=_glossary_count(dictionary_entries),
    )
    return {
        "latex": latex,
        "used_substitutions": used_substitutions,
        "used_notes": used_notes,
        "used_dictionary_entries": used_dictionary_entries,
        "dictionary_entries": dictionary_entries,
        "summary": summary.to_dict(),
        "chapter_count": summary.chapter_count,
        "paragraph_count": summary.paragraph_count,
    }


def write_full_book_artifacts(
    *,
    corpus_payload: dict[str, object],
    plan_payload: dict[str, object],
    notes_payload: dict[str, object],
    template_text: str,
    output_dir: Path,
    label: str,
    alignment_artifact: dict[str, object] | None = None,
    root: Path | None = None,
    book_author: str = "",
    book_subtitle: str = "",
    translation_cache_path: Path | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_full_book_latex(
        corpus_payload=corpus_payload,
        plan_payload=plan_payload,
        notes_payload=notes_payload,
        template_text=template_text,
        alignment_artifact=alignment_artifact,
        root=root,
        meta_output_dir=output_dir / "meta",
        meta_label=label,
        book_author=book_author,
        book_subtitle=book_subtitle,
        translation_cache_path=translation_cache_path,
    )

    tex_path = output_dir / f"{label}.tex"
    substitutions_path = output_dir / f"{label}_substitutions.json"
    notes_path = output_dir / f"{label}_notes.json"
    summary_path = output_dir / f"{label}_summary.json"
    dictionary_entries_path = output_dir.parent / "debug_dictionary_entries.json"

    tex_path.write_text(str(payload["latex"]), encoding="utf-8")
    substitutions_path.write_text(
        json.dumps([item.to_dict() for item in payload["used_substitutions"]], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    notes_path.write_text(
        json.dumps([item.to_dict() for item in payload["used_notes"]], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(payload["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    dictionary_entries_path.write_text(
        json.dumps([item.to_dict() for item in payload["dictionary_entries"]], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    payload["tex_path"] = tex_path
    payload["substitutions_path"] = substitutions_path
    payload["notes_path"] = notes_path
    payload["summary_path"] = summary_path
    payload["dictionary_entries_path"] = dictionary_entries_path
    return payload


def compile_full_book_pdf(*, tex_path: Path, output_dir: Path, cwd: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".aux", ".fdb_latexmk", ".fls", ".log", ".out", ".toc", ".pdf"):
        stale_path = output_dir / f"{tex_path.stem}{suffix}"
        if stale_path.exists():
            stale_path.unlink()
    subprocess.run(
        [
            "latexmk",
            "-lualatex",
            "-interaction=nonstopmode",
            "-file-line-error",
            f"-output-directory={output_dir}",
            str(tex_path),
        ],
        cwd=cwd,
        check=True,
    )
    pdf_path = output_dir / f"{tex_path.stem}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    return pdf_path


def _apply_renderable_candidates(
    paragraph: dict[str, object],
    candidates: list[dict[str, object]],
    notes_by_candidate: dict[str, dict[str, object]],
    new_candidate_ids: set[str],
) -> tuple[str, list[tuple[dict[str, object], dict[str, object] | None, bool]], list[dict[str, object]]]:
    paragraph_text = str(paragraph["text"])
    occupied: list[tuple[int, int]] = []
    hits: list[dict[str, object]] = []
    used_notes: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        try:
            span = _candidate_span(paragraph, candidate, occupied)
        except ValueError:
            continue
        occupied.append(span)
        note = notes_by_candidate.get(candidate_id)
        hits.append(
            {
                "start": span[0],
                "end": span[1],
                "candidate": candidate,
                "note": note,
                "is_new": candidate_id in new_candidate_ids,
                "target_text": str(candidate["target_text"]),
            }
        )

    hits = _merge_adjacent_noted_hits(paragraph_text, hits)
    rendered_parts: list[str] = []
    cursor = 0
    used_pairs: list[tuple[dict[str, object], dict[str, object] | None, bool]] = []
    for hit in sorted(hits, key=lambda item: int(item["start"])):
        start = int(hit["start"])
        end = int(hit["end"])
        note = hit["note"]
        is_new = bool(hit["is_new"])
        macro = _render_target_hit(hit)
        if note is not None:
            used_notes.append(note)
        rendered_parts.append(escape_latex(paragraph_text[cursor:start]))
        rendered_parts.append(macro)
        cursor = end
        used_pairs.append((hit["candidate"], note, is_new))
    rendered_parts.append(escape_latex(paragraph_text[cursor:]))
    return "".join(rendered_parts), used_pairs, used_notes


def _render_target_hit(hit: dict[str, object]) -> str:
    target_text = str(hit["target_text"])
    note = hit["note"]
    is_new = bool(hit["is_new"])
    if note is None:
        escaped = escape_latex(target_text)
        if is_new:
            return f"\\cznew{{{escaped}}}"
        return f"\\czplain{{{escaped}}}"

    anchor_text = str(note.get("target_anchor_text", "")).strip()
    note_text = escape_latex(str(note["note_text"]))
    if not anchor_text:
        return f"\\cznote{{{escape_latex(target_text)}}}{{{note_text}}}"
    anchor_start = target_text.find(anchor_text)
    if anchor_start <= 0 and target_text.strip() != anchor_text:
        return f"\\cznote{{{escape_latex(target_text)}}}{{{note_text}}}"
    if anchor_start == -1:
        return f"\\cznote{{{escape_latex(target_text)}}}{{{note_text}}}"
    anchor_end = anchor_start + len(anchor_text)
    prefix = target_text[:anchor_start]
    suffix = target_text[anchor_end:]
    parts: list[str] = []
    if prefix:
        parts.append(f"\\czplain{{{escape_latex(prefix)}}}")
    parts.append(f"\\cznote{{{escape_latex(anchor_text)}}}{{{note_text}}}")
    if suffix:
        parts.append(f"\\czplain{{{escape_latex(suffix)}}}")
    return "".join(parts)


def _render_chapter_heading(chapter_index: int, chapter_title: str, *, part_title: str | None = None) -> list[str]:
    lines: list[str] = []
    if part_title:
        lines.extend([f"\\part{{{escape_latex(part_title)}}}", ""])
    escaped_title = escape_latex(chapter_title)
    if chapter_index == 1 and chapter_title.strip().upper() == "WSTĘP":
        lines.extend([
            f"\\chapter*{{{escaped_title}}}",
            f"\\addcontentsline{{toc}}{{chapter}}{{{escaped_title}}}",
            "",
        ])
        return lines
    lines.extend([f"\\chapter{{{escaped_title}}}", ""])
    return lines


def _chapter_headings_for_corpus(
    *, corpus_payload: dict[str, object], chapter_count: int, root: Path | None
) -> list[ChapterHeading]:
    default_titles = [
        ChapterHeading(part_title=None, chapter_title=str(chapter.get("title", "")))
        for chapter in corpus_payload.get("chapters", [])
    ]
    metadata = corpus_payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return default_titles
    source_path = metadata.get("source_path")
    if not isinstance(source_path, str) or not source_path.strip():
        return default_titles
    epub_path = Path(source_path)
    if not epub_path.is_absolute():
        if root is None:
            return default_titles
        epub_path = root / epub_path
    if not epub_path.exists():
        return default_titles
    try:
        toc = _load_epub_toc(epub_path)
    except Exception:
        return default_titles
    if len(toc) != chapter_count:
        return default_titles
    return toc


def _load_epub_toc(epub_path: Path) -> list[ChapterHeading]:
    ns = {"ncx": "http://www.daisy.org/z3986/2005/ncx/"}
    with ZipFile(epub_path) as zf:
        root = ET.fromstring(zf.read("OPS/toc.ncx"))
    nav_map = root.find("ncx:navMap", ns)
    if nav_map is None:
        raise ValueError("EPUB toc.ncx missing navMap")

    headings: list[ChapterHeading] = []
    for nav_point in nav_map.findall("ncx:navPoint", ns):
        title = _ncx_label(nav_point, ns)
        if title in {"Strona tytułowa", "Przygody dobrego wojaka Szwejka. podczas wojny światowej"}:
            continue
        children = nav_point.findall("ncx:navPoint", ns)
        if not children:
            continue
        for index, child in enumerate(children):
            headings.append(
                ChapterHeading(
                    part_title=title if index == 0 else None,
                    chapter_title=_ncx_label(child, ns),
                )
            )
    return headings


def _ncx_label(node: ET.Element, namespaces: dict[str, str]) -> str:
    return node.findtext("ncx:navLabel/ncx:text", default="", namespaces=namespaces).strip()


def _units_by_source_chapter(plan_payload: dict[str, object]) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for unit in plan_payload.get("units", []):
        source_chapter = int(unit["chapter_pair"][0])
        grouped.setdefault(source_chapter, []).append(unit)
    return grouped


def _notes_by_candidate(notes_payload: dict[str, object]) -> dict[str, dict[str, object]]:
    notes_by_candidate: dict[str, dict[str, object]] = {}
    for note in notes_payload.get("notes", []):
        candidate_id = str(note["candidate_id"])
        if candidate_id in notes_by_candidate:
            raise ValueError(f"Ambiguous notes for candidate_id={candidate_id}")
        notes_by_candidate[candidate_id] = note
    return notes_by_candidate


def _introduced_candidate_ids(plan_payload: dict[str, object]) -> set[str]:
    introduced_families: set[str] = set()
    new_candidate_ids: set[str] = set()
    for unit in plan_payload.get("units", []):
        for candidate in unit.get("selected_candidates", []):
            family_ids = [str(item) for item in candidate.get("family_ids", [])]
            if any(family_id not in introduced_families for family_id in family_ids):
                new_candidate_ids.add(str(candidate["candidate_id"]))
            introduced_families.update(family_ids)
    return new_candidate_ids


def _match_unit_for_paragraph(chapter_units: list[dict[str, object]], paragraph_index: int) -> dict[str, object] | None:
    matches = [
        unit
        for unit in chapter_units
        if int(unit["source_paragraph_range"][0]) <= paragraph_index <= int(unit["source_paragraph_range"][1])
    ]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Ambiguous unit mapping for paragraph {paragraph_index}")
    return matches[0]


def _build_dictionary_content(dictionary_entries: list[DictionaryEntry]) -> str:
    if not dictionary_entries:
        return "Brak pozycji słownikowych."

    lines = [
        r"\begingroup",
        r"\small",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.1\baselineskip}",
        r"\raggedcolumns",
        r"\begin{multicols}{2}",
    ]
    for entry in dictionary_entries:
        lines.append(
            r"\noindent "
            + f"\\textbf{{\\czplain{{{escape_latex(entry.czech)}}}}} --- {escape_latex(entry.polish)}\\par"
        )
        if entry.explanation:
            lines.append(r"{\footnotesize " + escape_latex(entry.explanation) + r"\par}")
    lines.extend([r"\end{multicols}", r"\endgroup"])
    return "\n".join(lines).strip()


def _glossary_count(glossary_entries: list[DictionaryEntry]) -> int:
    return len(glossary_entries)
