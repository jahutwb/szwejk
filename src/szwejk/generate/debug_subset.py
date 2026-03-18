"""Render a LaTeX debug subset from corpus + selected plan substitutions + notes."""

from __future__ import annotations

from pathlib import Path
import json
import os
import re
import subprocess
import unicodedata
from dataclasses import dataclass

from szwejk.generate.pdf_meta import build_pdf_meta_section


DEBUG_CHAPTER_COUNT = 2


@dataclass(slots=True)
class DebugSubstitution:
    unit_index: int
    chapter_pair: tuple[int, int]
    paragraph_index: int
    candidate_id: str
    source_text: str
    target_text: str
    granularity: str
    note_id: str | None
    is_new_lemma: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "unit_index": self.unit_index,
            "chapter_pair": list(self.chapter_pair),
            "paragraph_index": self.paragraph_index,
            "candidate_id": self.candidate_id,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "granularity": self.granularity,
            "note_id": self.note_id,
            "is_new_lemma": self.is_new_lemma,
        }


@dataclass(slots=True)
class DebugNote:
    note_id: str
    unit_index: int
    chapter_pair: tuple[int, int]
    candidate_id: str
    source_anchor_text: str
    target_anchor_text: str
    note_text: str
    note_type: str

    def to_dict(self) -> dict[str, object]:
        return {
            "note_id": self.note_id,
            "unit_index": self.unit_index,
            "chapter_pair": list(self.chapter_pair),
            "candidate_id": self.candidate_id,
            "source_anchor_text": self.source_anchor_text,
            "target_anchor_text": self.target_anchor_text,
            "note_text": self.note_text,
            "note_type": self.note_type,
        }


def escape_latex(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def _normalized_letters(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    raw_positions: list[int] = []
    for index, char in enumerate(text):
        category = unicodedata.category(char)
        if category.startswith(("L", "N")):
            chars.append(char.casefold())
            raw_positions.append(index)
    return "".join(chars), raw_positions


def _sentence_locations(paragraph: dict[str, object]) -> dict[int, tuple[int, int]]:
    text = str(paragraph["text"])
    cursor = 0
    mapping: dict[int, tuple[int, int]] = {}
    for sentence in paragraph.get("sentences", []):
        sentence_text = str(sentence["text"])
        start = text.find(sentence_text, cursor)
        if start == -1:
            raise ValueError(
                f"Cannot locate sentence {sentence['index']} in paragraph {paragraph['index']}"
            )
        end = start + len(sentence_text)
        mapping[int(sentence["index"])] = (start, end)
        cursor = end
    return mapping


def _surface_token_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if unicodedata.category(char).startswith(("L", "N")):
            start = index
            index += 1
            while index < len(text) and unicodedata.category(text[index]).startswith(("L", "N")):
                index += 1
            spans.append((start, index))
            continue
        spans.append((index, index + 1))
        index += 1
    return spans


def _scope_sentence_index(candidate: dict[str, object]) -> int | None:
    scope_id = str(candidate.get("scope_id", ""))
    if not scope_id.startswith("sentence:"):
        return None
    return int(scope_id.split(":", 1)[1])


def _find_unique_word_match(text: str, token_text: str, window: tuple[int, int]) -> tuple[int, int] | None:
    segment = text[window[0] : window[1]]
    pattern = re.compile(rf"(?<!\w){re.escape(token_text)}(?!\w)", re.UNICODE)
    matches = list(pattern.finditer(segment))
    if len(matches) != 1:
        return None
    match = matches[0]
    return (window[0] + match.start(), window[0] + match.end())


def _candidate_sentence_window(paragraph_text: str, candidate: dict[str, object]) -> tuple[int, int] | None:
    metadata = dict(candidate.get("metadata", {}))
    sentence_text = str(metadata.get("source_sentence_text", "")).strip()
    if not sentence_text:
        return None
    exact_start = paragraph_text.find(sentence_text)
    if exact_start != -1:
        return (exact_start, exact_start + len(sentence_text))
    return _match_text_within_window(
        full_text=paragraph_text,
        window=(0, len(paragraph_text)),
        source_text=sentence_text,
        occupied=[],
        candidate_id=f"{candidate['candidate_id']}:sentence",
    )


def _candidate_span(
    paragraph: dict[str, object],
    candidate: dict[str, object],
    occupied: list[tuple[int, int]],
) -> tuple[int, int]:
    paragraph_text = str(paragraph["text"])
    if str(candidate["granularity"]) == "paragraph":
        return (0, len(paragraph_text))

    if str(candidate["granularity"]) == "sentence":
        sentence_locations = _sentence_locations(paragraph)
        sentence_index = _scope_sentence_index(candidate)
        if sentence_index is None or sentence_index not in sentence_locations:
            raise ValueError(f"Missing sentence location for {candidate['candidate_id']}")
        return sentence_locations[sentence_index]

    sentence_index = _scope_sentence_index(candidate)
    if sentence_index is not None:
        sentence_window = _candidate_sentence_window(paragraph_text, candidate)
        if sentence_window is not None:
            sent_start, sent_end = sentence_window
        else:
            sentence_locations = _sentence_locations(paragraph)
            if sentence_index not in sentence_locations:
                raise ValueError(f"Missing sentence location for {candidate['candidate_id']}")
            sent_start, sent_end = sentence_locations[sentence_index]
        if str(candidate["granularity"]) == "token":
            exact_word_span = _find_unique_word_match(paragraph_text, str(candidate["source_text"]), (sent_start, sent_end))
            if exact_word_span is not None and all(exact_word_span[1] <= left or exact_word_span[0] >= right for left, right in occupied):
                return exact_word_span
        sentence_text = paragraph_text[sent_start:sent_end]
        token_spans = _surface_token_spans(sentence_text)
        source_span = (int(candidate["source_span"][0]), int(candidate["source_span"][1]))
        if source_span[0] >= 1 and source_span[1] <= len(token_spans):
            raw_start = sent_start + token_spans[source_span[0] - 1][0]
            raw_end = sent_start + token_spans[source_span[1] - 1][1]
            if any(raw_end > left and raw_start < right for left, right in occupied):
                raise ValueError(f"Conflicting span for {candidate['candidate_id']}")
            return (raw_start, raw_end)
        local_span = _match_text_within_window(
            full_text=paragraph_text,
            window=(sent_start, sent_end),
            source_text=str(candidate["source_text"]),
            occupied=occupied,
            candidate_id=str(candidate["candidate_id"]),
        )
        if local_span is not None:
            return local_span
        paragraph_token_spans = _surface_token_spans(paragraph_text)
        if source_span[0] >= 1 and source_span[1] <= len(paragraph_token_spans):
            raw_start = paragraph_token_spans[source_span[0] - 1][0]
            raw_end = paragraph_token_spans[source_span[1] - 1][1]
            if all(raw_end <= left or raw_start >= right for left, right in occupied):
                return (raw_start, raw_end)
        source_text = str(candidate["source_text"])
        needle, _ = _normalized_letters(source_text)
        haystack, positions = _normalized_letters(paragraph_text)
        if needle:
            matches: list[tuple[int, int]] = []
            start = haystack.find(needle)
            while start != -1:
                raw_start = positions[start]
                raw_end = positions[start + len(needle) - 1] + 1
                if all(raw_end <= left or raw_start >= right for left, right in occupied):
                    matches.append((raw_start, raw_end))
                start = haystack.find(needle, start + 1)
            if len(matches) == 1:
                return matches[0]
        raise ValueError(
            f"Token span {source_span} out of bounds for {candidate['candidate_id']} in sentence {sentence_index}"
        )

    source_text = str(candidate["source_text"])
    needle, _ = _normalized_letters(source_text)
    haystack, positions = _normalized_letters(paragraph_text)
    if not needle:
        raise ValueError(f"Empty normalized source_text for {candidate['candidate_id']}")

    matches: list[tuple[int, int]] = []
    start = haystack.find(needle)
    while start != -1:
        raw_start = positions[start]
        raw_end = positions[start + len(needle) - 1] + 1
        if all(raw_end <= left or raw_start >= right for left, right in occupied):
            matches.append((raw_start, raw_end))
        start = haystack.find(needle, start + 1)

    if len(matches) != 1:
        raise ValueError(
            f"Ambiguous mapping for {candidate['candidate_id']}: {len(matches)} matches in paragraph"
        )
    return matches[0]


def _match_text_within_window(
    *,
    full_text: str,
    window: tuple[int, int],
    source_text: str,
    occupied: list[tuple[int, int]],
    candidate_id: str,
) -> tuple[int, int] | None:
    segment = full_text[window[0] : window[1]]
    needle, _ = _normalized_letters(source_text)
    haystack, positions = _normalized_letters(segment)
    if not needle:
        return None
    matches: list[tuple[int, int]] = []
    start = haystack.find(needle)
    while start != -1:
        raw_start = window[0] + positions[start]
        raw_end = window[0] + positions[start + len(needle) - 1] + 1
        if all(raw_end <= left or raw_start >= right for left, right in occupied):
            matches.append((raw_start, raw_end))
        start = haystack.find(needle, start + 1)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous sentence-local mapping for {candidate_id}: {len(matches)} matches")
    return None


def apply_selected_candidates(
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
        note = notes_by_candidate.get(candidate_id)
        span = _candidate_span(paragraph, candidate, occupied)
        occupied.append(span)
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


def _merge_adjacent_noted_hits(paragraph_text: str, hits: list[dict[str, object]]) -> list[dict[str, object]]:
    if not hits:
        return hits
    ordered = sorted(hits, key=lambda item: int(item["start"]))
    merged: list[dict[str, object]] = []
    index = 0
    while index < len(ordered):
        current = dict(ordered[index])
        if current["note"] is None:
            merged.append(current)
            index += 1
            continue
        run = [current]
        run_end = int(current["end"])
        next_index = index + 1
        while next_index < len(ordered):
            candidate = ordered[next_index]
            gap = paragraph_text[run_end : int(candidate["start"])]
            same_scope = str(candidate["candidate"].get("scope_id")) == str(current["candidate"].get("scope_id"))
            if candidate["note"] is None or not same_scope or gap.strip():
                break
            run.append(dict(candidate))
            run_end = int(candidate["end"])
            next_index += 1
        if len(run) == 1:
            merged.append(current)
            index += 1
            continue

        merged_target_parts = [str(run[0]["target_text"])]
        for left, right in zip(run, run[1:]):
            gap = paragraph_text[int(left["end"]) : int(right["start"])]
            merged_target_parts.append(gap)
            merged_target_parts.append(str(right["target_text"]))
        merged_source = paragraph_text[int(run[0]["start"]) : int(run[-1]["end"])]
        merged_target = "".join(merged_target_parts)
        merged_note = dict(run[0]["note"])
        merged_note["note_id"] = f"{merged_note['note_id']}-merged"
        merged_note["source_anchor_text"] = merged_source
        merged_note["target_anchor_text"] = merged_target
        merged_note["note_text"] = f"{merged_target} = {merged_source}"
        merged.append(
            {
                "start": int(run[0]["start"]),
                "end": int(run[-1]["end"]),
                "candidate": run[0]["candidate"],
                "note": merged_note,
                "is_new": True,
                "target_text": merged_target,
            }
        )
        index = next_index
    return merged


def build_debug_subset(
    root: Path,
    *,
    plan_path: Path | None = None,
    notes_path: Path | None = None,
) -> dict[str, object]:
    pl = json.loads((root / "data/corpus/pl_szwejk.json").read_text(encoding="utf-8"))
    plan = json.loads((plan_path or root / "data/reports/paragraph_hybridization_plan_p18_fw_pron_det_aux.json").read_text(encoding="utf-8"))
    notes = json.loads((notes_path or root / "data/reports/paragraph_notes_p18_fw_pron_det_aux.json").read_text(encoding="utf-8"))
    template = (root / "template/template.tex").read_text(encoding="utf-8")

    chapters = pl["chapters"][:DEBUG_CHAPTER_COUNT]
    detected_chapters = [
        {
            "chapter_index": int(chapter["index"]),
            "title": str(chapter.get("title", "")),
            "paragraph_count": len(chapter.get("paragraphs", [])),
        }
        for chapter in pl["chapters"][:3]
    ]
    selected_chapters = [
        {
            "chapter_index": int(chapter["index"]),
            "title": str(chapter.get("title", "")),
            "paragraph_count": len(chapter.get("paragraphs", [])),
        }
        for chapter in chapters
    ]
    units_by_pair: dict[tuple[int, int], list[dict[str, object]]] = {}
    for unit in plan.get("units", []):
        pair = (int(unit["chapter_pair"][0]), int(unit["chapter_pair"][1]))
        if pair in {(1, 1), (2, 2)}:
            units_by_pair.setdefault(pair, []).append(unit)

    notes_by_candidate: dict[str, dict[str, object]] = {}
    for note in notes.get("notes", []):
        pair = (int(note["chapter_pair"][0]), int(note["chapter_pair"][1]))
        if pair not in {(1, 1), (2, 2)}:
            continue
        candidate_id = str(note["candidate_id"])
        if candidate_id in notes_by_candidate:
            raise ValueError(f"Ambiguous notes for candidate_id={candidate_id}")
        notes_by_candidate[candidate_id] = note

    used_substitutions: list[DebugSubstitution] = []
    used_notes: list[DebugNote] = []
    introduced_families: set[str] = set()
    body_lines: list[str] = []
    paragraph_count = 0
    for chapter_index, chapter in enumerate(chapters, start=1):
        chapter_title = str(chapter["title"])
        if chapter_index == 1 and chapter_title.strip().upper() == "WSTĘP":
            body_lines.append(f"\\chapter*{{{escape_latex(chapter_title)}}}")
            body_lines.append(f"\\addcontentsline{{toc}}{{chapter}}{{{escape_latex(chapter_title)}}}")
        else:
            body_lines.append(f"\\chapter{{{escape_latex(chapter_title)}}}")
        body_lines.append("")
        chapter_units = units_by_pair.get((chapter_index, chapter_index), [])
        for paragraph in chapter.get("paragraphs", []):
            paragraph_count += 1
            paragraph_index = int(paragraph["index"])
            matches = [
                unit
                for unit in chapter_units
                if int(unit["source_paragraph_range"][0]) <= paragraph_index <= int(unit["source_paragraph_range"][1])
            ]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous unit mapping for chapter {chapter_index} paragraph {paragraph_index}")
            candidates = [] if not matches else _paragraph_scoped_candidates(paragraph, list(matches[0].get("selected_candidates", [])))
            if candidates:
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
            else:
                rendered = escape_latex(str(paragraph["text"]))
            body_lines.append(rendered)
            body_lines.append("")

    latex = (
        template.replace("%%BOOK_TITLE%%", escape_latex(str(pl["metadata"].get("title", ""))))
        .replace("%%BOOK_AUTHOR%%", "")
        .replace("%%BOOK_SUBTITLE%%", "")
        .replace("%%BOOK_CONTENT%%", "\n".join(body_lines).strip())
        .replace("%%GLOSSARY_CONTENT%%", "")
    )
    return {
        "detected_chapters": detected_chapters,
        "selected_chapters": selected_chapters,
        "paragraph_count": paragraph_count,
        "used_substitutions": used_substitutions,
        "used_notes": used_notes,
        "latex": latex,
        "plan": plan,
    }


def _paragraph_scoped_candidates(paragraph: dict[str, object], candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    scoped: list[dict[str, object]] = []
    for candidate in candidates:
        try:
            _candidate_span(paragraph, candidate, [])
        except Exception:
            continue
        scoped.append(candidate)
    return scoped


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    plan_override = os.environ.get("SZW_PLAN_PATH")
    notes_override = os.environ.get("SZW_NOTES_PATH")
    alignment_override = os.environ.get("SZW_ALIGNMENT_PATH")
    output_dir = root / "output"
    output_dir.mkdir(exist_ok=True)
    payload = build_debug_subset(
        root,
        plan_path=None if not plan_override else Path(plan_override),
        notes_path=None if not notes_override else Path(notes_override),
    )
    alignment_path = (
        Path(alignment_override)
        if alignment_override
        else root / "data/reports/full_hierarchical_alignment_e5_stanza.json"
    )
    alignment_artifact = json.loads(alignment_path.read_text(encoding="utf-8"))
    meta_dir = output_dir / "meta"
    meta_latex = build_pdf_meta_section(
        root=root,
        plan_payload=dict(payload["plan"]),
        alignment_artifact=alignment_artifact,
        output_dir=meta_dir,
        label="debug_intro_ch1",
    )
    payload["latex"] = str(payload["latex"]).replace("%%META_CONTENT%%", meta_latex)
    tex_path = output_dir / "debug_intro_ch1.tex"
    pdf_path = output_dir / "debug_intro_ch1.pdf"
    tex_path.write_text(str(payload["latex"]), encoding="utf-8")
    (output_dir / "debug_substitutions.json").write_text(
        json.dumps([item.to_dict() for item in payload["used_substitutions"]], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "debug_notes.json").write_text(
        json.dumps([item.to_dict() for item in payload["used_notes"]], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Detected chapters from corpus:")
    for chapter in payload["detected_chapters"]:
        print(json.dumps(chapter, ensure_ascii=False))
    print("Selected chapters:")
    for chapter in payload["selected_chapters"]:
        print(json.dumps(chapter, ensure_ascii=False))
    print(f"Paragraphs rendered: {payload['paragraph_count']}")
    print(f"Substitutions applied: {len(payload['used_substitutions'])}")
    print("First 10 substitutions:")
    for item in payload["used_substitutions"][:10]:
        print(json.dumps(item.to_dict(), ensure_ascii=False))
    print("First 10 notes attached:")
    for item in payload["used_notes"][:10]:
        print(json.dumps(item.to_dict(), ensure_ascii=False))

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
    text_dump = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    print("PDF text preview:")
    print(text_dump[:1200])


if __name__ == "__main__":
    main()
