#!/usr/bin/env python3
"""Compare staircase vs smooth hybridization plans with charts and reading samples."""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
from statistics import median

from szwejk.generate.debug_subset import _candidate_span, _merge_adjacent_noted_hits


CHAPTER_SAMPLES = [2, 15, 29]


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def render_policy_summary(plan: dict[str, object], czechness: dict[str, object]) -> dict[str, object]:
    paragraph_reports = list(czechness["paragraph_reports"])
    chapter_reports = list(czechness["chapter_reports"])
    distances = [float(unit["distance_after"]) for unit in plan["units"]]
    selected = [int(unit["selected_count"]) for unit in plan["units"]]
    local_after = [float(item["local_czechness_after"]) for item in paragraph_reports]
    final_chapter = next(item for item in chapter_reports if item["chapter_pair"] == [29, 29])
    return {
        "introduced_family_count": int(plan["summary"]["introduced_family_count"]),
        "selected_granularity_counts": dict(plan["summary"]["selected_granularity_counts"]),
        "avg_distance_after": round(sum(distances) / max(len(distances), 1), 6),
        "avg_selected_count": round(sum(selected) / max(len(selected), 1), 6),
        "weighted_local_czechness_after": round(
            sum(float(item["local_czechness_after"]) * float(item["family_occurrence_count"]) for item in paragraph_reports)
            / max(sum(float(item["family_occurrence_count"]) for item in paragraph_reports), 1.0),
            6,
        ),
        "chapter_29_local_czechness_after": float(final_chapter["local_czechness_after"]),
        "chapter_29_local_czechness_median": round(
            median(float(item["local_czechness_after"]) for item in paragraph_reports if item["chapter_pair"] == [29, 29]),
            6,
        ),
        "chapter_29_local_czechness_over_0_9": sum(
            1 for item in paragraph_reports if item["chapter_pair"] == [29, 29] and float(item["local_czechness_after"]) >= 0.9
        ),
        "unit_count": len(plan["units"]),
        "paragraph_reports_count": len(paragraph_reports),
        "local_after_min": min(local_after) if local_after else 0.0,
        "local_after_max": max(local_after) if local_after else 0.0,
    }


def build_samples(
    corpus: dict[str, object],
    plan: dict[str, object],
    notes: dict[str, object],
) -> list[dict[str, object]]:
    units_by_pair: dict[tuple[int, int], dict[int, dict[str, object]]] = {}
    for unit in plan["units"]:
        pair = (int(unit["chapter_pair"][0]), int(unit["chapter_pair"][1]))
        units_by_pair.setdefault(pair, {})[int(unit["source_paragraph_range"][0])] = unit

    notes_by_candidate = {str(note["candidate_id"]): note for note in notes.get("notes", [])}
    introduced_families: set[str] = set()
    samples: list[dict[str, object]] = []

    for chapter in corpus["chapters"]:
        chapter_index = int(chapter["index"])
        unit_lookup = units_by_pair.get((chapter_index, chapter_index), {})
        changed_paragraphs: list[dict[str, object]] = []
        for paragraph in chapter["paragraphs"]:
            paragraph_index = int(paragraph["index"])
            unit = unit_lookup.get(paragraph_index)
            if not unit:
                continue
            candidates = list(unit.get("selected_candidates", []))
            if not candidates:
                continue
            new_candidate_ids: set[str] = set()
            for candidate in candidates:
                family_ids = [str(item) for item in candidate.get("family_ids", [])]
                if any(family_id not in introduced_families for family_id in family_ids):
                    new_candidate_ids.add(str(candidate["candidate_id"]))
                introduced_families.update(family_ids)
            changed_paragraphs.append(
                {
                    "chapter_index": chapter_index,
                    "chapter_title": str(chapter["title"]),
                    "paragraph_index": paragraph_index,
                    "original_text": str(paragraph["text"]),
                    "rendered_text": render_plain_hybrid(paragraph, candidates, notes_by_candidate, new_candidate_ids),
                    "selected_granularity_counts": dict(Counter(candidate["granularity"] for candidate in candidates)),
                }
            )
        if chapter_index in CHAPTER_SAMPLES:
            samples.extend(changed_paragraphs[:5])
    return samples


def render_plain_hybrid(
    paragraph: dict[str, object],
    candidates: list[dict[str, object]],
    notes_by_candidate: dict[str, dict[str, object]],
    new_candidate_ids: set[str],
) -> str:
    paragraph_text = str(paragraph["text"])
    occupied: list[tuple[int, int]] = []
    hits: list[dict[str, object]] = []
    for candidate in candidates:
        try:
            span = _candidate_span(paragraph, candidate, occupied)
        except Exception:
            span = _fallback_span(paragraph_text, candidate, occupied)
        if span is None:
            continue
        occupied.append(span)
        note = notes_by_candidate.get(str(candidate["candidate_id"]))
        hits.append(
            {
                "start": span[0],
                "end": span[1],
                "candidate": candidate,
                "note": note,
                "is_new": str(candidate["candidate_id"]) in new_candidate_ids,
                "target_text": str(candidate["target_text"]),
            }
        )
    hits = _merge_adjacent_noted_hits(paragraph_text, hits)
    parts: list[str] = []
    cursor = 0
    note_counter = 0
    note_lines: list[str] = []
    for hit in sorted(hits, key=lambda item: int(item["start"])):
        parts.append(paragraph_text[cursor : int(hit["start"])])
        note = hit["note"]
        if note is not None:
            note_counter += 1
            parts.append(f"<<{hit['target_text']}>>[{note_counter}]")
            note_lines.append(f"[{note_counter}] {note['note_text']}")
        elif hit["is_new"]:
            parts.append(f"**{hit['target_text']}**")
        else:
            parts.append(str(hit["target_text"]))
        cursor = int(hit["end"])
    parts.append(paragraph_text[cursor:])
    rendered = "".join(parts)
    if note_lines:
        rendered += "\nNotes: " + " | ".join(note_lines)
    return rendered


def _fallback_span(
    paragraph_text: str,
    candidate: dict[str, object],
    occupied: list[tuple[int, int]],
) -> tuple[int, int] | None:
    source_text = str(candidate.get("source_text", ""))
    if not source_text:
        return None
    start = paragraph_text.find(source_text)
    while start != -1:
        end = start + len(source_text)
        if all(end <= left or start >= right for left, right in occupied):
            return (start, end)
        start = paragraph_text.find(source_text, start + 1)
    return None


def moving_average(values: list[float], window: int) -> list[float]:
    radius = window // 2
    output: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - radius)
        end = min(len(values), idx + radius + 1)
        bucket = values[start:end]
        output.append(sum(bucket) / max(len(bucket), 1))
    return output


def polyline_points(values: list[float], *, width: int, height: int, left: int, top: int, plot_width: int, plot_height: int) -> str:
    if not values:
        return ""
    points: list[str] = []
    max_index = max(len(values) - 1, 1)
    for idx, value in enumerate(values):
        x = left + (plot_width * idx / max_index)
        y = top + plot_height - (plot_height * max(0.0, min(1.0, value)))
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def write_compare_svg(path: Path, title: str, series: list[tuple[str, str, list[float]]]) -> None:
    width = 1400
    height = 700
    left = 80
    top = 60
    plot_width = 1260
    plot_height = 560
    colors = {"reader": "#355c7d", "smooth": "#c06c84", "target": "#6c7a89"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        f'<text x="{left}" y="32" font-size="24" font-family="Helvetica" fill="#111">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333" stroke-width="1"/>',
    ]
    for tick in range(6):
        y = top + plot_height - (plot_height * tick / 5)
        value = tick / 5
        lines.append(f'<line x1="{left-5}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e4dfd5" stroke-width="1"/>')
        lines.append(f'<text x="18" y="{y+4:.2f}" font-size="14" font-family="Helvetica" fill="#333">{value:.1f}</text>')
    legend_x = left + 20
    for idx, (label, key, values) in enumerate(series):
        y = height - 38 + (idx // 3) * 20
        x = legend_x + (idx % 3) * 260
        lines.append(f'<line x1="{x}" y1="{y}" x2="{x+28}" y2="{y}" stroke="{colors[key]}" stroke-width="4"/>')
        lines.append(f'<text x="{x+36}" y="{y+5}" font-size="16" font-family="Helvetica" fill="#222">{label}</text>')
        points = polyline_points(values, width=width, height=height, left=left, top=top, plot_width=plot_width, plot_height=plot_height)
        lines.append(f'<polyline fill="none" stroke="{colors[key]}" stroke-width="2.5" points="{points}"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_samples(path: Path, samples_by_policy: dict[str, list[dict[str, object]]]) -> None:
    lines = ["# Hybridization Samples", ""]
    for policy, samples in samples_by_policy.items():
        lines.append(f"## {policy}")
        lines.append("")
        for sample in samples:
            lines.append(f"### Rozdział {sample['chapter_index']}: {sample['chapter_title']} / akapit {sample['paragraph_index']}")
            lines.append("")
            lines.append("Oryginał:")
            lines.append(sample["original_text"])
            lines.append("")
            lines.append("Hybryda:")
            lines.append(sample["rendered_text"])
            lines.append("")
            lines.append(f"Granularności: {sample['selected_granularity_counts']}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--reader-plan", required=True)
    parser.add_argument("--smooth-plan", required=True)
    parser.add_argument("--reader-notes", required=True)
    parser.add_argument("--smooth-notes", required=True)
    parser.add_argument("--reader-czechness", required=True)
    parser.add_argument("--smooth-czechness", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_json(Path(args.corpus))
    reader_plan = load_json(Path(args.reader_plan))
    smooth_plan = load_json(Path(args.smooth_plan))
    reader_notes = load_json(Path(args.reader_notes))
    smooth_notes = load_json(Path(args.smooth_notes))
    reader_czechness = load_json(Path(args.reader_czechness))
    smooth_czechness = load_json(Path(args.smooth_czechness))

    summary = {
        "reader": render_policy_summary(reader_plan, reader_czechness),
        "smooth": render_policy_summary(smooth_plan, smooth_czechness),
    }
    (output_dir / "hybridization_policy_compare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    reader_future = [float(unit["actual_future_czechness_after"]) for unit in reader_plan["units"]]
    smooth_future = [float(unit["actual_future_czechness_after"]) for unit in smooth_plan["units"]]
    target = [float(unit["target_future_czechness_after"]) for unit in reader_plan["units"]]
    write_compare_svg(
        output_dir / "paragraph_future_czechness_raw_reader_vs_smooth.svg",
        "Paragraph Future Czechness: Reader vs Smooth",
        [
            ("Target", "target", target),
            ("Reader", "reader", reader_future),
            ("Smooth", "smooth", smooth_future),
        ],
    )

    reader_local = [float(item["local_czechness_after"]) for item in reader_czechness["paragraph_reports"]]
    smooth_local = [float(item["local_czechness_after"]) for item in smooth_czechness["paragraph_reports"]]
    write_compare_svg(
        output_dir / "paragraph_local_czechness_smoothed_51_reader_vs_smooth.svg",
        "Paragraph Local Czechness (Smoothed 51): Reader vs Smooth",
        [
            ("Reader", "reader", moving_average(reader_local, 51)),
            ("Smooth", "smooth", moving_average(smooth_local, 51)),
        ],
    )

    reader_chapter = [float(item["local_czechness_after"]) for item in reader_czechness["chapter_reports"]]
    smooth_chapter = [float(item["local_czechness_after"]) for item in smooth_czechness["chapter_reports"]]
    write_compare_svg(
        output_dir / "chapter_local_czechness_reader_vs_smooth.svg",
        "Chapter Local Czechness: Reader vs Smooth",
        [
            ("Reader", "reader", reader_chapter),
            ("Smooth", "smooth", smooth_chapter),
        ],
    )

    samples = {
        "reader": build_samples(corpus, reader_plan, reader_notes),
        "smooth": build_samples(corpus, smooth_plan, smooth_notes),
    }
    write_markdown_samples(output_dir / "hybridization_policy_compare_samples.md", samples)


if __name__ == "__main__":
    main()
