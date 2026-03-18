"""Lemma-family timeline artifact for whole-book hybridization planning."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json

from szwejk.generate.paragraph_hybridization import DEFAULT_BLOCKED_STANDALONE_UPOS, _build_units, load_alignment_artifact


def build_lemma_timeline_report(
    alignment_artifact: dict[str, object],
    *,
    blocked_standalone_upos: frozenset[str] = DEFAULT_BLOCKED_STANDALONE_UPOS,
) -> dict[str, object]:
    units = _build_units(alignment_artifact, blocked_standalone_upos=blocked_standalone_upos)
    family_positions: dict[str, list[dict[str, object]]] = defaultdict(list)
    total_occurrence_count = sum(int(unit["source_occurrence_count"]) for unit in units) or 1

    for unit in units:
        unit_index = int(unit["unit_index"])
        chapter_pair = list(unit["chapter_pair"])
        source_range = list(unit["source_paragraph_range"])
        target_range = list(unit["target_paragraph_range"])
        preview = str(unit.get("source_preview", ""))
        for family_id, count in sorted(unit["family_occurrences"].items()):
            for local_occurrence_index in range(1, int(count) + 1):
                family_positions[family_id].append(
                    {
                        "unit_index": unit_index,
                        "chapter_pair": chapter_pair,
                        "source_paragraph_range": source_range,
                        "target_paragraph_range": target_range,
                        "local_occurrence_index": local_occurrence_index,
                        "source_preview": preview,
                    }
                )

    families: list[dict[str, object]] = []
    for family_id, occurrences in sorted(family_positions.items()):
        total_family_count = len(occurrences)
        entries = []
        for occurrence_index, occurrence in enumerate(occurrences, start=1):
            remaining_after = total_family_count - occurrence_index
            urgency = 1.0 / (1.0 + remaining_after)
            future_gain_if_introduced_here = remaining_after / total_occurrence_count
            entries.append(
                {
                    **occurrence,
                    "occurrence_index": occurrence_index,
                    "remaining_after_occurrence": remaining_after,
                    "urgency": round(urgency, 6),
                    "future_gain_if_introduced_here": round(future_gain_if_introduced_here, 8),
                }
            )
        families.append(
            {
                "family_id": family_id,
                "total_count": total_family_count,
                "first_unit_index": int(entries[0]["unit_index"]),
                "last_unit_index": int(entries[-1]["unit_index"]),
                "occurrences": entries,
            }
        )

    return {
        "family_count": len(families),
        "total_occurrence_count": total_occurrence_count,
        "families": families,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build lemma timeline artifact from full alignment.")
    parser.add_argument("--alignment", required=True, help="Path to full hierarchical alignment artifact JSON.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument(
        "--blocked-standalone-upos",
        default="SCONJ,CCONJ,PART,ADP",
        help="Comma-separated UPOS tags passed through unit building.",
    )
    args = parser.parse_args()

    blocked = frozenset(item.strip().upper() for item in str(args.blocked_standalone_upos).split(",") if item.strip())
    artifact = load_alignment_artifact(args.alignment)
    payload = build_lemma_timeline_report(artifact, blocked_standalone_upos=blocked)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
