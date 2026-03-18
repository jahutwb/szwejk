"""PDF meta section and chart assets for hybridization diagnostics."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from szwejk.generate.czechness_report import build_czechness_report


def build_pdf_meta_section(
    *,
    root: Path,
    plan_payload: dict[str, object],
    alignment_artifact: dict[str, object],
    output_dir: Path,
    label: str,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    czechness = build_czechness_report(alignment_artifact, plan_payload)
    summary = _meta_summary(plan_payload, czechness)
    (output_dir / f"{label}_meta_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    future_path = output_dir / f"{label}_future_czechness.png"
    local_path = output_dir / f"{label}_local_czechness_smoothed_51.png"
    chapter_path = output_dir / f"{label}_chapter_local_czechness.png"
    chapter_simple_path = output_dir / f"{label}_chapter_simple_czechness.png"
    chapter_mix_path = output_dir / f"{label}_chapter_granularity_mix.png"
    cumulative_simple_path = output_dir / f"{label}_cumulative_simple_czechness.png"
    local_simple_target_path = output_dir / f"{label}_local_simple_target_vs_actual.png"

    _plot_future_czechness(plan_payload, future_path)
    _plot_local_czechness(czechness, local_path)
    _plot_chapter_czechness(czechness, chapter_path)
    _plot_chapter_simple_czechness(czechness, chapter_simple_path)
    _plot_chapter_granularity_mix(czechness, chapter_mix_path)
    _plot_cumulative_simple_czechness(plan_payload, cumulative_simple_path)
    _plot_local_simple_target_vs_actual(plan_payload, czechness, local_simple_target_path)

    policy_lines = [
        "\\chapter*{Meta}",
        "\\addcontentsline{toc}{chapter}{Meta}",
        "\\section*{Polityka hybrydyzacji}",
        "\\begin{itemize}",
        f"\\item Tryb: {escape_latex(str(plan_payload.get('mode', '')))}",
        f"\\item Granularity policy: {escape_latex(str(plan_payload.get('granularity_policy', 'staircase')))}",
        f"\\item Target power: {escape_latex(str(plan_payload.get('target_power', '')))}",
        f"\\item Zablokowane standalone UPOS: {escape_latex(', '.join(plan_payload.get('blocked_standalone_upos', [])))}",
        f"\\item Wprowadzone rodziny: {int(summary['introduced_family_count'])}",
        f"\\item Ważona lokalna czeskość lematyczna: {summary['weighted_local_czechness_after']}",
        f"\\item Prosta czeskość tekstu: {summary['weighted_simple_czechness_after']}",
        f"\\item Ważona widoczna czeskość: {summary['weighted_visible_czechness_after']}",
        f"\\item Rozdział 29 prosta czeskość: {summary['chapter_29_simple_czechness_after']}",
        f"\\item Rozdział 29 widoczna czeskość: {summary['chapter_29_visible_czechness_after']}",
        "\\end{itemize}",
        "\\section*{Opis sterowania}",
        "\\noindent Planer steruje przede wszystkim docelową czeskością tekstu widoczną na powierzchni, a znajomość rodzin lematów traktuje pomocniczo. Identyczne lub prawie identyczne podstawienia mogą wejść wcześnie, ale mają mały wpływ na metrykę czeskości; większe frazy, zdania i akapity zaczynają dominować dopiero wtedy, gdy realnie podnoszą widoczną czeskość czytanego tekstu.\\\\",
        "\\section*{Rozkład podmian}",
        "\\begin{description}",
    ]
    for key, value in summary["selected_granularity_counts"].items():
        policy_lines.append(f"\\item[{escape_latex(key)}] {int(value)}")
    policy_lines.extend(
        [
            "\\end{description}",
            "\\section*{Wykresy}",
            "\\begin{figure}[h!]",
            "\\centering",
            f"\\includegraphics[width=0.92\\textwidth]{{{_latex_path(future_path, output_dir.parent)}}}",
            "\\caption*{Future czechness po akapitach}",
            "\\end{figure}",
            "\\begin{figure}[h!]",
            "\\centering",
            f"\\includegraphics[width=0.92\\textwidth]{{{_latex_path(cumulative_simple_path, output_dir.parent)}}}",
            "\\caption*{Średnia skumulowana prosta czeskość: oczekiwana i rzeczywista}",
            "\\end{figure}",
            "\\begin{figure}[h!]",
            "\\centering",
            f"\\includegraphics[width=0.92\\textwidth]{{{_latex_path(local_simple_target_path, output_dir.parent)}}}",
            "\\caption*{Chwilowa prosta czeskość: target i rzeczywista}",
            "\\end{figure}",
            "\\begin{figure}[h!]",
            "\\centering",
            f"\\includegraphics[width=0.92\\textwidth]{{{_latex_path(local_path, output_dir.parent)}}}",
            "\\caption*{Local czechness po akapitach, średnia ruchoma 51}",
            "\\end{figure}",
            "\\begin{figure}[h!]",
            "\\centering",
            f"\\includegraphics[width=0.92\\textwidth]{{{_latex_path(chapter_path, output_dir.parent)}}}",
            "\\caption*{Widoczna czeskość po rozdziałach}",
            "\\end{figure}",
            "\\begin{figure}[h!]",
            "\\centering",
            f"\\includegraphics[width=0.92\\textwidth]{{{_latex_path(chapter_simple_path, output_dir.parent)}}}",
            "\\caption*{Prosta czeskość po rozdziałach}",
            "\\end{figure}",
            "\\begin{figure}[h!]",
            "\\centering",
            f"\\includegraphics[width=0.96\\textwidth]{{{_latex_path(chapter_mix_path, output_dir.parent)}}}",
            "\\caption*{Udział granularności w prostej czeskości po rozdziałach}",
            "\\end{figure}",
        ]
    )
    return "\n".join(policy_lines)


def build_meta_pdf_latex(
    *,
    root: Path,
    plan_payload: dict[str, object],
    alignment_artifact: dict[str, object],
    output_dir: Path,
    label: str,
    title: str = "Meta eksperymentu hybrydyzacji",
) -> str:
    meta_body = build_pdf_meta_section(
        root=root,
        plan_payload=plan_payload,
        alignment_artifact=alignment_artifact,
        output_dir=output_dir,
        label=label,
    )
    return "\n".join(
        [
            r"\documentclass[11pt,a5paper]{book}",
            r"\usepackage[margin=16mm]{geometry}",
            r"\usepackage{fontspec}",
            r"\usepackage{graphicx}",
            r"\usepackage{hyperref}",
            r"\usepackage{float}",
            r"\usepackage{bookmark}",
            r"\setmainfont{TeX Gyre Pagella}",
            r"\begin{document}",
            rf"\title{{{escape_latex(title)}}}",
            r"\author{}",
            r"\date{}",
            r"\maketitle",
            r"\tableofcontents",
            meta_body,
            r"\end{document}",
        ]
    )


def _meta_summary(plan_payload: dict[str, object], czechness: dict[str, object]) -> dict[str, object]:
    paragraph_reports = list(czechness["paragraph_reports"])
    chapter_reports = list(czechness["chapter_reports"])
    total_occ = sum(float(item["family_occurrence_count"]) for item in paragraph_reports) or 1.0
    chapter_29 = next((item for item in chapter_reports if item["chapter_pair"] == [29, 29]), None)
    return {
        "introduced_family_count": int(plan_payload["summary"]["introduced_family_count"]),
        "selected_granularity_counts": dict(plan_payload["summary"]["selected_granularity_counts"]),
        "weighted_local_czechness_after": round(
            sum(float(item["local_czechness_after"]) * float(item["family_occurrence_count"]) for item in paragraph_reports) / total_occ,
            6,
        ),
        "weighted_simple_czechness_after": round(
            _weighted_simple_average(paragraph_reports),
            6,
        ),
        "weighted_visible_czechness_after": round(
            _weighted_visible_average(paragraph_reports),
            6,
        ),
        "chapter_29_local_czechness_after": 0.0 if chapter_29 is None else float(chapter_29["local_czechness_after"]),
        "chapter_29_simple_czechness_after": 0.0 if chapter_29 is None else float(chapter_29.get("local_simple_czechness_after", 0.0)),
        "chapter_29_visible_czechness_after": 0.0 if chapter_29 is None else float(chapter_29.get("local_visible_czechness_after", 0.0)),
    }


def _plot_future_czechness(plan_payload: dict[str, object], path: Path) -> None:
    target = [float(unit["target_future_czechness_after"]) for unit in plan_payload["units"]]
    actual = [float(unit["actual_future_czechness_after"]) for unit in plan_payload["units"]]
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=160)
    ax.plot(target, label="target", color="#6c7a89", linewidth=1.4)
    ax.plot(actual, label="actual", color="#355c7d", linewidth=1.2)
    ax.set_ylim(0, 1.02)
    ax.set_title("Future czechness")
    ax.set_xlabel("blok akapitowy")
    ax.set_ylabel("udział")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_local_czechness(czechness: dict[str, object], path: Path) -> None:
    values = [float(item.get("local_visible_czechness_after", item["local_czechness_after"])) for item in czechness["paragraph_reports"]]
    smoothed = _moving_average(values, 51)
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=160)
    ax.plot(smoothed, color="#c06c84", linewidth=1.4)
    ax.set_ylim(0, 1.02)
    ax.set_title("Visible local czechness, smoothing=51")
    ax.set_xlabel("blok akapitowy")
    ax.set_ylabel("udział")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_chapter_czechness(czechness: dict[str, object], path: Path) -> None:
    values = [float(item.get("local_visible_czechness_after", item["local_czechness_after"])) for item in czechness["chapter_reports"]]
    xs = list(range(1, len(values) + 1))
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=160)
    ax.plot(xs, values, marker="o", color="#2a9d8f", linewidth=1.4)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(xs)
    ax.set_title("Visible local czechness by chapter")
    ax.set_xlabel("rozdział")
    ax.set_ylabel("udział")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_chapter_simple_czechness(czechness: dict[str, object], path: Path) -> None:
    values = [float(item.get("local_simple_czechness_after", 0.0)) for item in czechness["chapter_reports"]]
    xs = list(range(1, len(values) + 1))
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=160)
    ax.plot(xs, values, marker="o", color="#8d6e63", linewidth=1.4)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(xs)
    ax.set_title("Simple czechness by chapter")
    ax.set_xlabel("rozdział")
    ax.set_ylabel("udział")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_cumulative_simple_czechness(plan_payload: dict[str, object], path: Path) -> None:
    cumulative_target = [
        float(unit.get("target_cumulative_simple_czechness_after", unit["target_future_czechness_after"]))
        for unit in plan_payload["units"]
    ]
    actual = [float(unit.get("actual_simple_czechness_after", 0.0)) for unit in plan_payload["units"]]
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=160)
    ax.plot(cumulative_target, label="target cumulative", color="#6c7a89", linewidth=1.4)
    ax.plot(actual, label="actual", color="#8d6e63", linewidth=1.2)
    ax.set_ylim(0, 1.02)
    ax.set_title("Cumulative average simple czechness")
    ax.set_xlabel("blok akapitowy")
    ax.set_ylabel("udział")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_local_simple_target_vs_actual(
    plan_payload: dict[str, object],
    czechness: dict[str, object],
    path: Path,
) -> None:
    target = [
        float(unit.get("target_local_simple_czechness_after", unit["target_future_czechness_after"]))
        for unit in plan_payload["units"]
    ]
    actual = [float(item.get("local_simple_czechness_after", 0.0)) for item in czechness["paragraph_reports"]]
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=160)
    ax.plot(target, label="target local", color="#6c7a89", linewidth=1.4)
    ax.plot(actual, label="actual local", color="#b56576", linewidth=1.2)
    ax.set_ylim(0, 1.02)
    ax.set_title("Local simple czechness")
    ax.set_xlabel("blok akapitowy")
    ax.set_ylabel("udział")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_chapter_granularity_mix(czechness: dict[str, object], path: Path) -> None:
    chapter_reports = list(czechness["chapter_reports"])
    xs = list(range(1, len(chapter_reports) + 1))
    granularities = ["token", "phrase", "subtree", "sentence", "paragraph"]
    colors = {
        "token": "#5B8FF9",
        "phrase": "#61DDAA",
        "subtree": "#65789B",
        "sentence": "#F6BD16",
        "paragraph": "#E8684A",
    }
    fig, ax = plt.subplots(figsize=(11.2, 5.2), dpi=160)
    bottom = [0.0] * len(xs)
    for granularity in granularities:
        values = [
            float(report.get("local_simple_czechness_after", 0.0))
            * float(report.get("simple_share_by_granularity", {}).get(granularity, 0.0))
            for report in chapter_reports
        ]
        ax.bar(xs, values, bottom=bottom, color=colors[granularity], width=0.78, label=granularity)
        bottom = [left + right for left, right in zip(bottom, values)]
    ax.set_ylim(0, max(1.02, max(bottom, default=0.0) * 1.08))
    ax.set_xticks(xs)
    ax.set_title("Simple czechness by chapter, split by granularity")
    ax.set_xlabel("rozdział")
    ax.set_ylabel("udział")
    ax.legend(ncol=5, fontsize=8, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _moving_average(values: list[float], window: int) -> list[float]:
    radius = window // 2
    out: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - radius)
        end = min(len(values), idx + radius + 1)
        bucket = values[start:end]
        out.append(sum(bucket) / max(len(bucket), 1))
    return out


def _weighted_visible_average(paragraph_reports: list[dict[str, object]]) -> float:
    total = sum(float(item.get("source_word_count", 0)) for item in paragraph_reports)
    if total <= 0:
        return 0.0
    return sum(
        float(item.get("local_visible_czechness_after", 0.0)) * float(item.get("source_word_count", 0))
        for item in paragraph_reports
    ) / total


def _weighted_simple_average(paragraph_reports: list[dict[str, object]]) -> float:
    total = sum(float(item.get("source_word_count", 0)) for item in paragraph_reports)
    if total <= 0:
        return 0.0
    return sum(
        float(item.get("local_simple_czechness_after", 0.0)) * float(item.get("source_word_count", 0))
        for item in paragraph_reports
    ) / total


def _latex_path(path: Path, tex_root: Path) -> str:
    return str(path.relative_to(tex_root)).replace("\\", "/")


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
