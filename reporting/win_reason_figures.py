#!/usr/bin/env python3
"""Render paper-ready win-pathway and vote-mechanics figures from CSV outputs.

Run `reporting/win_reasons.py` first, then render these figures. For the
7-model paper focus set, the usual sequence is:

    uv run --with polars --with rich python reporting/win_reasons.py \
        --seasons 0,1 --cohort exemplars --cohort-match all \
        --out-dir reporting/out --markdown reporting/win_reasons.md

    uv run python reporting/win_reason_figures.py \
        --cohort exemplars --theme light --formats png,pdf

Default outputs:

    reporting/win_reason_pathways_exemplars_light.png
    reporting/vote_mechanics_exemplars_light.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


CATEGORY_ORDER = [
    ("crew_tasks", "Crew tasks", "#4C78A8"),
    ("crew_vote_out", "Crew vote-out", "#72B7B2"),
    ("imp_outnumber", "Imp kills", "#E45756"),
    ("imp_time_limit", "Imp time limit", "#F58518"),
    ("unknown", "Unknown", "#9D9D9D"),
]

COUNT_METRICS = [
    ("n_meetings", "Meetings/game"),
    ("n_ejections", "Ejections/game"),
    ("n_imp_ejections", "Imp ejections/game"),
]

PERCENT_METRICS = [
    ("vote_quality", "Vote quality"),
    ("vote_accuracy", "Vote accuracy"),
]

THEMES = {
    "light": {
        "bg": "#ffffff",
        "fg": "#1f2328",
        "muted": "#656d76",
        "grid": "#d8dee4",
        "bar": "#0969da",
        "bar_alt": "#d1242f",
    },
    "dark": {
        "bg": "#0d1117",
        "fg": "#e6edf3",
        "muted": "#8b949e",
        "grid": "#30363d",
        "bar": "#58a6ff",
        "bar_alt": "#f85149",
    },
}


def default_csv_path(input_dir: Path, kind: str, cohort: str) -> Path:
    suffix = "" if cohort == "all" else f"_{cohort}"
    return input_dir / f"win_reasons_{kind}{suffix}.csv"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Generate it with reporting/win_reasons.py first."
        )
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def season_label(version: int, rows: list[dict[str, str]]) -> str:
    totals = {int(float(r["engine_version"])): int(float(r["season_total"])) for r in rows}
    n = totals.get(version)
    labels = {0: "Short-context", 1: "Long-context"}
    label = labels.get(version, f"Engine {version}")
    return f"{label}\n(n={n})" if n else label


def style_axes(ax, theme: dict[str, str]) -> None:
    ax.set_facecolor(theme["bg"])
    ax.tick_params(colors=theme["fg"])
    for spine in ax.spines.values():
        spine.set_color(theme["grid"])
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def plot_win_pathways(
    rows: list[dict[str, str]],
    out_path: Path,
    theme_name: str,
    title: str,
) -> None:
    theme = THEMES[theme_name]
    versions = sorted({int(float(r["engine_version"])) for r in rows})
    by_version = {
        v: {r["category"]: float(r["pct"]) for r in rows if int(float(r["engine_version"])) == v}
        for v in versions
    }

    fig, ax = plt.subplots(figsize=(6.8, 4.2), facecolor=theme["bg"])
    style_axes(ax, theme)

    x = list(range(len(versions)))
    bottoms = [0.0] * len(versions)
    for category, label, color in CATEGORY_ORDER:
        vals = [by_version[v].get(category, 0.0) for v in versions]
        if not any(vals):
            continue
        ax.bar(x, vals, bottom=bottoms, label=label, color=color, width=0.58)
        for i, val in enumerate(vals):
            if val >= 6:
                ax.text(
                    x[i],
                    bottoms[i] + val / 2,
                    f"{val:.0f}%",
                    ha="center",
                    va="center",
                    color="#ffffff",
                    fontsize=9,
                    fontweight="bold",
                )
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_ylim(0, 100)
    ax.set_ylabel("Games (%)", color=theme["fg"])
    ax.set_xticks(x, [season_label(v, rows) for v in versions])
    ax.set_title(title, color=theme["fg"], pad=12, fontweight="bold")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
        labelcolor=theme["fg"],
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=theme["bg"])
    plt.close(fig)


def plot_vote_mechanics(
    rows: list[dict[str, str]],
    out_path: Path,
    theme_name: str,
    title: str,
) -> None:
    theme = THEMES[theme_name]
    versions = sorted({int(float(r["engine_version"])) for r in rows})
    by_version = {int(float(r["engine_version"])): r for r in rows}

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.9), facecolor=theme["bg"])
    for ax in axes:
        style_axes(ax, theme)

    width = 0.36
    for ax, metrics, as_percent, panel_title in [
        (axes[0], COUNT_METRICS, False, "Meeting and ejection volume"),
        (axes[1], PERCENT_METRICS, True, "Vote targeting"),
    ]:
        x = list(range(len(metrics)))
        for offset, version in [(-width / 2, versions[0]), (width / 2, versions[-1])]:
            vals = []
            for key, _ in metrics:
                raw = float(by_version[version][key])
                vals.append(raw * 100 if as_percent else raw)
            ax.bar(
                [i + offset for i in x],
                vals,
                width=width,
                label=f"S{version}",
                color=theme["bar"] if version == versions[0] else theme["bar_alt"],
            )
            for i, val in enumerate(vals):
                label = f"{val:.0f}%" if as_percent else f"{val:.2f}"
                ax.text(
                    i + offset,
                    val,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=theme["fg"],
                )

        ax.set_title(panel_title, color=theme["fg"], fontsize=11, fontweight="bold")
        ax.set_xticks(x, [label for _, label in metrics], rotation=20, ha="right")
        ax.set_ylabel("%" if as_percent else "Mean per game", color=theme["fg"])
        ax.legend(frameon=False, labelcolor=theme["fg"])

    fig.suptitle(title, color=theme["fg"], fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=theme["bg"])
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).parent / "out",
        help="Directory containing win_reasons_*.csv files (default: reporting/out).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory for rendered figures (default: reporting).",
    )
    parser.add_argument(
        "--cohort",
        default="exemplars",
        help="CSV cohort suffix to read and output suffix to write (default: exemplars).",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Override path to win_reasons_summary*.csv.",
    )
    parser.add_argument(
        "--mechanics-csv",
        type=Path,
        default=None,
        help="Override path to win_reasons_mechanics*.csv.",
    )
    parser.add_argument(
        "--theme",
        choices=sorted(THEMES),
        default="light",
        help="Figure theme (default: light).",
    )
    parser.add_argument(
        "--formats",
        default="png",
        help="Comma-separated output formats, e.g. png,pdf (default: png).",
    )
    args = parser.parse_args()

    summary_csv = args.summary_csv or default_csv_path(args.input_dir, "summary", args.cohort)
    mechanics_csv = args.mechanics_csv or default_csv_path(
        args.input_dir, "mechanics", args.cohort
    )
    summary_rows = read_csv_rows(summary_csv)
    mechanics_rows = read_csv_rows(mechanics_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    formats = [fmt.strip().lstrip(".") for fmt in args.formats.split(",") if fmt.strip()]
    suffix = f"_{args.cohort}_{args.theme}"
    for fmt in formats:
        plot_win_pathways(
            summary_rows,
            args.out_dir / f"win_reason_pathways{suffix}.{fmt}",
            args.theme,
            "Game-ending pathways by context regime",
        )
        plot_vote_mechanics(
            mechanics_rows,
            args.out_dir / f"vote_mechanics{suffix}.{fmt}",
            args.theme,
            "Voting mechanics by context regime",
        )


if __name__ == "__main__":
    main()
