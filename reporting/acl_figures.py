#!/usr/bin/env python3
"""Camera-ready ACL paper figures (vector PDF + SVG, no titles).

Emits three figures from the local season CSVs in `reporting/out/`, sized for a
single ACL column (~3.03in): fonts are large relative to the canvas so they
print at ~6.5-7pt when scaled to `\\columnwidth`.

1. `openskill_ratings_short_context.{pdf,svg}` — exemplar cohort, Season 0.
2. `openskill_ratings_long_context.{pdf,svg}`  — exemplar cohort, Season 1.
   Both share the same x-range and model ordering (sorted by long-context peak
   role rating) so the two figures are directly comparable side by side.
3. `win_rates_exemplars_short_vs_long.{pdf,svg}` — role win-rate dumbbells,
   short-context and long-context panels stacked.

Titles are intentionally omitted — captions live in the LaTeX source.

Usage:
    uv run reporting/acl_figures.py
    uv run reporting/acl_figures.py --csv-dir reporting/out --out-dir paper-figures
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from season_chart import (  # noqa: E402
    EXEMPLAR_MODEL_IDS,
    THEMES,
    _fade,
    _wilson_ci,
    build_rows,
    load_local_season,
    sort_by_s1_peak,
)

# Canvas is 620px wide; at \columnwidth (3.03in ≈ 218pt) 1px ≈ 0.352pt, so
# 20px type prints at ~7pt and 18px at ~6.3pt — the readable floor for ACL.
WIDTH = 620
FONT = 20
TICK_FONT = 18

THEME = THEMES["light"]
# Role colors mirror the live leaderboard's chart palette
# (amongus-leaderboard/frontend/src/lib/theme/amongUsPalette.ts).
IMP_COLOR = "#F21717"
IMP_OUTLINE = "#C51111"
CREW_COLOR = "#75DBF4"
CREW_OUTLINE = "#235685"
OVERALL_COLOR = "#FFDE2A"
OVERALL_OUTLINE = "#235685"
OVERALL_ERROR = "#EF7D0E"

BASE_LAYOUT = dict(
    paper_bgcolor=THEME["bg"],
    plot_bgcolor=THEME["panel"],
    font=dict(
        color=THEME["text"],
        size=FONT,
        family="Helvetica, Arial, sans-serif",
    ),
    width=WIDTH,
)


def build_rating_panel(
    rows: list[dict],
    snap_key: str,
    names: list[str],
    x_range: list[float],
) -> go.Figure:
    """Single-season OpenSkill chart: bar = conservative rating (μ − σ),
    one-sided whisker up to μ. Same visual language as the leaderboard."""
    fig = go.Figure()

    series = {
        "Overall": (
            "overall_rating",
            "overall_sigma",
            OVERALL_COLOR,
            OVERALL_OUTLINE,
            OVERALL_ERROR,
        ),
        "Impostor": ("imp_rating", "imp_sigma", IMP_COLOR, IMP_OUTLINE, IMP_OUTLINE),
        "Crewmate": (
            "crew_rating",
            "crew_sigma",
            CREW_COLOR,
            CREW_OUTLINE,
            CREW_OUTLINE,
        ),
    }
    for label, (mu_key, sigma_key, color, outline, err_color) in series.items():
        xs, ys, sigmas = [], [], []
        for r in rows:
            snap = r[snap_key]
            if snap is None:
                continue
            xs.append(snap[mu_key] - snap[sigma_key])
            ys.append(r["name"])
            sigmas.append(snap[sigma_key])
        fig.add_trace(
            go.Bar(
                x=xs,
                y=ys,
                orientation="h",
                name=label,
                marker=dict(color=color, line=dict(color=outline, width=1.25)),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=sigmas,
                    arrayminus=[0] * len(sigmas),
                    color=_fade(err_color, 0.85),
                    thickness=1.6,
                    width=5,
                ),
            )
        )

    fig.update_layout(
        **BASE_LAYOUT,
        barmode="group",
        bargap=0.28,
        bargroupgap=0.08,
        showlegend=True,
        legend=dict(
            orientation="h",
            y=1.02,
            x=0.5,
            xanchor="center",
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=TICK_FONT),
            itemsizing="constant",
            traceorder="normal",
        ),
        margin=dict(l=10, r=20, t=60, b=10),
        height=64 * len(names) + 140,
    )
    fig.update_xaxes(
        range=x_range,
        tickfont=dict(size=TICK_FONT, color=THEME["muted"]),
        gridcolor=THEME["grid"],
        zerolinecolor=THEME["grid"],
        showline=False,
        tickformat=",d",
    )
    fig.update_yaxes(
        tickfont=dict(size=FONT, color=THEME["text"]),
        showgrid=False,
        categoryorder="array",
        categoryarray=names,
        automargin=True,
    )
    return fig


def build_winrate_stacked(rows: list[dict], names: list[str]) -> go.Figure:
    """Win-rate dumbbells, short-context (top) and long-context (bottom)
    panels stacked with a shared x-axis. Marker size encodes √role-games."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=["Short-context", "Long-context"],
    )

    MIN_SIZE, MAX_SIZE = 9.0, 24.0
    all_ns = [
        snap[k]
        for r in rows
        for snap in (r["s0"], r["s1"])
        if snap is not None
        for k in ("imp_games", "crew_games")
    ]
    n_lo, n_hi = (min(all_ns), max(all_ns)) if all_ns else (1, 1)

    def _size_for(n: int) -> float:
        if n_hi == n_lo:
            return (MIN_SIZE + MAX_SIZE) / 2
        t = (math.sqrt(n) - math.sqrt(n_lo)) / (math.sqrt(n_hi) - math.sqrt(n_lo))
        return MIN_SIZE + t * (MAX_SIZE - MIN_SIZE)

    for plot_row, snap_key in ((1, "s0"), (2, "s1")):
        imp_x, imp_y, imp_sizes = [], [], []
        crew_x, crew_y, crew_sizes = [], [], []
        for r in rows:
            snap = r[snap_key]
            if snap is None:
                continue
            iv, cv = snap["imp_wr"], snap["crew_wr"]
            imp_x.append(iv)
            imp_y.append(r["name"])
            imp_sizes.append(_size_for(snap["imp_games"]))
            crew_x.append(cv)
            crew_y.append(r["name"])
            crew_sizes.append(_size_for(snap["crew_games"]))
            fig.add_trace(
                go.Scatter(
                    x=[iv, cv],
                    y=[r["name"], r["name"]],
                    mode="lines",
                    line=dict(color=IMP_COLOR if iv > cv else CREW_COLOR, width=4),
                    opacity=0.75,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=plot_row,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=imp_x,
                y=imp_y,
                mode="markers",
                name="Impostor",
                marker=dict(
                    color=IMP_COLOR,
                    size=imp_sizes,
                    symbol="triangle-up",
                    line=dict(color=IMP_OUTLINE, width=1.5),
                ),
                legendgroup="imp",
                showlegend=(plot_row == 1),
            ),
            row=plot_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=crew_x,
                y=crew_y,
                mode="markers",
                name="Crewmate",
                marker=dict(
                    color=CREW_COLOR,
                    size=crew_sizes,
                    symbol="circle",
                    line=dict(color=CREW_OUTLINE, width=1.5),
                ),
                legendgroup="crew",
                showlegend=(plot_row == 1),
            ),
            row=plot_row,
            col=1,
        )
        fig.add_vline(
            x=50,
            row=plot_row,
            col=1,
            line_dash="dot",
            line_color=THEME["fifty"],
            line_width=1.2,
        )

    fig.update_layout(
        **BASE_LAYOUT,
        legend=dict(
            orientation="h",
            y=1.045,
            x=0.5,
            xanchor="center",
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=TICK_FONT),
            itemsizing="constant",
        ),
        margin=dict(l=10, r=20, t=100, b=10),
        height=2 * (46 * len(names)) + 260,
        hovermode="closest",
    )
    fig.update_xaxes(
        range=[0, 105],
        ticksuffix="%",
        dtick=25,
        tickfont=dict(size=TICK_FONT, color=THEME["muted"]),
        gridcolor=THEME["grid"],
        zerolinecolor=THEME["grid"],
        showline=False,
    )
    # Shared x-axis hides the top panel's ticks by default; show both.
    fig.update_xaxes(showticklabels=True, row=1, col=1)
    fig.update_yaxes(
        tickfont=dict(size=FONT, color=THEME["text"]),
        showgrid=False,
        categoryorder="array",
        categoryarray=names,
        automargin=True,
    )
    for ann in fig.layout.annotations:
        ann.font = dict(size=FONT, color=THEME["text"])
        ann.x = 0.5
    return fig


def write_vector(fig: go.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("pdf", "svg"):
        path = out_dir / f"{stem}.{ext}"
        fig.write_image(str(path))
        print(f"  saved → {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-dir",
        default=str(Path(__file__).parent / "out"),
        help="Directory containing season_s0.csv / season_s1.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).parent.parent / "paper-figures"),
        help="Output directory (default: paper-figures/)",
    )
    parser.add_argument("--min-role-games", type=int, default=1)
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s0 = load_local_season(csv_dir, 0)
    s1 = load_local_season(csv_dir, 1)
    rows = build_rows(s0, s1, args.min_role_games, EXEMPLAR_MODEL_IDS)

    # ── OpenSkill ratings: one figure per context, shared order + x-range ──
    rating_rows = sort_by_s1_peak(rows, "imp_rating", "crew_rating")
    names = [r["name"] for r in rating_rows]
    max_mu = max(
        snap[k]
        for r in rating_rows
        for snap in (r["s0"], r["s1"])
        if snap is not None
        for k in ("imp_rating", "crew_rating", "overall_rating")
    )
    x_range = [0, max_mu * 1.08]

    print("OpenSkill ratings — short context (S0):")
    write_vector(
        build_rating_panel(rating_rows, "s0", names, x_range),
        out_dir,
        "openskill_ratings_short_context",
    )
    print("OpenSkill ratings — long context (S1):")
    write_vector(
        build_rating_panel(rating_rows, "s1", names, x_range),
        out_dir,
        "openskill_ratings_long_context",
    )

    # ── Win rates: exemplar cohort, short vs long stacked ──
    wr_rows = sort_by_s1_peak(rows, "imp_wr", "crew_wr")
    wr_names = [r["name"] for r in wr_rows]
    print("Win rates — exemplars, short vs long:")
    write_vector(
        build_winrate_stacked(wr_rows, wr_names),
        out_dir,
        "win_rates_exemplars_short_vs_long",
    )


if __name__ == "__main__":
    main()
