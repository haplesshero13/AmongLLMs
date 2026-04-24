#!/usr/bin/env python3
"""Season 0 → Season 1 visualizations.

Produces companion charts from live API data, ordered top-to-bottom by the
Season 1 peak of the plotted metric:

1. `season_comparison_{subset}_{theme}.{html,png}` — role win rates. Marker
   size encodes sample certainty (∝ √role games); hover surfaces the 95%
   Wilson CI.
2. `season_ratings_{subset}_{theme}.{html,png}` — OpenSkill role ratings as
   horizontal bar + whisker (bar = μ − σ conservative, whisker = σ).

Two model subsets are emitted by default (override with `--subset`):
  - `featured` = 20-model paper set (full leaderboard view)
  - `presentation` = 11-model 16:9 slide set (7 long-context human-AI
    participants + Human Brain 1.0 + 3 clean references).

Dark theme targets presentation use; `--theme light` produces the paper variant.

Usage:
    uv run reporting/season_chart.py                               # both subsets, dark
    uv run reporting/season_chart.py --theme light                 # both subsets, light
    uv run reporting/season_chart.py --subset presentation         # just the slide set
    uv run reporting/season_chart.py --api-base http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import urllib.request
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_API_BASE = "https://api.sdgarena.averyyen.dev"

# Curated 20 featured models — for paper-form tables and the full leaderboard
# view. Also covers every long-context human-AI participant.
FEATURED_MODEL_IDS = [
    "gpt-oss-120b",
    "trinity-large",
    "minimax-m2.5",
    "kimi-k2.5",
    "qwen3.5-plus-02-15",
    "nemotron-3-super",
    "claude-haiku-4.5",
    "llama-3.3-70b",
    "deepseek-v3.2",
    "step-3.5-flash",
    "grok-4.1-fast",
    "mimo-v2-flash",
    "glm-5",
    "claude-sonnet-4.5",
    "llama-4-maverick",
    "gemini-3.1-pro",
    "gpt-5.4",
    "gemini-3-flash",
    "claude-opus-4.6",
    "brain-1.0",
]

# Presentation-grade 10-model subset for 16:9 slides. The 7 models that faced
# humans in long-context trials, plus Human Brain 1.0 and three clean reference
# points: Gemini 3 Flash (top crewmate, high-n), Claude Sonnet 4.5 (most
# balanced, high-n), Grok 4.1 Fast (clear S0→S1 narrative, high-n).
PRESENTATION_MODEL_IDS = [
    # Human-AI long-context participants
    "claude-opus-4.6",
    "gpt-5.4",
    "gemini-3.1-pro",
    "llama-3.3-70b",
    "nemotron-3-super",
    "kimi-k2.5",
    "deepseek-v3.2",
    "brain-1.0",
    # Clean reference points
    "gemini-3-flash",
    "claude-sonnet-4.5",
    "grok-4.1-fast",
]

MODEL_SUBSETS = {
    "featured": FEATURED_MODEL_IDS,
    "presentation": PRESENTATION_MODEL_IDS,
}

HUMAN_MODEL_IDS = {"brain-1.0"}
HUMAN_LABEL_PREFIX = "\U0001f9e0 "  # brain emoji prepended to human row labels


# ── themes ────────────────────────────────────────────────────────────────────
THEMES = {
    "dark": {
        "bg": "#0d1117",
        "panel": "#0d1117",
        "text": "#e6edf3",
        "muted": "#8b949e",
        "grid": "#21262d",
        "fifty": "#30363d",
        "divider": "#30363d",
        "imp_s1": "#f85149",
        "crew_s1": "#58a6ff",
        "imp_s0": "#da6e68",
        "crew_s0": "#5b9bd5",
        "marker_border": "#0d1117",
    },
    "light": {
        "bg": "#ffffff",
        "panel": "#ffffff",
        "text": "#1f2328",
        "muted": "#656d76",
        "grid": "#eaecef",
        "fifty": "#afb8c1",
        "divider": "#d0d7de",
        "imp_s1": "#d1242f",
        "crew_s1": "#0969da",
        "imp_s0": "#e8989c",
        "crew_s0": "#8cb4ea",
        "marker_border": "#ffffff",
    },
}


# ── API helpers ───────────────────────────────────────────────────────────────
def fetch_json(url: str) -> object:
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def fetch_season(base_url: str, version: int) -> dict[str, dict]:
    # API caps per_page at 100; paginate if the leaderboard grows past that.
    merged: list[dict] = []
    page = 1
    while True:
        payload = fetch_json(
            f"{base_url}/api/leaderboard?page={page}&per_page=100&engine_version={version}"
        )
        data = payload["data"] if isinstance(payload, dict) else payload  # type: ignore[index]
        if not data:
            break
        merged.extend(data)
        if len(data) < 100:
            break
        page += 1
    return {m["model_id"]: m for m in merged}


def whole_board_stats(season: dict[str, dict]) -> dict:
    """Pooled role win rates across every model in the given season.

    Pooled (not per-model-averaged) — these numbers reflect the actual game-level
    balance, not an average-of-averages that depends on who happens to be in the
    subset we're plotting.
    """
    imp_w = imp_g = crew_w = crew_g = 0
    n_models = 0
    n_games_sum = 0
    for m in season.values():
        ig = int(m.get("impostor_games") or 0)
        cg = int(m.get("crewmate_games") or 0)
        if ig < 1 or cg < 1:
            continue
        n_models += 1
        imp_w += int(m.get("impostor_wins") or 0)
        imp_g += ig
        crew_w += int(m.get("crewmate_wins") or 0)
        crew_g += cg
        n_games_sum += int(m.get("games_played") or 0)

    imp_pct = (imp_w / imp_g * 100) if imp_g else float("nan")
    crew_pct = (crew_w / crew_g * 100) if crew_g else float("nan")
    # Weight by the 2-imp + 5-crew slot composition of a 7-player game.
    overall_pct = (2 * imp_pct + 5 * crew_pct) / 7 if imp_g and crew_g else float("nan")
    # Per-model games_played double-counts actual games because each game contributes
    # up to 7 player-observations. Approximate total distinct games = total slots / 7.
    distinct_games_est = (imp_g + crew_g) // 7 if (imp_g + crew_g) else 0
    return {
        "imp_pct": imp_pct,
        "crew_pct": crew_pct,
        "overall_pct": overall_pct,
        "n_models": n_models,
        "n_games": distinct_games_est,
    }


def _fade(hex_color: str, alpha: float) -> str:
    """Convert a #rrggbb hex to an rgba(r,g,b,alpha) string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson-score CI for a proportion p in [0, 1] with sample size n.

    Robust near 0 and 1 where the Wald interval breaks down. Returns (low, high)
    in the same [0, 1] space as `p`.
    """
    if n <= 0:
        return (p, p)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _season_snapshot(
    model: dict | None, min_role_games: int
) -> dict | None:
    """Pull everything we need for a season (win rates, sample sizes, ratings,
    sigmas), gated on the model having ≥min_role_games on BOTH sides."""
    if model is None:
        return None
    imp_games = model.get("impostor_games") or 0
    crew_games = model.get("crewmate_games") or 0
    if imp_games < min_role_games or crew_games < min_role_games:
        return None
    return {
        "imp_wr": float(model["impostor_win_rate"]),
        "crew_wr": float(model["crewmate_win_rate"]),
        "imp_games": imp_games,
        "crew_games": crew_games,
        "imp_rating": float(model["impostor_rating"]),
        "crew_rating": float(model["crewmate_rating"]),
        "imp_sigma": float(model["impostor_sigma"]),
        "crew_sigma": float(model["crewmate_sigma"]),
    }


def build_rows(
    s0: dict[str, dict],
    s1: dict[str, dict],
    min_role_games: int,
    model_ids: list[str],
) -> list[dict]:
    """One row per model in `model_ids`, with null-safe S0/S1 snapshots."""
    rows = []
    for mid in model_ids:
        m1 = s1.get(mid)
        m0 = s0.get(mid)
        if m1 is None:
            # No S1 data — skip entirely; inclusion sets assume S1 presence.
            print(f"  [warn] {mid} missing from Season 1 — skipping")
            continue
        is_human = mid in HUMAN_MODEL_IDS
        label = (HUMAN_LABEL_PREFIX if is_human else "") + m1["model_name"]
        rows.append(
            {
                "model_id": mid,
                "name": label,
                "is_human": is_human,
                "s0": _season_snapshot(m0, min_role_games),
                "s1": _season_snapshot(m1, min_role_games),
            }
        )
    return rows


def sort_by_s1_peak(rows: list[dict], imp_key: str, crew_key: str) -> list[dict]:
    """Sort ascending by S1 peak of the two named keys (draw bottom→top)."""

    def peak(r: dict) -> float:
        s1 = r["s1"]
        if s1 is None:
            return -math.inf
        return max(s1[imp_key], s1[crew_key])

    return sorted(rows, key=peak)


# ── chart: win-rate ───────────────────────────────────────────────────────────
def build_winrate_figure(
    rows: list[dict],
    theme_name: str,
    aspect: str = "auto",
    board_s0: dict | None = None,
    board_s1: dict | None = None,
) -> go.Figure:
    """Win-rate dot plot; `aspect='16x9'` forces a 1280×720 presentation frame.

    `board_s0` / `board_s1` are whole-board pooled stats (from
    `whole_board_stats`) used for the panel captions. Displaying subset-average
    numbers would be misleading when the subset cherry-picks strong models, so
    we always caption with the board-wide figure instead.
    """
    theme = THEMES[theme_name]
    names = [r["name"] for r in rows]

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.12,
        subplot_titles=[
            "Season 0 — Single-Turn Summarized Context",
            "Season 1 — Multi-Turn Long Context",
        ],
    )

    panels = [
        (1, "s0", theme["imp_s0"], theme["crew_s0"]),
        (2, "s1", theme["imp_s1"], theme["crew_s1"]),
    ]

    # Marker-size scaling: sqrt(n) so marker AREA scales with precision (1/Var).
    # We fix the range so "big dot" and "small dot" are interpretable across panels.
    MIN_SIZE, MAX_SIZE = 7.0, 22.0
    all_ns = [
        snap[k]
        for r in rows
        for snap in (r["s0"], r["s1"])
        if snap is not None
        for k in ("imp_games", "crew_games")
    ]
    n_lo = min(all_ns) if all_ns else 1
    n_hi = max(all_ns) if all_ns else 1

    def _size_for(n: int) -> float:
        if n_hi == n_lo:
            return (MIN_SIZE + MAX_SIZE) / 2
        t = (math.sqrt(n) - math.sqrt(n_lo)) / (math.sqrt(n_hi) - math.sqrt(n_lo))
        return MIN_SIZE + t * (MAX_SIZE - MIN_SIZE)

    for panel, snap_key, imp_c, crew_c in panels:
        imp_x, imp_y, imp_sizes, imp_hover = [], [], [], []
        crew_x, crew_y, crew_sizes, crew_hover = [], [], [], []

        for r in rows:
            snap = r[snap_key]
            name = r["name"]
            if snap is None:
                continue

            iv = snap["imp_wr"]
            cv = snap["crew_wr"]
            ni = snap["imp_games"]
            nc = snap["crew_games"]

            imp_lo, imp_hi = _wilson_ci(iv / 100, ni)
            crew_lo, crew_hi = _wilson_ci(cv / 100, nc)

            imp_x.append(iv)
            imp_y.append(name)
            imp_sizes.append(_size_for(ni))
            imp_hover.append(
                f"<b>{name}</b><br>Impostor: {iv:.1f}% "
                f"({int(round(ni * iv / 100))}/{ni} games)"
                f"<br>95% Wilson CI: [{imp_lo * 100:.1f}%, {imp_hi * 100:.1f}%]"
            )
            crew_x.append(cv)
            crew_y.append(name)
            crew_sizes.append(_size_for(nc))
            crew_hover.append(
                f"<b>{name}</b><br>Crewmate: {cv:.1f}% "
                f"({int(round(nc * cv / 100))}/{nc} games)"
                f"<br>95% Wilson CI: [{crew_lo * 100:.1f}%, {crew_hi * 100:.1f}%]"
            )

            fig.add_trace(
                go.Scatter(
                    x=[iv, cv],
                    y=[name, name],
                    mode="lines",
                    line=dict(color=imp_c if iv > cv else crew_c, width=4),
                    opacity=0.55,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=panel,
            )

        fig.add_trace(
            go.Scatter(
                x=imp_x,
                y=imp_y,
                mode="markers",
                name="Impostor",
                marker=dict(
                    color=imp_c,
                    size=imp_sizes,
                    symbol="triangle-up",
                    line=dict(color=theme["marker_border"], width=1.5),
                ),
                legendgroup="imp",
                showlegend=(panel == 2),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=imp_hover,
            ),
            row=1,
            col=panel,
        )

        fig.add_trace(
            go.Scatter(
                x=crew_x,
                y=crew_y,
                mode="markers",
                name="Crewmate",
                marker=dict(
                    color=crew_c,
                    size=crew_sizes,
                    symbol="circle",
                    line=dict(color=theme["marker_border"], width=1.5),
                ),
                legendgroup="crew",
                showlegend=(panel == 2),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=crew_hover,
            ),
            row=1,
            col=panel,
        )

        fig.add_vline(
            x=50,
            row=1,
            col=panel,
            line_dash="dot",
            line_color=theme["fifty"],
            line_width=1.2,
        )

    # Inter-panel vertical divider.
    fig.add_shape(
        type="line",
        x0=0.5,
        x1=0.5,
        y0=0,
        y1=1,
        xref="paper",
        yref="paper",
        line=dict(color=theme["divider"], width=2),
    )

    fig.update_layout(
        title=dict(
            text="Summarized vs Long Context — Among Us Role Win Rates",
            font=dict(size=24, color=theme["text"]),
            x=0.0,
            xanchor="left",
            pad=dict(l=12),
        ),
        paper_bgcolor=theme["bg"],
        plot_bgcolor=theme["panel"],
        font=dict(
            color=theme["text"],
            family="'Inter', 'Helvetica Neue', Arial, sans-serif",
        ),
        legend=dict(
            orientation="h",
            y=-0.08,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=13, color=theme["text"]),
            itemsizing="constant",
        ),
        margin=dict(l=150, r=40, t=100, b=90),
        height=720 if aspect == "16x9" else max(620, 36 * len(names) + 200),
        width=1280,
        hovermode="closest",
    )

    for ax in ["xaxis", "xaxis2"]:
        fig.update_layout(
            **{
                ax: dict(
                    range=[0, 105],
                    ticksuffix="%",
                    tickfont=dict(size=11, color=theme["muted"]),
                    gridcolor=theme["grid"],
                    zerolinecolor=theme["grid"],
                    showline=False,
                    dtick=25,
                )
            }
        )

    # Enforce explicit y-axis order matching our sort. Plotly otherwise orders
    # categories by trace-insertion, which bumps S1-only models to the bottom.
    fig.update_yaxes(
        tickfont=dict(size=13, color=theme["text"]),
        showgrid=False,
        categoryorder="array",
        categoryarray=names,
    )

    for ann in fig.layout.annotations:
        if ann.text and ("Season" in ann.text or "Context" in ann.text):
            ann.font = dict(size=15, color=theme["text"])

    # Size-encoding legend annotation — top-right above the plot area.
    fig.add_annotation(
        x=1.0,
        y=1.06,
        xref="paper",
        yref="paper",
        text=(
            f"<span style='color:{theme['muted']}'>"
            f"marker size = confidence (∝ √role games, larger = more certain)"
            f"</span>"
        ),
        showarrow=False,
        font=dict(size=11),
        xanchor="right",
    )

    # Board-wide pooled captions under each panel. These always reflect the
    # *full* season leaderboard, not just the subset we're plotting — the
    # subset can cherry-pick, the board-wide number can't.
    def _board_caption(stats: dict | None, imp_col: str, crew_col: str) -> str:
        if stats is None:
            return ""
        return (
            f"<span style='color:{theme['muted']}'>"
            f"Whole-board average (n={stats['n_models']} models, "
            f"{stats['n_games']} games):</span>  "
            f"<span style='color:{theme[imp_col]}'>Imp {stats['imp_pct']:.1f}%</span>"
            f"<span style='color:{theme['muted']}'>  ·  </span>"
            f"<span style='color:{theme[crew_col]}'>Crew {stats['crew_pct']:.1f}%</span>"
            f"<span style='color:{theme['muted']}'>  ·  "
            f"Overall {stats['overall_pct']:.1f}%</span>"
        )

    fig.add_annotation(
        x=0.22,
        y=-0.15,
        xref="paper",
        yref="paper",
        text=_board_caption(board_s0, "imp_s0", "crew_s0"),
        showarrow=False,
        font=dict(size=12),
        xanchor="center",
    )
    fig.add_annotation(
        x=0.78,
        y=-0.15,
        xref="paper",
        yref="paper",
        text=_board_caption(board_s1, "imp_s1", "crew_s1"),
        showarrow=False,
        font=dict(size=12),
        xanchor="center",
    )

    return fig


# ── chart: rating ─────────────────────────────────────────────────────────────
def build_rating_figure(
    rows: list[dict],
    theme_name: str,
    aspect: str = "auto",
) -> go.Figure:
    """OpenSkill role-rating chart using the same visual language as the live
    leaderboard: horizontal bars to the conservative rating (μ − σ), with a
    one-sided whisker extending from the bar end up to μ.
    """
    theme = THEMES[theme_name]
    names = [r["name"] for r in rows]

    # X-axis range: 0 on the left; right edge from max μ with padding.
    max_mu = 0.0
    for r in rows:
        for snap in (r["s0"], r["s1"]):
            if snap is None:
                continue
            max_mu = max(max_mu, snap["imp_rating"], snap["crew_rating"])
    x_range = [0, max_mu * 1.08]

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.12,
        subplot_titles=[
            "Season 0 — Single-Turn Summarized Context",
            "Season 1 — Multi-Turn Long Context",
        ],
    )

    panels = [
        (1, "s0", theme["imp_s0"], theme["crew_s0"]),
        (2, "s1", theme["imp_s1"], theme["crew_s1"]),
    ]

    for panel, snap_key, imp_c, crew_c in panels:
        imp_x, imp_y, imp_sigma, imp_hover = [], [], [], []
        crew_x, crew_y, crew_sigma, crew_hover = [], [], [], []

        for r in rows:
            snap = r[snap_key]
            name = r["name"]
            if snap is None:
                continue

            ir = snap["imp_rating"]
            cr = snap["crew_rating"]
            is_ = snap["imp_sigma"]
            cs = snap["crew_sigma"]

            imp_x.append(ir - is_)  # bar ends at conservative rating
            imp_y.append(name)
            imp_sigma.append(is_)
            imp_hover.append(
                f"<b>{name}</b><br>Impostor μ: {ir:,.0f}  σ: {is_:,.0f}"
                f"<br>Conservative (μ − σ): {ir - is_:,.0f}"
            )
            crew_x.append(cr - cs)
            crew_y.append(name)
            crew_sigma.append(cs)
            crew_hover.append(
                f"<b>{name}</b><br>Crewmate μ: {cr:,.0f}  σ: {cs:,.0f}"
                f"<br>Conservative (μ − σ): {cr - cs:,.0f}"
            )

        fig.add_trace(
            go.Bar(
                x=imp_x,
                y=imp_y,
                orientation="h",
                name="Impostor",
                marker=dict(
                    color=imp_c,
                    line=dict(color=_fade(imp_c, 0.9), width=1.25),
                ),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=imp_sigma,
                    arrayminus=[0] * len(imp_sigma),
                    color=_fade(imp_c, 0.85),
                    thickness=1.4,
                    width=5,
                ),
                legendgroup="imp",
                showlegend=(panel == 2),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=imp_hover,
            ),
            row=1,
            col=panel,
        )

        fig.add_trace(
            go.Bar(
                x=crew_x,
                y=crew_y,
                orientation="h",
                name="Crewmate",
                marker=dict(
                    color=crew_c,
                    line=dict(color=_fade(crew_c, 0.9), width=1.25),
                ),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=crew_sigma,
                    arrayminus=[0] * len(crew_sigma),
                    color=_fade(crew_c, 0.85),
                    thickness=1.4,
                    width=5,
                ),
                legendgroup="crew",
                showlegend=(panel == 2),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=crew_hover,
            ),
            row=1,
            col=panel,
        )

    # Inter-panel vertical divider.
    fig.add_shape(
        type="line",
        x0=0.5,
        x1=0.5,
        y0=0,
        y1=1,
        xref="paper",
        yref="paper",
        line=dict(color=theme["divider"], width=2),
    )

    fig.update_layout(
        barmode="group",
        bargap=0.28,
        bargroupgap=0.08,
        title=dict(
            text="Summarized vs Long Context — OpenSkill Role Ratings (bar = μ − σ, whisker = σ)",
            font=dict(size=22, color=theme["text"]),
            x=0.0,
            xanchor="left",
            pad=dict(l=12),
        ),
        paper_bgcolor=theme["bg"],
        plot_bgcolor=theme["panel"],
        font=dict(
            color=theme["text"],
            family="'Inter', 'Helvetica Neue', Arial, sans-serif",
        ),
        legend=dict(
            orientation="h",
            y=-0.06,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=13, color=theme["text"]),
            itemsizing="constant",
        ),
        margin=dict(l=150, r=40, t=100, b=60),
        height=720 if aspect == "16x9" else max(680, 42 * len(names) + 200),
        width=1280,
        hovermode="closest",
    )

    for ax in ["xaxis", "xaxis2"]:
        fig.update_layout(
            **{
                ax: dict(
                    range=x_range,
                    tickfont=dict(size=11, color=theme["muted"]),
                    gridcolor=theme["grid"],
                    zerolinecolor=theme["grid"],
                    showline=False,
                    tickformat=",d",
                )
            }
        )

    fig.update_yaxes(
        tickfont=dict(size=13, color=theme["text"]),
        showgrid=False,
        categoryorder="array",
        categoryarray=names,
    )

    for ann in fig.layout.annotations:
        if ann.text and ("Season" in ann.text or "Context" in ann.text):
            ann.font = dict(size=15, color=theme["text"])

    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-base",
        default=os.environ.get("ANALYSIS_API_BASE_URL", DEFAULT_API_BASE),
    )
    parser.add_argument(
        "--theme",
        choices=["dark", "light"],
        default="dark",
        help="dark = presentation, light = paper",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).parent),
        help="Output directory for html/png (default: reporting/)",
    )
    parser.add_argument(
        "--min-role-games",
        type=int,
        default=1,
        help=(
            "Minimum per-role game count a model must have in a season "
            "(BOTH sides) to appear in that season's panel (default: 1). "
            "Guards against tiny-sample artifacts like 0%%-from-0-games."
        ),
    )
    parser.add_argument(
        "--subset",
        choices=["both", "featured", "presentation"],
        default="both",
        help=(
            "Which model inclusion list to render. `featured` = full 20-model "
            "paper set; `presentation` = 11-model 16:9 slide subset (7 "
            "human-AI participants + Human Brain 1.0 + 3 clean references). "
            "Default `both` emits both."
        ),
    )
    args = parser.parse_args()

    print(f"Fetching seasons from {args.api_base} ...")
    s0 = fetch_season(args.api_base, 0)
    s1 = fetch_season(args.api_base, 1)
    print(f"  Season 0: {len(s0)} models  |  Season 1: {len(s1)} models")

    # Whole-board pooled stats for captions — computed once from unfiltered
    # seasons so the number is independent of which subset we're plotting.
    board_s0 = whole_board_stats(s0)
    board_s1 = whole_board_stats(s1)
    print(
        f"  Board-wide S0: imp {board_s0['imp_pct']:.1f}%  crew {board_s0['crew_pct']:.1f}%  "
        f"overall {board_s0['overall_pct']:.1f}%  (n={board_s0['n_models']} models, "
        f"{board_s0['n_games']} games)"
    )
    print(
        f"  Board-wide S1: imp {board_s1['imp_pct']:.1f}%  crew {board_s1['crew_pct']:.1f}%  "
        f"overall {board_s1['overall_pct']:.1f}%  (n={board_s1['n_models']} models, "
        f"{board_s1['n_games']} games)"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subsets = (
        ["featured", "presentation"] if args.subset == "both" else [args.subset]
    )

    for subset in subsets:
        model_ids = MODEL_SUBSETS[subset]
        # 16:9 layout only makes sense for the compact presentation set.
        aspect = "16x9" if subset == "presentation" else "auto"
        suffix = f"_{subset}_{args.theme}"
        rows = build_rows(s0, s1, args.min_role_games, model_ids)

        # Win-rate chart — sort by S1 peak role win rate.
        wr_rows = sort_by_s1_peak(rows, "imp_wr", "crew_wr")
        print(
            f"  [{subset}/win rate] {len(wr_rows)} models "
            f"(sorted by S1 peak WR, aspect={aspect})"
        )
        fig_wr = build_winrate_figure(
            wr_rows, args.theme, aspect=aspect,
            board_s0=board_s0, board_s1=board_s1,
        )
        wr_html = out_dir / f"season_comparison{suffix}.html"
        wr_png = out_dir / f"season_comparison{suffix}.png"
        fig_wr.write_html(str(wr_html), include_plotlyjs="cdn")
        print(f"    saved → {wr_html}")
        try:
            fig_wr.write_image(str(wr_png), scale=2)
            print(f"    saved → {wr_png}")
        except Exception as exc:
            print(f"    (PNG export skipped — {exc})")

        # Rating chart — sort by S1 peak role rating.
        rating_rows = sort_by_s1_peak(rows, "imp_rating", "crew_rating")
        print(
            f"  [{subset}/rating]   {len(rating_rows)} models "
            f"(sorted by S1 peak rating, aspect={aspect})"
        )
        fig_rt = build_rating_figure(rating_rows, args.theme, aspect=aspect)
        rt_html = out_dir / f"season_ratings{suffix}.html"
        rt_png = out_dir / f"season_ratings{suffix}.png"
        fig_rt.write_html(str(rt_html), include_plotlyjs="cdn")
        print(f"    saved → {rt_html}")
        try:
            fig_rt.write_image(str(rt_png), scale=2)
            print(f"    saved → {rt_png}")
        except Exception as exc:
            print(f"    (PNG export skipped — {exc})")


if __name__ == "__main__":
    main()
