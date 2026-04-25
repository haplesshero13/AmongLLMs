"""Season-level win-rate statistics for the Among Us leaderboard.

Pulls both seasons from the live API, restricts to a featured model set, and
computes the quantitative findings that back the paper:

  1. Per-season leaderboard (with Wilson 95% CIs on role win rates)
  2. Human Brain 1.0 vs chance (binomial test, both roles)
  3. Paired S0→S1 analysis across models present in both seasons:
       - mean role-WR delta with 95% CI (paired t)
       - sign test (direction-of-shift)
       - per-model Wilson CI overlap check (is any individual shift disjoint?)

Emits to three channels:
  - rich tables on the console
  - CSVs under --out-dir  (season_s0.csv, season_s1.csv, paired.csv, ci_overlap.csv)
  - an optional markdown summary (--markdown)

Usage:
  uv run --with polars --with rich \\
      python reporting/season_win_rates.py --out-dir reporting/out
  uv run --with polars --with rich \\
      python reporting/season_win_rates.py --markdown reporting/season_win_rates.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import urllib.request
from math import comb
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

DEFAULT_API_BASE = "https://api.sdgarena.averyyen.dev"

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

EXEMPLAR_MODEL_IDS = [
    "claude-opus-4.6",
    "gpt-5.4",
    "gemini-3.1-pro",
    "llama-3.3-70b",
    "nemotron-3-super",
    "kimi-k2.5",
    "deepseek-v3.2",
]

MODEL_SUBSETS: dict[str, list[str] | None] = {
    "all": None,
    "exemplars": EXEMPLAR_MODEL_IDS,
    "featured": FEATURED_MODEL_IDS,
}

HUMAN_MODEL_ID = "brain-1.0"
SHORT_CONTEXT_LABEL = "Short-context"
LONG_CONTEXT_LABEL = "Long-context"

# Critical-t at α=0.05 two-sided for small df. Avoids pulling scipy for one call.
_T_CRIT = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 29: 2.045, 30: 2.042,
}


# ── stats helpers ─────────────────────────────────────────────────────────────
def wilson_ci(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson-score CI for a proportion, in [0, 1]."""
    if games <= 0:
        return (0.0, 1.0)
    p = wins / games
    denom = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denom
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def binom_two_sided(k: int, n: int, p0: float = 0.5) -> float:
    """Exact two-sided binomial p-value under H0: p = p0."""
    if n <= 0:
        return 1.0
    obs = comb(n, k) * p0**k * (1 - p0) ** (n - k)
    total = 0.0
    for i in range(n + 1):
        pi = comb(n, i) * p0**i * (1 - p0) ** (n - i)
        if pi <= obs + 1e-12:
            total += pi
    return min(1.0, total)


def paired_t_ci(xs: list[float], alpha: float = 0.05) -> dict:
    """Mean, paired-t 95% CI, and t-statistic."""
    n = len(xs)
    if n < 2:
        return {"n": n, "mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "t": float("nan")}
    m = st.mean(xs)
    sd = st.stdev(xs)
    se = sd / math.sqrt(n) if n else float("nan")
    df = n - 1
    t = _T_CRIT.get(df) or _T_CRIT[min(_T_CRIT, key=lambda k: abs(k - df))]
    return {
        "n": n,
        "mean": m,
        "lo": m - t * se,
        "hi": m + t * se,
        "t": m / se if se > 0 else float("inf"),
    }


def sign_test_two_sided(k: int, n: int) -> float:
    """Two-sided sign test against p=0.5."""
    if n <= 0:
        return 1.0
    lo = min(k, n - k)
    hi = max(k, n - k)
    p = 0.0
    for i in range(0, lo + 1):
        p += comb(n, i) * 0.5**n
    for i in range(hi, n + 1):
        p += comb(n, i) * 0.5**n
    return min(1.0, p)


# ── data layer ────────────────────────────────────────────────────────────────
def fetch_leaderboard(base_url: str, version: int) -> pl.DataFrame:
    """Fetch a full season as a polars DataFrame (paginated past 100 rows)."""
    rows: list[dict] = []
    page = 1
    while True:
        url = (
            f"{base_url}/api/leaderboard"
            f"?page={page}&per_page=100&engine_version={version}"
        )
        with urllib.request.urlopen(url) as resp:
            payload = json.load(resp)
        data = payload["data"] if isinstance(payload, dict) else payload
        if not data:
            break
        rows.extend(data)
        if len(data) < 100:
            break
        page += 1
    return pl.DataFrame(rows)


def _wilson_pair(wins: int | None, games: int | None) -> tuple[float, float]:
    lo, hi = wilson_ci(int(wins or 0), int(games or 0))
    return (lo * 100, hi * 100)


def enrich(df: pl.DataFrame, season_label: str) -> pl.DataFrame:
    """Add Wilson CI + conservative rating columns and a season tag."""
    base = df.with_columns(pl.lit(season_label).alias("season"))

    if {"impostor_wins", "impostor_games"}.issubset(df.columns):
        base = base.with_columns(
            pl.struct(["impostor_wins", "impostor_games"])
            .map_elements(
                lambda s: _wilson_pair(s["impostor_wins"], s["impostor_games"])[0],
                return_dtype=pl.Float64,
            )
            .alias("imp_wr_lo"),
            pl.struct(["impostor_wins", "impostor_games"])
            .map_elements(
                lambda s: _wilson_pair(s["impostor_wins"], s["impostor_games"])[1],
                return_dtype=pl.Float64,
            )
            .alias("imp_wr_hi"),
        )
    if {"crewmate_wins", "crewmate_games"}.issubset(df.columns):
        base = base.with_columns(
            pl.struct(["crewmate_wins", "crewmate_games"])
            .map_elements(
                lambda s: _wilson_pair(s["crewmate_wins"], s["crewmate_games"])[0],
                return_dtype=pl.Float64,
            )
            .alias("crew_wr_lo"),
            pl.struct(["crewmate_wins", "crewmate_games"])
            .map_elements(
                lambda s: _wilson_pair(s["crewmate_wins"], s["crewmate_games"])[1],
                return_dtype=pl.Float64,
            )
            .alias("crew_wr_hi"),
        )

    if {"impostor_rating", "impostor_sigma"}.issubset(df.columns):
        base = base.with_columns(
            (pl.col("impostor_rating") - pl.col("impostor_sigma")).alias("imp_conservative")
        )
    if {"crewmate_rating", "crewmate_sigma"}.issubset(df.columns):
        base = base.with_columns(
            (pl.col("crewmate_rating") - pl.col("crewmate_sigma")).alias("crew_conservative")
        )
    if {"overall_rating", "overall_sigma"}.issubset(df.columns):
        base = base.with_columns(
            (pl.col("overall_rating") - pl.col("overall_sigma")).alias("overall_conservative")
        )

    return base


def filter_subset(
    df: pl.DataFrame,
    min_role_games: int,
    min_season_games: int,
    subset: str,
) -> pl.DataFrame:
    model_ids = MODEL_SUBSETS[subset]
    filtered = df
    if model_ids is not None:
        filtered = filtered.filter(pl.col("model_id").is_in(model_ids))
    return (
        filtered.filter(pl.col("games_played") >= min_season_games)
        .filter(pl.col("impostor_games") >= min_role_games)
        .filter(pl.col("crewmate_games") >= min_role_games)
    )


def condition_label(tag: str) -> str:
    return {
        "S0": SHORT_CONTEXT_LABEL,
        "S1": LONG_CONTEXT_LABEL,
    }.get(tag, tag)


# ── analyses ──────────────────────────────────────────────────────────────────
def human_brain_analysis(s1: pl.DataFrame) -> dict:
    row = s1.filter(pl.col("model_id") == HUMAN_MODEL_ID)
    if row.is_empty():
        return {}
    r = row.row(0, named=True)
    ni, nc = int(r["impostor_games"]), int(r["crewmate_games"])
    iw, cw = int(r["impostor_wins"]), int(r["crewmate_wins"])
    i_ci = wilson_ci(iw, ni)
    c_ci = wilson_ci(cw, nc)
    return {
        "name": r["model_name"],
        "imp_wins": iw,
        "imp_games": ni,
        "imp_rate": (iw / ni * 100) if ni else 0.0,
        "imp_ci": (i_ci[0] * 100, i_ci[1] * 100),
        "imp_p": binom_two_sided(iw, ni),
        "crew_wins": cw,
        "crew_games": nc,
        "crew_rate": (cw / nc * 100) if nc else 0.0,
        "crew_ci": (c_ci[0] * 100, c_ci[1] * 100),
        "crew_p": binom_two_sided(cw, nc),
        "total_wins": iw + cw,
        "total_games": ni + nc,
    }


def paired_frame(s0: pl.DataFrame, s1: pl.DataFrame) -> pl.DataFrame:
    """Inner-join paired models and compute per-model long−short deltas."""
    keep_cols = [
        "model_id",
        "model_name",
        "impostor_win_rate",
        "crewmate_win_rate",
        "impostor_games",
        "crewmate_games",
        "impostor_wins",
        "crewmate_wins",
    ]
    a = s0.select(keep_cols).rename({c: f"s0_{c}" for c in keep_cols if c != "model_id"})
    b = s1.select(keep_cols).rename({c: f"s1_{c}" for c in keep_cols if c != "model_id"})
    joined = a.join(b, on="model_id", how="inner").sort("s1_model_name")

    def overall(imp_col: str, crew_col: str) -> pl.Expr:
        # Weighted overall = 2 impostors + 5 crewmates per 7-player game.
        return (2 * pl.col(imp_col) + 5 * pl.col(crew_col)) / 7

    return joined.with_columns(
        (pl.col("s1_impostor_win_rate") - pl.col("s0_impostor_win_rate")).alias("imp_delta"),
        (pl.col("s1_crewmate_win_rate") - pl.col("s0_crewmate_win_rate")).alias("crew_delta"),
        (
            overall("s1_impostor_win_rate", "s1_crewmate_win_rate")
            - overall("s0_impostor_win_rate", "s0_crewmate_win_rate")
        ).alias("overall_delta"),
    )


def ci_overlap_frame(paired: pl.DataFrame) -> pl.DataFrame:
    """Per-model Wilson CI disjoint check between S0 and S1, per role."""
    rows = []
    for r in paired.iter_rows(named=True):
        i0 = wilson_ci(int(r["s0_impostor_wins"]), int(r["s0_impostor_games"]))
        i1 = wilson_ci(int(r["s1_impostor_wins"]), int(r["s1_impostor_games"]))
        c0 = wilson_ci(int(r["s0_crewmate_wins"]), int(r["s0_crewmate_games"]))
        c1 = wilson_ci(int(r["s1_crewmate_wins"]), int(r["s1_crewmate_games"]))
        rows.append(
            {
                "model_id": r["model_id"],
                "model_name": r["s1_model_name"],
                "imp_s0_lo": i0[0] * 100,
                "imp_s0_hi": i0[1] * 100,
                "imp_s1_lo": i1[0] * 100,
                "imp_s1_hi": i1[1] * 100,
                "imp_disjoint": i0[1] < i1[0] or i1[1] < i0[0],
                "crew_s0_lo": c0[0] * 100,
                "crew_s0_hi": c0[1] * 100,
                "crew_s1_lo": c1[0] * 100,
                "crew_s1_hi": c1[1] * 100,
                "crew_disjoint": c0[1] < c1[0] or c1[1] < c0[0],
            }
        )
    return pl.DataFrame(rows)


# ── display ───────────────────────────────────────────────────────────────────
def _delta_text(value: float, flip: bool = False, nd: int = 1) -> Text:
    """Color delta by sign. When flip=True, negative is the 'good' direction."""
    sign = "+" if value > 0 else ""
    s = f"{sign}{value:.{nd}f}"
    if value == 0:
        return Text(s, style="dim")
    positive_is_good = not flip
    good = (value > 0) == positive_is_good
    return Text(s, style="green" if good else "red")


def _sig_text(value: float, threshold: float = 0.05) -> Text:
    s = f"{value:.3f}"
    if value <= threshold:
        return Text(s, style="bold green")
    if value <= 0.10:
        return Text(s, style="yellow")
    return Text(s, style="dim")


def render_leaderboard(df: pl.DataFrame, title: str, console: Console) -> None:
    table = Table(title=title, header_style="bold cyan", border_style="dim")
    table.add_column("#", justify="right")
    table.add_column("Model")
    table.add_column("μ−σ", justify="right")
    table.add_column("Games", justify="right")
    table.add_column("Imp %", justify="right", style="red")
    table.add_column("Imp 95% CI", justify="right", style="dim red")
    table.add_column("Crew %", justify="right", style="blue")
    table.add_column("Crew 95% CI", justify="right", style="dim blue")

    sorted_df = df.sort("overall_conservative", descending=True)
    for i, r in enumerate(sorted_df.iter_rows(named=True), 1):
        table.add_row(
            str(i),
            r["model_name"],
            f"{r['overall_conservative']:,.0f}",
            str(r["games_played"]),
            f"{r['impostor_win_rate']:.1f}",
            f"[{r['imp_wr_lo']:.0f},{r['imp_wr_hi']:.0f}]",
            f"{r['crewmate_win_rate']:.1f}",
            f"[{r['crew_wr_lo']:.0f},{r['crew_wr_hi']:.0f}]",
        )
    console.print(table)


def render_human(stats: dict, console: Console) -> None:
    if not stats:
        console.print(f"[yellow]No human data found in {LONG_CONTEXT_LABEL}.[/yellow]")
        return
    table = Table(
        title="Human Brain 1.0 vs chance (p=0.5, two-sided binomial)",
        header_style="bold magenta",
        border_style="dim",
    )
    table.add_column("Role")
    table.add_column("Record", justify="right")
    table.add_column("Rate %", justify="right")
    table.add_column("Wilson 95% CI", justify="right")
    table.add_column("CI contains 50%?", justify="center")
    table.add_column("Binomial p", justify="right")

    for role, wins, games, rate, ci, p in [
        ("Impostor", stats["imp_wins"], stats["imp_games"], stats["imp_rate"],
         stats["imp_ci"], stats["imp_p"]),
        ("Crewmate", stats["crew_wins"], stats["crew_games"], stats["crew_rate"],
         stats["crew_ci"], stats["crew_p"]),
    ]:
        contains50 = ci[0] <= 50 <= ci[1]
        contains_cell = (
            Text("yes", style="yellow") if contains50 else Text("no", style="bold green")
        )
        table.add_row(
            role,
            f"{wins}/{games}",
            f"{rate:.1f}",
            f"[{ci[0]:.1f}, {ci[1]:.1f}]",
            contains_cell,
            _sig_text(p),
        )
    table.add_row(
        "Combined",
        f"{stats['total_wins']}/{stats['total_games']}",
        f"{stats['total_wins'] / stats['total_games'] * 100:.1f}",
        "—", "—", "—",
        style="dim",
    )
    console.print(table)


def render_paired_summary(paired: pl.DataFrame, console: Console) -> dict:
    imp_d = paired["imp_delta"].to_list()
    crew_d = paired["crew_delta"].to_list()
    over_d = paired["overall_delta"].to_list()
    imp_s0_mean = st.mean(paired["s0_impostor_win_rate"].to_list()) if imp_d else float("nan")
    imp_s1_mean = st.mean(paired["s1_impostor_win_rate"].to_list()) if imp_d else float("nan")
    crew_s0_mean = st.mean(paired["s0_crewmate_win_rate"].to_list()) if crew_d else float("nan")
    crew_s1_mean = st.mean(paired["s1_crewmate_win_rate"].to_list()) if crew_d else float("nan")

    imp_stat = paired_t_ci(imp_d)
    crew_stat = paired_t_ci(crew_d)
    over_stat = paired_t_ci(over_d)

    n = paired.height
    n_imp_up = sum(1 for d in imp_d if d > 0)
    n_crew_down = sum(1 for d in crew_d if d < 0)
    sign_imp_p = sign_test_two_sided(n_imp_up, n)
    sign_crew_p = sign_test_two_sided(n_crew_down, n)

    table = Table(
        title=(
            f"Paired {SHORT_CONTEXT_LABEL}→{LONG_CONTEXT_LABEL} deltas "
            f"(n={n} models in both conditions)"
        ),
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Quantity")
    table.add_column("Mean Δ", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("t", justify="right")
    table.add_column("CI excludes 0?", justify="center")

    for label, stat, flip in [
        ("Impostor WR Δ", imp_stat, False),
        # Crew decline is the expected/hypothesis direction → flip sign convention.
        ("Crewmate WR Δ", crew_stat, True),
        ("Overall WR Δ",  over_stat, True),
    ]:
        excludes0 = stat["lo"] > 0 or stat["hi"] < 0
        exc_cell = (
            Text("yes", style="bold green") if excludes0 else Text("no", style="yellow")
        )
        table.add_row(
            label,
            _delta_text(stat["mean"], flip=flip),
            f"[{stat['lo']:+.1f}, {stat['hi']:+.1f}]",
            f"{stat['t']:+.2f}",
            exc_cell,
        )
    console.print(table)

    st_table = Table(
        title="Sign tests (short-context to long-context shift)",
        header_style="bold cyan",
        border_style="dim",
    )
    st_table.add_column("Direction")
    st_table.add_column("Count", justify="right")
    st_table.add_column("p (two-sided)", justify="right")
    st_table.add_row("Impostor WR increased", f"{n_imp_up}/{n}", _sig_text(sign_imp_p))
    st_table.add_row("Crewmate WR decreased", f"{n_crew_down}/{n}", _sig_text(sign_crew_p))
    console.print(st_table)

    return {
        "n": n,
        "imp_s0_mean": imp_s0_mean,
        "imp_s1_mean": imp_s1_mean,
        "crew_s0_mean": crew_s0_mean,
        "crew_s1_mean": crew_s1_mean,
        "imp_stat": imp_stat,
        "crew_stat": crew_stat,
        "overall_stat": over_stat,
        "n_imp_up": n_imp_up,
        "n_crew_down": n_crew_down,
        "sign_imp_p": sign_imp_p,
        "sign_crew_p": sign_crew_p,
    }


def render_ci_overlap(ci_df: pl.DataFrame, console: Console) -> None:
    n = ci_df.height
    imp_disj = int(ci_df["imp_disjoint"].sum())
    crew_disj = int(ci_df["crew_disjoint"].sum())

    table = Table(
        title=(
            f"Per-model Wilson CI overlap check  "
            f"(imp disjoint: {imp_disj}/{n}, crew disjoint: {crew_disj}/{n})"
        ),
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Model")
    table.add_column("Imp short CI", justify="right", style="dim red")
    table.add_column("Imp long CI", justify="right", style="red")
    table.add_column("Imp disj?", justify="center")
    table.add_column("Crew short CI", justify="right", style="dim blue")
    table.add_column("Crew long CI", justify="right", style="blue")
    table.add_column("Crew disj?", justify="center")

    for r in ci_df.sort("model_name").iter_rows(named=True):
        imp_flag = (
            Text("yes", style="bold green") if r["imp_disjoint"] else Text("·", style="dim")
        )
        crew_flag = (
            Text("yes", style="bold green") if r["crew_disjoint"] else Text("·", style="dim")
        )
        table.add_row(
            r["model_name"],
            f"[{r['imp_s0_lo']:.0f},{r['imp_s0_hi']:.0f}]",
            f"[{r['imp_s1_lo']:.0f},{r['imp_s1_hi']:.0f}]",
            imp_flag,
            f"[{r['crew_s0_lo']:.0f},{r['crew_s0_hi']:.0f}]",
            f"[{r['crew_s1_lo']:.0f},{r['crew_s1_hi']:.0f}]",
            crew_flag,
        )
    console.print(table)


# ── outputs ───────────────────────────────────────────────────────────────────
def write_csvs(
    s0: pl.DataFrame,
    s1: pl.DataFrame,
    paired: pl.DataFrame,
    ci: pl.DataFrame,
    out_dir: Path,
    suffix: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    s0.write_csv(out_dir / f"season_s0{suffix}.csv")
    s1.write_csv(out_dir / f"season_s1{suffix}.csv")
    paired.write_csv(out_dir / f"paired{suffix}.csv")
    ci.write_csv(out_dir / f"ci_overlap{suffix}.csv")


def output_suffix(subset: str, min_role_games: int, min_season_games: int) -> str:
    parts: list[str] = []
    if subset != "featured":
        parts.append(subset)
    if min_season_games > 1:
        parts.append(f"min{min_season_games}games")
    if min_role_games > 1:
        parts.append(f"min{min_role_games}role")
    return "" if not parts else "_" + "_".join(parts)


def write_markdown(
    path: Path,
    s0: pl.DataFrame,
    s1: pl.DataFrame,
    human: dict,
    paired_summary: dict,
    ci: pl.DataFrame,
    subset: str,
    min_role_games: int,
    min_season_games: int,
) -> None:
    lines: list[str] = []
    lines.append("# Context-Regime Win-Rate Findings\n")
    lines.append(f"- Model subset: `{subset}`")
    lines.append(f"- Minimum condition games: {min_season_games}")
    lines.append(f"- Minimum role games: {min_role_games}")
    lines.append(f"- {SHORT_CONTEXT_LABEL} qualifying models: {s0.height}")
    lines.append(f"- {LONG_CONTEXT_LABEL} qualifying models: {s1.height}")
    lines.append(f"- Paired (both conditions): {paired_summary['n']}")
    lines.append("")

    lines.append("## Human Brain 1.0 vs chance (two-sided binomial)\n")
    if human:
        lines.append(
            f"- **Impostor:** {human['imp_wins']}/{human['imp_games']} "
            f"= {human['imp_rate']:.1f}% — "
            f"Wilson 95% CI [{human['imp_ci'][0]:.1f}, {human['imp_ci'][1]:.1f}] — "
            f"binomial p={human['imp_p']:.3f}"
        )
        lines.append(
            f"- **Crewmate:** {human['crew_wins']}/{human['crew_games']} "
            f"= {human['crew_rate']:.1f}% — "
            f"Wilson 95% CI [{human['crew_ci'][0]:.1f}, {human['crew_ci'][1]:.1f}] — "
            f"binomial p={human['crew_p']:.3f}"
        )
        lines.append(
            f"- **Combined:** {human['total_wins']}/{human['total_games']} = "
            f"{human['total_wins'] / human['total_games'] * 100:.1f}%"
        )
    else:
        lines.append(f"- _no human row found in {LONG_CONTEXT_LABEL}_")
    lines.append("")

    ps = paired_summary
    lines.append(f"## Paired {SHORT_CONTEXT_LABEL}→{LONG_CONTEXT_LABEL} analysis\n")
    lines.append("### Report bullets\n")
    lines.append(
        f"- Mean Impostor win rate rises from {ps['imp_s0_mean']:.1f}% "
        f"to {ps['imp_s1_mean']:.1f}%"
    )
    lines.append(
        f"- Mean crewmate win rate falls from {ps['crew_s0_mean']:.1f}% "
        f"to {ps['crew_s1_mean']:.1f}%"
    )
    lines.append(
        f"- {ps['n_imp_up']}/{ps['n']} models improve their Impostor win rate "
        "from short-context to long-context"
    )
    lines.append(
        f"- {ps['n_crew_down']}/{ps['n']} models worsen their crewmate win rate "
        "from short-context to long-context"
    )
    lines.append("")

    lines.append("| Quantity | Mean Δ | 95% CI | t |")
    lines.append("|---|---:|---:|---:|")
    for label, stat in [
        ("Impostor WR Δ", ps["imp_stat"]),
        ("Crewmate WR Δ", ps["crew_stat"]),
        ("Overall WR Δ", ps["overall_stat"]),
    ]:
        lines.append(
            f"| {label} | {stat['mean']:+.2f} | "
            f"[{stat['lo']:+.2f}, {stat['hi']:+.2f}] | {stat['t']:+.2f} |"
        )
    lines.append("")
    lines.append(
        f"- Sign test — imp WR increased in {ps['n_imp_up']}/{ps['n']} "
        f"models (two-sided p={ps['sign_imp_p']:.3f})"
    )
    lines.append(
        f"- Sign test — crew WR decreased in {ps['n_crew_down']}/{ps['n']} "
        f"models (two-sided p={ps['sign_crew_p']:.3f})"
    )
    lines.append("")

    lines.append("## Per-model Wilson CI disjointness\n")
    imp_disj = int(ci["imp_disjoint"].sum())
    crew_disj = int(ci["crew_disjoint"].sum())
    lines.append(f"- Impostor disjoint: {imp_disj}/{ci.height}")
    lines.append(f"- Crewmate disjoint: {crew_disj}/{ci.height}")
    lines.append("")
    lines.append("| Model | Imp short CI | Imp long CI | Imp disj | Crew short CI | Crew long CI | Crew disj |")
    lines.append("|---|---|---|:---:|---|---|:---:|")
    for r in ci.sort("model_name").iter_rows(named=True):
        lines.append(
            f"| {r['model_name']} | "
            f"[{r['imp_s0_lo']:.0f},{r['imp_s0_hi']:.0f}] | "
            f"[{r['imp_s1_lo']:.0f},{r['imp_s1_hi']:.0f}] | "
            f"{'OK' if r['imp_disjoint'] else '·'} | "
            f"[{r['crew_s0_lo']:.0f},{r['crew_s0_hi']:.0f}] | "
            f"[{r['crew_s1_lo']:.0f},{r['crew_s1_hi']:.0f}] | "
            f"{'OK' if r['crew_disjoint'] else '·'} |"
        )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--api-base",
        default=os.environ.get("ANALYSIS_API_BASE_URL", DEFAULT_API_BASE),
    )
    parser.add_argument(
        "--min-role-games",
        type=int,
        default=1,
        help="Require ≥ this many games per role per condition (default: 1)",
    )
    parser.add_argument(
        "--min-season-games",
        type=int,
        default=1,
        help="Require ≥ this many total games in each context condition (default: 1)",
    )
    parser.add_argument(
        "--subset",
        choices=sorted(MODEL_SUBSETS),
        default="featured",
        help=(
            "Model set to analyze: `all`, `exemplars`, or `featured` "
            "(default: featured)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="If set, write CSVs (season_s0, season_s1, paired, ci_overlap) here.",
    )
    parser.add_argument(
        "--markdown",
        default=None,
        help="If set, write a markdown summary to this path.",
    )
    parser.add_argument(
        "--no-ci-overlap",
        action="store_true",
        help="Suppress the per-model CI overlap table (the largest one).",
    )
    args = parser.parse_args()

    console = Console()
    console.print(
        Panel.fit(
            f"[bold]Among Us — context-regime win-rate report[/bold]\n"
            f"api={args.api_base}  |  subset={args.subset}  |  "
            f"min condition games={args.min_season_games}  |  "
            f"min role games={args.min_role_games}",
            border_style="cyan",
        )
    )

    raw_s0 = fetch_leaderboard(args.api_base, 0)
    raw_s1 = fetch_leaderboard(args.api_base, 1)
    console.print(
        f"Fetched {SHORT_CONTEXT_LABEL}={raw_s0.height}, "
        f"{LONG_CONTEXT_LABEL}={raw_s1.height} models total."
    )

    s0 = filter_subset(
        enrich(raw_s0, "S0"),
        args.min_role_games,
        args.min_season_games,
        args.subset,
    )
    s1 = filter_subset(
        enrich(raw_s1, "S1"),
        args.min_role_games,
        args.min_season_games,
        args.subset,
    )
    subset_size = MODEL_SUBSETS[args.subset]
    subset_label = len(subset_size) if subset_size is not None else "all"
    console.print(
        f"{args.subset.title()} set after filter: "
        f"{SHORT_CONTEXT_LABEL}={s0.height}, {LONG_CONTEXT_LABEL}={s1.height}"
        f" (model list = {subset_label})"
    )
    console.print()

    render_leaderboard(s1, LONG_CONTEXT_LABEL, console)
    console.print()
    render_leaderboard(s0, SHORT_CONTEXT_LABEL, console)
    console.print()

    human = human_brain_analysis(s1)
    render_human(human, console)
    console.print()

    paired = paired_frame(s0, s1)
    paired_summary = render_paired_summary(paired, console)
    console.print()

    ci = ci_overlap_frame(paired)
    if not args.no_ci_overlap:
        render_ci_overlap(ci, console)
        console.print()

    if args.out_dir:
        out_dir = Path(args.out_dir)
        suffix = output_suffix(
            args.subset,
            args.min_role_games,
            args.min_season_games,
        )
        write_csvs(s0, s1, paired, ci, out_dir, suffix=suffix)
        console.print(
            f"[green]OK[/green] Wrote CSVs to {out_dir} "
            f"(suffix='{suffix or '(none)'}')"
        )

    if args.markdown:
        md_path = Path(args.markdown)
        suffix = output_suffix(
            args.subset,
            args.min_role_games,
            args.min_season_games,
        )
        if suffix and suffix.lstrip("_") not in md_path.stem:
            md_path = md_path.with_stem(md_path.stem + suffix)
        write_markdown(
            md_path,
            s0,
            s1,
            human,
            paired_summary,
            ci,
            subset=args.subset,
            min_role_games=args.min_role_games,
            min_season_games=args.min_season_games,
        )
        console.print(f"[green]OK[/green] Wrote markdown summary to {md_path}")


if __name__ == "__main__":
    main()
