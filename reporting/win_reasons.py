"""Game-outcome and game-mechanics analysis across both seasons.

For every completed game in each season, fetches `summary.*` from the logs
endpoint and computes BOTH the final-reason breakdown AND the mechanistic
signals that underlie it (meetings, votes, ejections, kills, timing).

Outcome categories
------------------
  - crew_tasks        — "Crewmates win! (All task completed)"
  - crew_vote_out     — "Crewmates win! (Impostors eliminated)"
  - imp_outnumber     — "Impostors win! (Crewmates being outnumbered or tied...)"
  - imp_time_limit    — "Impostors win! (Time limit reached)"

Mechanistic per-game signals
----------------------------
Each game gets additional columns describing *how it was played*, not just
who won:

  - game_length        — final timestep observed
  - n_meetings         — number of meetings convened
  - n_votes            — total individual votes cast across all meetings
  - n_ejections        — meetings that actually ejected a player (non-skips)
  - n_imp_ejections    — ejections that removed an impostor
  - vote_accuracy      — n_imp_ejections / n_ejections (nan if 0 ejections)
  - vote_quality       — fraction of individual votes that targeted an impostor
  - n_kills            — impostor kills during the game
  - turn_of_first_imp_out — timestep of first impostor ejection, or None
  - kills_before_first_imp_out — kills that happened before then (all kills if never)

Emits:
  - rich console tables (outcomes, team shares, mechanics by season, mechanics
    conditioned on who won)
  - CSVs when --out-dir is set (per_game.csv, summary.csv, mechanics.csv)
  - optional markdown summary

Usage:
  uv run --with polars --with rich python reporting/win_reasons.py
  uv run --with polars --with rich python reporting/win_reasons.py \\
      --out-dir reporting/out --markdown reporting/win_reasons.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

DEFAULT_API_BASE = "https://api.sdgarena.averyyen.dev"

# Model cohorts (kept in sync with reporting/season_chart.py). Used to restrict
# the analysis to games where *all* participants come from the chosen cohort.
FEATURED_MODEL_IDS = [
    "gpt-oss-120b", "trinity-large", "minimax-m2.5", "kimi-k2.5",
    "qwen3.5-plus-02-15", "nemotron-3-super", "claude-haiku-4.5",
    "llama-3.3-70b", "deepseek-v3.2", "step-3.5-flash", "grok-4.1-fast",
    "mimo-v2-flash", "glm-5", "claude-sonnet-4.5", "llama-4-maverick",
    "gemini-3.1-pro", "gpt-5.4", "gemini-3-flash", "claude-opus-4.6",
    "brain-1.0",
]

PRESENTATION_MODEL_IDS = [
    # 7 long-context human-AI participants
    "claude-opus-4.6", "gpt-5.4", "gemini-3.1-pro", "llama-3.3-70b",
    "nemotron-3-super", "kimi-k2.5", "deepseek-v3.2",
    # Human participants
    "brain-1.0",
    # Clean references
    "gemini-3-flash", "claude-sonnet-4.5", "grok-4.1-fast",
]

MODEL_COHORTS: dict[str, list[str] | None] = {
    "all": None,  # no filter
    "featured": FEATURED_MODEL_IDS,
    "presentation": PRESENTATION_MODEL_IDS,
    # "common" is resolved at runtime — intersection of model_ids across the
    # requested seasons (see compute_common_cohort below). This is the right
    # cohort for paper-grade cross-season claims because it holds the
    # participant pool fixed; the confound from "Season N had different models"
    # is removed by construction.
    "common": None,  # placeholder — populated dynamically in main()
}

# Canonical outcome taxonomy.
REASON_BUCKETS = [
    ("All task completed", "crew_tasks", "Crewmates"),
    ("Impostors eliminated", "crew_vote_out", "Crewmates"),
    ("Crewmates being outnumbered", "imp_outnumber", "Impostors"),
    ("Time limit reached", "imp_time_limit", "Impostors"),
]

CATEGORY_LABELS = {
    "crew_tasks": "Crewmates completed all tasks",
    "crew_vote_out": "Crewmates voted out impostors",
    "imp_outnumber": "Impostors reached kill parity",
    "imp_time_limit": "Impostors ran out the clock",
    "unknown": "Unclassified",
}


def categorize(reason: str | None) -> tuple[str, str]:
    """Return (category_code, winning_team). Falls back to ('unknown', '?')."""
    if not reason:
        return ("unknown", "?")
    for needle, code, team in REASON_BUCKETS:
        if needle in reason:
            return (code, team)
    return ("unknown", "?")


# ── API ───────────────────────────────────────────────────────────────────────
def fetch_games(base_url: str, version: int) -> list[dict]:
    return json.loads(
        urllib.request.urlopen(
            f"{base_url}/api/games?engine_version={version}"
            f"&status=completed&limit=1000",
            timeout=30,
        ).read()
    )


def fetch_summary(base_url: str, game_id: str, retries: int = 1) -> dict | None:
    url = f"{base_url}/api/games/{game_id}/logs"
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=45) as resp:
                payload = json.load(resp)
            return payload.get("summary")
        except Exception:  # socket timeout, SSL error, JSON error, etc.
            pass
    # Give up after the retry budget — don't crash the run over a flaky fetch.
    return None


# ── per-game extraction ───────────────────────────────────────────────────────
def _role_map(summary: dict) -> dict[str, str]:
    """Map from display name ('Player 1: blue') to identity ('Impostor'/'Crewmate')."""
    m: dict[str, str] = {}
    for key, val in summary.items():
        if not key.startswith("Player "):
            continue
        if not isinstance(val, dict):
            continue
        name = val.get("name")
        identity = val.get("identity")
        if name and identity:
            m[name] = identity
    return m


def extract_mechanics(summary: dict) -> dict:
    """Mechanical features of a single game's play, computed from summary fields."""
    roles = _role_map(summary)
    voting = summary.get("voting_history") or []
    kills = summary.get("kill_history") or []
    turn_log = summary.get("turn_log") or []

    # Meetings and voting.
    n_meetings = len(voting)
    n_votes = 0
    n_ejections = 0
    n_imp_ejections = 0
    n_imp_target_votes = 0
    n_crew_target_votes = 0  # misfires
    turn_of_first_imp_out: int | None = None
    first_ejection_role: str | None = None
    turn_of_first_ejection: int | None = None
    for meeting in voting:
        votes = meeting.get("votes") or []
        n_votes += len(votes)
        for v in votes:
            target = v.get("target")
            if not target:
                continue
            role = roles.get(target)
            if role == "Impostor":
                n_imp_target_votes += 1
            elif role == "Crewmate":
                n_crew_target_votes += 1
        eliminated = meeting.get("eliminated")
        if eliminated:
            n_ejections += 1
            elim_role = roles.get(eliminated)
            if first_ejection_role is None and elim_role in ("Impostor", "Crewmate"):
                first_ejection_role = elim_role
                turn_of_first_ejection = meeting.get("timestep")
            if elim_role == "Impostor":
                n_imp_ejections += 1
                if turn_of_first_imp_out is None:
                    turn_of_first_imp_out = meeting.get("timestep")

    vote_accuracy = (n_imp_ejections / n_ejections) if n_ejections else None
    votes_with_target = n_imp_target_votes + n_crew_target_votes
    vote_quality = (n_imp_target_votes / votes_with_target) if votes_with_target else None

    # Kills.
    n_kills = len(kills)
    kills_before_first_imp_out = (
        sum(
            1 for k in kills
            if turn_of_first_imp_out is not None
            and int(k.get("timestep", 0)) < turn_of_first_imp_out
        )
        if turn_of_first_imp_out is not None
        else n_kills
    )

    # Game length: last timestep observed across any log stream.
    last_ts_candidates = []
    if turn_log:
        last_ts_candidates.append(
            max(int(t.get("timestep", 0)) for t in turn_log)
        )
    if voting:
        last_ts_candidates.append(
            max(int(m.get("timestep", 0)) for m in voting)
        )
    if kills:
        last_ts_candidates.append(
            max(int(k.get("timestep", 0)) for k in kills)
        )
    game_length = max(last_ts_candidates) if last_ts_candidates else 0

    # Turns spent in meeting vs task phases. `turn_log` was added by the
    # current engine (S1+); S0 came from an earlier baseline that didn't
    # populate it, so these three columns are effectively S1-only and will
    # be 0 for every S0 game.
    n_turns = len(turn_log)
    n_meeting_turns = sum(
        1 for t in turn_log if str(t.get("phase", "")).lower() == "meeting"
    )
    n_task_turns = sum(
        1 for t in turn_log if str(t.get("phase", "")).lower() == "task"
    )

    return {
        "game_length": game_length,
        "n_turns": n_turns,
        "n_meeting_turns": n_meeting_turns,
        "n_task_turns": n_task_turns,
        "n_meetings": n_meetings,
        "n_votes": n_votes,
        "n_ejections": n_ejections,
        "n_imp_ejections": n_imp_ejections,
        "n_imp_target_votes": n_imp_target_votes,
        "n_crew_target_votes": n_crew_target_votes,
        "vote_accuracy": vote_accuracy,
        "vote_quality": vote_quality,
        "n_kills": n_kills,
        "turn_of_first_imp_out": turn_of_first_imp_out,
        "turn_of_first_ejection": turn_of_first_ejection,
        "first_ejection_role": first_ejection_role,
        "kills_before_first_imp_out": kills_before_first_imp_out,
    }


def fetch_participant_models(base_url: str, version: int) -> tuple[set[str], dict[str, int]]:
    """Return (model_id set, per-model game counts) for a season.

    Counts help tell apart models that merely *appeared* in a season from those
    that played a non-trivial number of games, which matters when picking a
    cross-season cohort for paper claims.
    """
    games = fetch_games(base_url, version)
    counts: dict[str, int] = {}
    for g in games:
        seen_in_game: set[str] = set()
        for p in g.get("participants") or []:
            mid = p.get("model_id")
            if mid:
                seen_in_game.add(mid)
        for mid in seen_in_game:
            counts[mid] = counts.get(mid, 0) + 1
    return set(counts.keys()), counts


def compute_common_cohort(
    base_url: str,
    versions: list[int],
    console: Console,
    min_games: int = 1,
) -> set[str]:
    """Intersection of model_ids that played ≥ min_games in every season.

    Prints a transparency panel showing season sizes, the intersection, and
    which models are dropped (so the paper can cite exactly who is in/out).
    """
    per_season: dict[int, tuple[set[str], dict[str, int]]] = {
        v: fetch_participant_models(base_url, v) for v in versions
    }
    if not per_season:
        return set()

    # Model set filtered by min_games per season.
    qualified: dict[int, set[str]] = {
        v: {m for m, c in counts.items() if c >= min_games}
        for v, (_, counts) in per_season.items()
    }
    common = set.intersection(*qualified.values())

    lines = ["[bold]Common-cohort resolution[/bold]"]
    lines.append(f"  min_games per season: {min_games}")
    for v, (models, counts) in per_season.items():
        lines.append(
            f"  S{v}: {len(models)} models total, "
            f"{len(qualified[v])} with ≥{min_games} game(s)"
        )
    lines.append(f"  [green]Common:[/green] {len(common)} models")
    if common:
        lines.append("    " + ", ".join(sorted(common)))
    for v, (models, counts) in per_season.items():
        only = qualified[v] - common
        if only:
            lines.append(
                f"  [yellow]S{v}-only (dropped):[/yellow] "
                + ", ".join(f"{m} ({counts[m]}g)" for m in sorted(only))
            )
    console.print(Panel.fit("\n".join(lines), border_style="magenta"))
    return common


def _game_matches_cohort(game: dict, cohort: set[str], policy: str) -> bool:
    """Per-game membership test. `policy` is 'all', 'majority', or 'any'."""
    participants = game.get("participants") or []
    if not participants:
        return False
    in_cohort = sum(1 for p in participants if p.get("model_id") in cohort)
    if policy == "all":
        return in_cohort == len(participants)
    if policy == "majority":
        return in_cohort * 2 > len(participants)
    if policy == "any":
        return in_cohort >= 1
    raise ValueError(f"unknown cohort match policy: {policy}")


def collect_season(
    base_url: str,
    version: int,
    max_workers: int,
    console: Console,
    cohort: set[str] | None = None,
    match_policy: str = "all",
) -> pl.DataFrame:
    games = fetch_games(base_url, version)
    total = len(games)
    if cohort is not None:
        games = [g for g in games if _game_matches_cohort(g, cohort, match_policy)]
        console.print(
            f"  S{version}: {len(games)}/{total} games match cohort "
            f"(policy=[cyan]{match_policy}[/cyan])"
        )
    if not games:
        return pl.DataFrame()

    rows: list[dict] = []
    with Progress(
        TextColumn("[cyan]Season {task.fields[version]}[/cyan]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("fetch", total=len(games), version=version)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(fetch_summary, base_url, g["game_id"]): g for g in games
            }
            for fut in as_completed(futures):
                game = futures[fut]
                summary = fut.result()
                reason = None
                mech: dict = {}
                if summary is not None:
                    reason = summary.get("game_outcome", {}).get("reason")
                    mech = extract_mechanics(summary)
                category, winner = categorize(reason)
                rows.append(
                    {
                        "engine_version": version,
                        "game_id": game["game_id"],
                        "reason": reason,
                        "category": category,
                        "winner": winner,
                        **mech,
                    }
                )
                progress.update(task, advance=1)

    return pl.DataFrame(rows)


# ── aggregation ──────────────────────────────────────────────────────────────
def outcome_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Per-(season, category) counts + percent of that season's total."""
    if df.is_empty():
        return df
    totals = df.group_by("engine_version").agg(pl.len().alias("season_total"))
    return (
        df.group_by(["engine_version", "category", "winner"])
        .agg(pl.len().alias("count"))
        .join(totals, on="engine_version")
        .with_columns((pl.col("count") / pl.col("season_total") * 100).alias("pct"))
        .sort(["engine_version", "count"], descending=[False, True])
    )


MECHANIC_FIELDS = [
    ("game_length", "Game length (last timestep)"),
    ("n_turns", "Turns logged"),
    ("n_meeting_turns", "Meeting-phase turns"),
    ("n_task_turns", "Task-phase turns"),
    ("n_meetings", "Meetings per game"),
    ("n_votes", "Votes cast per game"),
    ("n_ejections", "Ejections per game"),
    ("n_imp_ejections", "Impostor ejections per game"),
    ("vote_accuracy", "Vote accuracy (imp ejects / ejects)"),
    ("vote_quality", "Vote quality (imp-targeted votes / votes)"),
    ("n_kills", "Kills per game"),
    ("kills_before_first_imp_out", "Kills before first impostor ejected"),
]


def mechanics_means(
    df: pl.DataFrame, group_cols: list[str]
) -> pl.DataFrame:
    """Group-wise means for each mechanic field."""
    if df.is_empty():
        return pl.DataFrame()
    cols = [pl.col(f).mean().alias(f) for f, _ in MECHANIC_FIELDS]
    cols.append(pl.len().alias("n_games"))
    return df.group_by(group_cols).agg(cols).sort(group_cols)


# ── rendering ────────────────────────────────────────────────────────────────
def render_outcomes(summary: pl.DataFrame, console: Console) -> None:
    if summary.is_empty():
        console.print("[yellow]No outcome data.[/yellow]")
        return
    versions = sorted(summary["engine_version"].unique().to_list())
    categories = [code for _, code, _ in REASON_BUCKETS] + ["unknown"]

    table = Table(
        title="Game-ending outcomes by season",
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Category")
    table.add_column("Winner")
    for v in versions:
        total = (
            summary.filter(pl.col("engine_version") == v)["season_total"]
            .head(1)
            .item()
        )
        table.add_column(f"S{v}  (n={total})", justify="right")
    table.add_column("Δ S1−S0 (pts)", justify="right")

    for cat in categories:
        cells = []
        pct_by_v: dict[int, float] = {}
        winner = ""
        for v in versions:
            row = summary.filter(
                (pl.col("engine_version") == v) & (pl.col("category") == cat)
            )
            if row.is_empty():
                cells.append("—")
                pct_by_v[v] = 0.0
            else:
                r = row.row(0, named=True)
                winner = r["winner"]
                cells.append(f"{r['count']:>3}  ({r['pct']:.1f}%)")
                pct_by_v[v] = r["pct"]
        if all(c == "—" for c in cells):
            continue
        delta = pct_by_v.get(1, 0.0) - pct_by_v.get(0, 0.0)
        delta_style = "green" if delta > 0 else ("red" if delta < 0 else "dim")
        row_style = (
            "red" if winner == "Impostors"
            else "blue" if winner == "Crewmates"
            else "dim"
        )
        table.add_row(
            CATEGORY_LABELS.get(cat, cat),
            winner,
            *cells,
            f"[{delta_style}]{delta:+.1f}[/{delta_style}]",
            style=row_style,
        )
    console.print(table)

    # Team-level totals.
    team_table = Table(
        title="Team win share by season",
        header_style="bold cyan",
        border_style="dim",
    )
    team_table.add_column("Team")
    for v in versions:
        total = (
            summary.filter(pl.col("engine_version") == v)["season_total"]
            .head(1)
            .item()
        )
        team_table.add_column(f"S{v}  (n={total})", justify="right")
    team_table.add_column("Δ S1−S0 (pts)", justify="right")

    for team, style in [("Crewmates", "blue"), ("Impostors", "red")]:
        pcts: dict[int, float] = {}
        cells = []
        for v in versions:
            sub = summary.filter(
                (pl.col("engine_version") == v) & (pl.col("winner") == team)
            )
            if sub.is_empty():
                cells.append("—")
                pcts[v] = 0.0
            else:
                ct = int(sub["count"].sum())
                pt = float(sub["pct"].sum())
                pcts[v] = pt
                cells.append(f"{ct:>3}  ({pt:.1f}%)")
        delta = pcts.get(1, 0.0) - pcts.get(0, 0.0)
        delta_style = "green" if delta > 0 else ("red" if delta < 0 else "dim")
        team_table.add_row(
            team, *cells, f"[{delta_style}]{delta:+.1f}[/{delta_style}]", style=style
        )
    console.print(team_table)


def _fmt(v: float | None, as_pct: bool = False, nd: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if as_pct:
        return f"{v * 100:.1f}%"
    return f"{v:.{nd}f}"


def render_mechanics(
    per_game: pl.DataFrame, console: Console, title_suffix: str = ""
) -> None:
    """Per-season mechanic means with S1−S0 delta."""
    grand = mechanics_means(per_game, ["engine_version"])
    if grand.is_empty():
        console.print("[yellow]No mechanics data.[/yellow]")
        return

    versions = sorted(grand["engine_version"].unique().to_list())
    table = Table(
        title=f"Game mechanics — per-season means{title_suffix}",
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Metric")
    for v in versions:
        n = int(grand.filter(pl.col("engine_version") == v)["n_games"].head(1).item())
        table.add_column(f"S{v}  (n={n})", justify="right")
    table.add_column("Δ S1−S0", justify="right")
    table.add_column("Rel Δ", justify="right")

    for field, label in MECHANIC_FIELDS:
        by_v: dict[int, float | None] = {}
        for v in versions:
            sub = grand.filter(pl.col("engine_version") == v)
            if sub.is_empty():
                by_v[v] = None
                continue
            val = sub[field].head(1).item()
            by_v[v] = val
        as_pct = field in ("vote_accuracy", "vote_quality")
        cells = [_fmt(by_v[v], as_pct=as_pct) for v in versions]
        v0, v1 = by_v.get(0), by_v.get(1)
        if v0 is None or v1 is None:
            delta_cell = "—"
            rel_cell = "—"
        else:
            delta = v1 - v0
            rel = (delta / v0 * 100) if v0 else None
            if as_pct:
                delta_txt = f"{delta * 100:+.1f} pts"
            else:
                delta_txt = f"{delta:+.2f}"
            style = "green" if delta > 0 else ("red" if delta < 0 else "dim")
            delta_cell = f"[{style}]{delta_txt}[/{style}]"
            rel_cell = (
                f"[{style}]{rel:+.0f}%[/{style}]"
                if rel is not None and abs(rel) < 1000
                else "—"
            )
        table.add_row(label, *cells, delta_cell, rel_cell)
    console.print(table)


def render_mechanics_by_winner(per_game: pl.DataFrame, console: Console) -> None:
    """Split each season's mechanics by who won — shows what winning looks like."""
    sub = per_game.filter(pl.col("winner").is_in(["Crewmates", "Impostors"]))
    if sub.is_empty():
        return
    grouped = mechanics_means(sub, ["engine_version", "winner"])

    versions = sorted(grouped["engine_version"].unique().to_list())
    teams = ["Crewmates", "Impostors"]

    for v in versions:
        table = Table(
            title=f"Mechanics conditioned on winner — Season {v}",
            header_style="bold cyan",
            border_style="dim",
        )
        table.add_column("Metric")
        for team in teams:
            row = grouped.filter(
                (pl.col("engine_version") == v) & (pl.col("winner") == team)
            )
            n = int(row["n_games"].head(1).item()) if not row.is_empty() else 0
            table.add_column(f"{team} win  (n={n})", justify="right")
        table.add_column("Crew−Imp", justify="right")

        for field, label in MECHANIC_FIELDS:
            vals: dict[str, float | None] = {}
            for team in teams:
                row = grouped.filter(
                    (pl.col("engine_version") == v) & (pl.col("winner") == team)
                )
                vals[team] = None if row.is_empty() else row[field].head(1).item()
            as_pct = field in ("vote_accuracy", "vote_quality")
            cells = [_fmt(vals[t], as_pct=as_pct) for t in teams]
            c, i = vals["Crewmates"], vals["Impostors"]
            if c is None or i is None:
                diff_cell = "—"
            else:
                d = c - i
                if as_pct:
                    txt = f"{d * 100:+.1f} pts"
                else:
                    txt = f"{d:+.2f}"
                style = "blue" if d > 0 else ("red" if d < 0 else "dim")
                diff_cell = f"[{style}]{txt}[/{style}]"
            table.add_row(label, *cells, diff_cell)
        console.print(table)


# ── deep dives ───────────────────────────────────────────────────────────────
def vote_decomposition_rows(per_game: pl.DataFrame) -> list[dict]:
    """Pooled per-season vote-quality / vote-accuracy / alignment ratio."""
    out: list[dict] = []
    for v in sorted(per_game["engine_version"].unique().to_list()):
        sub = per_game.filter(pl.col("engine_version") == v)
        total_imp_votes = int(sub["n_imp_target_votes"].sum())
        total_crew_votes = int(sub["n_crew_target_votes"].sum())
        total_target_votes = total_imp_votes + total_crew_votes
        total_ejections = int(sub["n_ejections"].sum())
        total_imp_ejections = int(sub["n_imp_ejections"].sum())
        vq = (total_imp_votes / total_target_votes) if total_target_votes else None
        va = (total_imp_ejections / total_ejections) if total_ejections else None
        ratio = (va / vq) if (vq and va is not None) else None
        out.append({
            "v": v,
            "n_games": sub.height,
            "target_votes": total_target_votes,
            "ejections": total_ejections,
            "vq": vq,
            "va": va,
            "ratio": ratio,
        })
    return out


def render_vote_decomposition(per_game: pl.DataFrame, console: Console) -> None:
    """Individual detection (vote_quality) vs. group alignment (vote_accuracy)
    — pooled across all games per season, with alignment ratio."""
    if per_game.is_empty():
        return
    rows = vote_decomposition_rows(per_game)

    table = Table(
        title="Vote decomposition — individual detection vs. group alignment",
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Metric")
    for r in rows:
        table.add_column(
            f"S{r['v']}  (n_games={r['n_games']})", justify="right"
        )
    if len(rows) == 2:
        table.add_column("Δ S1−S0", justify="right")

    def add(label: str, key: str, as_pct: bool) -> None:
        cells = []
        for r in rows:
            val = r[key]
            if val is None:
                cells.append("—")
            elif as_pct:
                cells.append(f"{val * 100:.1f}%")
            else:
                cells.append(f"{val:.3f}")
        if len(rows) == 2 and rows[0][key] is not None and rows[1][key] is not None:
            d = rows[1][key] - rows[0][key]
            style = "green" if d > 0 else ("red" if d < 0 else "dim")
            txt = f"{d * 100:+.1f} pts" if as_pct else f"{d:+.3f}"
            table.add_row(label, *cells, f"[{style}]{txt}[/{style}]")
        else:
            table.add_row(label, *cells)

    # Supporting counts row (no delta — raw sizes)
    sample_cells = [
        f"{r['target_votes']:,} votes / {r['ejections']:,} ejects"
        for r in rows
    ]
    if len(rows) == 2:
        table.add_row("[dim]Pool size[/dim]", *sample_cells, "[dim]—[/dim]")
    else:
        table.add_row("[dim]Pool size[/dim]", *sample_cells)

    add("Vote quality (indiv — % votes → impostor)", "vq", as_pct=True)
    add("Vote accuracy (group — % ejections → impostor)", "va", as_pct=True)
    add("Alignment ratio (accuracy / quality)", "ratio", as_pct=False)
    console.print(table)
    console.print(
        "[dim]  ratio ≈ 1.0 → group tracks individual signal cleanly;  "
        "ratio < 1 → coordination failure (individuals know, group misfires);  "
        "ratio > 1 → decisive minorities / strategic voting.[/dim]"
    )


def first_ejection_rows(per_game: pl.DataFrame) -> pl.DataFrame:
    """Counts + crew win rate per (season, first-ejection bucket)."""
    if per_game.is_empty():
        return pl.DataFrame()
    work = per_game.with_columns(
        pl.col("first_ejection_role").fill_null("none").alias("fej")
    )
    agg = (
        work.group_by(["engine_version", "fej"])
        .agg(
            pl.len().alias("n_games"),
            (pl.col("winner") == "Crewmates").sum().alias("crew_wins"),
            (pl.col("winner") == "Impostors").sum().alias("imp_wins"),
        )
        .with_columns((pl.col("crew_wins") / pl.col("n_games")).alias("crew_rate"))
        .sort(["engine_version", "fej"])
    )
    return agg


def render_first_ejection(per_game: pl.DataFrame, console: Console) -> None:
    """Crew win rate conditional on who got ejected first."""
    agg = first_ejection_rows(per_game)
    if agg.is_empty():
        return
    versions = sorted(agg["engine_version"].unique().to_list())
    buckets = [
        ("Impostor", "Impostor ejected first"),
        ("Crewmate", "Crewmate ejected first"),
        ("none", "No ejection occurred"),
    ]

    table = Table(
        title="First-ejection decisiveness — crew win rate given who was ejected first",
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("First ejection")
    for v in versions:
        n = int(
            per_game.filter(pl.col("engine_version") == v).height
        )
        table.add_column(f"S{v}  (n={n})", justify="right")
    if len(versions) == 2:
        table.add_column("Δ crew-rate (pts)", justify="right")

    for fej, label in buckets:
        cells = []
        rates: dict[int, float | None] = {}
        for v in versions:
            row = agg.filter(
                (pl.col("engine_version") == v) & (pl.col("fej") == fej)
            )
            if row.is_empty():
                cells.append("—")
                rates[v] = None
                continue
            r = row.row(0, named=True)
            rates[v] = r["crew_rate"]
            cells.append(
                f"{r['crew_wins']}/{r['n_games']}  ({r['crew_rate'] * 100:.1f}%)"
            )
        style_row = (
            "blue" if fej == "Impostor"
            else "red" if fej == "Crewmate"
            else "dim"
        )
        if (
            len(versions) == 2
            and rates.get(0) is not None
            and rates.get(1) is not None
        ):
            d = rates[1] - rates[0]
            delta_style = "green" if d > 0 else ("red" if d < 0 else "dim")
            table.add_row(
                label,
                *cells,
                f"[{delta_style}]{d * 100:+.1f}[/{delta_style}]",
                style=style_row,
            )
        else:
            table.add_row(label, *cells, style=style_row)
    console.print(table)
    console.print(
        "[dim]  Read off the decisiveness of first blood: how much does correct "
        "(or mistaken) first ejection lock in the winner?[/dim]"
    )


def render_imp_pathway(per_game: pl.DataFrame, console: Console) -> None:
    """Impostor-win pathway mechanics: outnumber (kill-rush) vs. time-limit (stall)."""
    imp = per_game.filter(pl.col("winner") == "Impostors")
    if imp.is_empty():
        return
    pathways = [
        ("imp_outnumber", "Outnumber (kill-rush)"),
        ("imp_time_limit", "Time-limit (stall)"),
    ]
    versions = sorted(imp["engine_version"].unique().to_list())

    for v in versions:
        seasonal = imp.filter(pl.col("engine_version") == v)
        counts = {p: seasonal.filter(pl.col("category") == p).height for p, _ in pathways}
        if sum(counts.values()) == 0:
            continue
        share = {
            p: counts[p] / sum(counts.values()) if sum(counts.values()) else 0.0
            for p, _ in pathways
        }

        table = Table(
            title=f"Impostor-win pathway mechanics — Season {v}",
            header_style="bold cyan",
            border_style="dim",
        )
        table.add_column("Metric")
        for code, label in pathways:
            table.add_column(
                f"{label}\n"
                f"n={counts[code]}  ({share[code] * 100:.1f}% of imp-wins)",
                justify="right",
            )
        table.add_column("Out − Time", justify="right")

        for field, label in MECHANIC_FIELDS:
            vals: dict[str, float | None] = {}
            for code, _ in pathways:
                sub = seasonal.filter(pl.col("category") == code)
                if sub.is_empty():
                    vals[code] = None
                    continue
                col_vals = sub[field].drop_nulls()
                vals[code] = float(col_vals.mean()) if col_vals.len() else None
            as_pct = field in ("vote_accuracy", "vote_quality")
            cells = [_fmt(vals[c], as_pct=as_pct) for c, _ in pathways]
            o, t = vals["imp_outnumber"], vals["imp_time_limit"]
            if o is None or t is None:
                diff = "—"
            else:
                d = o - t
                txt = f"{d * 100:+.1f} pts" if as_pct else f"{d:+.2f}"
                diff = f"[cyan]{txt}[/cyan]"
            table.add_row(label, *cells, diff)
        console.print(table)
    console.print(
        "[dim]  Outnumber = impostors won by reaching kill parity (aggressive kill-rush).  "
        "Time-limit = impostors won by stalling until the clock ran out (patient / "
        "avoid-detection).  Look for metrics where the two pathways diverge.[/dim]"
    )


# ── markdown output ──────────────────────────────────────────────────────────
def write_markdown(
    path: Path,
    per_game: pl.DataFrame,
    summary: pl.DataFrame,
) -> None:
    lines: list[str] = ["# Win-Reason & Game-Mechanics Report\n"]
    versions = sorted(summary["engine_version"].unique().to_list())

    lines.append("## Outcome distribution\n")
    lines.append("| Category | Winner |" + "".join(
        f" S{v} |" for v in versions
    ) + " Δ S1−S0 (pts) |")
    lines.append("|---|---|" + "---:|" * len(versions) + "---:|")

    for _, cat, _ in REASON_BUCKETS + [(None, "unknown", None)]:
        pct_by_v = {}
        winner = ""
        cells = []
        for v in versions:
            sub = summary.filter(
                (pl.col("engine_version") == v) & (pl.col("category") == cat)
            )
            if sub.is_empty():
                cells.append("—")
                pct_by_v[v] = 0.0
            else:
                r = sub.row(0, named=True)
                winner = r["winner"]
                cells.append(f"{r['count']} ({r['pct']:.1f}%)")
                pct_by_v[v] = r["pct"]
        if all(c == "—" for c in cells):
            continue
        delta = pct_by_v.get(1, 0.0) - pct_by_v.get(0, 0.0)
        lines.append(
            f"| {CATEGORY_LABELS.get(cat, cat)} | {winner} | "
            + " | ".join(cells)
            + f" | {delta:+.1f} |"
        )
    lines.append("")

    # Mechanics block.
    grand = mechanics_means(per_game, ["engine_version"])
    if not grand.is_empty():
        lines.append("## Mechanics — per-season means\n")
        header = "| Metric |" + "".join(
            f" S{v} (n={int(grand.filter(pl.col('engine_version') == v)['n_games'].head(1).item())}) |"
            for v in versions
        ) + " Δ S1−S0 |"
        lines.append(header)
        lines.append("|---|" + "---:|" * (len(versions) + 1))
        for field, label in MECHANIC_FIELDS:
            by_v = {}
            for v in versions:
                sub = grand.filter(pl.col("engine_version") == v)
                by_v[v] = None if sub.is_empty() else sub[field].head(1).item()
            as_pct = field in ("vote_accuracy", "vote_quality")
            cells = [_fmt(by_v[v], as_pct=as_pct) for v in versions]
            v0, v1 = by_v.get(0), by_v.get(1)
            if v0 is None or v1 is None:
                delta_txt = "—"
            else:
                d = v1 - v0
                delta_txt = f"{d * 100:+.1f} pts" if as_pct else f"{d:+.2f}"
            lines.append(f"| {label} | " + " | ".join(cells) + f" | {delta_txt} |")
        lines.append("")

    # ── Deep dive 1: vote decomposition ─────────────────────────────────────
    vd = vote_decomposition_rows(per_game)
    if vd:
        lines.append("## Vote decomposition — individual detection vs. group alignment\n")
        header = "| Metric |" + "".join(
            f" S{r['v']} (n_games={r['n_games']}) |" for r in vd
        ) + (" Δ S1−S0 |" if len(vd) == 2 else "")
        lines.append(header)
        lines.append("|---|" + "---:|" * (len(vd) + (1 if len(vd) == 2 else 0)))
        pool_cells = [
            f"{r['target_votes']:,} votes / {r['ejections']:,} ejects"
            for r in vd
        ]
        lines.append(
            "| Pool size | " + " | ".join(pool_cells)
            + (" | — |" if len(vd) == 2 else " |")
        )

        def _row(label: str, key: str, as_pct: bool) -> None:
            cells = []
            for r in vd:
                val = r[key]
                if val is None:
                    cells.append("—")
                elif as_pct:
                    cells.append(f"{val * 100:.1f}%")
                else:
                    cells.append(f"{val:.3f}")
            if len(vd) == 2 and vd[0][key] is not None and vd[1][key] is not None:
                d = vd[1][key] - vd[0][key]
                txt = f"{d * 100:+.1f} pts" if as_pct else f"{d:+.3f}"
                lines.append(f"| {label} | " + " | ".join(cells) + f" | {txt} |")
            else:
                lines.append(f"| {label} | " + " | ".join(cells) + " |")

        _row("Vote quality (indiv — % votes → impostor)", "vq", True)
        _row("Vote accuracy (group — % ejections → impostor)", "va", True)
        _row("Alignment ratio (accuracy / quality)", "ratio", False)
        lines.append(
            "\n*ratio ≈ 1 → group tracks individual signal; <1 → coordination "
            "failure (individuals know, group misfires); >1 → decisive minorities / "
            "strategic voting.*\n"
        )

    # ── Deep dive 2: first-ejection decisiveness ───────────────────────────
    fe = first_ejection_rows(per_game)
    if not fe.is_empty():
        lines.append("## First-ejection decisiveness\n")
        lines.append(
            "Crew win rate conditional on who was ejected first (or whether any "
            "ejection happened at all). Measures how much the first meeting's "
            "verdict locks in the outcome.\n"
        )
        fe_versions = sorted(fe["engine_version"].unique().to_list())
        totals = {
            v: int(per_game.filter(pl.col("engine_version") == v).height)
            for v in fe_versions
        }
        header = "| First ejection |" + "".join(
            f" S{v} (n={totals[v]}) |" for v in fe_versions
        ) + (" Δ crew-rate (pts) |" if len(fe_versions) == 2 else "")
        lines.append(header)
        lines.append(
            "|---|" + "---:|" * (len(fe_versions) + (1 if len(fe_versions) == 2 else 0))
        )
        buckets = [
            ("Impostor", "Impostor ejected first"),
            ("Crewmate", "Crewmate ejected first"),
            ("none", "No ejection occurred"),
        ]
        for fej, label in buckets:
            cells = []
            rates: dict[int, float | None] = {}
            for v in fe_versions:
                row = fe.filter(
                    (pl.col("engine_version") == v) & (pl.col("fej") == fej)
                )
                if row.is_empty():
                    cells.append("—")
                    rates[v] = None
                    continue
                r = row.row(0, named=True)
                rates[v] = r["crew_rate"]
                cells.append(
                    f"{r['crew_wins']}/{r['n_games']} ({r['crew_rate'] * 100:.1f}%)"
                )
            if (
                len(fe_versions) == 2
                and rates.get(0) is not None
                and rates.get(1) is not None
            ):
                d = rates[1] - rates[0]
                lines.append(
                    f"| {label} | " + " | ".join(cells) + f" | {d * 100:+.1f} |"
                )
            else:
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

    # ── Deep dive 3: impostor-win pathway mechanics ────────────────────────
    imp = per_game.filter(pl.col("winner") == "Impostors")
    if not imp.is_empty():
        lines.append("## Impostor-win pathway mechanics\n")
        lines.append(
            "For each season, split impostor-wins into **Outnumber** (kill-rush) "
            "vs. **Time-limit** (stall) and compare the mechanical profile.\n"
        )
        pathways = [
            ("imp_outnumber", "Outnumber (kill-rush)"),
            ("imp_time_limit", "Time-limit (stall)"),
        ]
        for v in sorted(imp["engine_version"].unique().to_list()):
            seasonal = imp.filter(pl.col("engine_version") == v)
            counts = {p: seasonal.filter(pl.col("category") == p).height for p, _ in pathways}
            total = sum(counts.values())
            if total == 0:
                continue
            lines.append(f"### Season {v}\n")
            header = "| Metric |" + "".join(
                f" {label} (n={counts[code]}, {counts[code] / total * 100:.0f}%) |"
                for code, label in pathways
            ) + " Out − Time |"
            lines.append(header)
            lines.append("|---|" + "---:|" * (len(pathways) + 1))
            for field, label in MECHANIC_FIELDS:
                vals: dict[str, float | None] = {}
                for code, _ in pathways:
                    sub = seasonal.filter(pl.col("category") == code)
                    col_vals = sub[field].drop_nulls()
                    vals[code] = float(col_vals.mean()) if col_vals.len() else None
                as_pct = field in ("vote_accuracy", "vote_quality")
                cells = [_fmt(vals[c], as_pct=as_pct) for c, _ in pathways]
                o, t = vals["imp_outnumber"], vals["imp_time_limit"]
                if o is None or t is None:
                    diff = "—"
                else:
                    d = o - t
                    diff = f"{d * 100:+.1f} pts" if as_pct else f"{d:+.2f}"
                lines.append(f"| {label} | " + " | ".join(cells) + f" | {diff} |")
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--api-base",
        default=os.environ.get("ANALYSIS_API_BASE_URL", DEFAULT_API_BASE),
    )
    parser.add_argument(
        "--seasons",
        default="0,1",
        help="Comma-separated engine versions to pull (default: 0,1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Concurrent fetches against /games/{id}/logs (default: 12)",
    )
    parser.add_argument(
        "--cohort",
        choices=list(MODEL_COHORTS.keys()),
        default="all",
        help=(
            "Named cohort to restrict to. `featured` = 20-model paper set; "
            "`presentation` = 11-model slide set; `common` = intersection of "
            "models that appeared in every requested season, computed at "
            "runtime (the right control for cross-season claims). Default "
            "`all` = no filter."
        ),
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=1,
        help=(
            "When --cohort=common, a model must have played at least this many "
            "games in every season to enter the intersection (default 1). "
            "Raise to exclude tourist models that barely played."
        ),
    )
    parser.add_argument(
        "--cohort-match",
        choices=["all", "majority", "any"],
        default="majority",
        help=(
            "How strictly a game must match the cohort. `all` = every "
            "participant must be in cohort (strictest, usually sparse); "
            "`majority` = over half of participants (default, balanced); "
            "`any` = at least one cohort participant (most permissive)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="If set, writes per_game.csv, summary.csv, mechanics.csv here.",
    )
    parser.add_argument(
        "--markdown",
        default=None,
        help="If set, writes a markdown summary here.",
    )
    parser.add_argument(
        "--no-winner-split",
        action="store_true",
        help="Skip the per-winner mechanics tables (they're the largest).",
    )
    args = parser.parse_args()

    console = Console()
    versions = [int(v) for v in args.seasons.split(",") if v.strip()]

    if args.cohort == "common":
        cohort_set: set[str] | None = compute_common_cohort(
            args.api_base, versions, console, min_games=args.min_games
        )
        cohort_size = f"{len(cohort_set)} models (intersection)"
        if not cohort_set:
            console.print("[red]Common cohort is empty — aborting.[/red]")
            return
    else:
        cohort_ids = MODEL_COHORTS[args.cohort]
        cohort_set = set(cohort_ids) if cohort_ids is not None else None
        cohort_size = f"{len(cohort_ids)} models" if cohort_ids else "no filter"
    console.print(
        Panel.fit(
            f"[bold]Among Us — game-outcome & mechanics report[/bold]\n"
            f"api={args.api_base}  |  workers={args.workers}  |  "
            f"cohort=[cyan]{args.cohort}[/cyan] ({cohort_size})  |  "
            f"match=[cyan]{args.cohort_match}[/cyan]",
            border_style="cyan",
        )
    )
    frames: list[pl.DataFrame] = []
    for v in versions:
        df = collect_season(
            args.api_base,
            v,
            args.workers,
            console,
            cohort=cohort_set,
            match_policy=args.cohort_match,
        )
        if df.is_empty():
            console.print(f"  S{v}: 0 games in analysis (cohort filter excluded all)")
            continue
        console.print(
            f"  S{v}: {df.height} games in analysis, "
            f"{int(df.filter(pl.col('category') == 'unknown').height)} unclassified"
        )
        frames.append(df)

    if not frames:
        console.print(
            "[red]No games matched the cohort filter across any season.[/red] "
            "The `presentation` cohort excludes most S0 games because several "
            "of its models (GPT-5.4, Gemini 3.1 Pro, Nemotron 3 Super, Human "
            "Brain 1.0) weren't present in S0. Try `--cohort featured` or "
            "`--cohort all`."
        )
        return

    per_game = pl.concat(frames) if frames else pl.DataFrame()
    summary = outcome_summary(per_game)
    console.print()

    render_outcomes(summary, console)
    console.print()
    render_mechanics(per_game, console)
    console.print()
    if not args.no_winner_split:
        render_mechanics_by_winner(per_game, console)
        console.print()

    render_vote_decomposition(per_game, console)
    console.print()
    render_first_ejection(per_game, console)
    console.print()
    render_imp_pathway(per_game, console)
    console.print()

    cohort_tag = f"_{args.cohort}" if args.cohort != "all" else ""
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        per_game.write_csv(out / f"win_reasons_per_game{cohort_tag}.csv")
        summary.write_csv(out / f"win_reasons_summary{cohort_tag}.csv")
        mechanics_means(per_game, ["engine_version"]).write_csv(
            out / f"win_reasons_mechanics{cohort_tag}.csv"
        )
        console.print(
            f"[green]OK[/green] Wrote CSVs to {out} (suffix='{cohort_tag or '(none)'}')"
        )

    if args.markdown:
        md = Path(args.markdown)
        # If user passed a plain path and a cohort, tuck the cohort into the
        # filename so repeated runs don't overwrite one another.
        if cohort_tag and "_" not in md.stem.split(".")[0]:
            md = md.with_stem(md.stem + cohort_tag)
        write_markdown(md, per_game, summary)
        console.print(f"[green]OK[/green] Wrote markdown to {md}")


if __name__ == "__main__":
    main()
