"""Calculate OpenSkill ratings from experiment logs.

Supports two formats:
1. Old JSONL format: Each line is a game entry with "Game N" as key
2. New single-JSON format: One JSON object with all games, each with game_outcome

Replays all games chronologically using the meta-agent approach for symmetric
team-level rating updates despite asymmetric team sizes.

Usage:
    uv run calculate_ratings.py expt-logs/2026-02-11_exp_5/summary.json
    uv run calculate_ratings.py expt-logs/2026-02-21_exp_3/summary.json --theme light
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from openskill.models import PlackettLuce

RATING_MODEL = PlackettLuce()

DEFAULT_MU = 25.0
DEFAULT_SIGMA = DEFAULT_MU / 3  # 8.333...

OPENROUTER_ROUTE_SUFFIXES = (":free", ":floor", ":nitro")


# =============================================================================
# Core rating algorithm
# =============================================================================


def compute_meta_agent_update(
    team_a: list[tuple[float, float]],
    team_b: list[tuple[float, float]],
    team_a_won: bool,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Compute updated (mu, sigma) for each player using the meta-agent approach.

    Instead of rating teams directly (which causes asymmetry in rating changes
    due to OpenSkill's probability model treating larger teams as favored), we
    collapse each team to a single meta-agent with averaged mu/sigma, run a 1v1
    match, and distribute the symmetric deltas to all team members.

    The delta is distributed using variance-weighted redistribution: players
    with higher uncertainty (sigma) receive larger updates.

    Args:
        team_a: List of (mu, sigma) for team A players.
        team_b: List of (mu, sigma) for team B players.
        team_a_won: True if team A won, False if team B won.

    Returns:
        (updated_team_a, updated_team_b) — each a list of (mu, sigma) tuples
        in the same order as the input.
    """
    model = RATING_MODEL

    # Build OpenSkill rating objects
    os_a = [model.rating(mu=mu, sigma=sigma) for mu, sigma in team_a]
    os_b = [model.rating(mu=mu, sigma=sigma) for mu, sigma in team_b]

    # Create meta-agents (average mu, sqrt of avg variance for sigma)
    def _meta(team: list) -> object:
        avg_mu = sum(r.mu for r in team) / len(team)
        avg_var = sum(r.sigma**2 for r in team) / len(team)
        return model.rating(mu=avg_mu, sigma=math.sqrt(avg_var))

    meta_a = _meta(os_a)
    meta_b = _meta(os_b)

    # 1v1 match — lower rank is better
    ranks = [0, 1] if team_a_won else [1, 0]
    new_a, new_b = model.rate([[meta_a], [meta_b]], ranks=ranks)

    # Team-level deltas and sigma ratios
    mu_delta_a = new_a[0].mu - meta_a.mu
    mu_delta_b = new_b[0].mu - meta_b.mu
    sigma_ratio_a = new_a[0].sigma / meta_a.sigma
    sigma_ratio_b = new_b[0].sigma / meta_b.sigma

    # Distribute deltas with variance weighting
    def _distribute(
        os_team: list, mu_delta: float, sigma_ratio: float
    ) -> list[tuple[float, float]]:
        total_var = sum(r.sigma**2 for r in os_team)
        pool = mu_delta * len(os_team)
        result = []
        for r in os_team:
            share = r.sigma**2 / total_var
            new_mu = r.mu + pool * share
            new_sigma = max(0.1, r.sigma * sigma_ratio)
            result.append((new_mu, new_sigma))
        return result

    updated_a = _distribute(os_a, mu_delta_a, sigma_ratio_a)
    updated_b = _distribute(os_b, mu_delta_b, sigma_ratio_b)

    return updated_a, updated_b


# =============================================================================
# Data Model
# =============================================================================


@dataclass
class ModelRating:
    impostor_mu: float = DEFAULT_MU
    impostor_sigma: float = DEFAULT_SIGMA
    impostor_games: int = 0
    impostor_wins: int = 0
    crewmate_mu: float = DEFAULT_MU
    crewmate_sigma: float = DEFAULT_SIGMA
    crewmate_games: int = 0
    crewmate_wins: int = 0

    @property
    def total_games(self) -> int:
        return self.impostor_games + self.crewmate_games

    @property
    def total_wins(self) -> int:
        return self.impostor_wins + self.crewmate_wins

    @property
    def overall_mu(self) -> float:
        if self.total_games == 0:
            return DEFAULT_MU
        w_imp = self.impostor_games / self.total_games
        w_crew = self.crewmate_games / self.total_games
        return w_imp * self.impostor_mu + w_crew * self.crewmate_mu

    @property
    def overall_sigma(self) -> float:
        if self.total_games == 0:
            return DEFAULT_SIGMA
        w_imp = self.impostor_games / self.total_games
        w_crew = self.crewmate_games / self.total_games
        return w_imp * self.impostor_sigma + w_crew * self.crewmate_sigma

    @property
    def conservative_rating(self) -> float:
        return self.overall_mu - self.overall_sigma

    @property
    def win_rate(self) -> float:
        return self.total_wins / self.total_games if self.total_games else 0.0

    @property
    def impostor_win_rate(self) -> float:
        return self.impostor_wins / self.impostor_games if self.impostor_games else 0.0

    @property
    def crewmate_win_rate(self) -> float:
        return self.crewmate_wins / self.crewmate_games if self.crewmate_games else 0.0


# =============================================================================
# File I/O & Helpers
# =============================================================================


def canonical_model_id(model_id: str) -> str:
    for suffix in OPENROUTER_ROUTE_SUFFIXES:
        if model_id.endswith(suffix):
            model_id = model_id[: -len(suffix)]
            break
    return model_id


def scale(mu: float) -> int:
    """Scale OpenSkill mu to display-friendly integer (25 -> 2500)."""
    return round(mu * 100)


def detect_format(filepath: str) -> str:
    """Detect whether the file is JSONL (old) or single JSON (new) format."""
    with open(filepath) as f:
        content = f.read()
    stripped = content.lstrip()
    if stripped and stripped[0] == "{":
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                for key, val in data.items():
                    if key.startswith("Game ") and isinstance(val, dict):
                        if "game_outcome" in val:
                            return "new"
                return "jsonl"
        except json.JSONDecodeError:
            pass
    return "jsonl"


def load_games(filepath: str) -> list[dict]:
    """Load games from summary file, supporting both old JSONL and new single-JSON formats.

    Old format: JSONL where each line is {"Game N": {...}}
    New format: Single JSON object {"Game 1": {..., "game_outcome": {...}}, "Game 2": ...}
    """
    fmt = detect_format(filepath)
    games = []

    if fmt == "new":
        # New format: single JSON object with all games
        with open(filepath) as f:
            data = json.load(f)
        for game_id, game_data in data.items():
            if not game_id.startswith("Game "):
                continue
            num = int(game_id.split()[-1])
            game_data["_game_id"] = game_id
            game_data["_game_num"] = num
            games.append(game_data)
    else:
        # Old format: JSONL
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                for game_id, game_data in entry.items():
                    # Extract game number for sorting
                    num = int(game_id.split()[-1])
                    game_data["_game_id"] = game_id
                    game_data["_game_num"] = num
                    games.append(game_data)

    games.sort(key=lambda g: g["_game_num"])
    return games


def extract_players(game: dict) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Extract (model_id, player_name) tuples for impostors and crewmates."""
    impostors = []
    crewmates = []
    for key, val in game.items():
        if not key.startswith("Player "):
            continue
        if not isinstance(val, dict) or "identity" not in val:
            continue
        model = canonical_model_id(val["model"])
        name = val.get("name", key)
        if val["identity"] == "Impostor":
            impostors.append((model, name))
        else:
            crewmates.append((model, name))
    return impostors, crewmates


def build_ranked_data(ratings: dict[str, ModelRating]) -> list[dict]:
    """Build sorted list of ranking dicts from raw ratings."""
    ranked = []
    for model_id, r in ratings.items():
        short = model_id.split("/")[-1] if "/" in model_id else model_id
        ranked.append(
            {
                "model": model_id,
                "name": short,
                "rating": scale(r.conservative_rating),
                "overall_mu": scale(r.overall_mu),
                "overall_sigma": scale(r.overall_sigma),
                "imp_mu": scale(r.impostor_mu),
                "imp_sigma": scale(r.impostor_sigma),
                "crew_mu": scale(r.crewmate_mu),
                "crew_sigma": scale(r.crewmate_sigma),
                "games": r.total_games,
                "wins": r.total_wins,
                "win_rate": r.win_rate * 100,
                "imp_games": r.impostor_games,
                "imp_wins": r.impostor_wins,
                "imp_wr": r.impostor_win_rate * 100,
                "crew_games": r.crewmate_games,
                "crew_wins": r.crewmate_wins,
                "crew_wr": r.crewmate_win_rate * 100,
            }
        )
    ranked.sort(key=lambda x: x["rating"], reverse=True)
    return ranked


# =============================================================================
# Game Replay
# =============================================================================


def get_winner(game: dict) -> str | None:
    """Extract winner from game data, supporting both old and new formats.

    Old format: game["winner"] is 1 (impostors) or 0 (crewmates)
    New format: game["game_outcome"]["winner"] is "Impostors" or "Crewmates"

    Returns:
        "Impostors" or "Crewmates" or None if not found
    """
    # Old format: winner field directly on game
    if "winner" in game:
        w = game["winner"]
        if w == 1:
            return "Impostors"
        elif w == 0:
            return "Crewmates"
        return w  # Already a string

    # New format: game_outcome.winner
    if "game_outcome" in game:
        outcome = game["game_outcome"]
        if isinstance(outcome, dict) and "winner" in outcome:
            return outcome["winner"]

    return None


def replay_with_history(
    games: list[dict],
) -> tuple[dict[str, ModelRating], dict[str, list[tuple[int, float]]]]:
    """Replay games and also track rating history per model over time."""

    ratings: dict[str, ModelRating] = {}
    # model -> list of (game_num, conservative_rating_scaled)
    history: dict[str, list[tuple[int, float]]] = {}

    for game in games:
        winner = get_winner(game)
        if winner is None:
            continue

        game_num = game["_game_num"]
        impostors_won = winner == "Impostors"
        impostors, crewmates = extract_players(game)
        if not impostors or not crewmates:
            continue

        for model, _ in impostors + crewmates:
            if model not in ratings:
                ratings[model] = ModelRating()
                short = model.split("/")[-1] if "/" in model else model
                history[short] = [(0, scale(DEFAULT_MU - DEFAULT_SIGMA))]

        # Build (mu, sigma) inputs and keep parallel references
        impostor_data = []
        for model, name in impostors:
            r = ratings[model]
            impostor_data.append((model, r))

        crewmate_data = []
        for model, name in crewmates:
            r = ratings[model]
            crewmate_data.append((model, r))

        imp_team = [(r.impostor_mu, r.impostor_sigma) for _, r in impostor_data]
        crew_team = [(r.crewmate_mu, r.crewmate_sigma) for _, r in crewmate_data]

        updated_imp, updated_crew = compute_meta_agent_update(
            imp_team, crew_team, team_a_won=impostors_won
        )

        for (model, rating), (new_mu, new_sigma) in zip(impostor_data, updated_imp):
            rating.impostor_mu = new_mu
            rating.impostor_sigma = new_sigma
            rating.impostor_games += 1
            if impostors_won:
                rating.impostor_wins += 1

        for (model, rating), (new_mu, new_sigma) in zip(crewmate_data, updated_crew):
            rating.crewmate_mu = new_mu
            rating.crewmate_sigma = new_sigma
            rating.crewmate_games += 1
            if not impostors_won:
                rating.crewmate_wins += 1

        # Record snapshot for all models after this game
        for model, r in ratings.items():
            short = model.split("/")[-1] if "/" in model else model
            history[short].append((game_num, scale(r.conservative_rating)))

    return ratings, history


def print_rankings(ranked: list[dict], num_games: int) -> None:
    """Print a formatted leaderboard to stdout."""
    print()
    print("=" * 120)
    print(
        f"{'Rank':<5} {'Model':<42} {'Rating':>7} {'Overall':>8} {'+-':>5}"
        f"  {'Imp':>7} {'+-':>5} {'Crew':>7} {'+-':>5}"
        f"  {'Games':>5} {'WR%':>5} {'ImpWR%':>6} {'CrewWR%':>7}"
    )
    print("-" * 120)

    for i, r in enumerate(ranked, 1):
        name = r["name"]
        if len(name) > 40:
            name = name[:37] + "..."
        print(
            f"{i:<5} {name:<42} {r['rating']:>7}"
            f" {r['overall_mu']:>8} {r['overall_sigma']:>5}"
            f"  {r['imp_mu']:>7} {r['imp_sigma']:>5}"
            f" {r['crew_mu']:>7} {r['crew_sigma']:>5}"
            f"  {r['games']:>5} {r['win_rate']:>5.1f}"
            f" {r['imp_wr']:>6.1f} {r['crew_wr']:>7.1f}"
        )

    print("=" * 120)
    print(f"\nRating = mu - 1*sigma (scaled x100)  |  {num_games} games\n")


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

# Palette: assign a consistent color to each model for all charts
MODEL_PALETTE = {
    "gpt-5-mini": "#74aa9c",
    "kimi-k2.5": "#4a7dff",
    "mistral-large-2512": "#ff7000",
    "gemini-3-flash-preview": "#4285f4",
    "llama-3.3-70b-instruct": "#792ee5",
    "claude-sonnet-4.5": "#d4a27f",
    "qwen3-next-80b-a3b-thinking": "#1a7f64",
}

FALLBACK_PALETTE = sns.color_palette("husl", 10)

CHART_THEMES = {
    "dark": {
        "bg": "#0d1117",
        "panel": "#0d1117",
        "text": "#e6edf3",
        "muted": "#8b949e",
        "grid": "#21262d",
        "header_bg": "#161b22",
        "row_a": "#0d1117",
        "row_b": "#161b22",
        "edge": "#30363d",
        "fifty": "#6e7681",
        "overall": "#8b949e",
        "impostor": "#f85149",
        "crewmate": "#58a6ff",
        "seaborn_style": "darkgrid",
    },
    "light": {
        "bg": "#ffffff",
        "panel": "#ffffff",
        "text": "#1f2328",
        "muted": "#656d76",
        "grid": "#eaecef",
        "header_bg": "#2d2d2d",
        "row_a": "#ffffff",
        "row_b": "#f7f7f7",
        "edge": "#d0d7de",
        "fifty": "#999999",
        "overall": "#555555",
        "impostor": "#e74c3c",
        "crewmate": "#3498db",
        "seaborn_style": "whitegrid",
    },
}


def _color_for(name: str, idx: int) -> str:
    return MODEL_PALETTE.get(name, FALLBACK_PALETTE[idx % len(FALLBACK_PALETTE)])


def _apply_theme(theme: dict) -> None:
    """Apply matplotlib rcParams for the chosen theme."""
    sns.set_theme(style=theme["seaborn_style"], font_scale=1.05)
    plt.rcParams.update(
        {
            "figure.facecolor": theme["bg"],
            "axes.facecolor": theme["panel"],
            "axes.edgecolor": theme["edge"],
            "axes.labelcolor": theme["text"],
            "axes.titlecolor": theme["text"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["text"],
            "text.color": theme["text"],
            "grid.color": theme["grid"],
            "savefig.facecolor": theme["bg"],
            "savefig.edgecolor": theme["bg"],
        }
    )


def plot_leaderboard_table(
    ranked: list[dict], num_games: int, out_path: str, theme: dict
) -> None:
    """Render a publication-quality leaderboard table as an image."""
    fig, ax = plt.subplots(figsize=(14, 0.6 * len(ranked) + 1.8))
    ax.axis("off")

    columns = [
        "Rank",
        "Model",
        "Rating",
        "Overall",
        "\u03c3",
        "Imp Rating",
        "Imp \u03c3",
        "Crew Rating",
        "Crew \u03c3",
        "Games",
        "Win%",
        "Imp W%",
        "Crew W%",
    ]
    cell_data = []
    row_colors = []
    for i, r in enumerate(ranked, 1):
        cell_data.append(
            [
                str(i),
                r["name"],
                f"{r['rating']:,}",
                f"{r['overall_mu']:,}",
                f"{r['overall_sigma']:,}",
                f"{r['imp_mu']:,}",
                f"{r['imp_sigma']:,}",
                f"{r['crew_mu']:,}",
                f"{r['crew_sigma']:,}",
                str(r["games"]),
                f"{r['win_rate']:.1f}",
                f"{r['imp_wr']:.1f}",
                f"{r['crew_wr']:.1f}",
            ]
        )
        row_colors.append(theme["row_b"] if i % 2 == 0 else theme["row_a"])

    table = ax.table(
        cellText=cell_data,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor(theme["header_bg"])
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        cell.set_edgecolor(theme["edge"])

    # Style rows
    for i in range(len(ranked)):
        color = _color_for(ranked[i]["name"], i)
        for j in range(len(columns)):
            cell = table[i + 1, j]
            cell.set_facecolor(row_colors[i])
            cell.set_edgecolor(theme["edge"])
            if j == 1:  # Model name column
                cell.set_text_props(color=color, fontweight="bold", ha="left")
            elif j == 2:  # Rating column
                cell.set_text_props(color=theme["text"], fontweight="bold")
            else:
                cell.set_text_props(color=theme["text"])

    ax.set_title(
        f"OpenSkill Leaderboard  ({num_games} games, mu - 1\u03c3)",
        fontsize=14,
        fontweight="bold",
        pad=20,
        color=theme["text"],
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=theme["bg"])
    plt.close(fig)
    print(f"  Saved table: {out_path}")


def plot_rating_bars(
    ranked: list[dict], num_games: int, out_path: str, theme: dict
) -> None:
    """Horizontal bar chart: overall rating with impostor/crewmate breakdown."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    names = [r["name"] for r in ranked][::-1]  # bottom-to-top
    n = len(names)

    panels = [
        ("Overall Rating", "overall_mu", "overall_sigma", axes[0]),
        ("Impostor Rating", "imp_mu", "imp_sigma", axes[1]),
        ("Crewmate Rating", "crew_mu", "crew_sigma", axes[2]),
    ]

    for title, mu_key, sigma_key, ax in panels:
        mus = [r[mu_key] for r in ranked][::-1]
        sigmas = [r[sigma_key] for r in ranked][::-1]
        colors = [_color_for(names[i], n - 1 - i) for i in range(n)]

        ax.barh(
            range(n),
            mus,
            xerr=sigmas,
            color=colors,
            edgecolor=theme["bg"],
            linewidth=0.8,
            capsize=3,
            error_kw={
                "elinewidth": 1.2,
                "capthick": 1.2,
                "ecolor": theme["muted"],
            },
        )

        for j, (mu_val, sig_val) in enumerate(zip(mus, sigmas)):
            ax.text(
                mu_val + sig_val + 30,
                j,
                f"{mu_val:,}",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=theme["text"],
            )

        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=10, color=theme["text"])
        ax.set_title(title, fontsize=12, fontweight="bold", color=theme["text"])
        ax.set_xlabel("Rating (mu x100)", color=theme["muted"])
        ax.axvline(
            x=2500, color=theme["fifty"], linestyle="--", linewidth=0.8, alpha=0.6
        )
        ax.set_facecolor(theme["panel"])
        for spine in ax.spines.values():
            spine.set_color(theme["edge"])

    fig.suptitle(
        f"Model Ratings Breakdown  ({num_games} games)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
        color=theme["text"],
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=theme["bg"])
    plt.close(fig)
    print(f"  Saved bars: {out_path}")


def plot_rating_history(
    history: dict[str, list[tuple[int, float]]],
    ranked: list[dict],
    num_games: int,
    out_path: str,
    theme: dict,
) -> None:
    """Line chart showing rating convergence over games."""
    fig, ax = plt.subplots(figsize=(12, 6))

    rank_order = [r["name"] for r in ranked]

    for i, name in enumerate(rank_order):
        if name not in history:
            continue
        data = history[name]
        xs = [d[0] for d in data]
        ys = [d[1] for d in data]
        color = _color_for(name, i)
        ax.plot(xs, ys, label=name, color=color, linewidth=2.0, alpha=0.88)

    ax.axhline(
        y=2500,
        color=theme["fifty"],
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
        label="Starting rating",
    )
    ax.set_xlabel("Game Number", fontsize=11, color=theme["muted"])
    ax.set_ylabel(
        "Conservative Rating (mu - 1\u03c3, x100)", fontsize=11, color=theme["muted"]
    )
    ax.set_title(
        f"Rating Convergence Over {num_games} Games",
        fontsize=14,
        fontweight="bold",
        color=theme["text"],
    )
    legend = ax.legend(loc="best", fontsize=9, framealpha=0.9)
    legend.get_frame().set_facecolor(theme["panel"])
    legend.get_frame().set_edgecolor(theme["edge"])
    for text in legend.get_texts():
        text.set_color(theme["text"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_facecolor(theme["panel"])
    for spine in ax.spines.values():
        spine.set_color(theme["edge"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=theme["bg"])
    plt.close(fig)
    print(f"  Saved history: {out_path}")


def plot_win_rates(
    ranked: list[dict], num_games: int, out_path: str, theme: dict
) -> None:
    """Grouped bar chart comparing overall / impostor / crewmate win rates."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 5.5))

    names = [r["name"] for r in ranked]
    n = len(names)
    x = np.arange(n)
    width = 0.25

    overall = [r["win_rate"] for r in ranked]
    imp = [r["imp_wr"] for r in ranked]
    crew = [r["crew_wr"] for r in ranked]

    bars1 = ax.bar(
        x - width, overall, width, label="Overall", color=theme["overall"], alpha=0.9
    )
    bars2 = ax.bar(
        x, imp, width, label="Impostor", color=theme["impostor"], alpha=0.9
    )
    bars3 = ax.bar(
        x + width, crew, width, label="Crewmate", color=theme["crewmate"], alpha=0.9
    )

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.8,
                f"{h:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=theme["text"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        names, rotation=25, ha="right", fontsize=9, color=theme["text"]
    )
    ax.set_ylabel("Win Rate (%)", color=theme["muted"])
    ax.set_ylim(0, 100)
    ax.set_title(
        f"Win Rates by Role  ({num_games} games)",
        fontsize=14,
        fontweight="bold",
        color=theme["text"],
    )
    legend = ax.legend(fontsize=10)
    legend.get_frame().set_facecolor(theme["panel"])
    legend.get_frame().set_edgecolor(theme["edge"])
    for text in legend.get_texts():
        text.set_color(theme["text"])
    ax.set_facecolor(theme["panel"])
    for spine in ax.spines.values():
        spine.set_color(theme["edge"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=theme["bg"])
    plt.close(fig)
    print(f"  Saved win rates: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", help="Path to summary.json from an experiment run")
    parser.add_argument(
        "--theme",
        choices=["dark", "light"],
        default="dark",
        help="dark = presentation, light = paper",
    )
    parser.add_argument(
        "--out-dir",
        help="Directory to write charts into (default: summary file's folder)",
    )
    args = parser.parse_args()

    filepath = args.summary
    out_dir = args.out_dir or str(Path(filepath).parent)
    theme = CHART_THEMES[args.theme]
    _apply_theme(theme)
    suffix = f"_{args.theme}"

    games = load_games(filepath)
    num_games = len(games)
    print(f"Loaded {num_games} games from {filepath}")

    ratings, history = replay_with_history(games)
    ranked = build_ranked_data(ratings)

    print_rankings(ranked, num_games)

    print("Exporting charts...")
    plot_leaderboard_table(
        ranked,
        num_games,
        os.path.join(out_dir, f"leaderboard_table{suffix}.png"),
        theme,
    )
    plot_rating_bars(
        ranked, num_games, os.path.join(out_dir, f"rating_bars{suffix}.png"), theme
    )
    plot_rating_history(
        history,
        ranked,
        num_games,
        os.path.join(out_dir, f"rating_history{suffix}.png"),
        theme,
    )
    plot_win_rates(
        ranked, num_games, os.path.join(out_dir, f"win_rates{suffix}.png"), theme
    )
    print("Done!")


if __name__ == "__main__":
    main()
