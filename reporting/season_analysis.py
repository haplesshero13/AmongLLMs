"""Fetch live season leaderboard data and print a markdown comparison summary."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from collections import Counter
from statistics import mean, median

DEFAULT_API_BASE = "https://api.sdgarena.averyyen.dev"


def canonical_provider(name: str) -> str:
    aliases = {
        "MiniMax": "Minimax",
        "Z-AI": "Z.AI",
    }
    return aliases.get(name, name)


def fetch_json(url: str) -> object:
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def conservative_score(model: dict) -> int:
    return int(model["overall_rating"] - model["overall_sigma"])


def format_table(headers: list[str], rows: list[list[object]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt(row: list[object]) -> str:
        return (
            "| "
            + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
            + " |"
        )

    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [fmt(headers), divider]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def summarize(base_url: str) -> str:
    seasons: list[dict] = fetch_json(f"{base_url}/api/seasons")  # type: ignore[assignment]
    season_data: dict[int, list[dict]] = {}
    for season in seasons:
        version = season["version"]
        leaderboard = fetch_json(
            f"{base_url}/api/leaderboard?page=1&per_page=100&engine_version={version}"
        )
        season_data[version] = leaderboard["data"]

    s0 = {model["model_id"]: model for model in season_data[0]}
    s1 = {model["model_id"]: model for model in season_data[1]}
    overlap_ids = sorted(set(s0) & set(s1))

    overlap_rows = []
    for model_id in overlap_ids:
        model0 = s0[model_id]
        model1 = s1[model_id]
        gap0 = int(abs(model0["impostor_rating"] - model0["crewmate_rating"]))
        gap1 = int(abs(model1["impostor_rating"] - model1["crewmate_rating"]))
        overlap_rows.append(
            {
                "model_id": model_id,
                "model_name": model1["model_name"],
                "provider": canonical_provider(model1["provider"]),
                "s0_rank": int(model0["current_rank"]),
                "s1_rank": int(model1["current_rank"]),
                "global_rank_change": int(
                    model0["current_rank"] - model1["current_rank"]
                ),
                "skill_change": int(
                    model1["overall_rating"] - model0["overall_rating"]
                ),
                "s0_cons": conservative_score(model0),
                "s1_cons": conservative_score(model1),
                "cons_change": conservative_score(model1) - conservative_score(model0),
                "s0_gap": gap0,
                "s1_gap": gap1,
                "gap_change": gap1 - gap0,
            }
        )

    s0_overlap_rank = {
        row["model_id"]: i
        for i, row in enumerate(
            sorted(overlap_rows, key=lambda row: row["s0_rank"]), start=1
        )
    }
    s1_overlap_rank = {
        row["model_id"]: i
        for i, row in enumerate(
            sorted(overlap_rows, key=lambda row: row["s1_rank"]), start=1
        )
    }
    for row in overlap_rows:
        row["s0_relative_rank"] = s0_overlap_rank[row["model_id"]]
        row["s1_relative_rank"] = s1_overlap_rank[row["model_id"]]
        row["relative_rank_change"] = row["s0_relative_rank"] - row["s1_relative_rank"]

    avg_abs_overlap_rank_change = mean(
        abs(s0_overlap_rank[row["model_id"]] - s1_overlap_rank[row["model_id"]])
        for row in overlap_rows
    )
    median_abs_overlap_rank_change = median(
        abs(s0_overlap_rank[row["model_id"]] - s1_overlap_rank[row["model_id"]])
        for row in overlap_rows
    )
    avg_abs_rank_change = mean(abs(row["global_rank_change"]) for row in overlap_rows)
    moved_at_least_5 = sum(
        1 for row in overlap_rows if abs(row["relative_rank_change"]) >= 5
    )
    top5_s0 = {row["model_id"] for row in overlap_rows if row["s0_relative_rank"] <= 5}
    top5_s1 = {row["model_id"] for row in overlap_rows if row["s1_relative_rank"] <= 5}
    top10_s0 = {
        row["model_id"] for row in overlap_rows if row["s0_relative_rank"] <= 10
    }
    top10_s1 = {
        row["model_id"] for row in overlap_rows if row["s1_relative_rank"] <= 10
    }

    biggest_climbers = sorted(
        overlap_rows,
        key=lambda row: (
            row["relative_rank_change"],
            row["skill_change"],
            row["cons_change"],
        ),
        reverse=True,
    )[:8]
    biggest_fallers = sorted(
        overlap_rows,
        key=lambda row: (
            row["relative_rank_change"],
            row["skill_change"],
            row["cons_change"],
        ),
    )[:8]
    more_balanced = sorted(
        overlap_rows, key=lambda row: (row["gap_change"], row["cons_change"])
    )[:8]
    more_specialized = sorted(
        overlap_rows,
        key=lambda row: (row["gap_change"], row["cons_change"]),
        reverse=True,
    )[:8]

    def season_top_rows(version: int, limit: int = 10) -> list[list[object]]:
        rows = []
        for model in season_data[version][:limit]:
            rows.append(
                [
                    model["current_rank"],
                    model["model_name"],
                    canonical_provider(model["provider"]),
                    conservative_score(model),
                    model["win_rate"],
                    f"{model['impostor_win_rate']}/{model['crewmate_win_rate']}",
                ]
            )
        return rows

    top10_provider_counts = {
        version: Counter(
            canonical_provider(model["provider"]) for model in season_data[version][:10]
        )
        for version in (0, 1)
    }

    new_in_s1 = [
        s1[model_id]
        for model_id in sorted(
            set(s1) - set(s0), key=lambda mid: s1[mid]["current_rank"]
        )
    ]
    dropped_after_s0 = [
        s0[model_id]
        for model_id in sorted(
            set(s0) - set(s1), key=lambda mid: s0[mid]["current_rank"]
        )
    ]

    sections = [
        "# Season Comparison Snapshot",
        "",
        "## Coverage",
        f"- Season 0 models ranked: {len(season_data[0])}",
        f"- Season 1 models ranked: {len(season_data[1])}",
        f"- Overlap across both seasons: {len(overlap_rows)}",
        f"- Season 0 completed games: {next(season['game_count'] for season in seasons if season['version'] == 0)}",
        f"- Season 1 completed games: {next(season['game_count'] for season in seasons if season['version'] == 1)}",
        "",
        "## Stability",
        f"- Mean absolute overlap-rank change: {avg_abs_overlap_rank_change:.2f}",
        f"- Median absolute overlap-rank change: {median_abs_overlap_rank_change:.2f}",
        f"- Models moving at least 5 relative slots: {moved_at_least_5} of {len(overlap_rows)}",
        f"- Top-5 overlap within the shared pool: {len(top5_s0 & top5_s1)} of 5",
        f"- Top-10 overlap within the shared pool: {len(top10_s0 & top10_s1)} of 10",
        f"- Mean absolute full-leaderboard rank change: {avg_abs_rank_change:.2f}",
        "",
        "## Top 10: Season 0",
        format_table(
            ["Rank", "Model", "Provider", "Conservative", "Win %", "Imp/Crew %"],
            season_top_rows(0),
        ),
        "",
        "## Top 10: Season 1",
        format_table(
            ["Rank", "Model", "Provider", "Conservative", "Win %", "Imp/Crew %"],
            season_top_rows(1),
        ),
        "",
        "## Biggest Climbers Into Season 1",
        format_table(
            ["Model", "Rel S0", "Rel S1", "Rel Delta", "Skill Delta", "Cons Delta"],
            [
                [
                    row["model_name"],
                    row["s0_relative_rank"],
                    row["s1_relative_rank"],
                    f"{row['relative_rank_change']:+d}",
                    f"{row['skill_change']:+d}",
                    f"{row['cons_change']:+d}",
                ]
                for row in biggest_climbers
            ],
        ),
        "",
        "## Biggest Fallers Into Season 1",
        format_table(
            ["Model", "Rel S0", "Rel S1", "Rel Delta", "Skill Delta", "Cons Delta"],
            [
                [
                    row["model_name"],
                    row["s0_relative_rank"],
                    row["s1_relative_rank"],
                    f"{row['relative_rank_change']:+d}",
                    f"{row['skill_change']:+d}",
                    f"{row['cons_change']:+d}",
                ]
                for row in biggest_fallers
            ],
        ),
        "",
        "## Most More Balanced In Season 1",
        format_table(
            ["Model", "Gap S0", "Gap S1", "Gap Delta", "Cons Delta"],
            [
                [
                    row["model_name"],
                    row["s0_gap"],
                    row["s1_gap"],
                    f"{row['gap_change']:+d}",
                    f"{row['cons_change']:+d}",
                ]
                for row in more_balanced
            ],
        ),
        "",
        "## Most More Specialized In Season 1",
        format_table(
            ["Model", "Gap S0", "Gap S1", "Gap Delta", "Cons Delta"],
            [
                [
                    row["model_name"],
                    row["s0_gap"],
                    row["s1_gap"],
                    f"{row['gap_change']:+d}",
                    f"{row['cons_change']:+d}",
                ]
                for row in more_specialized
            ],
        ),
        "",
        "## Provider Mix In Top 10",
        f"- Season 0: {dict(top10_provider_counts[0])}",
        f"- Season 1: {dict(top10_provider_counts[1])}",
        "",
        "## New In Season 1",
        format_table(
            ["Rank", "Model", "Provider", "Games", "Conservative"],
            [
                [
                    model["current_rank"],
                    model["model_name"],
                    canonical_provider(model["provider"]),
                    model["games_played"],
                    conservative_score(model),
                ]
                for model in new_in_s1
            ]
            or [["-", "-", "-", "-", "-"]],
        ),
        "",
        "## Missing From Season 1",
        format_table(
            ["Rank", "Model", "Provider", "Games", "Conservative"],
            [
                [
                    model["current_rank"],
                    model["model_name"],
                    canonical_provider(model["provider"]),
                    model["games_played"],
                    conservative_score(model),
                ]
                for model in dropped_after_s0[:12]
            ]
            or [["-", "-", "-", "-", "-"]],
        ),
    ]

    return "\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-base",
        default=os.environ.get("ANALYSIS_API_BASE_URL", DEFAULT_API_BASE),
        help="Base URL for the live API",
    )
    args = parser.parse_args()
    print(summarize(args.api_base), end="")


if __name__ == "__main__":
    main()
