"""Compare named judge panels against the checked-in human gold labels."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from .judge_agreement import (
        BEHAVIOR_NAMES,
        CANDIDATE_BY_KEY,
        DEFAULT_HUMAN_CSV,
        DEFAULT_OUTPUT_DIR,
        JUDGE_PANELS,
        _config_hash,
        _portable_path,
        _safe_file_part,
        _write_csv,
        _write_json,
        cohen_kappa,
        load_human_gold,
    )
else:
    from judge_agreement import (
        BEHAVIOR_NAMES,
        CANDIDATE_BY_KEY,
        DEFAULT_HUMAN_CSV,
        DEFAULT_OUTPUT_DIR,
        JUDGE_PANELS,
        _config_hash,
        _portable_path,
        _safe_file_part,
        _write_csv,
        _write_json,
        cohen_kappa,
        load_human_gold,
    )


def krippendorff_alpha_nominal(ratings_by_judge: list[list[bool]]) -> float:
    """Compute nominal Krippendorff alpha for complete Boolean ratings."""
    if len(ratings_by_judge) < 2 or not ratings_by_judge[0]:
        return math.nan
    width = len(ratings_by_judge[0])
    if any(len(row) != width for row in ratings_by_judge):
        raise ValueError("Judge rating rows have different lengths")

    observed_disagreements = 0
    observed_pairs = 0
    yes_total = 0
    rating_total = 0
    for item in zip(*ratings_by_judge, strict=True):
        yes = sum(item)
        no = len(item) - yes
        observed_disagreements += 2 * yes * no
        observed_pairs += len(item) * (len(item) - 1)
        yes_total += yes
        rating_total += len(item)
    if observed_pairs == 0 or rating_total < 2:
        return math.nan
    observed = observed_disagreements / observed_pairs
    no_total = rating_total - yes_total
    expected = (2 * yes_total * no_total) / (rating_total * (rating_total - 1))
    if math.isclose(expected, 0.0):
        return math.nan
    return 1 - observed / expected


def _load_candidate_results(
    output_dir: Path, candidate_key: str, units: list[Any]
) -> dict[tuple[str, int], dict[str, bool]]:
    candidate = CANDIDATE_BY_KEY[candidate_key]
    expected_config_hash = _config_hash(candidate)
    loaded: dict[tuple[str, int], dict[str, bool]] = {}
    for unit in units:
        path = (
            output_dir
            / "raw"
            / candidate_key
            / f"{_safe_file_part(unit.unit_id)}.json"
        )
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "ok":
            continue
        if result.get("config_hash") != expected_config_hash:
            raise ValueError(f"Stale cached configuration: {path}")
        predictions = result.get("predictions", {})
        if set(predictions) != set(BEHAVIOR_NAMES):
            raise ValueError(f"Incomplete predictions: {path}")
        loaded[(unit.game_id, unit.player_num)] = {
            behavior: bool(predictions[behavior]) for behavior in BEHAVIOR_NAMES
        }
    return loaded


def score_panels(
    human_csv: Path, output_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return panel summaries, per-behavior metrics, and audit predictions."""
    units = load_human_gold(human_csv)
    needed_candidates = sorted({key for panel in JUDGE_PANELS.values() for key in panel})
    predictions = {
        key: _load_candidate_results(output_dir, key, units) for key in needed_candidates
    }

    summaries: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for panel_name, judge_keys in JUDGE_PANELS.items():
        complete_units = [
            unit
            for unit in units
            if all((unit.game_id, unit.player_num) in predictions[key] for key in judge_keys)
        ]
        pooled_human: list[bool] = []
        pooled_majority: list[bool] = []
        pooled_judges: dict[str, list[bool]] = {key: [] for key in judge_keys}

        for behavior_index, behavior in enumerate(BEHAVIOR_NAMES):
            human_values: list[bool] = []
            majority_values: list[bool] = []
            judge_values: dict[str, list[bool]] = {key: [] for key in judge_keys}
            for unit in complete_units:
                identity = (unit.game_id, unit.player_num)
                votes = [predictions[key][identity][behavior] for key in judge_keys]
                majority = sum(votes) > len(votes) / 2
                human = unit.labels[behavior_index]
                for key, vote in zip(judge_keys, votes, strict=True):
                    judge_values[key].append(vote)
                    pooled_judges[key].append(vote)
                if human is not None:
                    human_values.append(human)
                    majority_values.append(majority)
                    pooled_human.append(human)
                    pooled_majority.append(majority)
                audit_rows.append(
                    {
                        "panel": panel_name,
                        "game_id": unit.game_id,
                        "player_num": unit.player_num,
                        "behavior": behavior,
                        "human": human,
                        "majority": majority,
                        "yes_votes": sum(votes),
                        "judge_count": len(votes),
                    }
                )
            behavior_rows.append(
                {
                    "panel": panel_name,
                    "behavior": behavior,
                    "cohen_kappa": cohen_kappa(human_values, majority_values),
                    "agreement": (
                        sum(a == b for a, b in zip(human_values, majority_values, strict=True))
                        / len(human_values)
                        if human_values
                        else math.nan
                    ),
                    "krippendorff_alpha": krippendorff_alpha_nominal(
                        [judge_values[key] for key in judge_keys]
                    ),
                    "label_pairs": len(human_values),
                }
            )

        summaries.append(
            {
                "panel": panel_name,
                "judges": ",".join(judge_keys),
                "judge_count": len(judge_keys),
                "cohen_kappa": cohen_kappa(pooled_human, pooled_majority),
                "agreement": (
                    sum(a == b for a, b in zip(pooled_human, pooled_majority, strict=True))
                    / len(pooled_human)
                    if pooled_human
                    else math.nan
                ),
                "krippendorff_alpha": krippendorff_alpha_nominal(
                    [pooled_judges[key] for key in judge_keys]
                ),
                "label_pairs": len(pooled_human),
                "units_complete": len(complete_units),
                "units_total": len(units),
                "unit_coverage": len(complete_units) / len(units),
            }
        )
    return summaries, behavior_rows, audit_rows


def _write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Judge panel human-agreement ablation",
        "",
        "All panels use the same 38 human-rated player-game rows and the same 13-label rubric.",
        "Provider pins and sampling settings are recorded in `run_manifest.json`.",
        "",
        "| Panel | Judges | Human κ | Agreement | Inter-judge α | Coverage |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        judges = row["judges"].replace(",", ", ")
        lines.append(
            f"| {row['panel']} | {judges} | {row['cohen_kappa']:.3f} | "
            f"{row['agreement']:.1%} | {row['krippendorff_alpha']:.3f} | "
            f"{row['unit_coverage']:.1%} |"
        )
    lines.extend(
        [
            "",
            "- `matched_original3` updates the original Gemini–GLM–Claude composition on the matched gold set.",
            "- `expanded5` adds Inkling and Grok to test whether the result survives a broader panel.",
            "- `agreement_top3` uses the three best complete candidates; none is an exact acting model.",
            "- `noncohort3` removes Gemini and Claude, whose labs also appear in the acting cohort.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-csv", type=Path, default=DEFAULT_HUMAN_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summaries, behavior_rows, audit_rows = score_panels(args.human_csv, args.output_dir)
    _write_csv(args.output_dir / "panel_summary.csv", summaries)
    _write_csv(args.output_dir / "panel_behavior_metrics.csv", behavior_rows)
    _write_csv(args.output_dir / "panel_predictions.csv", audit_rows)
    _write_json(
        args.output_dir / "panel_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "human_csv": _portable_path(args.human_csv),
            "panels": JUDGE_PANELS,
        },
    )
    _write_markdown(args.output_dir / "panel_report.md", summaries)
    for row in summaries:
        print(
            f"{row['panel']}: kappa={row['cohen_kappa']:.3f}, "
            f"agreement={row['agreement']:.1%}, "
            f"alpha={row['krippendorff_alpha']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
