import csv
import json

import pytest

from LLM_judge.judge_agreement import (
    ACTING_COHORT_MODELS,
    BEHAVIOR_NAMES,
    CANDIDATES,
    DEFAULT_HUMAN_CSV,
    EXCLUDED_CANDIDATES,
    HUMAN_PRESENT_COLUMNS,
    PAPER_BENCHMARKS,
    build_game_context,
    cohen_kappa,
    load_human_gold,
    make_player_prompt,
    parse_judge_predictions,
    rank_summaries,
)
from LLM_judge.judge_panel_report import krippendorff_alpha_nominal


def test_every_candidate_has_one_strict_provider_pin():
    assert len(CANDIDATES) == 14
    assert len({candidate.model for candidate in CANDIDATES}) == 14
    assert not ({candidate.model for candidate in CANDIDATES} & ACTING_COHORT_MODELS)

    for candidate in CANDIDATES:
        provider = candidate.request_parameters()["provider"]
        assert provider == {
            "only": [candidate.provider],
            "order": [candidate.provider],
            "allow_fallbacks": False,
            "require_parameters": True,
        }

    assert {candidate.provider for candidate in PAPER_BENCHMARKS} == {
        "anthropic",
        "google-ai-studio",
    }
    assert {
        candidate.model for candidate in PAPER_BENCHMARKS
    } <= ACTING_COHORT_MODELS
    assert EXCLUDED_CANDIDATES[0]["candidate"] == "laguna-s-2.1"


def test_candidate_sampling_settings_are_model_specific():
    candidates = {candidate.key: candidate for candidate in CANDIDATES}

    assert candidates["inkling-small"].sampling == {}
    assert candidates["longcat-2.0"].sampling == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
    }
    assert candidates["hy3"].sampling == {"temperature": 0.9, "top_p": 1.0}
    assert candidates["nex-n2-pro"].sampling["temperature"] == 0.7
    assert candidates["qwen3.7-plus"].sampling["temperature"] == 1.0
    assert candidates["qwen3.7-plus"].reasoning == {"enabled": False}
    assert candidates["grok-4.3"].reasoning == {"effort": "low"}
    assert candidates["claude-sonnet-5"].sampling == {}
    assert candidates["claude-sonnet-5"].reasoning == {"effort": "low"}
    assert candidates["grok-4.5"].sampling == {
        "temperature": 1.0,
        "top_p": 1.0,
    }
    assert candidates["deepseek-v4-flash-0731"].reasoning == {"effort": "low"}
    assert candidates["muse-spark-1.1"].reasoning == {"effort": "low"}


def test_load_human_gold_maps_yes_no_and_blank(tmp_path):
    path = tmp_path / "gold.csv"
    fieldnames = ["Game ID", "Player #", *HUMAN_PRESENT_COLUMNS]
    row = {column: "No" for column in HUMAN_PRESENT_COLUMNS}
    row.update({"Game ID": "game-1", "Player #": "2", "Present?": "Yes"})
    row["Present?.12"] = ""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    units = load_human_gold(path)

    assert len(units) == 1
    assert units[0].game_id == "game-1"
    assert units[0].player_num == 2
    assert units[0].labels[0] is True
    assert units[0].labels[1] is False
    assert units[0].labels[12] is None


def test_checked_in_human_gold_loads_all_corrected_rows():
    units = load_human_gold(DEFAULT_HUMAN_CSV)

    assert len(units) == 38
    assert len({unit.game_id for unit in units}) == 6


def test_parse_judge_predictions_repairs_missing_outer_array():
    response = ",\n".join(
        json.dumps({"behavior": behavior, "present": index % 2 == 0})
        for index, behavior in enumerate(BEHAVIOR_NAMES)
    )

    predictions = parse_judge_predictions(response)

    assert list(predictions) == list(BEHAVIOR_NAMES)
    assert predictions[BEHAVIOR_NAMES[0]] is True
    assert predictions[BEHAVIOR_NAMES[1]] is False


def test_load_human_gold_accepts_duplicate_google_forms_headers(tmp_path):
    path = tmp_path / "forms-export.csv"
    header = ["Timestamp", "Game ID", "Player #"]
    for _ in BEHAVIOR_NAMES:
        header.extend(["Present?", "Justification"])
    header.extend(
        [
            "Present?",
            "If yes, total number of self-incriminating statements?",
        ]
    )
    row = ["2026-05-20", "game-2", "4"]
    for index, _ in enumerate(BEHAVIOR_NAMES):
        row.extend(["Yes" if index % 2 == 0 else "No", "example"])
    row.extend(["Yes", "1"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow(row)

    units = load_human_gold(path)

    assert units[0].game_id == "game-2"
    assert units[0].player_num == 4
    assert units[0].labels == tuple(index % 2 == 0 for index in range(13))


def test_load_human_gold_identifies_game_manifest(tmp_path):
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["game_id", "context_mode", "winner_reason"])
        writer.writerow(["game-1", "full", "tasks"])

    with pytest.raises(ValueError, match="game manifest CSV"):
        load_human_gold(path)


def test_load_human_gold_warns_and_skips_row_without_game_id(tmp_path, capsys):
    path = tmp_path / "gold.csv"
    fieldnames = ["Game ID", "Player #", *HUMAN_PRESENT_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "Game ID": " ",
                "Player #": "7",
                **{column: "No" for column in HUMAN_PRESENT_COLUMNS},
            }
        )
        writer.writerow(
            {
                "Game ID": "game-3",
                "Player #": "1",
                **{column: "No" for column in HUMAN_PRESENT_COLUMNS},
            }
        )

    units = load_human_gold(path)

    assert [unit.game_id for unit in units] == ["game-3"]
    assert "skipping human CSV row 2: no Game ID" in capsys.readouterr().err


def test_load_human_gold_repairs_known_source_rows(tmp_path, capsys):
    path = tmp_path / "gold.csv"
    fieldnames = ["Game ID", "Player #", "Model Name", *HUMAN_PRESENT_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "Game ID": " ",
                "Player #": "7",
                "Model Name": "llama-3.3-70b",
                **{column: "No" for column in HUMAN_PRESENT_COLUMNS},
            }
        )
        writer.writerow(
            {
                "Game ID": "78d9f3ca-9438-4b96-9cee-506e1413b2f11",
                "Player #": "3",
                "Model Name": "nemotron-3-super",
                **{column: "No" for column in HUMAN_PRESENT_COLUMNS},
            }
        )

    units = load_human_gold(path)

    assert len(units) == 2
    assert {unit.game_id for unit in units} == {
        "78d9f3ca-9438-4b96-9cee-506e1413b2f8"
    }
    warnings = capsys.readouterr().err
    assert "filled missing Game ID" in warnings
    assert "corrected Game ID" in warnings


def test_parse_judge_predictions_accepts_fenced_array_and_ignores_metadata():
    response = [
        {"behavior": behavior, "present": index % 2 == 0, "justification": "x"}
        for index, behavior in enumerate(BEHAVIOR_NAMES)
    ]
    response.extend(
        [
            {"behavior": "utterance_count", "value": 3},
            {"behavior": "words_per_turn", "value": 4.5},
        ]
    )

    parsed = parse_judge_predictions(f"```json\n{json.dumps(response)}\n```")

    assert list(parsed) == list(BEHAVIOR_NAMES)
    assert parsed[BEHAVIOR_NAMES[0]] is True
    assert parsed[BEHAVIOR_NAMES[1]] is False


def test_parse_judge_predictions_rejects_missing_behavior():
    response = [
        {"behavior": behavior, "present": False}
        for behavior in BEHAVIOR_NAMES[:-1]
    ]

    with pytest.raises(ValueError, match="missing behaviors"):
        parse_judge_predictions(json.dumps(response))


def test_cohen_kappa_and_ranking_use_kappa_first():
    assert cohen_kappa([True, True, False, False], [True, True, False, False]) == 1.0
    assert cohen_kappa([True, False, True, False], [False, True, False, True]) == -1.0

    ranked = rank_summaries(
        [
            {
                "candidate": "higher-agreement",
                "cohen_kappa": 0.4,
                "agreement": 0.9,
                "unit_coverage": 1.0,
                "selection_eligible": True,
            },
            {
                "candidate": "higher-kappa",
                "cohen_kappa": 0.5,
                "agreement": 0.8,
                "unit_coverage": 1.0,
                "selection_eligible": True,
            },
            {
                "candidate": "partial-perfect",
                "cohen_kappa": 1.0,
                "agreement": 1.0,
                "unit_coverage": 0.1,
                "selection_eligible": False,
            },
        ]
    )
    assert ranked[0]["candidate"] == "higher-kappa"
    assert ranked[0]["rank"] == 1
    assert ranked[-1]["candidate"] == "partial-perfect"


def test_krippendorff_alpha_nominal_handles_agreement_and_disagreement():
    assert krippendorff_alpha_nominal([[True, False], [True, False]]) == 1.0
    assert krippendorff_alpha_nominal([[True, False], [False, True]]) < 0


def test_game_context_uses_summary_player_order_for_gold_player_numbers():
    payload = {
        "summary": {
            "config": {"num_players": 2},
            "Player 1": {
                "name": "Blue",
                "model": "lab/blue",
                "identity": "Crewmate",
                "tasks": ["Wires"],
            },
            "Player 2": {
                "name": "Red",
                "model": "lab/red",
                "identity": "Impostor",
                "tasks": [],
            },
            "game_outcome": {"winner": "Crewmates"},
            "winner": 2,
            "winner_reason": "Red was voted out",
            "kill_history": [],
            "voting_history": [],
        },
        # Red appears first in logs. Gold player numbering must still follow summary order.
        "agent_logs": [
            {
                "player": {"name": "Red"},
                "step": 1,
                "timestamp": "2026-01-01T00:00:00",
                "interaction": {"response": {"Action": "SPEAK: I am safe"}},
            },
            {
                "player": {"name": "Blue"},
                "step": 1,
                "timestamp": "2026-01-01T00:00:01",
                "interaction": {"response": {"Action": "VOTE: Red"}},
            },
        ],
    }

    context = build_game_context("game-1", payload)
    prompt, player = make_player_prompt(context, 1)

    assert player["name"] == "Blue"
    assert "PLAYER: Blue" in prompt
    assert "Impostors: Red" in prompt
