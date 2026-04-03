import os

import main
import pytest


def test_role_accepts_impostor_and_crewmate():
    parser = main.build_parser()

    args_impostor = parser.parse_args(["--role", "impostor"])
    args_crewmate = parser.parse_args(["--role", "crewmate"])

    assert args_impostor.role == "impostor"
    assert args_crewmate.role == "crewmate"


def test_role_rejects_invalid_value():
    parser = main.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--role", "engineer"])


def test_role_enables_human_and_sets_requested_role(monkeypatch):
    parser = main.build_parser()
    args = parser.parse_args(["--role", "impostor"])
    monkeypatch.setenv("FLASK_ENABLED", "True")

    configured_args = main.configure_args_from_cli(args)

    assert configured_args["include_human"] is True
    assert configured_args["human_role"] == "impostor"
    assert os.getenv("FLASK_ENABLED") == "False"


def test_no_role_keeps_previous_human_defaults(monkeypatch):
    parser = main.build_parser()
    args = parser.parse_args([])
    monkeypatch.setenv("FLASK_ENABLED", "True")

    configured_args = main.configure_args_from_cli(args)

    assert configured_args["include_human"] is False
    assert configured_args["human_role"] is None
    assert os.getenv("FLASK_ENABLED") == "True"


def test_role_with_single_game_ui_enabled_sets_ui_flag():
    parser = main.build_parser()
    args = parser.parse_args(
        ["--num_games", "1", "--display_ui", "True", "--role", "crewmate"]
    )

    configured_args = main.configure_args_from_cli(args)

    assert configured_args["UI"] is True
    assert configured_args["include_human"] is True
    assert configured_args["human_role"] == "crewmate"


def test_ui_stays_disabled_when_requesting_multiple_games():
    parser = main.build_parser()
    args = parser.parse_args(
        ["--num_games", "2", "--display_ui", "True", "--role", "impostor"]
    )

    configured_args = main.configure_args_from_cli(args)

    assert configured_args["UI"] is False


def test_display_ui_accepts_false_string():
    parser = main.build_parser()
    args = parser.parse_args(["--display_ui", "False"])
    assert args.display_ui is False


@pytest.mark.asyncio
async def test_multiple_games_passes_human_role_into_runtime(monkeypatch, tmp_path):
    captured = {}

    class DummyAmongUs:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        async def run_game(self):
            return None

    monkeypatch.setattr(
        main,
        "setup_experiment",
        lambda *args, **kwargs: ("test-exp", str(tmp_path)),
    )
    monkeypatch.setattr(main, "AmongUs", DummyAmongUs)
    monkeypatch.setenv("EXPERIMENT_PATH", str(tmp_path))

    configured_args = main.configure_args_from_cli(
        main.build_parser().parse_args(["--role", "crewmate"])
    )
    monkeypatch.setattr(main, "ARGS", configured_args)

    await main.multiple_games(experiment_name="test-exp", num_games=1, rate_limit=1)

    assert captured["kwargs"]["include_human"] is True
    assert captured["kwargs"]["human_role"] == "crewmate"
    assert captured["kwargs"]["log_dir"] == os.path.join(str(tmp_path), "game_1")


@pytest.mark.asyncio
async def test_multiple_games_passes_ui_object_when_enabled(monkeypatch, tmp_path):
    captured = {}
    sentinel_ui = object()

    class DummyAmongUs:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        async def run_game(self):
            return None

    monkeypatch.setattr(
        main,
        "setup_experiment",
        lambda *args, **kwargs: ("test-exp", str(tmp_path)),
    )
    monkeypatch.setattr(main, "MapUI", lambda *args, **kwargs: sentinel_ui)
    monkeypatch.setattr(main, "AmongUs", DummyAmongUs)
    monkeypatch.setenv("EXPERIMENT_PATH", str(tmp_path))

    configured_args = main.configure_args_from_cli(
        main.build_parser().parse_args(
            ["--num_games", "1", "--display_ui", "True", "--role", "crewmate"]
        )
    )
    monkeypatch.setattr(main, "ARGS", configured_args)

    await main.multiple_games(experiment_name="test-exp", num_games=1, rate_limit=1)

    assert captured["kwargs"]["UI"] is sentinel_ui
    assert captured["kwargs"]["human_role"] == "crewmate"
    assert captured["kwargs"]["log_dir"] == os.path.join(str(tmp_path), "game_1")


@pytest.mark.asyncio
async def test_multiple_games_uses_distinct_log_dir_per_game(monkeypatch, tmp_path):
    captured = {}

    class DummyAmongUs:
        def __init__(self, **kwargs):
            captured[kwargs["game_index"]] = kwargs["log_dir"]

        async def run_game(self):
            return None

    monkeypatch.setattr(
        main,
        "setup_experiment",
        lambda *args, **kwargs: ("test-exp", str(tmp_path)),
    )
    monkeypatch.setattr(main, "AmongUs", DummyAmongUs)
    monkeypatch.setattr(main, "ARGS", main.configure_args_from_cli(main.build_parser().parse_args([])))

    await main.multiple_games(experiment_name="test-exp", num_games=2, rate_limit=2)

    assert captured == {
        1: os.path.join(str(tmp_path), "game_1"),
        2: os.path.join(str(tmp_path), "game_2"),
    }
