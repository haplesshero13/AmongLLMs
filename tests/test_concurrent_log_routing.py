import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from amongagents.agent.agent import HumanAgent, LLMHumanAgent
from amongagents.envs.configs.game_config import THREE_MEMBER_GAME
from amongagents.envs.game import AmongUs


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_human_trials_run_module(module_name: str = "human_trials_run_test"):
    module_path = REPO_ROOT / "human_trials" / "run.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_human_agent_defaults_to_experiment_path(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPERIMENT_PATH", str(tmp_path))

    agent = HumanAgent(SimpleNamespace(name="Player 1"))

    assert agent.log_path == os.path.join(str(tmp_path), "agent-logs.json")
    assert agent.compact_log_path == os.path.join(
        str(tmp_path), "agent-logs-compact.json"
    )


def test_llm_human_agent_defaults_to_experiment_path(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPERIMENT_PATH", str(tmp_path))

    agent = LLMHumanAgent(SimpleNamespace(name="Player 2"))

    assert agent.log_path == os.path.join(str(tmp_path), "agent-logs.json")
    assert agent.compact_log_path == os.path.join(
        str(tmp_path), "agent-logs-compact.json"
    )


def test_amongus_test_mode_initializes_llm_human_agents(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPERIMENT_PATH", str(tmp_path))

    game = AmongUs(game_config=THREE_MEMBER_GAME, test=True)
    game.initialize_game()

    assert len(game.agents) == THREE_MEMBER_GAME["num_players"]
    assert all(isinstance(agent, LLMHumanAgent) for agent in game.agents)
    assert all(agent.log_path == os.path.join(str(tmp_path), "agent-logs.json") for agent in game.agents)


def test_human_trials_run_uses_human_trials_logs_root(monkeypatch, tmp_path):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    monkeypatch.syspath_prepend(str(REPO_ROOT / "human_trials"))
    monkeypatch.syspath_prepend(str(REPO_ROOT / "among-agents"))

    human_run = _load_human_trials_run_module()

    assert Path(human_run.LOGS_PATH) == REPO_ROOT / "human_trials" / "logs"

    logs_path = tmp_path / "human_trials" / "logs"
    logs_path.mkdir(parents=True)
    (logs_path / "game_2_2026-04-03_12-00-00").mkdir()
    (logs_path / "game_5_2026-04-03_12-05-00").mkdir()

    monkeypatch.setattr(human_run, "LOGS_PATH", str(logs_path))
    monkeypatch.setattr(human_run.RunGames, "setup_experiment_once", lambda self: None)

    run_games = human_run.RunGames()

    assert run_games.next_game_id == 6
