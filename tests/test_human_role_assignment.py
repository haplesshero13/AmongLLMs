import numpy as np
import pytest

from amongagents.agent.agent import HumanAgent
from amongagents.envs.configs.agent_config import ALL_LLM
from amongagents.envs.configs.game_config import SEVEN_MEMBER_GAME
from amongagents.envs.game import AmongUs


@pytest.fixture(autouse=True)
def experiment_path(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPERIMENT_PATH", str(tmp_path))


def test_human_crewmate_role_is_deterministic():
    game = AmongUs(
        game_config=SEVEN_MEMBER_GAME,
        include_human=True,
        human_role="crewmate",
        agent_config=ALL_LLM,
    )
    game.initialize_game()

    first_crewmate_idx = next(
        idx for idx, player in enumerate(game.players) if player.identity == "Crewmate"
    )
    assert game.human_index == first_crewmate_idx
    assert game.players[game.human_index].identity == "Crewmate"
    assert isinstance(game.agents[game.human_index], HumanAgent)


def test_human_impostor_role_is_deterministic():
    game = AmongUs(
        game_config=SEVEN_MEMBER_GAME,
        include_human=True,
        human_role="impostor",
        agent_config=ALL_LLM,
    )
    game.initialize_game()

    first_impostor_idx = next(
        idx for idx, player in enumerate(game.players) if player.identity == "Impostor"
    )
    assert game.human_index == first_impostor_idx
    assert game.players[game.human_index].identity == "Impostor"
    assert isinstance(game.agents[game.human_index], HumanAgent)


def test_no_human_role_keeps_random_human_selection(monkeypatch):
    game = AmongUs(
        game_config=SEVEN_MEMBER_GAME,
        include_human=True,
        human_role=None,
        agent_config=ALL_LLM,
    )
    game.current_phase = "task"
    game.initialize_players()

    calls = {}

    def fake_choice(options):
        calls["options"] = options
        return 0

    monkeypatch.setattr(np.random, "choice", fake_choice)
    game.initialize_agents()

    assert calls["options"] == len(game.players)
    assert game.human_index == 0
