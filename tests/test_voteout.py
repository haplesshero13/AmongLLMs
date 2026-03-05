
"""
Tests for voteout() ejection logic in game.py

HOW TO RUN (from project root /Users/Ella/AmongUs_Ella):

    Option 1 — Run directly:
        .venv/bin/python3 tests/test_voteout.py

    Option 2 — Run via pytest:
        .venv/bin/python3 -m pytest tests/test_voteout.py -v

    Option 3 — Run a single test class:
        .venv/bin/python3 -m pytest tests/test_voteout.py::TestTie -v

    Option 4 — Run a single test:
        .venv/bin/python3 -m pytest tests/test_voteout.py::TestTie::test_two_way_tie_ejects_no_one -v

NOTE 1: Do NOT use /usr/local/bin/python3 — that is the system Python
      and does not have pytest installed. Always use .venv/bin/python3.
"""

import os
import sys
import pytest
from collections import Counter
from unittest.mock import MagicMock

# Adjust path so we can import the game modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, os.path.join(project_root, "among-agents"))
sys.path.insert(0, project_root)

from amongagents.envs.player import Crewmate, Impostor


# ============================================================
# Helpers
# ============================================================
COLORS = ["red", "blue", "green", "purple", "orange", "yellow", "black", "white", "cyan", "lime"]


def make_players(num_crewmates=3, num_impostors=1):
    """Create a mixed list of Crewmate and Impostor players."""
    players = []
    for i in range(num_crewmates):
        players.append(Crewmate(
            name=f"Player {i + 1}",
            color=COLORS[i],
            location="Cafeteria",
            personality="neutral",
        ))
    for i in range(num_impostors):
        idx = num_crewmates + i
        players.append(Impostor(
            name=f"Player {idx + 1}",
            color=COLORS[idx],
            location="Cafeteria",
            personality="neutral",
        ))
    return players


class FakeGame:
    """
    Minimal stand-in for AmongUs with just enough state
    for voteout() to run. Delegates to the real method.
    """

    def __init__(self, players):
        self.players = players
        self.timestep = 5
        self.meeting_number = 1
        self.current_phase = "meeting"
        self.game_config = {"discussion_rounds": 3}
        self.discussion_rounds_left = 0
        self.votes = Counter()
        self.vote_info_one_round = {}
        self.voting_history = []
        self.activity_log = []
        self.important_activity_log = []
        self.pending_system_announcement = None
        self.UI = None

    def voteout(self):
        from amongagents.envs.game import AmongUs
        AmongUs.voteout(self)


def build_votes(game, target, voters):
    """Helper: set up votes so `voters` all vote for `target`."""
    game.votes[target] = len(voters)
    game.vote_info_one_round = {v: target.name for v in voters}


# ============================================================
# 1. Impostor voted out
# ============================================================
class TestImpostorEjected:
    def test_impostor_ejected_by_unanimous_vote(self):
        players = make_players(3, 1)
        game = FakeGame(players)
        impostor = players[3]

        build_votes(game, impostor, ["Player 1", "Player 2", "Player 3"])
        game.voteout()

        assert impostor.is_alive is False
        assert "was ejected" in game.pending_system_announcement
        assert impostor.name in game.pending_system_announcement

    def test_impostor_ejected_by_simple_majority(self):
        players = make_players(4, 1)
        game = FakeGame(players)
        impostor = players[4]

        # 3 vote impostor, 1 votes crewmate, 1 skips
        game.votes[impostor] = 3
        game.votes[players[0]] = 1
        game.vote_info_one_round = {
            "Player 1": impostor.name,
            "Player 2": impostor.name,
            "Player 3": impostor.name,
            "Player 4": players[0].name,
            "Player 5": "SKIP",
        }
        game.voteout()

        assert impostor.is_alive is False
        assert "was ejected" in game.pending_system_announcement

    def test_impostor_ejected_with_minimum_votes(self):
        """Even a single vote can eject if no one else gets more."""
        players = make_players(3, 1)
        game = FakeGame(players)
        impostor = players[3]

        game.votes[impostor] = 1
        game.vote_info_one_round = {
            "Player 1": impostor.name,
            "Player 2": "SKIP",
            "Player 3": "SKIP",
            "Player 4": "SKIP",
        }
        game.voteout()

        assert impostor.is_alive is False

    def test_impostor_marked_as_reported_death(self):
        players = make_players(3, 1)
        game = FakeGame(players)
        impostor = players[3]

        build_votes(game, impostor, ["Player 1", "Player 2", "Player 3"])
        game.voteout()

        assert impostor.reported_death is True

    def test_ejection_announcement_contains_impostor_color(self):
        players = make_players(3, 1)
        game = FakeGame(players)
        impostor = players[3]  # color = "purple"

        build_votes(game, impostor, ["Player 1", "Player 2", "Player 3"])
        game.voteout()

        assert f"({impostor.color})" in game.pending_system_announcement


# ============================================================
# 2. Crewmate voted out
# ============================================================
class TestCrewmateEjected:
    def test_crewmate_ejected_by_majority(self):
        players = make_players(3, 1)
        game = FakeGame(players)
        crewmate = players[0]  # innocent

        build_votes(game, crewmate, ["Player 2", "Player 3", "Player 4"])
        game.voteout()

        assert crewmate.is_alive is False
        assert "was ejected" in game.pending_system_announcement
        assert crewmate.name in game.pending_system_announcement

    def test_crewmate_marked_as_reported_death(self):
        players = make_players(3, 1)
        game = FakeGame(players)
        crewmate = players[1]

        build_votes(game, crewmate, ["Player 1", "Player 4"])
        game.voteout()

        assert crewmate.is_alive is False
        assert crewmate.reported_death is True

    def test_crewmate_ejection_announcement_contains_color(self):
        players = make_players(3, 1)
        game = FakeGame(players)
        crewmate = players[0]  # color = "red"

        build_votes(game, crewmate, ["Player 2", "Player 3", "Player 4"])
        game.voteout()

        assert f"({crewmate.color})" in game.pending_system_announcement

    def test_impostor_survives_when_crewmate_ejected(self):
        players = make_players(3, 1)
        game = FakeGame(players)
        crewmate = players[0]
        impostor = players[3]

        build_votes(game, crewmate, ["Player 2", "Player 3", "Player 4"])
        game.voteout()

        assert impostor.is_alive is True


# ============================================================
# 3. Tie — no one ejected
# ============================================================
class TestTie:
    def test_two_way_tie_ejects_no_one(self):
        players = make_players(3, 1)
        game = FakeGame(players)

        game.votes[players[0]] = 2
        game.votes[players[3]] = 2
        game.vote_info_one_round = {
            "Player 1": players[3].name,
            "Player 2": players[3].name,
            "Player 3": players[0].name,
            "Player 4": players[0].name,
        }
        game.voteout()

        for p in players:
            assert p.is_alive is True
        assert "No one was ejected" in game.pending_system_announcement

    def test_three_way_tie_ejects_no_one(self):
        players = make_players(5, 1)
        game = FakeGame(players)

        game.votes[players[0]] = 2
        game.votes[players[1]] = 2
        game.votes[players[2]] = 2
        game.vote_info_one_round = {
            "Player 1": players[1].name,
            "Player 2": players[2].name,
            "Player 3": players[0].name,
            "Player 4": players[1].name,
            "Player 5": players[0].name,
            "Player 6": players[2].name,
        }
        game.voteout()

        for p in players:
            assert p.is_alive is True
        assert "No one was ejected" in game.pending_system_announcement

    def test_tie_between_impostor_and_crewmate(self):
        """Even if one tied player is the impostor, a tie means no ejection."""
        players = make_players(3, 1)
        game = FakeGame(players)
        impostor = players[3]

        game.votes[players[0]] = 2
        game.votes[impostor] = 2
        game.vote_info_one_round = {
            "Player 1": impostor.name,
            "Player 2": impostor.name,
            "Player 3": players[0].name,
            "Player 4": players[0].name,
        }
        game.voteout()

        assert impostor.is_alive is True
        assert players[0].is_alive is True
        assert "No one was ejected" in game.pending_system_announcement


# ============================================================
# 4. No one voted out — skips and empty votes
# ============================================================
class TestNoEjection:
    def test_all_skip_ejects_no_one(self):
        players = make_players(3, 1)
        game = FakeGame(players)

        game.vote_info_one_round = {
            "Player 1": "SKIP",
            "Player 2": "SKIP",
            "Player 3": "SKIP",
            "Player 4": "SKIP",
        }
        game.voteout()

        for p in players:
            assert p.is_alive is True
        assert "No one was ejected" in game.pending_system_announcement

    def test_no_votes_at_all(self):
        players = make_players(3, 1)
        game = FakeGame(players)

        game.vote_info_one_round = {}
        game.voteout()

        for p in players:
            assert p.is_alive is True
        assert "No one was ejected" in game.pending_system_announcement

    def test_partial_skip_with_single_vote_still_ejects(self):
        """If most skip but one player gets a vote, they should be ejected."""
        players = make_players(3, 1)
        game = FakeGame(players)

        game.votes[players[0]] = 1
        game.vote_info_one_round = {
            "Player 2": players[0].name,
            "Player 3": "SKIP",
            "Player 4": "SKIP",
        }
        game.voteout()

        assert players[0].is_alive is False


# ============================================================
# 5. Voting history and activity log
# ============================================================
class TestVotingHistory:
    def test_ejection_recorded_in_history(self):
        players = make_players(3, 1)
        game = FakeGame(players)
        target = players[0]

        build_votes(game, target, ["Player 2", "Player 3", "Player 4"])
        game.voteout()

        assert len(game.voting_history) == 1
        record = game.voting_history[0]
        assert record["timestep"] == 5
        assert record["meeting_number"] == 1
        assert record["eliminated"] == target.name

    def test_tie_records_no_elimination(self):
        players = make_players(3, 1)
        game = FakeGame(players)

        game.votes[players[0]] = 2
        game.votes[players[1]] = 2
        game.vote_info_one_round = {
            "Player 3": players[0].name,
            "Player 4": players[0].name,
            "Player 1": players[1].name,
            "Player 2": players[1].name,
        }
        game.voteout()

        record = game.voting_history[0]
        assert record["eliminated"] is None

    def test_skip_records_no_elimination(self):
        players = make_players(3, 1)
        game = FakeGame(players)

        game.vote_info_one_round = {"Player 1": "SKIP", "Player 2": "SKIP"}
        game.voteout()

        record = game.voting_history[0]
        assert record["eliminated"] is None

    def test_activity_log_has_vote_summary(self):
        players = make_players(3, 1)
        game = FakeGame(players)

        build_votes(game, players[0], ["Player 2", "Player 3"])
        game.voteout()

        vote_records = [
            r for r in game.activity_log if r.get("action") == "VOTE_SUMMARY"
        ]
        assert len(vote_records) == 1

    def test_multiple_meetings_accumulate_history(self):
        players = make_players(3, 1)
        game = FakeGame(players)

        # First meeting — skip
        game.vote_info_one_round = {"Player 1": "SKIP"}
        game.voteout()

        # Second meeting — eject someone
        game.meeting_number = 2
        game.timestep = 10
        game.votes = Counter()
        build_votes(game, players[0], ["Player 2", "Player 3", "Player 4"])
        game.voteout()

        assert len(game.voting_history) == 2
        assert game.voting_history[0]["eliminated"] is None
        assert game.voting_history[1]["eliminated"] == players[0].name


# ============================================================
# 6. Phase resets after voteout
# ============================================================
class TestPhaseReset:
    def test_phase_resets_to_task(self):
        players = make_players(3, 1)
        game = FakeGame(players)

        game.vote_info_one_round = {"Player 1": "SKIP"}
        game.voteout()

        assert game.current_phase == "task"

    def test_phase_resets_after_ejection(self):
        players = make_players(3, 1)
        game = FakeGame(players)

        build_votes(game, players[0], ["Player 2", "Player 3", "Player 4"])
        game.voteout()

        assert game.current_phase == "task"

    def test_discussion_rounds_reset(self):
        players = make_players(3, 1)
        game = FakeGame(players)

        game.vote_info_one_round = {"Player 1": "SKIP"}
        game.voteout()

        assert game.discussion_rounds_left == game.game_config["discussion_rounds"]

    def test_votes_counter_cleared_after_ejection(self):
        players = make_players(3, 1)
        game = FakeGame(players)

        build_votes(game, players[0], ["Player 2", "Player 3", "Player 4"])
        game.voteout()

        assert len(game.votes) == 0

    def test_votes_counter_cleared_after_tie(self):
        players = make_players(3, 1)
        game = FakeGame(players)

        game.votes[players[0]] = 2
        game.votes[players[1]] = 2
        game.vote_info_one_round = {
            "Player 3": players[0].name,
            "Player 4": players[0].name,
            "Player 1": players[1].name,
            "Player 2": players[1].name,
        }
        game.voteout()

        assert len(game.votes) == 0

    def test_votes_counter_cleared_after_skip(self):
        players = make_players(3, 1)
        game = FakeGame(players)

        game.vote_info_one_round = {"Player 1": "SKIP"}
        game.voteout()

        assert len(game.votes) == 0


# ============================================================
# 7. UI callbacks
# ============================================================
class TestUICallback:
    def test_ui_called_on_ejection(self):
        players = make_players(3, 1)
        game = FakeGame(players)
        game.UI = MagicMock()

        target = players[0]
        build_votes(game, target, ["Player 2", "Player 3", "Player 4"])
        game.voteout()

        game.UI.show_ejected_player.assert_called_once()
        call_arg = game.UI.show_ejected_player.call_args[0][0]
        assert "was ejected" in call_arg

    def test_ui_called_on_no_ejection(self):
        players = make_players(3, 1)
        game = FakeGame(players)
        game.UI = MagicMock()

        game.vote_info_one_round = {"Player 1": "SKIP", "Player 2": "SKIP"}
        game.voteout()

        game.UI.show_ejected_player.assert_called_once()
        call_arg = game.UI.show_ejected_player.call_args[0][0]
        assert "No one was ejected" in call_arg

    def test_ui_called_on_tie(self):
        players = make_players(3, 1)
        game = FakeGame(players)
        game.UI = MagicMock()

        game.votes[players[0]] = 2
        game.votes[players[1]] = 2
        game.vote_info_one_round = {
            "Player 3": players[0].name,
            "Player 4": players[0].name,
            "Player 1": players[1].name,
            "Player 2": players[1].name,
        }
        game.voteout()

        game.UI.show_ejected_player.assert_called_once()
        call_arg = game.UI.show_ejected_player.call_args[0][0]
        assert "No one was ejected" in call_arg

    def test_no_ui_no_crash_on_ejection(self):
        players = make_players(3, 1)
        game = FakeGame(players)
        game.UI = None

        build_votes(game, players[0], ["Player 2", "Player 3", "Player 4"])
        game.voteout()

        assert players[0].is_alive is False
        assert "was ejected" in game.pending_system_announcement

    def test_no_ui_no_crash_on_skip(self):
        players = make_players(3, 1)
        game = FakeGame(players)
        game.UI = None

        game.vote_info_one_round = {"Player 1": "SKIP"}
        game.voteout()

        assert "No one was ejected" in game.pending_system_announcement

    def test_no_ui_no_crash_on_tie(self):
        players = make_players(3, 1)
        game = FakeGame(players)
        game.UI = None

        game.votes[players[0]] = 2
        game.votes[players[1]] = 2
        game.vote_info_one_round = {
            "Player 3": players[0].name,
            "Player 4": players[0].name,
            "Player 1": players[1].name,
            "Player 2": players[1].name,
        }
        game.voteout()

        for p in players:
            assert p.is_alive is True
        assert "No one was ejected" in game.pending_system_announcement


# ============================================================
# 8. Edge cases and error handling
# ============================================================
class TestEdgeCases:
    def test_two_impostors_one_ejected_other_survives(self):
        players = make_players(3, 2)
        game = FakeGame(players)
        imp1, imp2 = players[3], players[4]

        build_votes(game, imp1, ["Player 1", "Player 2", "Player 3"])
        game.voteout()

        assert imp1.is_alive is False
        assert imp2.is_alive is True

    def test_only_two_players_left(self):
        """Smallest possible vote scenario."""
        players = make_players(1, 1)
        game = FakeGame(players)

        game.votes[players[1]] = 1
        game.vote_info_one_round = {"Player 1": players[1].name}
        game.voteout()

        assert players[1].is_alive is False

    def test_large_game_majority_vote(self):
        players = make_players(8, 2)
        game = FakeGame(players)
        target = players[9]  # second impostor

        voters = [f"Player {i+1}" for i in range(8)]
        build_votes(game, target, voters)
        game.voteout()

        assert target.is_alive is False
        assert "was ejected" in game.pending_system_announcement

    def test_dead_players_not_revived_by_voteout(self):
        """A player who is already dead stays dead after voteout."""
        players = make_players(3, 1)
        game = FakeGame(players)

        # Kill a player before the vote
        players[1].is_alive = False
        players[1].reported_death = True

        game.vote_info_one_round = {"Player 1": "SKIP", "Player 3": "SKIP"}
        game.voteout()

        assert players[1].is_alive is False

    def test_pending_announcement_is_set(self):
        """Ensure pending_system_announcement is always set (never None)."""
        players = make_players(3, 1)
        game = FakeGame(players)

        game.vote_info_one_round = {"Player 1": "SKIP"}
        game.voteout()

        assert game.pending_system_announcement is not None

    def test_pending_announcement_set_on_ejection(self):
        players = make_players(3, 1)
        game = FakeGame(players)

        build_votes(game, players[0], ["Player 2", "Player 3", "Player 4"])
        game.voteout()

        assert game.pending_system_announcement is not None
        assert len(game.pending_system_announcement) > 0

#============================================================
# 9. Ejection announcement content (drives frontend banner)
# ============================================================
class TestEjectionAnnouncement:
    def test_ejection_announcement_says_was_ejected(self):
        players = make_players(3, 1)
        game = FakeGame(players)
        target = players[0]

        build_votes(game, target, ["Player 2", "Player 3", "Player 4"])
        game.voteout()

        assert "was ejected" in game.pending_system_announcement

    def test_no_ejection_announcement_says_no_one(self):
        players = make_players(3, 1)
        game = FakeGame(players)

        game.vote_info_one_round = {"Player 1": "SKIP", "Player 2": "SKIP"}
        game.voteout()

        assert "No one was ejected" in game.pending_system_announcement

    def test_tie_announcement_says_no_one(self):
        players = make_players(3, 1)
        game = FakeGame(players)

        game.votes[players[0]] = 2
        game.votes[players[1]] = 2
        game.vote_info_one_round = {
            "Player 3": players[0].name,
            "Player 4": players[0].name,
            "Player 1": players[1].name,
            "Player 2": players[1].name,
        }
        game.voteout()

        assert "No one was ejected" in game.pending_system_announcement

    def test_announcement_contains_player_name(self):
        players = make_players(3, 1)
        game = FakeGame(players)
        target = players[2]

        build_votes(game, target, ["Player 1", "Player 4"])
        game.voteout()

        assert target.name in game.pending_system_announcement

    def test_announcement_contains_player_color(self):
        players = make_players(3, 1)
        game = FakeGame(players)
        target = players[0]  # color = "red"

        build_votes(game, target, ["Player 2", "Player 3", "Player 4"])
        game.voteout()

        assert f"({target.color})" in game.pending_system_announcement

    def test_announcement_not_none_after_any_vote(self):
        """Frontend relies on this field existing — it should never be None after voteout."""
        scenarios = [
            {"skip": True},
            {"tie": True},
            {"eject": True},
        ]
        for scenario in scenarios:
            players = make_players(3, 1)
            game = FakeGame(players)

            if scenario.get("skip"):
                game.vote_info_one_round = {"Player 1": "SKIP"}
            elif scenario.get("tie"):
                game.votes[players[0]] = 2
                game.votes[players[1]] = 2
                game.vote_info_one_round = {
                    "Player 3": players[0].name,
                    "Player 4": players[0].name,
                    "Player 1": players[1].name,
                    "Player 2": players[1].name,
                }
            else:
                build_votes(game, players[0], ["Player 2", "Player 3", "Player 4"])

            game.voteout()
            assert game.pending_system_announcement is not None, \
                f"Announcement was None for scenario: {scenario}"

    def test_announcement_is_string(self):
        """Frontend calls .includes() on this — must be a string."""
        players = make_players(3, 1)
        game = FakeGame(players)

        build_votes(game, players[0], ["Player 2", "Player 3", "Player 4"])
        game.voteout()

        assert isinstance(game.pending_system_announcement, str)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))