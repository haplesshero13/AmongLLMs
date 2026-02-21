"""Unit tests for the long_context module.

Tests for:
- JSON parsing and validation
- Action matching logic
- Token counting
- Retry logic (mock API failures)
- Per-player JSONL logging
"""

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import the modules under test
from amongagents.long_context.token_counter import count_tokens, estimate_tokens_chars
from amongagents.long_context.prompts import (
    build_system_prompt,
    build_user_turn,
    build_correction_prompt,
)
from amongagents.long_context.agent import LongContextAgent


# =============================================================================
# Token Counter Tests
# =============================================================================

def test_estimate_tokens_chars():
    """Test character-based token estimation."""
    text = "Hello world"
    # 11 chars / 4 = 2.75 -> 2 tokens
    assert estimate_tokens_chars(text) == 2

def test_count_tokens_basic():
    """Test token counting with a simple message."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    # Should not raise an exception
    tokens = count_tokens(messages, "gpt-4o")
    assert tokens > 0

def test_count_tokens_empty():
    """Test token counting with empty messages."""
    tokens = count_tokens([], "gpt-4o")
    # Overhead only: 2 (reply priming)
    assert tokens >= 2


# =============================================================================
# Prompt Builder Tests
# =============================================================================

def test_build_user_turn():
    """Test building user turn content."""
    result = build_user_turn(5, "Location info\nActions...")
    assert "=== Turn 5 ===" in result
    assert "Location info" in result
    assert "Actions..." in result


def test_build_correction_prompt():
    """Test building correction prompt."""
    from amongagents.envs.action import MoveTo

    available_actions = [MoveTo("Cafeteria", "Weapons")]
    prompt = build_correction_prompt("Invalid JSON", 2, available_actions)

    assert "Attempt 2/3" in prompt
    assert "Invalid JSON" in prompt
    assert "MOVE from Cafeteria to Weapons" in prompt


# =============================================================================
# Agent Method Tests
# =============================================================================

def test_parse_json_response_valid():
    """Test parsing valid JSON response."""
    result = LongContextAgent._parse_json_response(None, '{"thinking": "test", "action": "MOVE"}')
    assert result == {"thinking": "test", "action": "MOVE"}


def test_parse_json_response_code_block():
    """Test parsing JSON from markdown code block."""
    response = '```json\n{"thinking": "test", "action": "MOVE"}\n```'
    result = LongContextAgent._parse_json_response(None, response)
    assert result == {"thinking": "test", "action": "MOVE"}


def test_parse_json_response_embedded():
    """Test extracting JSON from embedded text."""
    response = 'Some text {"thinking": "test", "action": "MOVE"} more text'
    result = LongContextAgent._parse_json_response(None, response)
    assert result == {"thinking": "test", "action": "MOVE"}


def test_parse_json_response_invalid():
    """Test that invalid JSON returns None."""
    response = "This is not JSON at all"
    result = LongContextAgent._parse_json_response(None, response)
    assert result is None


def test_match_action_move():
    """Test matching MOVE action."""
    from amongagents.envs.action import MoveTo

    available_actions = [MoveTo("Cafeteria", "Weapons")]

    # Exact match
    result = LongContextAgent._match_action(None, "MOVE from Cafeteria to Weapons", available_actions)
    assert result is available_actions[0]

    # Case insensitive
    result = LongContextAgent._match_action(None, "move from cafeteria to weapons", available_actions)
    assert result is available_actions[0]


def test_match_action_speak():
    """Test matching SPEAK action and setting message."""
    from amongagents.envs.action import Speak

    action = Speak("Cafeteria")
    available_actions = [action]

    result = LongContextAgent._match_action(None, "SPEAK: Hello everyone!", available_actions)
    assert result is action
    assert action.message == "Hello everyone!"


def test_match_action_skip_vote():
    """Test matching SKIP VOTE action."""
    from amongagents.envs.action import SkipVote

    action = SkipVote("Cafeteria")
    available_actions = [action]

    result = LongContextAgent._match_action(None, "SKIP VOTE", available_actions)
    assert result is action


def test_match_action_none():
    """Test that non-matching action returns None."""
    from amongagents.envs.action import MoveTo

    available_actions = [MoveTo("Cafeteria", "Weapons")]

    result = LongContextAgent._match_action(None, "INVALID ACTION", available_actions)
    assert result is None


# =============================================================================
# Integration-style Tests with Mocked API
# =============================================================================

def _make_agent(tmpdir, **kwargs):
    """Helper to create a LongContextAgent with mocked player."""
    mock_player = MagicMock()
    mock_player.name = kwargs.get("name", "Player 1: red")
    mock_player.identity = kwargs.get("identity", "Crewmate")
    mock_player.location = "Cafeteria"
    mock_player.personality = None
    mock_player.get_available_actions.return_value = kwargs.get("actions", [])
    mock_player.all_info_prompt.return_value = "Test info"

    os.environ["EXPERIMENT_PATH"] = tmpdir
    os.environ["OPENROUTER_API_KEY"] = "test-key"

    agent_config = {
        "CREWMATE_LLM_CHOICES": ["test-model"],
        "IMPOSTOR_LLM_CHOICES": ["test-model"],
        "temperature": 1.0,
    }

    return LongContextAgent(
        player=mock_player,
        tools=[],
        game_index=1,
        agent_config=agent_config,
        list_of_impostors=[],
        model="test-model",
    )


@pytest.mark.asyncio
async def test_choose_action_success():
    """Test successful action selection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from amongagents.envs.action import SkipVote
        skip_action = SkipVote("Cafeteria")

        agent = _make_agent(tmpdir)
        agent._send_request = AsyncMock(
            return_value='{"thinking": "test", "action": "SKIP VOTE"}'
        )
        agent._match_action = MagicMock(return_value=skip_action)

        action = await agent.choose_action(1)

        assert action is skip_action
        assert len(agent.chat_history) == 2  # user + assistant messages added


@pytest.mark.asyncio
async def test_retry_logic_then_success():
    """Test that retry logic works and eventually succeeds."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from amongagents.envs.action import SkipVote
        skip_action = SkipVote("Cafeteria")

        agent = _make_agent(tmpdir, actions=[skip_action])
        agent._send_request = AsyncMock(side_effect=[
            "Not JSON",
            '{"thinking": "test", "action": "SKIP VOTE"}',
        ])

        action = await agent.choose_action(1)

        assert isinstance(action, SkipVote)
        assert len(agent.issues) == 1
        assert agent.issues[0]["resolved"] is True


@pytest.mark.asyncio
async def test_retry_logic_exhausted_raises():
    """Test that all retries exhausted raises RuntimeError in task phase."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from amongagents.envs.action import MoveTo
        agent = _make_agent(tmpdir, actions=[MoveTo("Cafeteria", "Weapons")])
        agent._send_request = AsyncMock(return_value="Invalid JSON")
        agent._match_action = MagicMock(return_value=None)

        with pytest.raises(RuntimeError) as exc_info:
            await agent.choose_action(1)

        assert "format validation failed after 3 retries" in str(exc_info.value)
        assert len(agent.issues) == 3


@pytest.mark.asyncio
async def test_retry_logic_exhausted_voting_fallback():
    """Test that all retries exhausted falls back to SKIP VOTE in voting phase."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from amongagents.envs.action import SkipVote
        agent = _make_agent(tmpdir, actions=[SkipVote("Cafeteria")])
        agent._send_request = AsyncMock(return_value="Invalid JSON")
        agent._match_action = MagicMock(return_value=None)

        action = await agent.choose_action(1)

        assert isinstance(action, SkipVote)
        assert len(agent.issues) == 3


# =============================================================================
# JSONL Logging Tests
# =============================================================================

def test_log_turn_creates_jsonl():
    """Test that _log_turn appends to a single agent-logs.jsonl."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(tmpdir)

        agent._log_turn('{"thinking": "first", "action": "MOVE"}', step=1)
        agent._log_turn('{"thinking": "second", "action": "SKIP VOTE"}', step=5)

        assert os.path.exists(agent.log_path)
        assert agent.log_path.endswith(".jsonl")

        with open(agent.log_path) as f:
            lines = [json.loads(line) for line in f]

        assert len(lines) == 2

        # Every line has game_index, player, identity, model
        for line in lines:
            assert line["game_index"] == 1
            assert line["player"] == "Player 1: red"
            assert line["identity"] == "Crewmate"
            assert line["model"] == "test-model"
            assert "token_tracking" in line

        assert lines[0]["step"] == 1
        assert "thinking" in lines[0]["response"]
        assert lines[1]["step"] == 5


def test_log_turn_no_system_prompt():
    """Log entries must NOT contain system_prompt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(tmpdir)
        agent._log_turn('{"thinking": "x", "action": "MOVE"}', step=1)

        with open(agent.log_path) as f:
            lines = [json.loads(line) for line in f]

        assert "system_prompt" not in lines[0]


def test_log_turn_no_duplicate_response_fields():
    """Log entries must have only 'response', not 'full_response'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(tmpdir)
        agent._log_turn('{"thinking": "x", "action": "MOVE"}', step=1)

        with open(agent.log_path) as f:
            lines = [json.loads(line) for line in f]

        assert "response" in lines[0]
        assert "full_response" not in lines[0]


# =============================================================================
# Token Tracking Tests
# =============================================================================

def test_token_tracking_updates():
    """Test that token tracking is updated correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(tmpdir)

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message"},
        ]
        response = '{"thinking": "test", "action": "MOVE"}'

        agent._update_token_tracking(messages, response, 5)

        assert agent.tokens_this_turn > 0
        assert agent.tokens_cumulative == agent.tokens_this_turn
        assert len(agent.token_log) == 1
        assert agent.token_log[0]["timestep"] == 5


def test_token_summary():
    """Test token_summary property."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(tmpdir)

        agent.token_log = [
            {"timestep": 1, "tokens_total_this_turn": 100},
            {"timestep": 2, "tokens_total_this_turn": 150},
            {"timestep": 3, "tokens_total_this_turn": 120},
        ]
        agent.tokens_cumulative = 370

        summary = agent.token_summary

        assert summary["total_tokens"] == 370
        assert summary["turns"] == 3
        assert summary["avg_tokens_per_turn"] == 370 / 3
        assert summary["max_tokens_single_turn"] == 150
        assert len(summary["log"]) == 3


# =============================================================================
# Issue Tracking Tests
# =============================================================================

def test_record_issue():
    """Test issue recording."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(tmpdir)

        issue = agent._record_issue(
            "format",
            "Invalid JSON",
            1,
            timestep=5,
            response_snippet="Invalid..."
        )

        assert len(agent.issues) == 1
        assert issue["type"] == "format"
        assert issue["error"] == "Invalid JSON"
        assert issue["attempt"] == 1
        assert issue["timestep"] == 5
        assert issue["response_snippet"] == "Invalid..."
        assert issue["resolved"] is False


# =============================================================================
# System Prompt Tests
# =============================================================================

def test_build_system_prompt_crewmate():
    """Test building crewmate system prompt."""
    mock_player = MagicMock()
    mock_player.name = "Player 1: red"
    mock_player.identity = "Crewmate"
    mock_player.personality = None

    prompt = build_system_prompt(
        player=mock_player,
        list_of_impostors=["Player 2: blue"],
        kill_cooldown=3,
        num_impostors=1,
        num_players=5,
    )

    assert "You are a Crewmate" in prompt
    assert "Player 1: red" in prompt
    assert "5 players" in prompt
    assert "JSON" in prompt


def test_build_system_prompt_impostor():
    """Test building impostor system prompt with teammates."""
    mock_player = MagicMock()
    mock_player.name = "Player 1: red"
    mock_player.identity = "Impostor"
    mock_player.personality = None

    prompt = build_system_prompt(
        player=mock_player,
        list_of_impostors=["Player 1: red", "Player 2: blue"],
        kill_cooldown=3,
        num_impostors=2,
        num_players=5,
    )

    assert "You are an Impostor" in prompt
    assert "Player 1: red" in prompt
    assert "YOUR FELLOW IMPOSTOR(S): Player 2: blue" in prompt
    assert "JSON" in prompt


def test_build_system_prompt_impostor_solo():
    """Test building impostor system prompt when alone."""
    mock_player = MagicMock()
    mock_player.name = "Player 1: red"
    mock_player.identity = "Impostor"
    mock_player.personality = None

    prompt = build_system_prompt(
        player=mock_player,
        list_of_impostors=["Player 1: red"],
        kill_cooldown=3,
        num_impostors=1,
        num_players=5,
    )

    assert "You are the ONLY Impostor" in prompt
