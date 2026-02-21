"""Unit tests for the long_context module.

Tests for:
- JSON parsing and validation
- Action matching logic
- Retry logic (mock API failures)
- Per-player JSONL logging (unified thinking + action format)
- Usage tracking from API responses
"""

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import the modules under test
from amongagents.long_context.model_info import ModelInfo
from amongagents.long_context.prompts import (
    build_system_prompt,
    build_user_turn,
    build_correction_prompt,
)
from amongagents.long_context.agent import LongContextAgent
from amongagents.long_context.short_context_agent import ShortContextAgent
from amongagents.long_context.prompts import (
    build_system_prompt_short_context,
    build_short_context_user_turn,
    SHORT_CONTEXT_JSON_FORMAT,
    SHORT_CONTEXT_JSON_FORMAT_REASONING,
    JSON_OUTPUT_FORMAT,
)


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

    agent = LongContextAgent(
        player=mock_player,
        tools=[],
        game_index=1,
        agent_config=agent_config,
        list_of_impostors=[],
        model="test-model",
    )
    # Mark setup as done with a dummy system prompt (skip API call)
    agent.system_prompt = "Test system prompt"
    agent._setup_done = True
    return agent


def _mock_api_result(content, reasoning=None, usage=None):
    """Build a mock return value for _send_request."""
    msg = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning"] = reasoning
    return {
        "message": msg,
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


@pytest.mark.asyncio
async def test_choose_action_success():
    """Test successful action selection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from amongagents.envs.action import SkipVote
        skip_action = SkipVote("Cafeteria")

        agent = _make_agent(tmpdir)
        agent._send_request = AsyncMock(
            return_value=_mock_api_result('{"thinking": "test", "action": "SKIP VOTE"}')
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
            _mock_api_result("Not JSON"),
            _mock_api_result('{"thinking": "test", "action": "SKIP VOTE"}'),
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
        agent._send_request = AsyncMock(return_value=_mock_api_result("Invalid JSON"))
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
        agent._send_request = AsyncMock(return_value=_mock_api_result("Invalid JSON"))
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

        usage1 = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        usage2 = {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}

        agent._log_turn(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "turn1"}],
            thinking="I should move", action="MOVE from Cafeteria to Weapons",
            step=1, usage=usage1,
        )
        agent._log_turn(
            [{"role": "user", "content": "turn2"}],
            thinking="I'll skip", action="SKIP VOTE",
            step=5, usage=usage2,
        )

        assert os.path.exists(agent.log_path)
        assert agent.log_path.endswith(".jsonl")

        with open(agent.log_path) as f:
            lines = [json.loads(line) for line in f]

        assert len(lines) == 2

        # Every line has game_index, player, identity, model, thinking, action
        for line in lines:
            assert line["game_index"] == 1
            assert line["player"] == "Player 1: red"
            assert line["identity"] == "Crewmate"
            assert line["model"] == "test-model"
            assert "thinking" in line
            assert "action" in line
            assert "usage" in line

        assert lines[0]["step"] == 1
        assert lines[0]["thinking"] == "I should move"
        assert lines[0]["action"] == "MOVE from Cafeteria to Weapons"
        assert lines[1]["step"] == 5
        assert lines[1]["action"] == "SKIP VOTE"


def test_log_turn_contains_only_input_messages():
    """Messages array should only contain input (system/user), not assistant output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(tmpdir)
        messages = [
            {"role": "system", "content": "You are a test agent"},
            {"role": "user", "content": "What do you do?"},
        ]
        agent._log_turn(messages, thinking="x", action="MOVE", step=1)

        with open(agent.log_path) as f:
            lines = [json.loads(line) for line in f]

        entry = lines[0]
        assert "messages" in entry
        assert entry["messages"][0]["role"] == "system"
        # No assistant message in the messages array
        for msg in entry["messages"]:
            assert msg["role"] != "assistant"
        # Thinking and action are separate top-level fields
        assert entry["thinking"] == "x"
        assert entry["action"] == "MOVE"


def test_log_has_no_response_or_reasoning_fields():
    """Log entries should use thinking+action, not response+reasoning."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(tmpdir)
        agent._log_turn(
            [{"role": "user", "content": "test"}],
            thinking="Some reasoning", action="SKIP VOTE", step=1,
        )

        with open(agent.log_path) as f:
            lines = [json.loads(line) for line in f]

        entry = lines[0]
        # New format: thinking + action
        assert "thinking" in entry
        assert "action" in entry
        # Old format: response + reasoning should NOT exist
        assert "response" not in entry
        assert "reasoning" not in entry


def test_log_reasoning_model_thinking_from_native():
    """For reasoning models, thinking comes from native reasoning tokens."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(tmpdir)
        # Simulate reasoning model
        agent.model_info = ModelInfo(
            context_length=1000000,
            supports_reasoning=True,
            supports_include_reasoning=True,
        )
        agent._send_request = AsyncMock(return_value=_mock_api_result(
            '{"action": "SKIP VOTE"}',
            reasoning="Let me think step by step..."
        ))

        import asyncio
        from amongagents.envs.action import SkipVote
        agent.player.get_available_actions.return_value = [SkipVote("Cafeteria")]

        asyncio.get_event_loop().run_until_complete(agent.choose_action(1))

        with open(agent.log_path) as f:
            lines = [json.loads(line) for line in f]

        # Thinking should come from reasoning tokens, not JSON field
        assert lines[0]["thinking"] == "Let me think step by step..."
        assert lines[0]["action"] == "SKIP VOTE"


def test_log_standard_model_thinking_from_json():
    """For non-reasoning models, thinking comes from the JSON thinking field."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(tmpdir)
        # No model_info = non-reasoning model
        agent.model_info = None
        agent._send_request = AsyncMock(return_value=_mock_api_result(
            '{"thinking": "I suspect Player 3", "action": "SKIP VOTE"}'
        ))

        import asyncio
        from amongagents.envs.action import SkipVote
        agent.player.get_available_actions.return_value = [SkipVote("Cafeteria")]

        asyncio.get_event_loop().run_until_complete(agent.choose_action(1))

        with open(agent.log_path) as f:
            lines = [json.loads(line) for line in f]

        # Thinking should come from JSON "thinking" field
        assert lines[0]["thinking"] == "I suspect Player 3"
        assert lines[0]["action"] == "SKIP VOTE"


# =============================================================================
# Usage Tracking Tests
# =============================================================================

def test_record_usage():
    """Test that usage from API is recorded correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(tmpdir)
        usage = {"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600}

        agent._record_usage(usage, timestep=3)

        assert agent.tokens_cumulative == 600
        assert len(agent.usage_log) == 1
        assert agent.usage_log[0]["timestep"] == 3
        assert agent.usage_log[0]["prompt_tokens"] == 500
        assert agent.usage_log[0]["completion_tokens"] == 100


def test_record_usage_cumulative():
    """Test that cumulative token count accumulates across turns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(tmpdir)

        agent._record_usage({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}, timestep=1)
        agent._record_usage({"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}, timestep=2)

        assert agent.tokens_cumulative == 430
        assert len(agent.usage_log) == 2


def test_token_summary():
    """Test token_summary property."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(tmpdir)

        agent.usage_log = [
            {"timestep": 1, "total_tokens": 100},
            {"timestep": 2, "total_tokens": 150},
            {"timestep": 3, "total_tokens": 120},
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


# =============================================================================
# ModelInfo Tests
# =============================================================================

def test_model_info_properties():
    """Test that model_info properties work correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(tmpdir)

        # Before model info is fetched
        agent.model_info = None
        assert agent.context_length is None
        assert agent.supports_reasoning is False
        assert agent.supports_include_reasoning is False

        # After setting model info
        agent.model_info = ModelInfo(
            context_length=1000000,
            max_completion_tokens=128000,
            supports_reasoning=True,
            supports_include_reasoning=True,
            supported_parameters=["reasoning", "include_reasoning", "temperature"],
        )

        assert agent.context_length == 1000000
        assert agent.supports_reasoning is True
        assert agent.supports_include_reasoning is True


# =============================================================================
# Setup Tests
# =============================================================================

@pytest.mark.asyncio
@patch("amongagents.long_context.agent.get_model_info", new_callable=AsyncMock)
async def test_setup_fetches_model_info(mock_gmi):
    """Test that setup() fetches model info and builds system prompt."""
    mock_gmi.return_value = ModelInfo(
        context_length=131072,
        supports_reasoning=True,
        supports_include_reasoning=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        mock_player = MagicMock()
        mock_player.name = "Player 1: red"
        mock_player.identity = "Crewmate"
        mock_player.personality = None

        os.environ["EXPERIMENT_PATH"] = tmpdir
        os.environ["OPENROUTER_API_KEY"] = "test-key"

        agent = LongContextAgent(
            player=mock_player,
            tools=[],
            game_index=1,
            agent_config={"CREWMATE_LLM_CHOICES": ["test-model"], "IMPOSTOR_LLM_CHOICES": ["test-model"], "temperature": 1.0},
            list_of_impostors=[],
            model="test-model",
        )

        assert agent._setup_done is False
        assert agent.system_prompt is None

        await agent.setup()

        assert agent._setup_done is True
        assert agent.system_prompt is not None
        assert agent.supports_reasoning is True
        mock_gmi.assert_called_once_with("test-model", "test-key")


# =============================================================================
# Regression: 2-impostor game must show correct counts in all prompts
# =============================================================================

def test_seven_player_two_impostor_prompt_values():
    """Regression test for experiment 2026-02-21_exp_2.

    In that game, SEVEN_MEMBER_GAME config had num_impostors=2 and
    kill_cooldown=3, but ALL players' system prompts said "1 Impostor(s)"
    and "0-timestep cooldown".  This test ensures build_system_prompt
    correctly propagates those values.
    """
    # --- Setup: two impostors, five crewmates (mirrors SEVEN_MEMBER_GAME) ---
    list_of_impostors = ["Player 2: red", "Player 6: yellow"]
    game_kwargs = dict(
        list_of_impostors=list_of_impostors,
        kill_cooldown=3,
        num_impostors=2,
        num_players=7,
    )

    # -- Crewmate prompt --
    crew = MagicMock()
    crew.name = "Player 1: pink"
    crew.identity = "Crewmate"
    crew.personality = None

    crew_prompt = build_system_prompt(player=crew, **game_kwargs)

    assert "2 Impostor(s)" in crew_prompt, (
        f"Crewmate prompt should say '2 Impostor(s)' but says: "
        f"{[l for l in crew_prompt.splitlines() if 'Impostor' in l][:3]}"
    )
    assert "3-timestep cooldown" in crew_prompt, (
        f"Crewmate prompt should say '3-timestep cooldown' but says: "
        f"{[l for l in crew_prompt.splitlines() if 'cooldown' in l][:3]}"
    )

    # -- Impostor 1 (red) prompt --
    imp1 = MagicMock()
    imp1.name = "Player 2: red"
    imp1.identity = "Impostor"
    imp1.personality = None

    imp1_prompt = build_system_prompt(player=imp1, **game_kwargs)

    assert "2 Impostor(s)" in imp1_prompt, (
        f"Impostor prompt should say '2 Impostor(s)' but says: "
        f"{[l for l in imp1_prompt.splitlines() if 'Impostor' in l][:3]}"
    )
    assert "3-timestep cooldown" in imp1_prompt, (
        f"Impostor prompt should say '3-timestep cooldown' but says: "
        f"{[l for l in imp1_prompt.splitlines() if 'cooldown' in l][:3]}"
    )
    # Red should be told about yellow as teammate
    assert "YOUR FELLOW IMPOSTOR(S): Player 6: yellow" in imp1_prompt, (
        f"Red should know about yellow. Got: "
        f"{[l for l in imp1_prompt.splitlines() if 'IMPOSTOR' in l or 'ONLY' in l][:3]}"
    )
    assert "ONLY Impostor" not in imp1_prompt, (
        "Red should NOT be told they're the ONLY impostor in a 2-impostor game"
    )

    # -- Impostor 2 (yellow) prompt --
    imp2 = MagicMock()
    imp2.name = "Player 6: yellow"
    imp2.identity = "Impostor"
    imp2.personality = None

    imp2_prompt = build_system_prompt(player=imp2, **game_kwargs)

    assert "2 Impostor(s)" in imp2_prompt
    assert "YOUR FELLOW IMPOSTOR(S): Player 2: red" in imp2_prompt, (
        f"Yellow should know about red. Got: "
        f"{[l for l in imp2_prompt.splitlines() if 'IMPOSTOR' in l or 'ONLY' in l][:3]}"
    )
    assert "ONLY Impostor" not in imp2_prompt


@pytest.mark.asyncio
@patch("amongagents.long_context.agent.get_model_info", new_callable=AsyncMock)
async def test_setup_propagates_game_config_to_prompt(mock_gmi):
    """End-to-end test: LongContextAgent.__init__ -> setup() -> system_prompt.

    Verifies that num_impostors, kill_cooldown, and num_players passed to the
    constructor actually appear in the built system prompt after setup().
    This reproduces the full code path used by game.py::initialize_agents().
    """
    mock_gmi.return_value = ModelInfo(
        context_length=131072,
        supports_reasoning=False,
        supports_include_reasoning=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate Impostor player in a 2-impostor game
        mock_player = MagicMock()
        mock_player.name = "Player 2: red"
        mock_player.identity = "Impostor"
        mock_player.personality = None

        os.environ["EXPERIMENT_PATH"] = tmpdir
        os.environ["OPENROUTER_API_KEY"] = "test-key"

        agent = LongContextAgent(
            player=mock_player,
            tools=[],
            game_index=1,
            agent_config={
                "CREWMATE_LLM_CHOICES": ["test-model"],
                "IMPOSTOR_LLM_CHOICES": ["test-model"],
                "temperature": 1.0,
            },
            list_of_impostors=["Player 2: red", "Player 6: yellow"],
            model="test-model",
            kill_cooldown=3,
            num_impostors=2,
            num_players=7,
        )

        await agent.setup()

        assert "2 Impostor(s)" in agent.system_prompt, (
            f"System prompt should contain '2 Impostor(s)'. "
            f"Prompt snippet: {agent.system_prompt[:300]}"
        )
        assert "3-timestep cooldown" in agent.system_prompt, (
            f"System prompt should contain '3-timestep cooldown'. "
            f"Prompt snippet: {agent.system_prompt[:500]}"
        )
        assert "7 players" in agent.system_prompt
        assert "YOUR FELLOW IMPOSTOR(S): Player 6: yellow" in agent.system_prompt
        assert "ONLY Impostor" not in agent.system_prompt


# =============================================================================
# ShortContextAgent Tests
# =============================================================================

def _make_short_agent(tmpdir, **kwargs):
    """Helper to create a ShortContextAgent with mocked player."""
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

    agent = ShortContextAgent(
        player=mock_player,
        tools=[],
        game_index=1,
        agent_config=agent_config,
        list_of_impostors=[],
        model="test-model",
    )
    # Mark setup as done with a dummy system prompt (skip API call)
    agent.system_prompt = "Test system prompt"
    agent._setup_done = True
    return agent


@pytest.mark.asyncio
async def test_short_context_agent_memory_carry():
    """Verify that memory from turn N appears in turn N+1's user message."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from amongagents.envs.action import SkipVote
        skip_action = SkipVote("Cafeteria")

        agent = _make_short_agent(tmpdir, actions=[skip_action])

        # Turn 1: model returns memory
        agent._send_request = AsyncMock(return_value=_mock_api_result(
            '{"memory": "Player 3 was near the body", "thinking": "suspicious", "action": "SKIP VOTE"}'
        ))

        await agent.choose_action(1)
        assert agent.processed_memory == "Player 3 was near the body"

        # Turn 2: verify the memory is injected into the user message
        calls = []
        original_send = agent._send_request

        async def capture_send(messages):
            calls.append(messages)
            return _mock_api_result(
                '{"memory": "updated memory", "thinking": "still suspicious", "action": "SKIP VOTE"}'
            )

        agent._send_request = capture_send
        await agent.choose_action(2)

        # The user message in turn 2 should contain the memory from turn 1
        user_msg = calls[0][1]["content"]  # messages[1] is the user turn
        assert "Player 3 was near the body" in user_msg, (
            f"Memory from turn 1 should appear in turn 2 user message. Got: {user_msg[:200]}"
        )


@pytest.mark.asyncio
async def test_short_context_agent_no_history_accumulation():
    """Verify chat_history stays empty (context is not accumulated)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from amongagents.envs.action import SkipVote
        skip_action = SkipVote("Cafeteria")

        agent = _make_short_agent(tmpdir, actions=[skip_action])
        agent._send_request = AsyncMock(return_value=_mock_api_result(
            '{"memory": "some memory", "thinking": "test", "action": "SKIP VOTE"}'
        ))

        await agent.choose_action(1)
        await agent.choose_action(2)
        await agent.choose_action(3)

        # ShortContextAgent should NOT accumulate chat history
        assert len(agent.chat_history) == 0, (
            f"ShortContextAgent should not accumulate history, but has {len(agent.chat_history)} messages"
        )


@pytest.mark.asyncio
async def test_short_context_sends_only_system_plus_user():
    """Verify each turn sends only system + current user (no prior turns)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from amongagents.envs.action import SkipVote
        skip_action = SkipVote("Cafeteria")

        agent = _make_short_agent(tmpdir, actions=[skip_action])

        calls = []

        async def capture_send(messages):
            calls.append(messages)
            return _mock_api_result(
                '{"memory": "m", "thinking": "t", "action": "SKIP VOTE"}'
            )

        agent._send_request = capture_send

        await agent.choose_action(1)
        await agent.choose_action(2)

        # Each call should have exactly 2 messages: system + user
        for i, msg_list in enumerate(calls):
            assert len(msg_list) == 2, (
                f"Turn {i+1} should send 2 messages (system + user), got {len(msg_list)}"
            )
            assert msg_list[0]["role"] == "system"
            assert msg_list[1]["role"] == "user"


def test_short_context_json_format_includes_memory():
    """Verify the short-context JSON output schema includes memory field."""
    assert '"memory"' in SHORT_CONTEXT_JSON_FORMAT
    assert '"thinking"' in SHORT_CONTEXT_JSON_FORMAT
    assert '"action"' in SHORT_CONTEXT_JSON_FORMAT


def test_short_context_json_format_reasoning_includes_memory():
    """Verify the reasoning variant also includes memory field."""
    assert '"memory"' in SHORT_CONTEXT_JSON_FORMAT_REASONING
    assert '"action"' in SHORT_CONTEXT_JSON_FORMAT_REASONING
    # Reasoning variant should NOT have thinking (model uses native reasoning)
    assert '"thinking"' not in SHORT_CONTEXT_JSON_FORMAT_REASONING


def test_build_system_prompt_short_context_has_memory():
    """Verify build_system_prompt_short_context produces prompt with memory field."""
    mock_player = MagicMock()
    mock_player.name = "Player 1: red"
    mock_player.identity = "Crewmate"
    mock_player.personality = None

    prompt = build_system_prompt_short_context(
        player=mock_player,
        list_of_impostors=[],
        kill_cooldown=0,
        num_impostors=1,
        num_players=5,
    )

    assert '"memory"' in prompt, "Short-context system prompt must include memory field"
    assert '"action"' in prompt
    # Should NOT contain the standard format (it should have been replaced)
    assert JSON_OUTPUT_FORMAT not in prompt, (
        "Short-context prompt should not contain the standard JSON format"
    )


def test_build_system_prompt_short_context_reasoning_has_memory():
    """Verify reasoning variant of short-context prompt includes memory field."""
    mock_player = MagicMock()
    mock_player.name = "Player 1: red"
    mock_player.identity = "Crewmate"
    mock_player.personality = None

    prompt = build_system_prompt_short_context(
        player=mock_player,
        list_of_impostors=[],
        kill_cooldown=0,
        num_impostors=1,
        num_players=5,
        supports_reasoning=True,
    )

    assert '"memory"' in prompt, "Short-context reasoning prompt must include memory field"
    assert JSON_OUTPUT_FORMAT not in prompt


def test_build_short_context_user_turn():
    """Verify memory is included in the user turn message."""
    result = build_short_context_user_turn(5, "Location info", "Player 3 is suspicious")
    assert "=== Turn 5 ===" in result
    assert "Location info" in result
    assert "Player 3 is suspicious" in result
    assert "Previous memory:" in result
