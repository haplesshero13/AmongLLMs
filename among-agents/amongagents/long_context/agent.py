"""LongContextAgent - Multi-turn conversational Among Us agent.

Key differences from LLMAgent (Season 0):
- Full chat history maintained (no summarization/truncation)
- Structured JSON output (not free-text with [Action] tags)
- Token usage captured from OpenRouter API responses
- Per-player JSONL logging with unified `thinking` + `action` fields
"""

import json
import os
import random
import re
from datetime import datetime
from typing import Any, Optional

import aiohttp

from amongagents.envs.action import SkipVote
from amongagents.long_context.prompts import (
    build_system_prompt,
    build_user_turn,
    build_correction_prompt,
)
from amongagents.long_context.model_info import get_model_info, ModelInfo


class LongContextAgent:
    """Multi-turn conversational Among Us agent with full context history."""

    def __init__(
        self,
        player,
        tools,
        game_index: int,
        agent_config: dict,
        list_of_impostors: list[str],
        model: Optional[str] = None,
        kill_cooldown: int = 0,
        num_impostors: int = 1,
        num_players: int = 7,
        log_dir=None,
    ):
        self.player = player
        self.game_index = game_index
        self.issues = []

        # Model selection (same logic as LLMAgent)
        if model is None:
            if player.identity == "Crewmate":
                model = random.choice(agent_config["CREWMATE_LLM_CHOICES"])
            else:
                model = random.choice(agent_config["IMPOSTOR_LLM_CHOICES"])
        self.model = model
        self.temperature = agent_config.get("temperature", 1.0)

        # API config
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

        # Prompt + model info are set during async setup(), before the game loop starts.
        self.system_prompt = None
        self.model_info: Optional[ModelInfo] = None
        self._setup_done: bool = False

        # Save init params for deferred setup
        self._init_list_of_impostors = list_of_impostors
        self._init_kill_cooldown = kill_cooldown
        self._init_num_impostors = num_impostors
        self._init_num_players = num_players

        # Full conversation history (no truncation)
        self.chat_history: list[dict] = []

        # Single JSONL log for all players
        effective_log_dir = log_dir or os.getenv("EXPERIMENT_PATH") or "."
        self.log_path = os.path.join(effective_log_dir, "agent-logs.jsonl")

        # Token usage tracking (populated from API response `usage` field)
        self.tokens_cumulative: int = 0
        self.usage_log: list[dict] = []

    # -------------------------------------------------------------------------
    # Async setup — must be called before the game loop starts
    # -------------------------------------------------------------------------

    async def setup(self):
        """Fetch model capabilities and build the system prompt.

        Must be called once before the first choose_action(). This is
        separated from __init__ because it requires async I/O.
        """
        if self._setup_done:
            return

        # Fetch model info from OpenRouter /api/v1/models
        if self.api_key:
            self.model_info = await get_model_info(self.model, self.api_key)

        # Build system prompt with correct JSON format for this model
        self.system_prompt = build_system_prompt(
            player=self.player,
            list_of_impostors=self._init_list_of_impostors,
            kill_cooldown=self._init_kill_cooldown,
            num_impostors=self._init_num_impostors,
            num_players=self._init_num_players,
            supports_reasoning=self.supports_reasoning,
        )
        self._setup_done = True

        if self.model_info:
            print(
                f"  [{self.player.name}] {self.model}: "
                f"ctx={self.model_info.context_length:,}, "
                f"reasoning={self.supports_reasoning}"
            )

    # -------------------------------------------------------------------------
    # Convenience properties derived from model_info
    # -------------------------------------------------------------------------

    @property
    def context_length(self) -> Optional[int]:
        return self.model_info.context_length if self.model_info else None

    @property
    def supports_reasoning(self) -> bool:
        return self.model_info.supports_reasoning if self.model_info else False

    @property
    def supports_include_reasoning(self) -> bool:
        return self.model_info.supports_include_reasoning if self.model_info else False

    # -------------------------------------------------------------------------
    # Core action selection
    # -------------------------------------------------------------------------

    async def choose_action(self, timestep: int):
        """Main entry point called by game.py::agent_step().

        Returns:
            Action object selected by the agent

        Raises:
            RuntimeError: If all retry attempts fail in task phase
        """
        # Safety net — setup() should have been called already
        if not self._setup_done:
            await self.setup()

        available_actions = self.player.get_available_actions()
        all_info = self.player.all_info_prompt()

        # Build current user turn content
        current_user_content = build_user_turn(timestep, all_info)

        # Base messages = system + full history + current turn
        base_messages = [
            {"role": "system", "content": self.system_prompt},
            *self.chat_history,
            {"role": "user", "content": current_user_content},
        ]

        # Retry loop (3 format attempts)
        messages = base_messages.copy()
        last_error = None

        for attempt in range(3):
            result = await self._send_request(messages)
            message = result["message"]
            usage = result["usage"]
            response = message.get("content", "")

            # Parse JSON
            parsed = self._parse_json_response(response)
            if parsed is None:
                error = f"Response is not valid JSON. Got: {response[:150]!r}"
                last_error = error
                self._record_issue(
                    "format",
                    error,
                    attempt + 1,
                    timestep=timestep,
                    response_snippet=response[:200],
                )
                messages = base_messages + [
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": build_correction_prompt(
                            error,
                            attempt + 1,
                            available_actions,
                            supports_reasoning=self.supports_reasoning,
                        ),
                    },
                ]
                continue

            # Match action
            action_str = parsed.get("action", "")
            action = self._match_action(action_str, available_actions)
            if action is None:
                error = (
                    f"Action '{action_str[:80]}' not found in available actions. "
                    f"Available: {[repr(a) for a in available_actions[:5]]}"
                )
                last_error = error
                self._record_issue(
                    "format",
                    error,
                    attempt + 1,
                    timestep=timestep,
                    response_snippet=response[:200],
                )
                messages = base_messages + [
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": build_correction_prompt(
                            error,
                            attempt + 1,
                            available_actions,
                            supports_reasoning=self.supports_reasoning,
                        ),
                    },
                ]
                continue

            # SUCCESS
            if attempt > 0:
                print(
                    f"[LongContext Retry SUCCESS attempt {attempt + 1}] "
                    f"{self.player.name}: {repr(action)}"
                )
                if self.issues:
                    self.issues[-1]["resolved"] = True
                    self.issues[-1]["resolved_on_attempt"] = attempt + 1

            # Append to permanent history (only role + content, no metadata)
            self.chat_history.append({"role": "user", "content": current_user_content})
            self.chat_history.append({"role": "assistant", "content": response})

            # Track token usage from API response
            self._record_usage(usage, timestep)

            # Determine `thinking`:
            #   - Reasoning models: native reasoning tokens from the API
            #   - Non-reasoning models: "thinking" field from the JSON response
            if self.supports_reasoning:
                thinking = message.get("reasoning", "")
            else:
                thinking = parsed.get("thinking", "")

            # Log: input messages + unified thinking + action
            if len(self.chat_history) == 2:
                log_messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": current_user_content},
                ]
            else:
                log_messages = [
                    {"role": "user", "content": current_user_content},
                ]

            self._log_turn(
                log_messages,
                thinking=thinking,
                action=action_str,
                step=timestep,
                usage=usage,
            )

            return action

        # All 3 attempts failed
        is_voting = all(a.name in ["VOTE", "SKIP VOTE"] for a in available_actions)
        if is_voting:
            skip = SkipVote(current_location=self.player.location)
            print(
                f"\n[LongContext FALLBACK] {self.player.name} defaulting to SKIP VOTE "
                f"after 3 failed retries. Last error: {last_error}"
            )

            self.chat_history.append({"role": "user", "content": current_user_content})
            self.chat_history.append(
                {"role": "assistant", "content": '{"action": "SKIP VOTE"}'}
            )

            if len(self.chat_history) == 2:
                log_messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": current_user_content},
                ]
            else:
                log_messages = [
                    {"role": "user", "content": current_user_content},
                ]

            self._log_turn(
                log_messages,
                thinking=f"[FALLBACK] {last_error}",
                action="SKIP VOTE",
                step=timestep,
            )
            return skip

        raise RuntimeError(
            f"LongContextAgent format validation failed after 3 retries for "
            f"{self.player.name} ({self.model}). Last error: {last_error}"
        )

    # -------------------------------------------------------------------------
    # API communication
    # -------------------------------------------------------------------------

    async def _send_request(self, messages: list[dict]) -> dict:
        """POST to OpenRouter. 5 API retries (same as LLMAgent).

        Returns:
            Dict with "message" (the assistant message) and "usage" (token counts from API).

        Raises:
            RuntimeError: After all retries exhausted
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "repetition_penalty": 1,
            "top_k": 0,
        }

        if self.supports_reasoning:
            payload["reasoning"] = {"enabled": True}
        if self.supports_include_reasoning:
            payload["include_reasoning"] = True

        last_error = None
        async with aiohttp.ClientSession() as session:
            for attempt in range(5):
                try:
                    async with session.post(
                        self.api_url, headers=headers, data=json.dumps(payload)
                    ) as resp:
                        text = await resp.text()

                        if resp.status != 200:
                            error_msg = text[:200]
                            try:
                                error_data = json.loads(text)
                                error_msg = error_data.get("error", {}).get(
                                    "message", text[:200]
                                )
                            except:
                                pass

                            last_error = f"HTTP {resp.status}: {error_msg}"
                            self._record_issue(
                                "api", last_error, attempt + 1, http_status=resp.status
                            )

                            if resp.status in (401, 403, 404):
                                break
                            continue

                        data = json.loads(text)
                        if "choices" not in data or not data["choices"]:
                            last_error = "No choices in response"
                            self._record_issue("api", last_error, attempt + 1)
                            continue

                        choice = data["choices"][0]
                        message = choice.get("message", {})

                        content = message.get("content", "")
                        if content and content.strip():
                            # Build clean message dict; include reasoning only if present
                            clean_msg = {"role": "assistant", "content": content}
                            reasoning = message.get("reasoning")
                            if reasoning:
                                clean_msg["reasoning"] = reasoning

                            # Capture usage from API response
                            usage = data.get("usage", {})

                            return {"message": clean_msg, "usage": usage}

                        last_error = "Empty response content"
                        self._record_issue("api", last_error, attempt + 1)

                except Exception as e:
                    last_error = f"Exception: {str(e)}"
                    self._record_issue(
                        "api", last_error, attempt + 1, exception_type=type(e).__name__
                    )
                    continue

        raise RuntimeError(
            f"API request failed after 5 retries for {self.model}. Last error: {last_error}"
        )

    # -------------------------------------------------------------------------
    # Response parsing and action matching
    # -------------------------------------------------------------------------

    def _parse_json_response(self, response: str) -> Optional[dict]:
        """Try to extract a JSON dict from the response string.

        Tries: 1) Direct parse  2) ```json block  3) First { ... } block
        """
        # 1. Direct parse
        try:
            data = json.loads(response.strip())
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        # 2. Extract ```json ... ``` block
        code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except Exception:
                pass

        # 3. Extract first { ... } block (non-greedy)
        brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except Exception:
                pass

        return None

    def _match_action(self, action_str: str, available_actions: list):
        """Match action_str against available_actions.

        Handles special cases for SPEAK, CALL MEETING/REPORT, VOTE, and SKIP VOTE.
        """
        action_str_lower = action_str.lower()
        action_str_norm = " ".join(action_str.split())

        for action in available_actions:
            action_repr = repr(action)
            action_repr_lower = action_repr.lower()
            action_repr_norm = " ".join(action_repr.split())

            # Exact match
            if action_repr in action_str:
                return action
            # Case-insensitive match
            if action_repr_lower in action_str_lower:
                return action
            # Normalized whitespace match
            if action_repr_norm.lower() in action_str_norm.lower():
                return action

            # Handle SPEAK action specially
            if action.name == "SPEAK" and "speak" in action_str_lower:
                speak_match = re.search(
                    r"speak[:\s]+(.+)", action_str, re.IGNORECASE | re.DOTALL
                )
                if speak_match:
                    action.provide_message(speak_match.group(1).strip())
                    return action

            # Handle CALL MEETING / REPORT DEAD BODY
            if action.name == "CALL MEETING":
                if hasattr(action, "is_report") and action.is_report:
                    if re.search(
                        r"REPORT\s+(?:DEAD\s+)?BODY", action_str, re.IGNORECASE
                    ):
                        return action
                elif hasattr(action, "is_report") and not action.is_report:
                    if re.search(r"CALL\s+MEETING", action_str, re.IGNORECASE):
                        return action

            # Handle VOTE action specially
            if action.name == "VOTE":
                vote_match = re.search(r"vote\s+(.+)", action_str, re.IGNORECASE)
                if vote_match:
                    target = vote_match.group(1).strip().lower()
                    if (
                        hasattr(action, "other_player")
                        and action.other_player.name.lower() in target
                    ):
                        return action

            # Handle SKIP VOTE
            if action.name == "SKIP VOTE":
                if "skip" in action_str_lower and "vote" in action_str_lower:
                    return action

        return None

    # -------------------------------------------------------------------------
    # Token usage tracking (from API response)
    # -------------------------------------------------------------------------

    def _record_usage(self, usage: dict, timestep: int) -> None:
        """Record token usage from the OpenRouter API response.

        The `usage` dict typically contains:
        - prompt_tokens: tokens in the input
        - completion_tokens: tokens in the output
        - total_tokens: sum of the above
        Plus provider-specific detail like completion_tokens_details.reasoning_tokens.
        """
        total = usage.get("total_tokens", 0)
        self.tokens_cumulative += total

        record = {
            "timestep": timestep,
            **usage,
            "tokens_cumulative": self.tokens_cumulative,
            "history_messages": len(self.chat_history),
        }

        # Warn if approaching context limit
        if self.context_length:
            prompt_tokens = usage.get("prompt_tokens", 0)
            if prompt_tokens:
                pct = round(prompt_tokens / self.context_length * 100.0, 1)
                record["context_pct_used"] = pct
                if pct > 80:
                    print(
                        f"\n[LongContext INFO] {self.player.name} ({self.model}): "
                        f"context {pct}% full "
                        f"({prompt_tokens:,} / {self.context_length:,} tokens, turn {timestep})"
                    )

        self.usage_log.append(record)

    @property
    def token_summary(self) -> dict:
        """Return summary of token usage across all turns."""
        return {
            "total_tokens": self.tokens_cumulative,
            "turns": len(self.usage_log),
            "avg_tokens_per_turn": (
                self.tokens_cumulative / max(len(self.usage_log), 1)
            ),
            "max_tokens_single_turn": max(
                (r.get("total_tokens", 0) for r in self.usage_log), default=0
            ),
            "log": self.usage_log,
        }

    # -------------------------------------------------------------------------
    # Issue tracking (same interface as LLMAgent)
    # -------------------------------------------------------------------------

    def _record_issue(
        self,
        issue_type,
        error_msg,
        attempt,
        timestep=None,
        response_snippet=None,
        **kwargs,
    ):
        """Record an issue (API error or format error) for later reporting."""
        issue = {
            "type": issue_type,
            "player": self.player.name,
            "model": self.model,
            "attempt": attempt,
            "error": error_msg,
            "resolved": False,
        }
        if timestep is not None:
            issue["timestep"] = timestep
        if response_snippet is not None:
            issue["response_snippet"] = response_snippet
        issue.update(kwargs)
        self.issues.append(issue)
        return issue

    # -------------------------------------------------------------------------
    # Per-player JSONL logging
    # -------------------------------------------------------------------------

    def _log_turn(
        self,
        messages: list[dict],
        *,
        thinking: str,
        action: str,
        step: int,
        usage: Optional[dict] = None,
    ):
        """Append one JSONL line to agent-logs.jsonl.

        Log format is unified regardless of model type:
        - `messages`: input context (system + user turns)
        - `thinking`: the model's reasoning (from native reasoning tokens OR
          from the "thinking" JSON field, depending on model capability)
        - `action`: the matched action string
        - `usage`: raw token usage from the OpenRouter API
        """
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        entry = {
            "game_index": self.game_index,
            "step": step,
            "timestamp": str(datetime.now()),
            "player": self.player.name,
            "identity": self.player.identity,
            "model": self.model,
            "messages": messages,
            "thinking": thinking,
            "action": action,
        }

        if usage:
            entry["usage"] = usage

        with open(self.log_path, "a") as f:
            json.dump(entry, f, separators=(",", ":"))
            f.write("\n")
            f.flush()
        print(".", end="", flush=True)

    # -------------------------------------------------------------------------
    # Compatibility stubs (used by game.py for HumanAgent/LLMAgent parity)
    # -------------------------------------------------------------------------

    def choose_observation_location(self, map):
        """Called by game.py when action is ViewMonitor."""
        import random

        return random.choice(list(map))
