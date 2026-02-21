"""LongContextAgent - Multi-turn conversational Among Us agent.

Key differences from LLMAgent (Season 0):
- Full chat history maintained (no summarization/truncation)
- Structured JSON output (not free-text with [Action] tags)
- Per-turn and cumulative token tracking
- Per-player JSONL logging (no system_prompt repetition)
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
from amongagents.long_context.token_counter import count_tokens, get_context_length


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

        # Build system prompt
        self.system_prompt = build_system_prompt(
            player=player,
            list_of_impostors=list_of_impostors,
            kill_cooldown=kill_cooldown,
            num_impostors=num_impostors,
            num_players=num_players,
        )

        # Full conversation history (no truncation)
        self.chat_history: list[dict] = []

        # Single JSONL log for all players
        experiment_path = os.getenv("EXPERIMENT_PATH", ".")
        self.log_path = os.path.join(experiment_path, "agent-logs.jsonl")

        # Token tracking
        self.tokens_this_turn: int = 0
        self.tokens_cumulative: int = 0
        self.token_log: list[dict] = []
        self.context_length: Optional[int] = None
        self._context_length_fetched: bool = False

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
        # Fetch context length on first call (lazy initialization)
        if not self._context_length_fetched and self.api_key:
            self.context_length = await get_context_length(self.model, self.api_key or "")
            self._context_length_fetched = True

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
            response = await self._send_request(messages)

            # Parse JSON
            parsed = self._parse_json_response(response)
            if parsed is None:
                error = f"Response is not valid JSON. Got: {response[:150]!r}"
                last_error = error
                self._record_issue("format", error, attempt + 1, timestep=timestep,
                                   response_snippet=response[:200])
                messages = base_messages + [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": build_correction_prompt(
                        error, attempt + 1, available_actions)},
                ]
                continue

            # Match action
            action_str = parsed.get("action", "")
            action = self._match_action(action_str, available_actions)
            if action is None:
                error = (f"Action '{action_str[:80]}' not found in available actions. "
                         f"Available: {[repr(a) for a in available_actions[:5]]}")
                last_error = error
                self._record_issue("format", error, attempt + 1, timestep=timestep,
                                   response_snippet=response[:200])
                messages = base_messages + [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": build_correction_prompt(
                        error, attempt + 1, available_actions)},
                ]
                continue

            # SUCCESS
            if attempt > 0:
                print(f"[LongContext Retry SUCCESS attempt {attempt+1}] "
                      f"{self.player.name}: {repr(action)}")
                if self.issues:
                    self.issues[-1]["resolved"] = True
                    self.issues[-1]["resolved_on_attempt"] = attempt + 1

            # Update token tracking
            self._update_token_tracking(base_messages, response, timestep)

            # Append to permanent history
            self.chat_history.append({"role": "user", "content": current_user_content})
            self.chat_history.append({"role": "assistant", "content": response})

            # Log interaction
            self._log_turn(response, timestep)

            return action

        # All 3 attempts failed
        is_voting = all(a.name in ["VOTE", "SKIP VOTE"] for a in available_actions)
        if is_voting:
            skip = SkipVote(current_location=self.player.location)
            print(f"\n[LongContext FALLBACK] {self.player.name} defaulting to SKIP VOTE "
                  f"after 3 failed retries. Last error: {last_error}")
            self._log_turn(f"[FALLBACK] SKIP VOTE. Last error: {last_error}", timestep)
            return skip

        raise RuntimeError(
            f"LongContextAgent format validation failed after 3 retries for "
            f"{self.player.name} ({self.model}). Last error: {last_error}"
        )

    # -------------------------------------------------------------------------
    # API communication
    # -------------------------------------------------------------------------

    async def _send_request(self, messages: list[dict]) -> str:
        """POST to OpenRouter. 5 API retries (same as LLMAgent).

        Returns:
            Response content string

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

        last_error = None
        async with aiohttp.ClientSession() as session:
            for attempt in range(5):
                try:
                    async with session.post(
                        self.api_url, headers=headers,
                        data=json.dumps(payload)
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
                            self._record_issue("api", last_error, attempt + 1,
                                               http_status=resp.status)

                            # Don't retry on auth/permission errors
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
                            return content

                        last_error = "Empty response content"
                        self._record_issue("api", last_error, attempt + 1)

                except Exception as e:
                    last_error = f"Exception: {str(e)}"
                    self._record_issue("api", last_error, attempt + 1,
                                       exception_type=type(e).__name__)
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
        code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except Exception:
                pass

        # 3. Extract first { ... } block (non-greedy)
        brace_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
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
                speak_match = re.search(r'speak[:\s]+(.+)', action_str,
                                        re.IGNORECASE | re.DOTALL)
                if speak_match:
                    action.provide_message(speak_match.group(1).strip())
                    return action

            # Handle CALL MEETING / REPORT DEAD BODY
            if action.name == "CALL MEETING":
                if hasattr(action, "is_report") and action.is_report:
                    if re.search(r'REPORT\s+(?:DEAD\s+)?BODY', action_str, re.IGNORECASE):
                        return action
                elif hasattr(action, "is_report") and not action.is_report:
                    if re.search(r'CALL\s+MEETING', action_str, re.IGNORECASE):
                        return action

            # Handle VOTE action specially
            if action.name == "VOTE":
                vote_match = re.search(r'vote\s+(.+)', action_str, re.IGNORECASE)
                if vote_match:
                    target = vote_match.group(1).strip().lower()
                    if (hasattr(action, "other_player") and
                            action.other_player.name.lower() in target):
                        return action

            # Handle SKIP VOTE
            if action.name == "SKIP VOTE":
                if "skip" in action_str_lower and "vote" in action_str_lower:
                    return action

        return None

    # -------------------------------------------------------------------------
    # Token tracking
    # -------------------------------------------------------------------------

    def _update_token_tracking(self, messages: list[dict],
                                response: str, timestep: int) -> None:
        """Update token tracking for this turn."""
        tokens_sent = count_tokens(messages, self.model)
        tokens_recv = count_tokens(
            [{"role": "assistant", "content": response}], self.model
        )
        self.tokens_this_turn = tokens_sent + tokens_recv
        self.tokens_cumulative += self.tokens_this_turn

        token_record: dict[str, Any] = {
            "timestep": timestep,
            "tokens_sent": tokens_sent,
            "tokens_received": tokens_recv,
            "tokens_total_this_turn": self.tokens_this_turn,
            "tokens_cumulative": self.tokens_cumulative,
            "history_length": len(self.chat_history),
        }

        # Add context window info if available
        if self.context_length:
            token_record["context_length"] = self.context_length
            token_record["context_pct_used"] = float(round(
                tokens_sent / self.context_length * 100.0, 1
            ))

            # Print warning if >80% full
            if token_record["context_pct_used"] > 80:
                print(f"\n[LongContext INFO] {self.player.name} ({self.model}): "
                      f"context {token_record['context_pct_used']}% full "
                      f"({tokens_sent:,} / {self.context_length:,} tokens used, turn {timestep})")

        self.token_log.append(token_record)

    @property
    def token_summary(self) -> dict:
        """Return summary of token usage across all turns."""
        return {
            "total_tokens": self.tokens_cumulative,
            "turns": len(self.token_log),
            "avg_tokens_per_turn": (
                self.tokens_cumulative / max(len(self.token_log), 1)
            ),
            "max_tokens_single_turn": max(
                (r["tokens_total_this_turn"] for r in self.token_log), default=0
            ),
            "log": self.token_log,
        }

    # -------------------------------------------------------------------------
    # Issue tracking (same interface as LLMAgent)
    # -------------------------------------------------------------------------

    def _record_issue(self, issue_type, error_msg, attempt,
                      timestep=None, response_snippet=None, **kwargs):
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

    def _log_turn(self, response: str, step: int):
        """Append one JSONL line to agent-logs.jsonl."""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        entry = {
            "game_index": self.game_index,
            "step": step,
            "timestamp": str(datetime.now()),
            "player": self.player.name,
            "identity": self.player.identity,
            "model": self.model,
            "response": response,
            "token_tracking": {
                "this_turn": self.tokens_this_turn,
                "cumulative": self.tokens_cumulative,
                "history_messages": len(self.chat_history),
            },
        }

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
