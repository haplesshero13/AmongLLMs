"""ShortContextAgent - JSON output + summarization-based context management.

Combines the structured JSON output from LongContextAgent with the
summarization approach of the old LLMAgent. Instead of keeping full
chat history, carries a "memory" field across turns.

Key differences from LongContextAgent:
- No chat_history accumulation (sends system + current turn each call)
- Model outputs a "memory" field that gets injected into the next turn
- Suitable for models with smaller context windows
"""

from typing import Optional

from amongagents.envs.action import SkipVote
from amongagents.long_context.agent import LongContextAgent
from amongagents.long_context.prompts import (
    build_correction_prompt,
    build_short_context_user_turn,
    build_system_prompt_short_context,
)


class ShortContextAgent(LongContextAgent):
    """JSON-output agent with memory-based context (no full history)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Carried-over memory from the model's own "memory" JSON field
        self.processed_memory: str = ""
        # Don't accumulate history — we override choose_action to skip it
        self.chat_history: list[dict] = []

    async def setup(self):
        """Build system prompt with short-context JSON format (includes memory field)."""
        if self._setup_done:
            return

        from amongagents.long_context.model_info import get_model_info

        if self.api_key:
            self.model_info = await get_model_info(self.model, self.api_key)

        self.system_prompt = build_system_prompt_short_context(
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
                f"  [{self.player.name}] {self.model} (short-ctx): "
                f"ctx={self.model_info.context_length:,}, "
                f"reasoning={self.supports_reasoning}"
            )

    async def choose_action(self, timestep: int):
        """Select an action using system + current turn only (no history).

        The model's "memory" field from the previous turn is injected into
        the current user message, giving the model a way to carry context
        without full chat history.
        """
        if not self._setup_done:
            await self.setup()

        available_actions = self.player.get_available_actions()
        all_info = self.player.all_info_prompt()

        # Build user turn with memory prefix
        current_user_content = build_short_context_user_turn(
            timestep, all_info, self.processed_memory
        )

        # System + single user turn (no history)
        base_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": current_user_content},
        ]

        messages = base_messages.copy()
        last_error = None

        for attempt in range(3):
            result = await self._send_request(messages)
            message = result["message"]
            usage = result["usage"]
            response = message.get("content", "")

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
                    f"[ShortContext Retry SUCCESS attempt {attempt + 1}] "
                    f"{self.player.name}: {repr(action)}"
                )
                if self.issues:
                    self.issues[-1]["resolved"] = True
                    self.issues[-1]["resolved_on_attempt"] = attempt + 1

            # Update carried memory from the model's output
            new_memory = parsed.get("memory", "")
            if new_memory:
                self.processed_memory = new_memory

            # Track token usage
            self._record_usage(usage, timestep)

            # Determine thinking
            if self.supports_reasoning:
                thinking = message.get("reasoning", "")
            else:
                thinking = parsed.get("thinking", "")

            # Log (always system + user since we don't accumulate history)
            log_messages = [
                {"role": "system", "content": self.system_prompt},
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
                f"\n[ShortContext FALLBACK] {self.player.name} defaulting to SKIP VOTE "
                f"after 3 failed retries. Last error: {last_error}"
            )

            log_messages = [
                {"role": "system", "content": self.system_prompt},
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
            f"ShortContextAgent format validation failed after 3 retries for "
            f"{self.player.name} ({self.model}). Last error: {last_error}"
        )
