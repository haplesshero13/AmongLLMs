"""System prompts and message templates for LongContextAgent.

This module contains the system prompts for crewmates and impostors, along with
templates for user messages and correction prompts.
"""

from typing import Any

from amongagents.agent.prompts import (
    IMPOSTOR_PROMPT,
    CREWMATE_PROMPT,
    PERSONALITY_PROMPT,
    CrewmatePersonalities,
    ImpostorPersonalities,
)


# =============================================================================
# JSON Output Format Instructions
# =============================================================================

JSON_OUTPUT_FORMAT = """\
OUTPUT FORMAT:
Respond with ONLY a valid JSON object — no other text, no markdown, no code blocks.

{{
  "thinking": "<your reasoning about the current situation and what to do>",
  "action": "<EXACTLY one action string copied from the Available actions list>"
}}
"""

JSON_OUTPUT_FORMAT_REASONING = """\
OUTPUT FORMAT:
Think before you act using your native reasoning output.
Respond with ONLY a valid JSON object — no other text, no markdown, no code blocks.

{{
  "action": "<EXACTLY one action string copied from the Available actions list>"
}}
"""

SHORT_CONTEXT_JSON_FORMAT = """\
OUTPUT FORMAT:
Respond with ONLY a valid JSON object — no other text, no markdown, no code blocks.

{{
  "memory": "<concise summary of key observations, suspicions, and past events you want to remember>",
  "thinking": "<your reasoning about the current situation and what to do>",
  "action": "<EXACTLY one action string copied from the Available actions list>"
}}
"""

SHORT_CONTEXT_JSON_FORMAT_REASONING = """\
OUTPUT FORMAT:
Think before you act using your native reasoning output.
Respond with ONLY a valid JSON object — no other text, no markdown, no code blocks.

{{
  "memory": "<concise summary of key observations, suspicions, and past events you want to remember>",
  "action": "<EXACTLY one action string copied from the Available actions list>"
}}
"""


# =============================================================================
# Base Role Prompts (composed from agent.prompts + JSON output format)
# =============================================================================

LONG_CONTEXT_IMPOSTOR_SYSTEM_PROMPT = IMPOSTOR_PROMPT + "\n" + JSON_OUTPUT_FORMAT


LONG_CONTEXT_CREWMATE_SYSTEM_PROMPT = CREWMATE_PROMPT + "\n" + JSON_OUTPUT_FORMAT


# =============================================================================
# Prompt Builders
# =============================================================================


def build_system_prompt(
    player,
    list_of_impostors: list[str],
    kill_cooldown: int = 0,
    num_impostors: int = 1,
    num_players: int = 7,
    supports_reasoning: bool = False,
) -> str:
    """Build the system prompt for a player.

    Args:
        player: The player object (Crewmate or Impostor)
        list_of_impostors: List of impostor names
        kill_cooldown: Kill cooldown in timesteps
        num_impostors: Number of impostors in the game
        num_players: Total number of players

    Returns:
        Formatted system prompt string
    """
    prompt_vars: dict[str, Any] = dict(
        name=player.name,
        kill_cooldown=kill_cooldown,
        num_impostors=num_impostors,
        num_players=num_players,
    )

    output_format_prompt = (
        JSON_OUTPUT_FORMAT_REASONING if supports_reasoning else JSON_OUTPUT_FORMAT
    )

    if player.identity == "Crewmate":
        system_prompt = LONG_CONTEXT_CREWMATE_SYSTEM_PROMPT.format(**prompt_vars)

        # Add personality if present
        if player.personality is not None:
            system_prompt = system_prompt.replace(
                JSON_OUTPUT_FORMAT,
                PERSONALITY_PROMPT.format(
                    personality=CrewmatePersonalities[player.personality]
                )
                + "\n\n"
                + output_format_prompt,
            )
        else:
            system_prompt = system_prompt.replace(
                JSON_OUTPUT_FORMAT, output_format_prompt
            )

    elif player.identity == "Impostor":
        # Format teammate information
        teammates = [imp for imp in list_of_impostors if imp != player.name]
        if teammates:
            prompt_vars["impostor_teammates_text"] = (
                f"YOUR FELLOW IMPOSTOR(S): {', '.join(teammates)}"
            )
        else:
            prompt_vars["impostor_teammates_text"] = (
                "You are the ONLY Impostor in this game."
            )

        system_prompt = LONG_CONTEXT_IMPOSTOR_SYSTEM_PROMPT.format(**prompt_vars)

        # Add personality if present
        if player.personality is not None:
            system_prompt = system_prompt.replace(
                JSON_OUTPUT_FORMAT,
                PERSONALITY_PROMPT.format(
                    personality=ImpostorPersonalities[player.personality]
                )
                + "\n\n"
                + output_format_prompt,
            )
        else:
            system_prompt = system_prompt.replace(
                JSON_OUTPUT_FORMAT, output_format_prompt
            )
    else:
        raise ValueError(f"Unknown player identity: {player.identity}")

    return system_prompt


def build_user_turn(timestep: int, all_info: str) -> str:
    """Build the user message content for a turn.

    Args:
        timestep: Current game timestep
        all_info: The player's all_info_prompt() output

    Returns:
        Formatted user message content
    """
    return f"=== Turn {timestep} ===\n{all_info}"


def build_correction_prompt(
    error_message: str,
    attempt: int,
    available_actions: list,
    supports_reasoning: bool = False,
) -> str:
    """Build a correction prompt for retry attempts.

    Args:
        error_message: Description of what went wrong
        attempt: Current attempt number (1-3)
        available_actions: List of available Action objects

    Returns:
        Formatted correction prompt
    """
    action_list = "\n".join([f"  - {repr(a)}" for a in available_actions])

    if supports_reasoning:
        json_req = '{"action": "<EXACTLY one action from the list below>"}'
    else:
        json_req = '{"thinking": "<your reasoning>", "action": "<EXACTLY one action from the list below>"}'

    return f"""\
Attempt {attempt}/3. Error: {error_message}

Respond with ONLY a valid JSON object: {json_req}

Available actions (copy exactly):
{action_list}
"""


def build_short_context_user_turn(timestep: int, all_info: str, memory: str) -> str:
    """Build the user message for a short-context turn (includes memory).

    Args:
        timestep: Current game timestep
        all_info: The player's all_info_prompt() output
        memory: Carried-over memory from previous turns

    Returns:
        Formatted user message content
    """
    return f"=== Turn {timestep} ===\n{all_info}\nPrevious memory:\n{memory}\n"


def build_system_prompt_short_context(
    player,
    list_of_impostors: list[str],
    kill_cooldown: int = 0,
    num_impostors: int = 1,
    num_players: int = 7,
    supports_reasoning: bool = False,
) -> str:
    """Build system prompt for ShortContextAgent (uses memory-based JSON format).

    Same as build_system_prompt but injects SHORT_CONTEXT_JSON_FORMAT
    (which includes a "memory" field) instead of JSON_OUTPUT_FORMAT.
    """
    # Build the base prompt using the standard function (always non-reasoning
    # so the prompt contains JSON_OUTPUT_FORMAT, which we then swap out).
    system_prompt = build_system_prompt(
        player=player,
        list_of_impostors=list_of_impostors,
        kill_cooldown=kill_cooldown,
        num_impostors=num_impostors,
        num_players=num_players,
        supports_reasoning=False,
    )

    # Replace with the short-context JSON format (includes memory field).
    # Note: .format() is needed because the constants use {{ }} for literal braces,
    # but build_system_prompt already called .format() so the prompt has single braces.
    short_fmt = (
        SHORT_CONTEXT_JSON_FORMAT_REASONING
        if supports_reasoning
        else SHORT_CONTEXT_JSON_FORMAT
    )
    formatted_std = JSON_OUTPUT_FORMAT.format()
    formatted_short = short_fmt.format()

    system_prompt = system_prompt.replace(formatted_std, formatted_short)

    return system_prompt
