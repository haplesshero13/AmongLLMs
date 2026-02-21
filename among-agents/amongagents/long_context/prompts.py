"""System prompts and message templates for LongContextAgent.

This module contains the system prompts for crewmates and impostors, along with
templates for user messages and correction prompts.
"""

from typing import Any

from amongagents.agent.prompts import (
    MAP_CONFIG,
    CONNECTION_INFO,
    GAME_END_CONDITIONS,
    SHARED_ABILITIES,
    PERSONALITY_PROMPT,
    CrewmatePersonalities,
    ImpostorPersonalities,
)


# =============================================================================
# JSON Output Format Instructions
# =============================================================================

JSON_OUTPUT_FORMAT = """\
OUTPUT FORMAT:
You must respond with ONLY a JSON object — no other text, no markdown, no code blocks.

{{
  "thinking": "<your reasoning about the current situation and what to do>",
  "action": "<EXACTLY one action string copied from the Available actions list>"
}}

For SPEAK actions: "action": "SPEAK: <your message here>"
For VOTE actions: "action": "VOTE Player X: color"
For SKIP VOTE: "action": "SKIP VOTE"

Do not include any text outside the JSON object.
"""


# =============================================================================
# Base Role Prompts
# =============================================================================

LONG_CONTEXT_IMPOSTOR_SYSTEM_PROMPT = """\
You are {name}. You are an Impostor on the spaceship Skeld.
There are {num_players} players, {num_impostors} Impostor(s).

{impostor_teammates_text}

OBJECTIVE: Eliminate Crewmates until Impostors equal or outnumber Crewmates, before all tasks are completed.

""" + GAME_END_CONDITIONS + """
GAME PHASES:
- Task Phase: Move around, perform fake tasks, and KILL isolated Crewmates. You can VENT to travel quickly between connected vent locations.
- Meeting Phase: Discuss (SPEAK), then VOTE to eject a player or SKIP VOTE. Deceive others and redirect suspicion.

IMPOSTOR ABILITIES (only you can do these):
- KILL: Eliminate a Crewmate in the same room ({kill_cooldown}-timestep cooldown between kills)
- VENT: Travel instantly between connected vent locations
- COMPLETE FAKE TASK: Pretend to do tasks (doesn't actually complete them)

""" + SHARED_ABILITIES + """
IMPORTANT:
- WITNESSES can see your kills and report them! Kill only when isolated.
- Voted-out players are EJECTED and do not leave behind a body. Only Impostor KILLS leave dead bodies that can be discovered and reported.

""" + MAP_CONFIG + "\n" + CONNECTION_INFO + """
Note: Only Impostors can KILL and VENT.

""" + JSON_OUTPUT_FORMAT


LONG_CONTEXT_CREWMATE_SYSTEM_PROMPT = """\
You are {name}. You are a Crewmate on the spaceship Skeld.
There are {num_players} players, {num_impostors} Impostor(s).

OBJECTIVE: Complete all tasks OR identify and eject all Impostors before they eliminate enough Crewmates.

""" + GAME_END_CONDITIONS + """
## IMPORTANT ##
- Impostors KILL Crewmates in the same room ({kill_cooldown}-timestep cooldown between kills). If Impostors equal or outnumber Crewmates, you lose!
- Impostors can VENT between non-adjacent rooms. If you see someone vent, they are an Impostor!
- Voted-out players are EJECTED and do not leave behind a body. Only Impostor KILLS leave dead bodies that can be discovered and reported.

GAME PHASES:
- Task Phase: COMPLETE TASK at task locations, MOVE to gather evidence, REPORT DEAD BODY if you find one, or CALL MEETING in Cafeteria.
- Meeting Phase: SPEAK to share observations, then VOTE to eject suspected Impostors or SKIP VOTE if unsure.

CREWMATE ABILITY (only Crewmates can do this):
- COMPLETE TASK: Do your assigned tasks to help the crew win

""" + SHARED_ABILITIES + """
""" + MAP_CONFIG + "\n" + CONNECTION_INFO + """
Note: Only Impostors can KILL and VENT. If you see either, they are definitely an Impostor!

""" + JSON_OUTPUT_FORMAT


# =============================================================================
# Prompt Builders
# =============================================================================

def build_system_prompt(
    player,
    list_of_impostors: list[str],
    kill_cooldown: int = 0,
    num_impostors: int = 1,
    num_players: int = 7,
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
    
    if player.identity == "Crewmate":
        system_prompt = LONG_CONTEXT_CREWMATE_SYSTEM_PROMPT.format(**prompt_vars)
        
        # Add personality if present
        if player.personality is not None:
            system_prompt = system_prompt.replace(
                JSON_OUTPUT_FORMAT,
                PERSONALITY_PROMPT.format(
                    personality=CrewmatePersonalities[player.personality]
                ) + "\n\n" + JSON_OUTPUT_FORMAT
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
                ) + "\n\n" + JSON_OUTPUT_FORMAT
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


def build_correction_prompt(error_message: str, attempt: int, available_actions: list) -> str:
    """Build a correction prompt for retry attempts.
    
    Args:
        error_message: Description of what went wrong
        attempt: Current attempt number (1-3)
        available_actions: List of available Action objects
        
    Returns:
        Formatted correction prompt
    """
    action_list = "\n".join([f"  - {repr(a)}" for a in available_actions])
    
    return f"""\
Your response could not be parsed correctly.

Attempt {attempt}/3.
Error: {error_message}

You MUST respond with a valid JSON object:
{{
  "thinking": "<your reasoning>",
  "action": "<EXACTLY one of the following>"
}}

Available actions (copy the action string exactly):
{action_list}

Do not include any text outside the JSON object.
"""
