"""Game configuration presets."""

from amongagents.envs.configs.game_config import (
    THREE_MEMBER_GAME,
    FIVE_MEMBER_GAME,
    SEVEN_MEMBER_GAME,
)
from amongagents.envs.configs.agent_config import ALL_LLM, ALL_RANDOM
from amongagents.envs.configs.map_config import map_coords
from amongagents.envs.configs.task_config import task_config

__all__ = [
    "THREE_MEMBER_GAME",
    "FIVE_MEMBER_GAME",
    "SEVEN_MEMBER_GAME",
    "ALL_LLM",
    "ALL_RANDOM",
    "map_coords",
    "task_config",
]
