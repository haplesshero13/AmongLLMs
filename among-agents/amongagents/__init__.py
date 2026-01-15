"""
AmongAgents - Among Us game simulation for AI agent research.

Core exports for running games with AI and human agents.
"""

from amongagents.envs.game import AmongUs
from amongagents.agent.agent import LLMAgent, RandomAgent, HumanAgent

__version__ = "0.1.0"

__all__ = [
    "AmongUs",
    "LLMAgent",
    "RandomAgent",
    "HumanAgent",
]
