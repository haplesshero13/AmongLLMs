#!/usr/bin/env python3
# Usage examples:
#   uv run main.py --num_games 1 --models "model1,model2,model3,model4,model5" --unique --game_size 5
#   uv run main.py --num_games 10 --crewmate_llm "openai/gpt-4o" --impostor_llm "anthropic/claude-3.5-sonnet"

import asyncio
import copy
import os
import random
import sys
from typing import List, Optional

sys.path.append(os.path.join(os.path.abspath("."), "among-agents"))

import argparse
import datetime
import subprocess

from amongagents.envs.configs.agent_config import ALL_LLM
from amongagents.envs.configs.game_config import (
    FIVE_MEMBER_GAME,
    SEVEN_MEMBER_GAME,
)
from amongagents.envs.configs.map_config import map_coords
from amongagents.envs.game import AmongUs
from amongagents.UI.MapUI import MapUI
from dotenv import load_dotenv

from utils import setup_experiment

ROOT_PATH = os.path.abspath(".")
LOGS_PATH = os.path.join(ROOT_PATH, "expt-logs")
ASSETS_PATH = os.path.join(ROOT_PATH, "among-agents", "amongagents", "assets")
BLANK_MAP_IMAGE = os.path.join(ASSETS_PATH, "blankmap.png")

load_dotenv()

DATE = datetime.datetime.now().strftime("%Y-%m-%d")
COMMIT_HASH = (
    subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
)

# only used as a fallback if those lists are missing or if you
# explicitly set the tournament_style to "1on1" (which randomly
# picks one model from the big list for each role
BIG_LIST_OF_MODELS: List[str] = [
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-opus",
    "anthropic/claude-3.7-sonnet:thinking",
    "anthropic/claude-3.7-sonnet",
    "openai/o3",
    "openai/o4-mini-high",
    "openai/gpt-4o",
    "deepseek/deepseek-r1",
    "deepseek/deepseek-chat-v3-0324",
    "deepseek/deepseek-r1-distill-llama-70b",
    "google/gemini-2.5-pro-preview-03-25",
    "google/gemini-2.0-flash-001",
    "google/gemma-3-4b-it",
    "qwen/qwen3-235b-a22b",
    "qwen/qwen-2.5-7b-instruct",
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-small-3.1-24b-instruct",
    "x-ai/grok-3-beta",
    "microsoft/phi-4",
]

DEFAULT_ARGS = {
    "game_config": SEVEN_MEMBER_GAME,
    "include_human": False,
    "human_role": None,
    "test": False,
    "personality": False,
    "agent_config": {
        "Impostor": "LLM",
        "Crewmate": "LLM",
        "IMPOSTOR_LLM_CHOICES": BIG_LIST_OF_MODELS,
        "CREWMATE_LLM_CHOICES": BIG_LIST_OF_MODELS,
    },
    "UI": False,
}
ARGS = copy.deepcopy(DEFAULT_ARGS)


async def multiple_games(experiment_name=None, num_games=1, rate_limit=50):
    experiment_name = setup_experiment(
        experiment_name, LOGS_PATH, DATE, COMMIT_HASH, ARGS
    )
    ui = MapUI(BLANK_MAP_IMAGE, map_coords, debug=False) if ARGS["UI"] else None
    with open(
        os.path.join(os.environ["EXPERIMENT_PATH"], "experiment-details.txt"), "a"
    ) as experiment_file:
        experiment_file.write(f"\nExperiment args: {ARGS}\n")

    semaphore = asyncio.Semaphore(rate_limit)

    async def run_limited_game(game_index):
        async with semaphore:
            try:
                if ARGS.get("tournament_style") == "1on1":
                    # Randomly select one model for each role for this specific game
                    game_config = ARGS["agent_config"].copy()
                    game_config["CREWMATE_LLM_CHOICES"] = [
                        random.choice(BIG_LIST_OF_MODELS)
                    ]
                    game_config["IMPOSTOR_LLM_CHOICES"] = [
                        random.choice(BIG_LIST_OF_MODELS)
                    ]
                else:
                    game_config = ARGS["agent_config"]

                game = AmongUs(
                    game_config=ARGS["game_config"],
                    include_human=ARGS["include_human"],
                    human_role=ARGS.get("human_role"),
                    test=ARGS["test"],
                    personality=ARGS["personality"],
                    agent_config=game_config,
                    UI=ui,
                    game_index=game_index,
                )
                await game.run_game()
            except Exception as e:
                print(f"Game {game_index} failed with error: {e}")
                import traceback

                traceback.print_exc()

    tasks = [run_limited_game(i) for i in range(1, num_games + 1)]
    await asyncio.gather(*tasks)


def parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        "Expected a boolean value (true/false, 1/0, yes/no)."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an AmongUs experiment.")
    parser.add_argument(
        "--name", type=str, default=None, help="Optional name for the experiment."
    )
    parser.add_argument(
        "--num_games", type=int, default=2, help="Number of games to run."
    )
    parser.add_argument(
        "--display_ui",
        type=parse_bool_arg,
        nargs="?",
        const=True,
        default=False,
        help="Display the map UI (single-game runs only).",
    )
    parser.add_argument(
        "--crewmate_llm", type=str, default=None, help="Crewmate LLM model."
    )
    parser.add_argument(
        "--impostor_llm", type=str, default=None, help="Impostor LLM model."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models for all players (e.g., 'model1,model2,model3').",
    )
    parser.add_argument(
        "--unique",
        action="store_true",
        help="Assign each player a unique model from the list (no duplicates).",
    )
    parser.add_argument(
        "--game_size",
        type=int,
        default=7,
        choices=[5, 7],
        help="Number of players: 5 or 7 (default: 7).",
    )
    parser.add_argument("--streamlit", type=bool, default=False, help="Streamlit.")
    parser.add_argument(
        "--tournament_style", type=str, default="random", help="random or 1on1."
    )
    parser.add_argument(
        "--long_context",
        action="store_true",
        help="Use LongContextAgent instead of standard LLMAgent (multi-turn conversation format).",
    )
    parser.add_argument(
        "--short_context",
        action="store_true",
        help="Use ShortContextAgent (JSON output + memory-based context, no full history).",
    )
    parser.add_argument(
        "--role",
        type=str,
        default=None,
        choices=["impostor", "crewmate"],
        help="Launch with one human player and request that role (CLI prompts, optionally with --display_ui True).",
    )
    return parser


def configure_args_from_cli(args: argparse.Namespace) -> dict:
    configured_args = copy.deepcopy(DEFAULT_ARGS)

    # Set game config based on size
    if args.game_size == 5:
        configured_args["game_config"] = FIVE_MEMBER_GAME
    else:
        configured_args["game_config"] = SEVEN_MEMBER_GAME

    configured_args["UI"] = bool(args.display_ui) and args.num_games == 1

    # Handle model selection
    if args.models:
        # Parse comma-separated model list
        model_list = [m.strip() for m in args.models.split(",")]
        configured_args["agent_config"]["CREWMATE_LLM_CHOICES"] = model_list
        configured_args["agent_config"]["IMPOSTOR_LLM_CHOICES"] = model_list
    elif args.crewmate_llm or args.impostor_llm:
        # Legacy single-model flags
        if args.crewmate_llm:
            configured_args["agent_config"]["CREWMATE_LLM_CHOICES"] = [
                args.crewmate_llm
            ]
        if args.impostor_llm:
            configured_args["agent_config"]["IMPOSTOR_LLM_CHOICES"] = [
                args.impostor_llm
            ]

    # Set assignment mode
    if args.unique:
        configured_args["agent_config"]["assignment_mode"] = "unique"

        # Validate: unique mode requires at least as many models as players
        num_players = configured_args["game_config"]["num_players"]
        model_list = configured_args["agent_config"]["CREWMATE_LLM_CHOICES"]
        if len(model_list) < num_players:
            print(
                f"Error: --unique requires at least {num_players} models for a {num_players}-player game."
            )
            print(f"       You provided {len(model_list)} model(s): {model_list}")
            sys.exit(1)

    configured_args["tournament_style"] = args.tournament_style

    # Handle agent type selection
    if args.long_context:
        configured_args["agent_config"]["Impostor"] = "LongContext"
        configured_args["agent_config"]["Crewmate"] = "LongContext"
        print("Using LongContextAgent")
    elif args.short_context:
        configured_args["agent_config"]["Impostor"] = "ShortContext"
        configured_args["agent_config"]["Crewmate"] = "ShortContext"
        print("Using ShortContextAgent")

    # Role contract at the entrypoint: choosing a role implies a human-controlled game.
    if args.role is not None:
        configured_args["include_human"] = True
        configured_args["human_role"] = args.role
        # The main.py human flow is terminal-driven; force CLI human input mode.
        os.environ["FLASK_ENABLED"] = "False"

    return configured_args


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    ARGS = configure_args_from_cli(args)

    asyncio.run(multiple_games(experiment_name=args.name, num_games=args.num_games))
