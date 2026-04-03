#!/usr/bin/env python3

import asyncio
import datetime
import os
import random
import subprocess
import sys
import threading
import uuid
from typing import Dict, List, Optional

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

sys.path.append(os.path.join(project_root, "among-agents"))
sys.path.append(project_root)

from amongagents.envs.configs.game_config import (
    FIVE_MEMBER_GAME,
    SEVEN_MEMBER_GAME,
    THREE_MEMBER_GAME,
)
from amongagents.envs.game import AmongUs
from config import BIG_LIST_OF_MODELS, CONFIG, DEFAULT_GAME_ARGS
from dotenv import load_dotenv

from utils import setup_experiment

ROOT_PATH = script_dir
LOGS_PATH = os.path.join(ROOT_PATH, CONFIG["logs_path"])

load_dotenv()

experiment_name = CONFIG["experiment_name"]


class RunGames:
    """
    A class to manage multiple AmongUs game instances.
    """

    # Class-level flag to track if setup has been done
    _setup_done = False

    def __init__(self, game_args: Optional[Dict] = None):
        """
        Initialize the RunGames class.

        Args:
            game_args: Optional dictionary of game arguments. If not provided, uses DEFAULT_GAME_ARGS.
        """
        self.games: Dict[int, AmongUs] = {}
        self.game_args = game_args if game_args else DEFAULT_GAME_ARGS.copy()
        self.next_game_id = self._detect_next_game_id()
        self.setup_experiment_once()

    def setup_experiment_once(self):
        """Setup experiment only once"""
        # Use a file-based lock to ensure this only runs once, even across module reloads
        global experiment_name

        # If experiment_name is None, call setup_experiment directly
        if experiment_name is None:
            print(f"Setting up experiment with auto-generated name")
            experiment_name, _ = setup_experiment(
                experiment_name,
                LOGS_PATH,
                CONFIG["date"],
                CONFIG["commit_hash"],
                self.game_args,
            )
            return

        # Otherwise, check if experiment directory already exists
        experiment_dir = os.path.join(LOGS_PATH, experiment_name)
        if os.path.exists(experiment_dir):
            return

        # Check if lock file exists
        lock_file = os.path.join(LOGS_PATH, f"{experiment_name}_setup.lock")
        if os.path.exists(lock_file):
            print(f"Setup already in progress or completed for {experiment_name}")
            return

        try:
            # Ensure LOGS_PATH exists
            os.makedirs(LOGS_PATH, exist_ok=True)

            # Create lock file
            with open(lock_file, "w") as f:
                f.write(f"Setup started at {datetime.datetime.now()}")

            print(f"Setting up experiment {experiment_name}")
            # The setup_experiment function now returns the experiment name
            experiment_name, _ = setup_experiment(
                experiment_name,
                LOGS_PATH,
                CONFIG["date"],
                CONFIG["commit_hash"],
                self.game_args,
            )

            # Remove lock file after successful setup
            if os.path.exists(lock_file):
                os.remove(lock_file)

        except Exception as e:
            print(f"Error during experiment setup: {e}")
            # Try to remove lock file even if setup fails
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                except:
                    pass

    def _detect_next_game_id(self) -> int:
        """Scan existing log folders to find the next game ID."""
        max_id = 0
        if os.path.exists(LOGS_PATH):
            import re
            for name in os.listdir(LOGS_PATH):
                match = re.match(r"game_(\d+)_", name)
                if match:
                    max_id = max(max_id, int(match.group(1)))
        next_id = max_id + 1
        print(f"[RunGames] Detected next game ID: {next_id}")
        return next_id

    def get_next_game_id(self):
        """Get the next game ID"""
        return self.next_game_id

    def increment_game_id(self):
        """Increment the game ID counter"""
        self.next_game_id += 1

    def log_game_start(self, game_id: int, log_dir: str = None):
        """Log the start of a game to the experiment details file"""
        if log_dir:
            with open(
                os.path.join(log_dir, "experiment-details.txt"),
                "a",
            ) as experiment_file:
                experiment_file.write(f"\nGame {game_id} started.\n")

    def create_game(
        self, game_id: Optional[int] = None, custom_args: Optional[Dict] = None, log_dir: Optional[str] = None
    ) -> AmongUs:
        """
        Create a new AmongUs game instance.

        Args:
            game_id: Optional game ID. If not provided, uses the next available ID.
            custom_args: Optional dictionary of custom game arguments to override defaults.

        Returns:
            AmongUs: The created game instance.
        """
        if game_id is None:
            game_id = self.get_next_game_id()
            self.increment_game_id()

        print(f"[RunGames] Attempting to create game with ID: {game_id}")

        # Merge custom args with default args if provided
        game_args = self.game_args.copy()
        if custom_args:
            print(
                f"[RunGames] Overriding default args with custom args for game {game_id}: {custom_args}"
            )
            # Make sure to handle nested dicts like agent_config correctly
            game_args["agent_config"] = game_args.get("agent_config", {}).copy()
            for key, value in custom_args.items():
                if key == "agent_config" and isinstance(value, dict):
                    game_args["agent_config"].update(value)
                else:
                    game_args[key] = value
        else:
            print(f"[RunGames] Using default game args for game {game_id}: {game_args}")

        # Get tournament style
        tournament_style = game_args.get("tournament_style", "random")
        print(f"[RunGames] Tournament style for game {game_id}: {tournament_style}")

        # Prepare agent config based on tournament style
        agent_config = game_args.get("agent_config", {}).copy()
        print(f"[RunGames] Initial agent_config for game {game_id}: {agent_config}")

        if tournament_style == "1on1":
            # Randomly select one model for each role for this specific game
            crew_choice = random.choice(BIG_LIST_OF_MODELS)
            impostor_choice = random.choice(BIG_LIST_OF_MODELS)
            agent_config["CREWMATE_LLM_CHOICES"] = [crew_choice]
            agent_config["IMPOSTOR_LLM_CHOICES"] = [impostor_choice]
            print(
                f"[RunGames] 1on1 style selected. Crew: {crew_choice}, Impostor: {impostor_choice}"
            )
        elif tournament_style == "random":
            # Ensure choices are lists
            if "CREWMATE_LLM_CHOICES" not in agent_config or not isinstance(
                agent_config["CREWMATE_LLM_CHOICES"], list
            ):
                agent_config["CREWMATE_LLM_CHOICES"] = [
                    random.choice(BIG_LIST_OF_MODELS)
                ]
                print(
                    f"[RunGames] Random style: Defaulting crew choice: {agent_config['CREWMATE_LLM_CHOICES'][0]}"
                )
            if "IMPOSTOR_LLM_CHOICES" not in agent_config or not isinstance(
                agent_config["IMPOSTOR_LLM_CHOICES"], list
            ):
                agent_config["IMPOSTOR_LLM_CHOICES"] = [
                    random.choice(BIG_LIST_OF_MODELS)
                ]
                print(
                    f"[RunGames] Random style: Defaulting impostor choice: {agent_config['IMPOSTOR_LLM_CHOICES'][0]}"
                )
            # If lists exist but maybe came from JSON as single strings
            if not agent_config["CREWMATE_LLM_CHOICES"]:
                agent_config["CREWMATE_LLM_CHOICES"] = [
                    random.choice(BIG_LIST_OF_MODELS)
                ]
            if not agent_config["IMPOSTOR_LLM_CHOICES"]:
                agent_config["IMPOSTOR_LLM_CHOICES"] = [
                    random.choice(BIG_LIST_OF_MODELS)
                ]
            print(
                f"[RunGames] Random style using Crew: {agent_config['CREWMATE_LLM_CHOICES']}, Impostor: {agent_config['IMPOSTOR_LLM_CHOICES']}"
            )

        # Resolve game_config if it's a string name
        current_game_config = game_args["game_config"]
        if isinstance(current_game_config, str):
            print(f"[RunGames] Resolving game_config string: {current_game_config}")
            config_map = {
                "FIVE_MEMBER_GAME": FIVE_MEMBER_GAME,
                "SEVEN_MEMBER_GAME": SEVEN_MEMBER_GAME,
                "THREE_MEMBER_GAME": THREE_MEMBER_GAME,
            }
            resolved_config = config_map.get(current_game_config)
            if resolved_config:
                game_args["game_config"] = resolved_config
                print(f"[RunGames] Resolved to: {resolved_config}")
            else:
                print(
                    f"[RunGames] Error: Unknown game_config string '{current_game_config}'. Using default."
                )
                game_args["game_config"] = FIVE_MEMBER_GAME  # Fallback

        # Ensure game_args has all required keys for AmongUs constructor
        final_args_for_game = {
            "game_config": game_args["game_config"],
            "include_human": game_args.get("include_human", False),
            "test": game_args.get("test", False),
            "personality": game_args.get("personality", False),
            "agent_config": agent_config,  # Use the processed agent_config
            "UI": None,  # Explicitly None for server
            "game_index": game_id,
            "log_dir": log_dir,
        }
        print(
            f"[RunGames] Final arguments for AmongUs.__init__ for game {game_id}: {final_args_for_game}"
        )

        # Create the game instance
        try:
            print(
                f"[RunGames] Initializing AmongUs game instance for game {game_id}..."
            )
            game = AmongUs(**final_args_for_game)
            print(
                f"[RunGames] AmongUs game instance for game {game_id} created successfully."
            )
        except Exception as e:
            print(
                f"[RunGames] !!! Error initializing AmongUs game instance for game {game_id}: {e}"
            )
            # Re-raise the exception so the caller knows creation failed
            raise

        # Store the game in the dictionary
        self.games[game_id] = game
        print(f"[RunGames] Game {game_id} stored in active games dict.")

        # Log the game start
        self.log_game_start(game_id, log_dir=log_dir)

        return game

    async def run_game(
        self, game_id: Optional[int] = None, custom_args: Optional[Dict] = None
    ) -> Dict:
        """
        Run a single game.

        Args:
            game_id: Optional game ID. If not provided, uses the next available ID.
            custom_args: Optional dictionary of custom game arguments to override defaults.

        Returns:
            Dict: The game summary.
        """
        game = self.create_game(game_id, custom_args)
        await game.run_game()
        return game.summary_json

    def get_game(self, game_id: int) -> Optional[AmongUs]:
        """
        Get a game instance by ID.

        Args:
            game_id: The ID of the game to retrieve.

        Returns:
            Optional[AmongUs]: The game instance, or None if not found.
        """
        return self.games.get(game_id)

    def get_game_count(self) -> int:
        """
        Get the number of games created.

        Returns:
            int: The number of games.
        """
        return len(self.games)

    async def run_multiple_games(
        self, count: int, custom_args: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Run multiple games in sequence.

        Args:
            count: The number of games to run.
            custom_args: Optional dictionary of custom game arguments to override defaults.

        Returns:
            List[Dict]: List of game summaries.
        """
        results = []
        for _ in range(count):
            result = await self.run_game(custom_args=custom_args)
            results.append(result)
        return results


def main():
    """Example usage of the RunGames class"""
    print("Starting Among Us: A Sandbox for Agentic Deception")

    # Create a RunGames instance
    run_games = RunGames()

    # Run a single game
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    game_results = loop.run_until_complete(run_games.run_game())
    loop.close()

    print("\nGame completed!")
    print("Game results:", game_results)


if __name__ == "__main__":
    main()


# claude to write a Flask server instead of main
# write game state (string) that returns what to show in the frontend
#
