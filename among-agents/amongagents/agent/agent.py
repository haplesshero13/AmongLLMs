import ast
import asyncio
import json
import os
import random
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import aiohttp
import numpy as np
import requests

from amongagents.agent.neutral_prompts import *

# Set Flask environment variable to True by default
if "FLASK" not in os.environ:
    os.environ["FLASK"] = "True"

# Global dictionary to store futures for human actions, keyed by game_id
human_action_futures: Dict[int, asyncio.Future] = {}


class Agent:
    def __init__(self, player):
        self.player = player

    def respond(self, message):
        return "..."

    def choose_action(self):
        return None


class LLMAgent(Agent):
    def __init__(
        self, player, tools, game_index, agent_config, list_of_impostors, model=None
    ):
        super().__init__(player)
        if player.identity == "Crewmate":
            system_prompt = CREWMATE_PROMPT.format(name=player.name)
            if player.personality is not None:
                system_prompt += PERSONALITY_PROMPT.format(
                    personality=CrewmatePersonalities[player.personality]
                )
            system_prompt += CREWMATE_EXAMPLE
            if model is None:
                model = random.choice(agent_config["CREWMATE_LLM_CHOICES"])
        elif player.identity == "Impostor":
            system_prompt = IMPOSTOR_PROMPT.format(name=player.name)
            if player.personality is not None:
                system_prompt += PERSONALITY_PROMPT.format(
                    personality=ImpostorPersonalities[player.personality]
                )
            system_prompt += IMPOSTOR_EXAMPLE
            system_prompt += f"List of impostors: {list_of_impostors}"
            if model is None:
                model = random.choice(agent_config["IMPOSTOR_LLM_CHOICES"])

        self.system_prompt = system_prompt
        self.model = model
        self.temperature = 0.7
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.summarization = "No thought process has been made."
        self.processed_memory = "No memory has been processed."
        self.chat_history = []
        self.tools = tools
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_path = os.getenv("EXPERIMENT_PATH") + "/agent-logs.json"
        self.compact_log_path = (
            os.getenv("EXPERIMENT_PATH") + "/agent-logs-compact.json"
        )
        self.game_index = game_index

    def log_interaction(self, sysprompt, prompt, original_response, step):
        """
        Helper method to store model interactions in properly nested JSON format.
        Handles deep nesting and properly parses all string-formatted dictionaries.

        Args:
            prompt (str): The input prompt containing dictionary-like strings
            response (str): The model response containing bracketed sections
            step (str): The game step number
        """

        def parse_dict_string(s):
            if isinstance(s, str):
                # Replace any single quotes with double quotes for valid JSON
                s = s.replace("'", '"')
                s = s.replace('"', '"')
                # Properly escape newlines for JSON
                s = s.replace("\\n", "\\\\n")
                try:
                    # Try parsing as JSON first
                    try:
                        return json.loads(s)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try ast.literal_eval
                        return ast.literal_eval(s)
                except:
                    # If parsing fails, keep original string
                    return s
            return s

        def extract_action(text):
            """Extract action from response text."""
            if "[Action]" in text:
                action_parts = text.split("[Action]")
                thought = action_parts[0].strip()
                action = action_parts[1].strip()
                return {"thought": thought, "action": action}
            return text

        # Parse the prompt
        if isinstance(prompt, str):
            try:
                prompt = parse_dict_string(prompt)
            except:
                pass
        if isinstance(original_response, str):
            sections = {}
            current_section = None
            current_content = []

            for line in original_response.split("\n"):
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    if current_section:
                        sections[current_section] = " ".join(current_content).strip()
                        current_content = []
                    current_section = line[1:-1]  # Remove brackets
                elif line and current_section:
                    current_content.append(line)

            if current_section and current_content:
                sections[current_section] = " ".join(current_content).strip()

            new_response = sections if sections else original_response

            # Parse any dictionary strings in the response sections and handle [Action]
            if isinstance(new_response, dict):
                for key, value in new_response.items():
                    if isinstance(value, str):
                        new_response[key] = extract_action(value)
                    else:
                        new_response[key] = parse_dict_string(value)

        # Create the interaction object with proper nesting
        interaction = {
            "game_index": "Game " + str(self.game_index),
            "step": step,
            "timestamp": str(datetime.now()),
            "player": {
                "name": self.player.name,
                "identity": self.player.identity,
                "personality": self.player.personality,
                "model": self.model,
                "location": self.player.location,
            },
            "interaction": {
                "system_prompt": sysprompt,
                "prompt": prompt,
                "response": new_response,
                "full_response": original_response,
            },
        }

        # Write to file with minimal whitespace but still readable
        with open(self.log_path, "a") as f:
            json.dump(interaction, f, indent=2, separators=(",", ": "))
            f.write("\n")
            f.flush()
        with open(self.compact_log_path, "a") as f:
            json.dump(interaction, f, separators=(",", ": "))
            f.write("\n")
            f.flush()

        print(".", end="", flush=True)

    async def send_request(self, messages):
        """Send a POST request to OpenRouter API with the provided messages."""
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
            for attempt in range(10):
                try:
                    async with session.post(
                        self.api_url, headers=headers, data=json.dumps(payload)
                    ) as response:
                        if response is None:
                            last_error = f"Response is None for {self.model}"
                            print(f"[API Error] {last_error}")
                            continue

                        # Try to get response body for error messages
                        response_text = await response.text()

                        if response.status != 200:
                            # Parse error details from response
                            try:
                                error_data = json.loads(response_text)
                                error_msg = error_data.get("error", {}).get("message", response_text[:200])
                            except:
                                error_msg = response_text[:200]

                            last_error = f"HTTP {response.status} for {self.model}: {error_msg}"
                            print(f"[API Error] {last_error}")

                            # Don't retry on auth/permission errors - they won't succeed
                            if response.status in (401, 403, 404):
                                break
                            continue

                        data = json.loads(response_text)
                        if "choices" not in data:
                            last_error = f"'choices' key not in response for {self.model}"
                            print(f"[API Error] {last_error}")
                            continue
                        if not data["choices"]:
                            last_error = f"'choices' key is empty in response for {self.model}"
                            print(f"[API Error] {last_error}")
                            continue
                        return data["choices"][0]["message"]["content"]
                except Exception as e:
                    last_error = f"Exception for {self.model}: {str(e)}"
                    print(f"[API Error] {last_error} (attempt {attempt + 1}/10)")
                    continue

            # All retries exhausted - raise an error instead of silently failing
            error_msg = f"API request failed after all retries for model {self.model}. Last error: {last_error}"
            print(f"\n[FATAL API ERROR] {error_msg}")
            raise RuntimeError(error_msg)

    def respond(self, message):
        all_info = self.player.all_info_prompt()
        prompt = f"{all_info}\n{message}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.send_request(messages)

    async def choose_action(self, timestep):
        available_actions = self.player.get_available_actions()
        all_info = self.player.all_info_prompt()
        # phase = "Meeting phase" if len(available_actions) == 1 else "Task phase"
        phase = (
            "Meeting phase"
            if len(available_actions) == 1
            or all(a.name == "VOTE" for a in available_actions)
            else "Task phase"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Summarization: {self.summarization}\n\n{all_info}\n\nMemory: {self.processed_memory}\
                    \n\nPhase: {phase}. Return your output.",
            },
        ]

        # log everything needed to reproduce the interaction
        full_prompt = {
            "Summarization": self.summarization,
            "All Info": all_info,
            "Memory": self.processed_memory,
            "Phase": phase,
        }

        response = await self.send_request(messages)

        self.log_interaction(
            sysprompt=self.system_prompt,
            prompt=full_prompt,
            original_response=response,
            step=timestep,
        )

        pattern = r"^\[Condensed Memory\]((.|\n)*)\[Thinking Process\]((.|\n)*)\[Action\]((.|\n)*)$"
        match = re.search(pattern, response)
        if match:
            memory = match.group(1).strip()
            summarization = match.group(3).strip()
            output_action = match.group(5).strip()
            self.summarization = summarization
            self.processed_memory = memory
        else:
            output_action = response.strip()

        for action in available_actions:
            if repr(action) in output_action:
                return action
            elif "SPEAK: " in repr(action) and "SPEAK: " in output_action:
                message = output_action.split("SPEAK: ")[1]
                action.message = message
                return action
            else:
                action.message = "..."
        return action

    def choose_observation_location(self, map):
        if isinstance(map, (list, tuple)):
            return random.choice(map)
        else:
            # For sets, dicts, or other non-sequence types
            return random.choice(list(map))


class RandomAgent(Agent):
    def __init__(self, player):
        super().__init__(player)

    def choose_action(self):
        available_actions = self.player.get_available_actions()
        action = np.random.choice(available_actions)
        if action.name == "speak":
            message = "Hello, I am a crewmate."
            action.provide_message(message)
        return action

    def choose_observation_location(self, map):
        return random.sample(map, 1)[0]


class HumanAgent(Agent):
    def __init__(
        self,
        player,
        tools=None,
        game_index=0,
        agent_config=None,
        list_of_impostors=None,
    ):
        super().__init__(player)
        self.model = "homosapiens/brain-1.0"
        self.tools = tools
        self.game_index = game_index
        self.summarization = "No thought process has been made."
        self.processed_memory = "No memory has been processed."
        self.log_path = os.getenv("EXPERIMENT_PATH") + "/agent-logs.json"
        self.compact_log_path = (
            os.getenv("EXPERIMENT_PATH") + "/agent-logs-compact.json"
        )
        self.current_available_actions = []
        self.current_step = 0
        self.max_steps = 50  # Default value, will be updated from game config
        self.action_future = None  # Store the future as an instance variable
        self.condensed_memory = (
            ""  # Store the condensed memory (scratchpad) between turns
        )

    def update_max_steps(self, max_steps):
        """Update the max_steps value from the game config."""
        self.max_steps = max_steps

    async def choose_action(self, timestep: int):
        """
        Chooses an action, either via web interface (if FLASK_ENABLED=True)
        or command line (if FLASK_ENABLED=False).
        """
        use_flask = os.getenv("FLASK_ENABLED", "True") == "True"
        all_info = self.player.all_info_prompt()
        self.current_available_actions = self.player.get_available_actions()
        self.current_step = timestep

        if use_flask:
            # --- Web Interface Logic ---
            action_prompt = (
                "Waiting for human action via web interface.\nAvailable actions:\n"
                + "\n".join(
                    [
                        f"{i + 1}: {str(action)}"
                        for i, action in enumerate(self.current_available_actions)
                    ]
                )
            )
            full_prompt = {
                "All Info": all_info,
                "Available Actions": action_prompt,
                "Current Step": f"{timestep}/{self.max_steps}",
                "Current Player": self.player.name,
            }

            loop = asyncio.get_event_loop()
            self.action_future = loop.create_future()  # Store in instance variable

            # Use game_id from the server instead of game_index
            # The game_id is passed to the HumanAgent when it's created
            game_id = getattr(self, "game_id", self.game_index)
            human_action_futures[game_id] = self.action_future

            print(f"[Agent] Created future for game {game_id}")
            print(f"[Agent] Available futures: {list(human_action_futures.keys())}")

            print(
                f"\n[Game {game_id}] Human player {self.player.name}'s turn. Waiting for action via web interface..."
            )
            print(
                f"Available actions: {[str(a) for a in self.current_available_actions]}"
            )

            try:
                chosen_action_data = await self.action_future
                action_idx = chosen_action_data.get("action_index")
                action_message = chosen_action_data.get("message")
                condensed_memory = chosen_action_data.get("condensed_memory", "")
                thinking_process = chosen_action_data.get("thinking_process", "")

                # Update the condensed memory if provided
                if condensed_memory:
                    self.condensed_memory = condensed_memory

                if (
                    action_idx is None
                    or action_idx < 0
                    or action_idx >= len(self.current_available_actions)
                ):
                    print(
                        f"[Game {game_id}] Invalid action index received: {action_idx}. Defaulting to first action."
                    )
                    selected_action = self.current_available_actions[0]
                else:
                    selected_action = self.current_available_actions[action_idx]

                # Format the response log to match LLMAgent format
                response_log = ""
                if self.condensed_memory:
                    response_log += f"[Condensed Memory]\n{self.condensed_memory}\n\n"
                if thinking_process:
                    response_log += f"[Thinking Process]\n{thinking_process}\n\n"

                response_log += f"[Action] {str(selected_action)}"

                # Check if action requires a message (e.g., SPEAK)
                # Use str() and check for attributes robustly
                is_speak_action = False
                if hasattr(selected_action, "name"):  # Check attribute exists
                    is_speak_action = selected_action.name == "SPEAK"
                elif "SPEAK" in str(selected_action):  # Fallback to string check
                    is_speak_action = True

                if is_speak_action and action_message:
                    if hasattr(selected_action, "provide_message"):
                        selected_action.provide_message(action_message)
                    elif hasattr(
                        selected_action, "message"
                    ):  # Fallback to setting attribute
                        selected_action.message = action_message
                    response_log += f" {action_message}"

                # Update the prompt to not include "Waiting for human action via web interface"
                full_prompt = {
                    "All Info": all_info,
                    "Available Actions": "\n".join(
                        [
                            f"{i + 1}: {str(action)}"
                            for i, action in enumerate(self.current_available_actions)
                        ]
                    ),
                    "Current Step": f"{timestep}/{self.max_steps}",
                    "Current Player": self.player.name,
                }

                self.log_interaction(
                    sysprompt="Human Agent (Web)",
                    prompt=full_prompt,
                    original_response=response_log,
                    step=timestep,
                )

                # Clear the future and actions only after successful action selection
                if game_id in human_action_futures:
                    print(
                        f"[Agent] Deleting future for game {game_id} after successful action"
                    )
                    del human_action_futures[game_id]
                self.current_available_actions = []
                self.action_future = None

                return selected_action

            except asyncio.CancelledError:
                print(f"[Game {game_id}] Human action cancelled.")
                # Clean up on cancellation
                if game_id in human_action_futures:
                    print(
                        f"[Agent] Deleting future for game {game_id} after cancellation"
                    )
                    del human_action_futures[game_id]
                self.current_available_actions = []
                self.action_future = None
                raise
        else:
            # --- Command Line Interface Logic ---
            action_prompt = "Available actions:\n" + "\n".join(
                [
                    f"{i + 1}: {str(action)}"
                    for i, action in enumerate(self.current_available_actions)
                ]
            )
            full_prompt = {"All Info": all_info, "Available Actions": action_prompt}

            print(
                f"\n--- [Game {self.game_index}] Player: {self.player.name} ({self.player.identity if self.player.identity else 'Role Unknown'}) ---"
            )
            print(all_info)
            print("\nChoose an action:")
            for i, action in enumerate(self.current_available_actions):
                print(f"{i + 1}: {str(action)}")
            print("(Enter 0 to stop game)")

            stop_triggered = False
            valid_input = False
            selected_action = None
            action_idx_chosen = -1

            while (not stop_triggered) and (not valid_input):
                try:
                    user_input = input("> ")
                    action_idx_chosen = int(user_input)
                    if action_idx_chosen == 0:
                        stop_triggered = True
                    elif action_idx_chosen < 1 or action_idx_chosen > len(
                        self.current_available_actions
                    ):
                        print(
                            f"Invalid input. Please enter a number between 1 and {len(self.current_available_actions)} (or 0 to stop)."
                        )
                    else:
                        valid_input = True
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    continue

            if stop_triggered:
                print("Stopping game as requested by user.")
                # How to signal stop? Raise exception? Return specific value?
                # For now, raise an exception that the game loop might catch.
                raise KeyboardInterrupt("Game stopped by user via CLI.")

            selected_action = self.current_available_actions[action_idx_chosen - 1]
            response_log = f"[Action] {str(selected_action)}"

            # Check if action requires a message using string check
            is_speak_action = False
            if hasattr(selected_action, "name"):
                is_speak_action = selected_action.name == "SPEAK"
            elif "SPEAK" in str(selected_action):
                is_speak_action = True

            if is_speak_action:
                print("Enter your message:")
                action_message = input("> ")
                if hasattr(selected_action, "provide_message"):
                    selected_action.provide_message(action_message)
                elif hasattr(selected_action, "message"):
                    selected_action.message = action_message
                else:
                    print("Warning: Could not set message for SPEAK action.")
                response_log += f" {action_message}"

            self.log_interaction(
                sysprompt="Human Agent (CLI)",
                prompt=full_prompt,
                original_response=response_log,
                step=timestep,
            )

            self.current_available_actions = []  # Clear actions after use
            return selected_action  # Return synchronously within async def

    def get_current_state_for_web(self) -> Dict[str, Any]:
        """
        Returns the necessary state for the web UI when it's the human's turn.
        Uses string checks for action properties.
        """
        available_actions_web = []
        for action in self.current_available_actions:
            action_str = str(action)
            requires_message = False
            if hasattr(action, "name"):
                requires_message = action.name == "SPEAK"
            elif "SPEAK" in action_str:
                requires_message = True

            available_actions_web.append(
                {"name": action_str, "requires_message": requires_message}
            )

        return {
            "is_human_turn": True,
            "player_name": self.player.name,
            "player_info": self.player.all_info_prompt(),
            "available_actions": available_actions_web,
            "current_step": f"{self.current_step}/{self.max_steps}",
            "current_player": self.player.name,
            "condensed_memory": self.condensed_memory,  # Include the condensed memory in the state
        }

    def respond(self, message):
        print(message)
        response = input()
        return response

    def choose_observation_location(self, map):
        map_list = list(map)
        print("Please select the room you wish to observe:")
        for i, room in enumerate(map_list):
            print(f"{i}: {room}")
        while True:
            try:
                index = int(input())
                if index < 0 or index >= len(map_list):
                    print(
                        f"Invalid input. Please enter a number between 0 and {len(map_list) - 1}."
                    )
                else:
                    return map_list[index]
            except:
                print("Invalid input. Please enter a number.")

    def log_interaction(self, sysprompt, prompt, original_response, step):
        """
        Helper method to store model interactions in properly nested JSON format.
        Handles deep nesting and properly parses all string-formatted dictionaries.
        Correctly separates Memory, Thinking, and Action sections.
        """
        sections = {}

        # Clean the original response slightly for easier parsing
        response_text = original_response.strip()

        # Use regex to find sections robustly, ignoring case for tags
        action_match = re.search(
            r"\[Action\](.*)", response_text, re.DOTALL | re.IGNORECASE
        )
        memory_match = re.search(
            r"\[Condensed Memory\](.*?)(\[(Thinking Process|Action)\]|$)",
            response_text,
            re.DOTALL | re.IGNORECASE,
        )
        thinking_match = re.search(
            r"\[Thinking Process\](.*?)(\[(Condensed Memory|Action)\]|$)",
            response_text,
            re.DOTALL | re.IGNORECASE,
        )

        # Initialize keys to ensure they exist, defaulting to empty string
        sections["Condensed Memory"] = ""
        sections["Thinking Process"] = ""

        # Extract content based on matches, overwriting defaults if found
        if memory_match:
            sections["Condensed Memory"] = memory_match.group(1).strip()

        if thinking_match:
            sections["Thinking Process"] = thinking_match.group(1).strip()

        if action_match:
            action_text = action_match.group(1).strip()
            # Remove leading number format like "1. "
            action_text_cleaned = re.sub(r"^\d+\.\s*", "", action_text).strip()

            # Assign the full cleaned action string directly, regardless of message presence
            if action_text_cleaned:
                sections["Action"] = action_text_cleaned
            # If action_text_cleaned is empty after stripping number, don't add Action section

        # Handle cases where tags might be missing or text exists outside tags
        # (This logic might need refinement depending on expected variations)
        # For now, prioritize explicitly tagged sections.

        # Create the interaction object with proper nesting
        interaction = {
            "game_index": "Game " + str(self.game_index),
            "step": step,
            "timestamp": str(datetime.now()),
            "player": {
                "name": self.player.name,
                "identity": self.player.identity,
                "personality": self.player.personality,
                "model": self.model,
                "location": self.player.location,
            },
            "interaction": {
                "system_prompt": sysprompt,
                "prompt": prompt,
                "response": sections,
                "full_response": original_response,
            },
        }

        # Ensure log directories exist
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.compact_log_path), exist_ok=True)

        # Write to file with minimal whitespace but still readable
        try:
            with open(self.log_path, "a") as f:
                json.dump(interaction, f, indent=2, separators=(",", ": "))
                f.write("\n")
                f.flush()
            with open(self.compact_log_path, "a") as f:
                json.dump(interaction, f, separators=(",", ":"))
                f.write("\n")
                f.flush()
        except Exception as e:
            print(f"Error writing to log file: {e}")  # Add error logging

        print(".", end="", flush=True)


class LLMHumanAgent(HumanAgent, LLMAgent):
    def __init__(
        self,
        player,
        tools=None,
        game_index=0,
        agent_config=None,
        list_of_impostors=None,
    ):
        super().__init__(player, tools, game_index, agent_config, list_of_impostors)

    async def choose_action(self, timestep):
        return await HumanAgent.choose_action(self, timestep)

    def respond(self, message):
        return HumanAgent.respond(self, message)

    def log_interaction(self, sysprompt, prompt, original_response, step):
        return HumanAgent.log_interaction(
            self, sysprompt, prompt, original_response, step
        )
