#!/usr/bin/env python3

import os
import sys
import asyncio
import json
import traceback
from typing import Dict, Optional, Any, List
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

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
from amongagents.agent.agent import HumanAgent, human_action_futures,  human_monitor_futures, human_monitor_rooms
from dotenv import load_dotenv

from utils import setup_experiment
from config import CONFIG, DEFAULT_GAME_ARGS
from run import RunGames
from gdrive import upload_logs_to_drive

app = FastAPI(title="Among Us Game Server")

from fastapi.staticfiles import StaticFiles

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount assets directory
assets_path = os.path.join(project_root, "human_trials", "assets")
app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

load_dotenv()

# Create a singleton instance of RunGames
run_games = None


def get_run_games():
    global run_games
    if run_games is None:
        run_games = RunGames()
    return run_games


# Global variables
active_games = {}
running_games = set()

game_tasks: Dict[int, asyncio.Task] = {}


class GameStartRequest(BaseModel):
    game_config: str = "SEVEN_MEMBER_GAME"
    include_human: bool = True
    tournament_style: str = "random"
    impostor_model: Optional[str] = None
    crewmate_model: Optional[str] = None


class HumanActionRequest(BaseModel):
    action_index: int
    message: Optional[str] = None
    condensed_memory: Optional[str] = ""
    thinking_process: Optional[str] = ""


def get_game_config_by_name(name: str) -> Optional[Dict]:
    if name == "FIVE_MEMBER_GAME":
        return FIVE_MEMBER_GAME
    elif name == "SEVEN_MEMBER_GAME":
        return SEVEN_MEMBER_GAME
    elif name == "THREE_MEMBER_GAME":
        return THREE_MEMBER_GAME
    return None


def get_human_player(game: AmongUs) -> Optional[tuple[HumanAgent, int]]:
    if not hasattr(game, "players"):
        return None
    for i, agent in enumerate(game.agents):
        if isinstance(agent, HumanAgent):
            return agent, i
    return None


async def run_game_background(game_id: int):
    if game_id not in active_games:
        print(f"[Server] Error: Game {game_id} not found for background run.")
        return

    game_info = active_games[game_id]
    game = game_info["game"]

    try:
        print(f"[Server] Starting background task for game {game_id}.")
        running_games.add(game_id)  # Add to running games set
        await game.run_game()
        game_info["status"] = "completed"
        try:
            logs_path = os.path.join(script_dir, "logs")
            folder_id = os.environ.get("GCS_BUCKET_NAME")
            upload_logs_to_drive(logs_path, folder_id)
        except Exception as e:
            print(f"[Server] Error uploading logs to Google Drive for game {game_id}: {e}")
        print(f"[Server] Game {game_id} completed successfully.")
        game_info["results"] = (
            game.summary_json if hasattr(game, "summary_json") else {}
        )
    except asyncio.CancelledError:
        game_info["status"] = "cancelled"
        print(f"[Server] Game {game_id} task was cancelled.")
    except Exception as e:
        game_info["status"] = "error"
        game_info["error_message"] = str(e)
        print(f"[Server] Error running game {game_id}: {e}")
        traceback.print_exc()
    finally:
        running_games.discard(game_id)  # Remove from running games set
        if game_id in game_tasks:
            del game_tasks[game_id]
        print(f"[Server] Background task for game {game_id} finished.")


@app.get("/")
async def serve_index():
    return FileResponse("game.html")


@app.post("/api/start_game")
async def start_game(request: GameStartRequest):
    try:
        custom_args = DEFAULT_GAME_ARGS.copy()
        custom_args["agent_config"] = custom_args["agent_config"].copy()

        game_config = get_game_config_by_name(request.game_config)
        if not game_config:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid game_config name: {request.game_config}",
            )

        custom_args["game_config"] = game_config
        custom_args["include_human"] = request.include_human
        custom_args["tournament_style"] = request.tournament_style

        if request.impostor_model:
            custom_args["agent_config"]["IMPOSTOR_LLM_CHOICES"] = [
                request.impostor_model
            ]
        if request.crewmate_model:
            custom_args["agent_config"]["CREWMATE_LLM_CHOICES"] = [
                request.crewmate_model
            ]

        game_id = get_run_games().get_next_game_id()
        game = get_run_games().create_game(game_id=game_id, custom_args=custom_args)

        active_games[game_id] = {
            "game": game,
            "config": custom_args,
            "status": "created",
            "error_message": None,
            "results": None,
        }

        response_config = {
            "game_config": request.game_config,
            "include_human": custom_args["include_human"],
            "tournament_style": custom_args["tournament_style"],
            "impostor_model": custom_args["agent_config"]["IMPOSTOR_LLM_CHOICES"][0]
            if custom_args["agent_config"]["IMPOSTOR_LLM_CHOICES"]
            else "Default",
            "crewmate_model": custom_args["agent_config"]["CREWMATE_LLM_CHOICES"][0]
            if custom_args["agent_config"]["CREWMATE_LLM_CHOICES"]
            else "Default",
        }

        print(f"[Server] Game {game_id} created successfully.")

        return {"game_id": game_id, "status": "created", "config": response_config}

    except Exception as e:
        print(f"[Server] Error in start_game: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/run_game/{game_id}")
async def run_game_endpoint(game_id: int, background_tasks: BackgroundTasks):
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

    game_info = active_games[game_id]
    if game_info["status"] not in ["created", "error"]:
        raise HTTPException(
            status_code=400, detail=f"Game {game_id} is already {game_info['status']}"
        )

    if game_id in game_tasks and not game_tasks[game_id].done():
        raise HTTPException(
            status_code=400,
            detail=f"Game {game_id} background task is already running.",
        )

    game_info["status"] = "running"
    game_info["error_message"] = None

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    game_tasks[game_id] = loop.create_task(run_game_background(game_id))

    print(f"[Server] Queued background task for game {game_id}.")
    return {"game_id": game_id, "status": "running"}


@app.get("/api/game/{game_id}/state")
async def get_game_state(game_id: int):
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")

    game_info = active_games[game_id]
    game = game_info["game"]
    human_player_result = get_human_player(game)

    # Check if game has completed/errored/cancelled
    game_status = game_info.get(
        "status", "running" if game_id in running_games else "waiting"
    )

    is_spectating = True
    if human_player_result:
        human_agent, _ = human_player_result
        # If the human agent is still in the list of agents the game is iterating over
        if human_agent in game.agents:
            human_state = human_agent.get_current_state_for_web()
            is_spectating = not human_agent.player.is_alive

    # Initialize state object
    state = {
        "timestep": game.timestep,
        "current_phase": game.current_phase,
        "is_human_turn": game.is_human_turn,
        "available_actions": [],
        "status": game_status,
        "max_timesteps": game.game_config.get(
            "max_timesteps", 50
        ),  # Add max_timesteps from game config
    }

    # Always send ejection annoucement regardless if player is alive or spectating, so the front-end can decide whether to show it in the banner or not based on if it's the human's turn and if they're alive
    state["ejection_announcement"] = game.pending_system_announcement or ""

    # =========== Begin Status Bar ============  
    # Always send human player's current location for the status bar, even if they're spectating, so the front-end can decide whether to show it based on if it's the human's turn and if they're alive
    if human_player_result:
        human_agent, _ = human_player_result
        state["current_location"] = human_agent.player.location
    else:
        state["current_location"] = ""
    # =========== End status bar ============ 
    
    # For Ejection Popup for front-end catching vote tallys and ejection results
    if human_player_result:
        human_agent, _ = human_player_result
        if human_agent in game.agents:
            human_state = human_agent.get_current_state_for_web()
            state["player_info"] = human_state.get("player_info", "")
    if game.current_phase == "meeting":
        # Everyone is moved to Cafeteria in meeting_phase(), but you probably want "alive attendees"
        alive = [p.name for p in game.players if getattr(p, "is_alive", False)]
        dead  = [p.name for p in game.players if not getattr(p, "is_alive", True)]

        state["meeting_attendees"] = alive
        state["meeting_dead"] = dead  # optional (ghosts)
    else:
        state["meeting_attendees"] = []
        state["meeting_dead"] = []

    # Add results and error message if game is completed/errored
    if game_status in ["completed", "error", "cancelled"]:
        if "results" in game_info:
            state["results"] = game_info["results"]
        if "error_message" in game_info:
            state["error_message"] = game_info["error_message"]

    # Add human-specific information if it's their turn
    if game.is_human_turn and human_player_result is not None:
        human_agent, human_index = human_player_result
        human_state = human_agent.get_current_state_for_web()
        state.update(human_state)
    else:
        # If it's not the human's turn, ensure is_human_turn is False
        state["is_human_turn"] = False

        # Set the current player name if available
        if hasattr(game, "current_player") and game.current_player is not None:
            state["current_player"] = game.current_player
        elif (
            hasattr(game, "current_player_index")
            and game.current_player_index is not None
        ):
            # Try to get the player name from the current_player_index
            if hasattr(game, "players") and 0 <= game.current_player_index < len(
                game.players
            ):
                state["current_player"] = game.players[game.current_player_index].name

        # Check if the current player is the human player
        if human_player_result is not None:
            human_agent, human_index = human_player_result
            current_player_name = state.get("current_player", "")
            human_player_name = human_agent.player.name

            # If the current player is the human player, include their available actions
            if current_player_name == human_player_name:
                # Get the available actions for the human player
                human_agent.current_available_actions = (
                    human_agent.player.get_available_actions()
                )
                human_state = human_agent.get_current_state_for_web()
                state["is_alive"] = human_state.get("is_alive", "")

                # Update the state with the human player's available actions
                state["available_actions"] = human_state.get("available_actions", [])
                state["player_info"] = human_state.get("player_info", "")
                state["condensed_memory"] = human_state.get("condensed_memory", "")


    if game_id in human_monitor_futures:
        state["waiting_for_monitor_room"] = True
        state["monitor_rooms"] = human_monitor_rooms.get(game_id, [])
        state["is_human_turn"] = True  # Keep showing it's the human's turn
    state["is_spectating"] = is_spectating


    log = []
    # if getattr(game, "important_activity_log", None):
    #     log = game.important_activity_log
    # elif getattr(game, "activity_log", None):
    log = game.activity_log

    state["spectator_log_len"] = len(log)

    tail = log[-10:] if log else []
    state["spectator_feed"] = []
    start_index = max(0, len(log) - len(tail))

    for i, r in enumerate(tail):
        action_obj = r.get("action")
        if hasattr(action_obj, "action_text"):
            action_text = action_obj.action_text()
        else:
            action_text = str(action_obj)

        state["spectator_feed"].append({
            "id": start_index + i,
            "timestep": r.get("timestep"),
            "player": getattr(r.get("player"), "name", r.get("player")),
            "action": action_text,
            "phase": r.get("phase"),
            "round": r.get("round", None),
        })

    return state


@app.get("/api/game/{game_id}/human_info")
async def get_human_player_info(game_id: int):
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")

    game = active_games[game_id]["game"]
    human_player_result = get_human_player(game)

    if human_player_result is None:
        raise HTTPException(
            status_code=404, detail="No human player found in this game"
        )

    human_agent, human_index = human_player_result

    teammates = []
    # If the human is an Impostor, find the other Impostors
    if human_agent.player.identity.lower() == "impostor":
        for p in game.players:
            if p.identity.lower() == "impostor" and p.name != human_agent.player.name:
                teammates.append(p.name)

    # Get the human player information
    response = {
        "human_index": human_index,
        "player_name": human_agent.player.name,
        "player_color": human_agent.player.color,
        "player_identity": human_agent.player.identity,
        "teammates": teammates
    }

    return response


@app.post("/api/game/{game_id}/action")
async def submit_human_action(game_id: int, action: HumanActionRequest):
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

    print(f"[Server] Received action submission for game {game_id}")
    print(f"[Server] Action index: {action.action_index}, Message: {action.message}")
    print(f"[Server] Available futures: {list(human_action_futures.keys())}")

    if game_id not in human_action_futures:
        print(f"[Server] Error: No future found for game {game_id}")
        return HTTPException(
            status_code=400,
            detail=f"Not currently waiting for human action in game {game_id}",
        )

    future = human_action_futures[game_id]
    if future.done():
        print(f"[Server] Error: Future for game {game_id} is already done")
        return HTTPException(
            status_code=400,
            detail=f"Not currently waiting for human action in game {game_id}",
        )

    try:
        print(f"[Server] Setting result for game {game_id}")
        action_data = {
            "action_index": action.action_index,
            "message": action.message,
            "condensed_memory": action.condensed_memory,
            "thinking_process": action.thinking_process,
        }
        future.set_result(action_data)
        return {"status": "success"}
    except Exception as e:
        print(f"[Server] Error setting result for game {game_id}: {str(e)}")
        traceback.print_exc()
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(future.cancel)
        except Exception as cancel_e:
            print(f"[Server] Error cancelling future for game {game_id}: {cancel_e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit action: {e}")

@app.get("/api/game/{game_id}/monitor_rooms")
async def get_monitor_rooms(game_id: int):
    """Return available rooms when the game is waiting for monitor room selection."""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")

    if game_id not in human_monitor_futures:
        raise HTTPException(status_code=400, detail="Not waiting for monitor room selection")

    rooms = human_monitor_rooms.get(game_id, [])
    return {"rooms": rooms, "waiting_for_room": True}


class MonitorRoomRequest(BaseModel):
    room: str


@app.post("/api/game/{game_id}/monitor_room")
async def submit_monitor_room(game_id: int, request: MonitorRoomRequest):
    """Submit the chosen room for ViewMonitor."""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")

    if game_id not in human_monitor_futures:
        raise HTTPException(status_code=400, detail="Not waiting for monitor room selection")

    future = human_monitor_futures[game_id]
    if future.done():
        raise HTTPException(status_code=400, detail="Monitor selection already completed")

    rooms = human_monitor_rooms.get(game_id, [])
    if request.room not in rooms:
        raise HTTPException(status_code=400, detail=f"Invalid room: {request.room}")

    game = active_games[game_id]["game"]
    human_player_result = get_human_player(game)
    obs_count_before = 0
    if human_player_result:
        obs_count_before = len(human_player_result[0].player.observation_history)

    try:
        future.set_result(request.room)

        # Wait briefly for the game loop to execute ViewMonitor and append the observation
        for _ in range(20):  # Try for up to 2 seconds
            await asyncio.sleep(0.1)
            if human_player_result:
                current_count = len(human_player_result[0].player.observation_history)
                if current_count > obs_count_before:
                    # New observation was added — return it
                    monitor_result = human_player_result[0].player.observation_history[-1]
                    return {"status": "success", "room": request.room, "monitor_result": monitor_result}

        # Fallback if we couldn't get the result in time
        return {"status": "success", "room": request.room, "monitor_result": None}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit room: {e}")


@app.post("/api/game/{game_id}/end")
async def end_game(game_id: int):
    """End a game early, upload logs to Drive, and clean up."""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")

    # Cancel the running game task
    if game_id in game_tasks and not game_tasks[game_id].done():
        game_tasks[game_id].cancel()

    # Resolve any pending human futures so the game loop can exit
    if game_id in human_action_futures and not human_action_futures[game_id].done():
        human_action_futures[game_id].cancel()
    if game_id in human_monitor_futures and not human_monitor_futures[game_id].done():
        human_monitor_futures[game_id].cancel()

    game_info = active_games[game_id]
    game_info["status"] = "cancelled"

    # Upload logs to Drive
    try:
        logs_path = os.path.join(script_dir, "logs")
        folder_id = os.environ.get("GCS_BUCKET_NAME")
        upload_logs_to_drive(logs_path, folder_id)
        print(f"[Server] Logs uploaded to Drive for game {game_id}.")
    except Exception as e:
        print(f"[Server] Failed to upload logs for game {game_id}: {e}")

    return {"status": "cancelled", "game_id": game_id}



if __name__ == "__main__":
    import logging

    log = logging.getLogger("uvicorn")
    log.setLevel(logging.ERROR)

    port = int(os.environ.get("PORT", 8888))
    print(f"Starting Among Us FastAPI Server on port {port}...")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="error",
        access_log=False,
    )
