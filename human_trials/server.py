#!/usr/bin/env python3

import os
import sys
import asyncio
import json
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

from amongagents.envs.configs.game_config import FIVE_MEMBER_GAME, SEVEN_MEMBER_GAME, THREE_MEMBER_GAME
from amongagents.envs.game import AmongUs
from amongagents.agent.agent import HumanAgent, human_action_futures
from dotenv import load_dotenv

from utils import setup_experiment
from config import CONFIG, DEFAULT_GAME_ARGS
from run import RunGames

app = FastAPI(title="Among Us Game Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    game_config: str = "FIVE_MEMBER_GAME"
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
    if not hasattr(game, 'players'):
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
        print(f"[Server] Game {game_id} completed successfully.")
        game_info["results"] = game.summary_json if hasattr(game, "summary_json") else {}
    except asyncio.CancelledError:
        game_info["status"] = "cancelled"
        print(f"[Server] Game {game_id} task was cancelled.")
    except Exception as e:
        game_info["status"] = "error"
        game_info["error_message"] = str(e)
        print(f"[Server] Error running game {game_id}: {e}")
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
            raise HTTPException(status_code=400, detail=f"Invalid game_config name: {request.game_config}")
        
        custom_args["game_config"] = game_config
        custom_args["include_human"] = request.include_human
        custom_args["tournament_style"] = request.tournament_style
        
        if request.impostor_model:
            custom_args["agent_config"]["IMPOSTOR_LLM_CHOICES"] = [request.impostor_model]
        if request.crewmate_model:
            custom_args["agent_config"]["CREWMATE_LLM_CHOICES"] = [request.crewmate_model]

        game_id = get_run_games().get_next_game_id()
        game = get_run_games().create_game(game_id=game_id, custom_args=custom_args)
        
        active_games[game_id] = {
            "game": game,
            "config": custom_args,
            "status": "created",
            "error_message": None,
            "results": None
        }
        
        response_config = {
            "game_config": request.game_config,
            "include_human": custom_args["include_human"],
            "tournament_style": custom_args["tournament_style"],
            "impostor_model": custom_args["agent_config"]["IMPOSTOR_LLM_CHOICES"][0] if custom_args["agent_config"]["IMPOSTOR_LLM_CHOICES"] else "Default",
            "crewmate_model": custom_args["agent_config"]["CREWMATE_LLM_CHOICES"][0] if custom_args["agent_config"]["CREWMATE_LLM_CHOICES"] else "Default"
        }

        print(f"[Server] Game {game_id} created successfully.")

        return {"game_id": game_id, "status": "created", "config": response_config}
    
    except Exception as e:
        print(f"[Server] Error in start_game: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/run_game/{game_id}")
async def run_game_endpoint(game_id: int, background_tasks: BackgroundTasks):
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")
    
    game_info = active_games[game_id]
    if game_info["status"] not in ["created", "error"]:
        raise HTTPException(status_code=400, detail=f"Game {game_id} is already {game_info['status']}")
    
    if game_id in game_tasks and not game_tasks[game_id].done():
        raise HTTPException(status_code=400, detail=f"Game {game_id} background task is already running.")

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
    game_status = game_info.get("status", "running" if game_id in running_games else "waiting")

    # Initialize state object
    state = {
        "timestep": game.timestep,
        "current_phase": game.current_phase,
        "is_human_turn": game.is_human_turn,
        "available_actions": [],
        "status": game_status,
        "max_timesteps": game.game_config.get("max_timesteps", 50)  # Add max_timesteps from game config
    }

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
        if hasattr(game, 'current_player') and game.current_player is not None:
            state["current_player"] = game.current_player
        elif hasattr(game, 'current_player_index') and game.current_player_index is not None:
            # Try to get the player name from the current_player_index
            if hasattr(game, 'players') and 0 <= game.current_player_index < len(game.players):
                state["current_player"] = game.players[game.current_player_index].name
        
        # Check if the current player is the human player
        if human_player_result is not None:
            human_agent, human_index = human_player_result
            current_player_name = state.get("current_player", "")
            human_player_name = human_agent.player.name
            
            # If the current player is the human player, include their available actions
            if current_player_name == human_player_name:
                # Get the available actions for the human player
                human_agent.current_available_actions = human_agent.player.get_available_actions()
                human_state = human_agent.get_current_state_for_web()
                
                # Update the state with the human player's available actions
                state["available_actions"] = human_state.get("available_actions", [])
                state["player_info"] = human_state.get("player_info", "")
                state["condensed_memory"] = human_state.get("condensed_memory", "")
    
    return state

@app.get("/api/game/{game_id}/human_info")
async def get_human_player_info(game_id: int):
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = active_games[game_id]["game"]
    human_player_result = get_human_player(game)
    
    if human_player_result is None:
        raise HTTPException(status_code=404, detail="No human player found in this game")
    
    human_agent, human_index = human_player_result
    
    # Get the human player information
    response = {
        "human_index": human_index,
        "player_name": human_agent.player.name,
        "player_color": human_agent.player.color,
        "player_identity": human_agent.player.identity
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
        return HTTPException(status_code=400, detail=f"Not currently waiting for human action in game {game_id}")
        
    future = human_action_futures[game_id]
    if future.done():
        print(f"[Server] Error: Future for game {game_id} is already done")
        return HTTPException(status_code=400, detail=f"Not currently waiting for human action in game {game_id}")
        
    try:
        print(f"[Server] Setting result for game {game_id}")
        action_data = {
            "action_index": action.action_index,
            "message": action.message,
            "condensed_memory": action.condensed_memory,
            "thinking_process": action.thinking_process
        }
        future.set_result(action_data)
        return {"status": "success"}
    except Exception as e:
        print(f"[Server] Error setting result for game {game_id}: {str(e)}")
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(future.cancel)
        except Exception as cancel_e:
            print(f"[Server] Error cancelling future for game {game_id}: {cancel_e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit action: {e}")

if __name__ == "__main__":
    import logging
    log = logging.getLogger('uvicorn')
    log.setLevel(logging.ERROR)
    
    print("Starting Among Us FastAPI Server...")
    uvicorn.run("server:app", host='0.0.0.0', port=3000, reload=True, log_level="error", access_log=False)
