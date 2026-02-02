# app.py
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet")


class WebMapUI:
    def __init__(self, map_image_dir, room_coords, debug=False):
        self.map_image_dir = map_image_dir
        self.room_coords = room_coords
        self.debug = debug
        self.game_states = {}  # Store states for multiple games

    def update_game_state(self, env, game_index):
        game_state = {
            "game_index": game_index,
            "rooms": {},
            "task_progress": env.task_assignment.check_task_completion(),
            "game_over": env.check_game_over(),
            "activities": env.activity_log,
        }

        for room, roominfo in self.room_coords.items():
            players = env.map.get_players_in_room(room, include_new_deaths=True)
            game_state["rooms"][room] = {
                "coords": roominfo["coords"],
                "players": [
                    {"color": player.color, "is_alive": player.is_alive}
                    for player in players
                ],
            }

        self.game_states[game_index] = game_state
        socketio.emit("game_update", game_state)

    def draw_map(self, env):
        # This replaces the Tkinter draw_map method
        game_index = env.game_index
        self.update_game_state(env, game_index)

    def report(self, text):
        # This replaces the Tkinter report method
        socketio.emit("report", {"text": text})

    def quit_UI(self):
        # No need to do anything here for web UI
        pass


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    print("Client connected")
    # Send current states of all games
    for game_state in web_ui.game_states.values():
        emit("game_update", game_state)
