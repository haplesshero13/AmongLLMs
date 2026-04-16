import datetime
import subprocess
import uuid

# Generate unique session ID
SESSION_ID = str(uuid.uuid4())[:8]

# Get experiment date and Git commit hash
DATE = datetime.datetime.now().strftime("%Y-%m-%d")
try:
    COMMIT_HASH = (
    subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
)
except (subprocess.CalledProcessError, FileNotFoundError):
    COMMIT_HASH = "unknown"

# List of available models for 1on1 tournament style
BIG_LIST_OF_MODELS = [
    "anthropic/claude-opus-4.6",
    "openai/gpt-5.4",
    "google/gemini-3.1-pro-preview",
    "meta-llama/llama-3.3-70b-instruct",
    "moonshotai/kimi-k2.5",
    "deepseek/deepseek-v3.2",
]

# Timeout settings for human player turns
HUMAN_TURN_TIMEOUT_SECONDS = 30 * 60   # 30 minutes per turn before game ends

# Actual default game configuration for Human Trials
DEFAULT_GAME_ARGS = {
    "game_config": "SEVEN_MEMBER_GAME",  # 7 players with 2 impostors
    "include_human": True,  # Set to True for human players
    "test": False,
    "personality": False,
    "agent_config": {
        "Impostor": "LongContext",
        "Crewmate": "LongContext",
        "IMPOSTOR_LLM_CHOICES": [
            "anthropic/claude-opus-4.6",
            "openai/gpt-5.4",
            "google/gemini-3.1-pro-preview",
            "meta-llama/llama-3.3-70b-instruct",
            "moonshotai/kimi-k2.5",
            "deepseek/deepseek-v3.2",
        ],
        "CREWMATE_LLM_CHOICES": [
            "anthropic/claude-opus-4.6",
            "openai/gpt-5.4",
            "google/gemini-3.1-pro-preview",
            "meta-llama/llama-3.3-70b-instruct",
            "moonshotai/kimi-k2.5",
            "deepseek/deepseek-v3.2",
        ],
        "assignment_mode": "unique",  # 'random' picks with replacement, 'unique' picks without replacement
    },
    "UI": False,
    "Streamlit": False,  # Set to False for command line
    "tournament_style": "random",  # Default tournament style
}

# Configuration dictionary
CONFIG = {
    "session_id": SESSION_ID,
    "date": DATE,
    "commit_hash": COMMIT_HASH,
    "experiment_name": "human_trials",
    "logs_path": "logs",
    "game_args": DEFAULT_GAME_ARGS,
}
