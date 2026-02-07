import datetime
import subprocess
import uuid

# Generate unique session ID
SESSION_ID = str(uuid.uuid4())[:8]

# Get experiment date and Git commit hash
DATE = datetime.datetime.now().strftime("%Y-%m-%d")
COMMIT_HASH = (
    subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
)

# List of available models for 1on1 tournament style
BIG_LIST_OF_MODELS = [
    "microsoft/phi-4",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.7-sonnet:thinking",
    "openai/o3-mini-high",
    "openai/gpt-4o-mini",
    "deepseek/deepseek-r1-distill-llama-70b",
    "qwen/qwen-2.5-7b-instruct",
    "mistralai/mistral-7b-instruct",
    "deepseek/deepseek-r1",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openai/gpt-oss-20b:free",
]

# Actual default game configuration for Human Trials
DEFAULT_GAME_ARGS = {
    "game_config": "SEVEN_MEMBER_GAME",  # 7 players with 2 impostors
    "include_human": True,  # Set to True for human players
    "test": False,
    "personality": False,
    "agent_config": {
        "Impostor": "LLM",
        "Crewmate": "LLM",
        "IMPOSTOR_LLM_CHOICES": [
            "openai/gpt-oss-120b",
            "moonshotai/kimi-k2.5",
            "mistralai/mistral-large-2512",
            "google/gemini-3-flash-preview",
            "meta-llama/llama-3.3-70b-instruct",
            "anthropic/claude-sonnet-4.5",
        ],
        "CREWMATE_LLM_CHOICES": [
            "openai/gpt-oss-120b",
            "moonshotai/kimi-k2.5",
            "mistralai/mistral-large-2512",
            "google/gemini-3-flash-preview",
            "meta-llama/llama-3.3-70b-instruct",
            "anthropic/claude-sonnet-4.5",
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
