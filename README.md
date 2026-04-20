# _AmongUs_: A Sandbox for Measuring and Detecting Agentic Deception

[Paper link.](https://arxiv.org/abs/2504.04072)

This project introduces the game "Among Us" as a model organism for lying and deception and studies how AI agents learn to express lying and deception, while evaluating the effectiveness of AI safety techniques to detect and control out-of-distribution deception.

## Overview

The aim is to simulate the popular multiplayer game "Among Us" using AI agents and analyze their behavior, particularly their ability to deceive and lie, which is central to the game's mechanics.

<img src="https://static.wikia.nocookie.net/among-us-wiki/images/f/f5/Among_Us_space_key_art_redesign.png" alt="Among Us" width="400"/>

## Setup

1. Clone the repository.

2. Install [uv](https://docs.astral.sh/uv/) if you haven't already:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Set up the environment and install dependencies:
   ```bash
   uv sync
   ```

## Run Games

To run the sandbox and log games of various LLMs playing against each other with free models, run:

```bash
uv run main.py --crewmate_llm "openai/gpt-oss-20b:free" --impostor_llm "meta-llama/llama-3.3-70b-instruct:free"
```

You will need to add a `.env` file with an [OpenRouter](https://openrouter.ai/) API key.

### Sample Commands

Run a single game with the UI enabled:

```bash
uv run main.py --num_games 1 --display_ui True
```

Run as a human crewmate from the main entrypoint:

```bash
uv run main.py --num_games 1 --role crewmate --crewmate_llm "meta-llama/llama-3.3-70b-instruct:free" --impostor_llm "meta-llama/llama-3.3-70b-instruct:free"
```

Run as a human impostor with the map UI enabled:

```bash
uv run main.py --num_games 1 --display_ui True --role impostor --crewmate_llm "meta-llama/llama-3.3-70b-instruct:free" --impostor_llm "meta-llama/llama-3.3-70b-instruct:free"
```

`--role` accepts only `impostor` or `crewmate` and enables a human-controlled player in that role.

Run a game using long-context agents (multi-turn conversation format, keeps full history):

```bash
uv run main.py --num_games 1 --long_context
```

Run a game using short-context agents (JSON output + memory-based context, no full history):

```bash
uv run main.py --num_games 1 --short_context
```

Run 10 games with free models (using Llama or GPT-based open-source models):

```bash
uv run main.py --num_games 10 --crewmate_llm "openai/gpt-oss-20b:free" --impostor_llm "meta-llama/llama-3.3-70b-instruct:free"
```
Run 20 games with 7 member game on different models (paid models) in an AI vs AI mode:

```bash
uv run main.py --num_games 20 --models "openai/gpt-5-mini","moonshotai/kimi-k2.5","mistralai/mistral-large-2512","google/gemini-3-flash-preview","meta-llama/llama-3.3-70b-instruct","anthropic/claude-opus-4.6","qwen/qwen3-next-80b-a3b-thinking" --unique --game_size 7 --name ai_vs_ai_trials
```

Run a tournament with multiple models:

```bash
uv run main.py --num_games 100 --tournament_style "1on1"
```

Alternatively, you can download 400 full-game logs (for `Phi-4-15b` and `Llama-3.3-70b-instruct`) and 810 game summaries from the [HuggingFace](https://huggingface.co/datasets/7vik/AmongUs) dataset to reproduce the results in the paper (and evaluate your own techniques!).

## Run Human Trials

The `human_trials/` directory contains a web-based interface that allows humans to play Among Us with or against AI agents. This is useful for testing agent behavior and gathering human evaluation data.

To run the human trials interface:

1. Start the FastAPI server:

   ```bash
   cd human_trials/
   uv run server.py
   ```

2. Open your browser and navigate to:

   ```
   http://localhost:3000
   ```

3. Follow the on-screen instructions to create a game and join as a human player alongside AI agents.

### Configuring Models for Human Trials

To specify which models AI agents use in the human trial interface, modify the `DEFAULT_GAME_ARGS` in `human_trials/config.py`. You can set specific models for both Impostors and Crewmates by updating the `agent_config`.

For example, to use specific OpenRouter models like `meta-llama/llama-3.3-70b-instruct:free` and `openai/gpt-oss-120b:free`, configure them as follows:

```AmongLLMs/human_trials/config.py#L35-41
    "agent_config": {
        "Impostor": "LLM",
        "Crewmate": "LLM",
        "IMPOSTOR_LLM_CHOICES": ["meta-llama/llama-3.3-70b-instruct:free"],
        "CREWMATE_LLM_CHOICES": ["openai/gpt-oss-120b:free"],
        "assignment_mode": "unique",  # Use 'unique' to ensure different models per agent
    },
```

### Model Assignment Modes

In `human_trials/config.py`, the `assignment_mode` key in `agent_config` controls how models are assigned to AI agents:

- **`random` (default)**: Each agent independently picks a random model from the provided list. This may result in the same model being used by multiple agents in the same game.
- **`unique`**: The system shuffles the provided list of models and assigns each agent a unique model from that list (no repetition).

> **Note:** When using `unique` mode, ensure you provide enough models in the `IMPOSTOR_LLM_CHOICES` and `CREWMATE_LLM_CHOICES` lists to cover all AI agents (e.g., 2 impostors and 4-5 crewmates depending on game size).

The interface provides a real-time view of the game state, allows you to make moves, participate in meetings, and vote on suspected impostors just like the AI agents.

## Testing

```bash
# Run all unit tests (skips integration tests by default)
uv run pytest

# Run integration tests (requires OPENROUTER_API_KEY)
uv run pytest -m integration

# Run all tests including integration
uv run pytest -m ""
```

## Deception ELO

After running (or downloading) the games, to reproduce our Deception ELO results, run the following notebook:

```
reports/2025_02_26_deception_ELO_v3_ci.ipynb
```

The other report files can be used to reproduce the respective results.

## Caching Activations

Once the (full) game logs are in place, use the following command to cache the activations of the LLMs:

```bash
uv run linear-probes/cache_activations.py --dataset <dataset_name>
```

This loads up the HuggingFace models and caches the activations of the specified layers for each game action step. This step is computationally expensive, so it is recommended to run this using GPUs.

Use `configs.py` to specify the model and layer to cache, and other configuration options.

## LLM-based Evaluation (for Lying, Awareness, Deception, and Planning)

To evaluate the game actions by passing agent outputs to an LLM, run:

```bash
bash evaluations/run_evals.sh
```

You will need to add a `.env` file with an OpenAI API key.

Alternatively, you can download the ground truth labels from the [HuggingFace](https://huggingface.co/datasets/7vik/AmongUs).

(TODO)

## Training Linear Probes

Once the activations are cached, training linear probes is easy. Just run:

```bash
uv run linear-probes/train_all_probes.py
```

You can choose which datasets to train probes on - by default, it will train on all datasets.

## Evaluating Linear Probes

To evaluate the linear probes, run:

```bash
uv run linear-probes/eval_all_probes.py
```

You can choose which datasets to evaluate probes on - by default, it will evaluate on all datasets.

It will store the results in `linear-probes/results/`, which are used to generate the plots in the paper.

## Sparse Autoencoders (SAEs)

We use the [Goodfire API](https://goodfire.ai/) to evaluate SAE features on the game logs. To do this, run the notebook:

```
reports/2025_02_27_sparse_autoencoders.ipynb
```

You will need to add a `.env` file with a Goodfire API key.

## Project Structure

```plaintext
.
├── CONTRIBUTING.md         # Contribution guidelines
├── Dockerfile               # Docker setup for project environment
├── LICENSE                  # License information
├── README.md                # Project documentation (this file)
├── CLAUDE.md                # Instructions for Claude Code
├── pyproject.toml           # Python project configuration and dependencies
├── uv.lock                  # Lock file for reproducible dependency resolution
├── among-agents/            # Main code for the Among Us agents
│   ├── README.md            # Documentation for agent implementation
│   ├── pyproject.toml       # Package configuration
│   └── amongagents/         # Core agent and environment modules
├── evaluations/             # LLM-based evaluation scripts
├── expt-logs/               # Experiment logs
├── human_trials/            # Web interface for human players
├── linear-probes/           # Linear probe training and evaluation
├── main.py                  # Main entry point for running the game
├── reports/                 # Analysis notebooks and results
├── tests/                   # Unit tests for project functionality
└── utils.py                 # Utility functions
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under CC0 1.0 Universal - see [LICENSE](LICENSE).

## Acknowledgments

- Our game logic uses a bunch of code from [AmongAgents](https://github.com/cyzus/among-agents).

If you face any bugs or issues with this codebase, please contact Satvik Golechha (7vik) at zsatvik@gmail.com.
