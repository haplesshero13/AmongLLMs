# *AmongUs*: A Sandbox for Measuring and Detecting Agentic Deception

[Paper link.](https://arxiv.org/abs/2504.04072)

This project introduces the game "Among Us" as a model organism for lying and deception and studies how AI agents learn to express lying and deception, while evaluating the effectiveness of AI safety techniques to detect and control out-of-distribution deception.

## Overview

The aim is to simulate the popular multiplayer game "Among Us" using AI agents and analyze their behavior, particularly their ability to deceive and lie, which is central to the game's mechanics.

<img src="https://static.wikia.nocookie.net/among-us-wiki/images/f/f5/Among_Us_space_key_art_redesign.png" alt="Among Us" width="400"/>

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/7vik/AmongUs.git
   cd AmongUs
   ```

2. Set up the environment:
   ```bash
   conda create -n amongus python=3.10
   conda activate amongus
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run Games

To run the sandbox and log games of various LLMs playing against each other, run:

```
main.py
```
You will need to add a `.env` file with an [OpenRouter](https://openrouter.ai/) API key.

Alternatively, you can download 400 full-game logs (for `Phi-4-15b` and `Llama-3.3-70b-instruct`) and 810 game summaries from the [HuggingFace](https://huggingface.co/datasets/7vik/AmongUs) dataset to reproduce the results in the paper (and evaluate your own techniques!).

## Deception ELO

After running (or downloading) the games, to reproduce our Deception ELO results, run the following notebook:

```
reports/2025_02_26_deception_ELO_v3_ci.ipynb
```

The other report files can be used to reproduce the respective results.

## Caching Activations

Once the (full) game logs are in place, use the following command to cache the activations of the LLMs:

```
python linear-probes/cache_activations.py --dataset <dataset_name>
```

This loads up the HuggingFace models and caches the activations of the specified layers for each game action step. This step is computationally expensive, so it is recommended to run this using GPUs.

Use `configs.py` to specify the model and layer to cache, and other configuration options.

## LLM-based Evaluation (for Lying, Awareness, Deception, and Planning)

To evaluate the game actions by passing agent outputs to an LLM, run:

```
bash evaluations/run_evals.sh
```
You will need to add a `.env` file with an OpenAI API key.

Alternatively, you can download the ground truth labels from the [HuggingFace](https://huggingface.co/datasets/7vik/AmongUs).

(TODO)

## Training Linear Probes

Once the activations are cached, training linear probes is easy. Just run:

```
python linear-probes/train_all_probes.py
```
You can choose which datasets to train probes on - by default, it will train on all datasets.

## Evaluating Linear Probes

To evaluate the linear probes, run:

```
python linear-probes/eval_all_probes.py
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
├── among-agents             # Main code for the Among Us agents
│   ├── README.md            # Documentation for agent implementation
│   ├── amongagents          # Core agent and environment modules
│   ├── envs                 # Game environment and configurations
│   ├── evaluation           # Evaluation scripts for agent performance
│   ├── notebooks            # Jupyter notebooks for running experiments
│   ├── requirements.txt     # Python dependencies for agents
│   └── setup.py             # Setup script for agent package
├── expt-logs                # Experiment logs
├── k8s                      # Kubernetes configurations for deployment
├── main.py                  # Main entry point for running the game
├── notebooks                # Additional notebooks (not part of the main project)
├── reports                  # Experiment reports
├── requirements.txt         # Python dependencies for main project
├── tests                    # Unit tests for project functionality
└── utils.py                 # Utility functions
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under CC0 1.0 Universal - see [LICENSE](LICENSE).

## Acknowledgments

- Our game logic uses a bunch of code from [AmongAgents](https://github.com/cyzus/among-agents).

If you face any bugs or issues with this codebase, please contact Satvik Golechha (7vik) at zsatvik@gmail.com.
