"""
LLM-as-Judge evaluation pipeline for AmongLLMs.

Usage:
    python evaluation.py [game_folder]

If game_folder is omitted, the most-recent unprocessed game from R2 is evaluated.
The aggregated final judgement is uploaded to:
    s3://amongus-leaderboard/results/<game_folder>/judged_game.json
"""

import json
import os
import re
import glob
import asyncio
import sys

import numpy as np
from openai import AsyncOpenAI
from dotenv import load_dotenv

from data import get_r2_client, load_all_games, load_new_games, fetch_game_logs
from parsing import parse_game_logs, create_game_log, get_player_experience_str
from prompts import FRAMING_TEXT, CHECKLIST_RUBRIC

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
R2_BUCKET = "amongus-leaderboard"

# ---------------------------------------------------------------------------
# Judge models — 3 independent judges for majority vote
# ---------------------------------------------------------------------------

JUDGE_MODELS = {
    "judge_1": "meta-llama/llama-3.3-70b-instruct",
    "judge_2": "google/gemini-2.0-flash-001",
    "judge_3": "z-ai/glm-5.1",
}

# ---------------------------------------------------------------------------
# LLM judging
# ---------------------------------------------------------------------------


async def judge_response(prompt, client, model):
    await asyncio.sleep(1)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": FRAMING_TEXT + "\n\n" + CHECKLIST_RUBRIC},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content


async def evaluate_player(player_id, game_data, model_str, client):
    print(f"  Evaluating {player_id}...")
    prompt = get_player_experience_str(game_data, player_id)
    try:
        response_text = await judge_response(prompt, client, model_str)
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        return player_id, json.loads(cleaned_text.strip())
    except Exception as e:
        print(f"  Error evaluating {player_id}: {e}")
        return player_id, {"error": str(e)}


async def evaluate_all(game_data, game_folder: str, client):
    """Run every judge model over every player and write per-judge JSON files."""
    for judge_id, model_str in JUDGE_MODELS.items():
        print(f"\n  Running judge: {model_str}...")

        tasks = [
            evaluate_player(player.get("name"), game_data, model_str, client)
            for player in game_data.get("players", [])
        ]
        results = await asyncio.gather(*tasks)

        game_evaluations = dict(results)
        model_name = model_str.split("/")[-1]
        output_file = f"judge_game_{game_folder}_{model_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(game_evaluations, f, indent=2, ensure_ascii=False)
        print(f"  Saved {output_file}")


# ---------------------------------------------------------------------------
# Loading and aggregating judge results
# ---------------------------------------------------------------------------

MODEL_DISPLAY_NAMES = {
    "mistralai/mistral-large-2512": "Mistralai",
    "openai/gpt-5-mini": "GPT5-mini",
    "z-ai/glm-4.7-flash": "GLM-4.7",
    "qwen/qwen3-max-thinking": "Qwen3",
    "anthropic/claude-sonnet-4.6": "Claude-sonnet-4.6",
    "google/gemini-3-flash-preview": "Gemini-3-flash",
    "meta-llama/llama-4-maverick": "Llama-4-maverick",
    "anthropic/claude-opus-4.6": "Claude-opus-4.6",
    "openai/gpt-5.4": "GPT-5.4",
    "google/gemini-3.1-pro-preview:": "Gemini-3.1-pro",
    "meta-llama/llama-3.3-70b-instruct": "Llama-3.3",
    "nvidia/nemotron-3-super-120b-a12b": "Nemotron-3-super",
    "moonshotai/kimi-k2.5": "Kimi-K2.5",
    "deepseek/deepseek-v3.2": "DeepSeek-V3.2",
}


def build_player_model_map(game_data):
    return {
        player["name"]: MODEL_DISPLAY_NAMES.get(player.get("model"), player.get("model"))
        for player in game_data.get("players", [])
    }


def normalize_behavior(name: str) -> str:
    return re.sub(r"\s*:?\s*\(.*?\)", "", name).strip().rstrip(":").strip()


def _get_behavior_name(entry: dict) -> str:
    """Handle both 'behavior' and 'behvior' (typo) keys."""
    return entry.get("behavior") or entry.get("behvior", "")


def load_judge_data(filepath: str, player_model_map: dict = None) -> dict:
    """Parse a judge_game_*.json into {player_name: [{behavior, present, justification}, ...]}"""
    with open(filepath) as f:
        raw = json.load(f)

    players = {}
    for player_key, player_data in raw.items():
        if isinstance(player_data, list):
            players[player_key] = player_data
        elif isinstance(player_data, dict):
            resp = player_data.get("raw_response", "")
            resp = re.sub(r"^```json\s*", "", resp.strip())
            resp = re.sub(r"\s*```$", "", resp.strip())
            try:
                players[player_key] = json.loads(resp)
            except json.JSONDecodeError:
                print(f"  ⚠ Could not parse response for {player_key}, skipping.")
        else:
            print(f"  ⚠ Unexpected data type for {player_key}: {type(player_data)}, skipping.")

    if player_model_map:
        players = {player_model_map.get(k, k): v for k, v in players.items()}

    return players


def load_all_judge_files(game_folder: str, player_model_map: dict = None) -> dict:
    """Load all per-judge JSON files for a specific game_folder."""
    pattern = f"judge_game_{game_folder}_*.json"
    all_judges = {}
    for filepath in glob.glob(pattern):
        # strip prefix and suffix to get a short model name
        stem = filepath.replace(f"judge_game_{game_folder}_", "").replace(".json", "")
        all_judges[stem] = load_judge_data(filepath, player_model_map)
    return all_judges


def build_matrix(players: dict):
    """Return (player_names, behavior_names, np matrix of 0/1)."""
    player_names = list(players.keys())
    behavior_names = [_get_behavior_name(b) for b in players[player_names[0]]]
    matrix = np.zeros((len(player_names), len(behavior_names)), dtype=int)
    for i, pname in enumerate(player_names):
        for j, beh in enumerate(players[pname]):
            matrix[i, j] = 1 if beh.get("present") else 0
    return player_names, behavior_names, matrix


def aggregate_judge_results(all_judges: dict, threshold: float = 0.5) -> dict:
    """Majority vote across judges. threshold=0.5 means >50% must agree."""
    first_judge = next(iter(all_judges.values()))
    final = {}

    for player_name, behaviors in first_judge.items():
        final[player_name] = []
        for beh in behaviors:
            beh_name = normalize_behavior(_get_behavior_name(beh))
            votes = []
            justifications = []

            for model_name, judge_data in all_judges.items():
                player_data = judge_data.get(player_name, [])
                for b in player_data:
                    if normalize_behavior(_get_behavior_name(b)) == beh_name:
                        votes.append(1 if b.get("present") else 0)
                        justifications.append(f"[{model_name}]: {b.get('justification', '')}")
                        break

            present = (sum(votes) / len(votes)) > threshold if votes else False
            if len(votes) != len(all_judges):
                print(
                    f"  ⚠ {player_name} | {beh_name}: "
                    f"only {len(votes)}/{len(all_judges)} judges returned this behavior"
                )

            final[player_name].append({
                "behavior": beh_name,
                "present": present,
                "votes": f"{sum(votes)}/{len(votes)}",
                "justifications": justifications,
            })

    return final


# ---------------------------------------------------------------------------
# Upload to Cloudflare R2
# ---------------------------------------------------------------------------

def upload_judgement_to_r2(final_judgement: dict, game_folder: str):
    """Upload the aggregated judgement JSON to results/<game_folder>/judged_game.json in R2."""
    r2 = get_r2_client()
    key = f"results/{game_folder}/judged_game.json"
    body = json.dumps(final_judgement, indent=2, ensure_ascii=False).encode("utf-8")
    r2.put_object(Bucket=R2_BUCKET, Key=key, Body=body, ContentType="application/json")
    print(f"[R2] Uploaded → s3://{R2_BUCKET}/{key}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    r2_client = get_r2_client()

    # Determine which game(s) to process
    if len(sys.argv) > 1:
        game_folder = sys.argv[1]
        entries = fetch_game_logs(r2_client, R2_BUCKET, game_folder)
        if entries is None:
            print(f"Could not load game: {game_folder}")
            return
        games = {game_folder: entries}
    else:
        games = load_new_games(bucket=R2_BUCKET, client=r2_client)
        if not games:
            print("No new games to evaluate.")
            return

    for game_folder, entries in games.items():
        print(f"\n=== Evaluating game: {game_folder} ===")

        parsed = parse_game_logs(entries)
        game_data = create_game_log(
            entries,
            parsed["agent_logs_df"],
            parsed["players"],
            parsed["meeting_transcripts"],
        )

        # Step 1: run all 3 judge models
        await evaluate_all(game_data, game_folder, client)

        # Step 2: aggregate with majority vote
        player_model_map = build_player_model_map(game_data)
        all_judges = load_all_judge_files(game_folder, player_model_map)
        if not all_judges:
            print(f"  No judge files found for {game_folder}, skipping aggregation.")
            continue

        final_judgement = aggregate_judge_results(all_judges)

        # Step 3: save locally
        local_path = f"judge_game_{game_folder}_final.json"
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(final_judgement, f, indent=2, ensure_ascii=False)
        print(f"Saved local copy → {local_path}")

        # Step 4: upload to Cloudflare R2
        upload_judgement_to_r2(final_judgement, game_folder)


if __name__ == "__main__":
    asyncio.run(main())
