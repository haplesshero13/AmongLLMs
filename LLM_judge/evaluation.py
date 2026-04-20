"""
LLM-as-Judge evaluation pipeline for AmongLLMs.

Usage:
    python evaluation.py [game_folder] [--judges m1,m2,m3]

If game_folder is omitted, the most-recent unprocessed game from R2 is evaluated.
If --judges is omitted, the default JUDGE_MODELS dict below is used. When given,
it must be a comma-separated list of exactly 3 OpenRouter model strings.

The aggregated final judgement is uploaded to:
    s3://amongus-leaderboard/results/<game_folder>/judged_game.json
"""

import json
import os
import re
import glob
import asyncio
import argparse

import numpy as np
from openai import AsyncOpenAI
from dotenv import load_dotenv

from data import get_r2_client, load_all_games, load_new_games, fetch_game_logs, mark_game_processed
from parsing import parse_game_logs, create_game_log, get_player_experience_str
from prompts import FRAMING_TEXT, CHECKLIST_RUBRIC

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
R2_BUCKET = "amongus-leaderboard"

# ---------------------------------------------------------------------------
# Judge models — 3 independent judges for majority vote
# ---------------------------------------------------------------------------

JUDGE_MODELS = {
    "judge_1": "anthropic/claude-opus-4.6:nitro",
    "judge_2": "google/gemini-3.1-pro-preview:nitro",
    "judge_3": "z-ai/glm-5.1:nitro",
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


def _clean_json_response(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    cleaned = text.strip()
    # Fallback: find a JSON array anywhere in the response
    if not cleaned or not cleaned.startswith("["):
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            cleaned = m.group(0).strip()
    return cleaned


async def evaluate_player(player_id, game_data, model_str, client, max_retries: int = 3):
    print(f"  Evaluating {player_id}...")
    prompt = get_player_experience_str(game_data, player_id)
    if prompt is None:
        print(f"  ⚠ No narrative found for {player_id}, skipping.")
        return player_id, {"error": f"Player '{player_id}' not found in game_data"}

    for attempt in range(1, max_retries + 1):
        try:
            response_text = await judge_response(prompt, client, model_str)
            cleaned = _clean_json_response(response_text)
            if not cleaned:
                print(f"  Raw response was: {repr(response_text[:300])}")
                raise ValueError("Model returned an empty response")
            return player_id, json.loads(cleaned)
        except (json.JSONDecodeError, ValueError) as e:
            if attempt < max_retries:
                print(f"  ↻ {player_id} attempt {attempt} failed ({e}), retrying...")
                await asyncio.sleep(2 * attempt)
            else:
                print(f"  Error evaluating {player_id} after {max_retries} attempts: {e}")
                return player_id, {"error": str(e)}
        except Exception as e:
            # Non-parse errors (API/network) — don't retry
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
            # Skip entries that are evaluation errors (not real responses)
            if "error" in player_data and len(player_data) == 1:
                print(f"  ⚠ Skipping {player_key}: evaluation error — {player_data['error']}")
                continue
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


def load_current_judge_files(game_folder: str, player_model_map: dict = None) -> dict:
    """Load only the judge files produced by the current JUDGE_MODELS (no stale old files)."""
    all_judges = {}
    for judge_id, model_str in JUDGE_MODELS.items():
        model_name = model_str.split("/")[-1]
        filepath = f"judge_game_{game_folder}_{model_name}.json"
        if os.path.exists(filepath):
            all_judges[model_name] = load_judge_data(filepath, player_model_map)
        else:
            print(f"  ⚠ Missing judge file: {filepath}")
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
    """Majority vote across judges. threshold=0.5 means >50% must agree.

    Uses the UNION of all players seen across all judges so that a single
    judge failing for a player doesn't drop that player from the final result.
    Behavior list is taken from whichever judge has the most behaviors for
    that player (most complete response).
    """
    # Collect all player names seen across every judge
    all_player_names = set()
    for judge_data in all_judges.values():
        all_player_names.update(judge_data.keys())

    final = {}
    for player_name in sorted(all_player_names):
        # Find the judge with the fullest behavior list for this player
        best_behaviors = []
        for judge_data in all_judges.values():
            player_behaviors = judge_data.get(player_name, [])
            if isinstance(player_behaviors, list) and len(player_behaviors) > len(best_behaviors):
                best_behaviors = player_behaviors

        if not best_behaviors:
            print(f"  ⚠ No valid judge data for {player_name}, skipping.")
            continue

        final[player_name] = []
        for beh in best_behaviors:
            beh_name = normalize_behavior(_get_behavior_name(beh))
            votes = []
            justifications = []

            for model_name, judge_data in all_judges.items():
                player_data = judge_data.get(player_name, [])
                if not isinstance(player_data, list):
                    continue
                for b in player_data:
                    if normalize_behavior(_get_behavior_name(b)) == beh_name:
                        votes.append(1 if b.get("present") else 0)
                        justifications.append(f"[{model_name}]: {b.get('justification', '')}")
                        break

            present = (sum(votes) / len(votes)) > threshold if votes else False
            if votes and len(votes) != len(all_judges):
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

def upload_judgement_to_r2(local_path: str, game_folder: str):
    """Upload the local final JSON file to results/<game_folder>/judged_game.json in R2.

    Reads directly from the saved local file so R2 is guaranteed to be
    byte-for-byte identical to what's on disk.
    """
    r2 = get_r2_client()
    key = f"results/{game_folder}/judged_game.json"
    with open(local_path, "rb") as f:
        r2.put_object(Bucket=R2_BUCKET, Key=key, Body=f.read(), ContentType="application/json")
    print(f"[R2] Uploaded → s3://{R2_BUCKET}/{key}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args):
    if args.judges:
        models = [m.strip() for m in args.judges.split(",") if m.strip()]
        if len(models) != 3:
            raise SystemExit(
                f"--judges requires exactly 3 models, got {len(models)}: {models}"
            )
        global JUDGE_MODELS
        JUDGE_MODELS = {f"judge_{i+1}": m for i, m in enumerate(models)}
        print(f"Using override judges: {JUDGE_MODELS}")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    r2_client = get_r2_client()

    # Determine which game(s) to process
    if args.game_folder is not None:
        game_folder = args.game_folder
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

        # Step 2: aggregate with majority vote (only current JUDGE_MODELS files)
        player_model_map = build_player_model_map(game_data)
        all_judges = load_current_judge_files(game_folder, player_model_map)
        if not all_judges:
            print(f"  No judge files found for {game_folder}, skipping aggregation.")
            continue

        final_judgement = aggregate_judge_results(all_judges)

        # Step 3: save locally
        local_path = f"judge_game_{game_folder}_final.json"
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(final_judgement, f, indent=2, ensure_ascii=False)
        print(f"Saved local copy → {local_path}")

        # Step 4: upload the saved file to Cloudflare R2
        upload_judgement_to_r2(local_path, game_folder)

        # Step 5: only mark processed after successful upload
        if args.game_folder is None:  # don't update manifest for explicit game_folder args
            mark_game_processed(game_folder)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge evaluation pipeline for AmongLLMs.",
    )
    parser.add_argument(
        "game_folder",
        nargs="?",
        default=None,
        help="Game folder name in R2 (e.g. game_7_2026-04-14_12-00-00). "
             "If omitted, the most-recent unprocessed game is evaluated.",
    )
    parser.add_argument(
        "--judges",
        default=None,
        help="Comma-separated list of exactly 3 model strings to use as judges "
             "(e.g. 'anthropic/claude-opus-4.6,openai/gpt-5.4,google/gemini-3.1-pro-preview'). "
             "If omitted, the default JUDGE_MODELS dict in this file is used.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
