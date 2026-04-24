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
from collections import Counter
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from data import get_r2_client, load_all_games, load_new_games, fetch_game_logs, mark_game_processed
from parsing import parse_game_logs, create_game_log, get_player_experience_str, get_player_experience_with_ground_truth
from prompts import FRAMING_TEXT, CHECKLIST_RUBRIC, LANGUAGE_DATA_RUBRIC, BELIEF_TRACKING_ANALYSIS

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
R2_BUCKET = "amongus-leaderboard"

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = DATA_DIR / "results"

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


async def judge_response(prompt, client, model, rubric=CHECKLIST_RUBRIC):
    await asyncio.sleep(1)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": FRAMING_TEXT + "\n\n" + rubric},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrapping from LLM responses."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _clean_json_response(text: str) -> str:
    cleaned = _strip_markdown_fences(text)
    if not cleaned:
        return cleaned
    if cleaned.startswith("["):
        return cleaned
    # Fallback: find a JSON array anywhere in the response
    m = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if m:
        return m.group(0).strip()
    # Some models return a JSON object wrapping the array
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        return m.group(0).strip()
    return ""


async def evaluate_player(player_id, game_data, model_str, client, max_retries: int = 3, rubric=CHECKLIST_RUBRIC):
    prompt = get_player_experience_str(game_data, player_id)
    if prompt is None:
        tqdm.write(f"  ⚠ No narrative found for {player_id}, skipping.")
        return player_id, {"error": f"Player '{player_id}' not found in game_data"}

    for attempt in range(1, max_retries + 1):
        try:
            response_text = await judge_response(prompt, client, model_str, rubric=rubric)
            cleaned = _clean_json_response(response_text)
            if not cleaned:
                tqdm.write(f"  Raw response (empty after cleaning): {repr(response_text[:500])}")
                raise ValueError("Model returned an empty response")
            return player_id, json.loads(cleaned)
        except (json.JSONDecodeError, ValueError) as e:
            if attempt < max_retries:
                tqdm.write(f"  ↻ {player_id} attempt {attempt} failed ({e}), retrying...")
                if attempt == 1:
                    tqdm.write(f"    Raw response preview: {repr(response_text[:500])}")
                await asyncio.sleep(2 * attempt)
            else:
                tqdm.write(f"  Error evaluating {player_id} after {max_retries} attempts: {e}")
                tqdm.write(f"    Final raw response: {repr(response_text[:500])}")
                return player_id, {"error": str(e)}
        except Exception as e:
            # Non-parse errors (API/network) — don't retry
            tqdm.write(f"  Error evaluating {player_id}: {e}")
            return player_id, {"error": str(e)}


async def evaluate_all(game_data, game_folder: str, client):
    """Run every judge model over every player and write per-judge JSON files."""
    for judge_id, model_str in JUDGE_MODELS.items():
        model_name = model_str.split("/")[-1]
        tasks = [
            evaluate_player(player.get("name"), game_data, model_str, client)
            for player in game_data.get("players", [])
        ]
        results = await async_tqdm.gather(
            *tasks,
            desc=f"    checklist · {model_name} ({judge_id})",
            leave=False,
            unit="pl",
        )

        game_evaluations = dict(results)
        output_file = DATA_DIR / f"judge_game_{game_folder}_{model_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(game_evaluations, f, indent=2, ensure_ascii=False)
        tqdm.write(f"  Saved {output_file}")


async def evaluate_all_language(game_data, game_folder: str, client):
    """Run every judge model over every player with the LANGUAGE_DATA_RUBRIC."""
    for judge_id, model_str in JUDGE_MODELS.items():
        model_name = model_str.split("/")[-1]
        tasks = [
            evaluate_player(player.get("name"), game_data, model_str, client,
                            rubric=LANGUAGE_DATA_RUBRIC)
            for player in game_data.get("players", [])
        ]
        results = await async_tqdm.gather(
            *tasks,
            desc=f"    language  · {model_name} ({judge_id})",
            leave=False,
            unit="pl",
        )

        game_evaluations = dict(results)
        output_file = DATA_DIR / f"judge_game_{game_folder}_{model_name}_language.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(game_evaluations, f, indent=2, ensure_ascii=False)
        tqdm.write(f"  Saved {output_file}")


def _clean_json_response_object(text: str) -> str:
    """Like _clean_json_response but expects a JSON object ({...}) instead of array."""
    cleaned = _strip_markdown_fences(text)
    if not cleaned:
        return cleaned
    if cleaned.startswith("{"):
        return cleaned
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        return m.group(0).strip()
    return ""


async def evaluate_player_belief(player_id, game_data, model_str, client, max_retries: int = 3):
    """Evaluate a single player with the BELIEF_TRACKING_ANALYSIS rubric."""
    prompt = get_player_experience_with_ground_truth(game_data, player_id)
    if prompt is None:
        tqdm.write(f"  ⚠ No narrative found for {player_id}, skipping.")
        return player_id, {"error": f"Player '{player_id}' not found in game_data"}

    for attempt in range(1, max_retries + 1):
        try:
            response_text = await judge_response(prompt, client, model_str,
                                                  rubric=BELIEF_TRACKING_ANALYSIS)
            cleaned = _clean_json_response_object(response_text)
            if not cleaned:
                tqdm.write(f"  Raw response (empty after cleaning): {repr(response_text[:500])}")
                raise ValueError("Model returned an empty response")
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
            return player_id, parsed
        except (json.JSONDecodeError, ValueError) as e:
            if attempt < max_retries:
                tqdm.write(f"  ↻ {player_id} belief attempt {attempt} failed ({e}), retrying...")
                if attempt == 1:
                    tqdm.write(f"    Raw response preview: {repr(response_text[:500])}")
                await asyncio.sleep(2 * attempt)
            else:
                tqdm.write(f"  Error evaluating belief for {player_id} after {max_retries} attempts: {e}")
                tqdm.write(f"    Final raw response: {repr(response_text[:500])}")
                return player_id, {"error": str(e)}
        except Exception as e:
            tqdm.write(f"  Error evaluating belief for {player_id}: {e}")
            return player_id, {"error": str(e)}


async def evaluate_all_beliefs(game_data, game_folder: str, client):
    """Run every judge model over every player with the BELIEF_TRACKING_ANALYSIS rubric."""
    for judge_id, model_str in JUDGE_MODELS.items():
        model_name = model_str.split("/")[-1]
        tasks = [
            evaluate_player_belief(player.get("name"), game_data, model_str, client)
            for player in game_data.get("players", [])
        ]
        results = await async_tqdm.gather(
            *tasks,
            desc=f"    belief    · {model_name} ({judge_id})",
            leave=False,
            unit="pl",
        )

        game_evaluations = dict(results)
        output_file = DATA_DIR / f"judge_game_{game_folder}_{model_name}_belief.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(game_evaluations, f, indent=2, ensure_ascii=False)
        tqdm.write(f"  Saved {output_file}")


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
        filepath = DATA_DIR / f"judge_game_{game_folder}_{model_name}.json"
        if filepath.exists():
            all_judges[model_name] = load_judge_data(str(filepath), player_model_map)
        else:
            print(f"  ⚠ Missing judge file: {filepath}")
    return all_judges


def load_current_judge_files_language(game_folder: str, player_model_map: dict = None) -> dict:
    """Load language rubric judge files produced by the current JUDGE_MODELS."""
    all_judges = {}
    for judge_id, model_str in JUDGE_MODELS.items():
        model_name = model_str.split("/")[-1]
        filepath = DATA_DIR / f"judge_game_{game_folder}_{model_name}_language.json"
        if filepath.exists():
            all_judges[model_name] = load_judge_data(str(filepath), player_model_map)
        else:
            print(f"  ⚠ Missing language judge file: {filepath}")
    return all_judges


def load_current_belief_judge_files(game_folder: str, player_model_map: dict = None) -> dict:
    """Load belief tracking judge files produced by the current JUDGE_MODELS.

    Unlike the checklist/language loaders, the per-player data is a dict (not list),
    so we parse the raw JSON directly rather than using load_judge_data().
    """
    all_judges = {}
    for judge_id, model_str in JUDGE_MODELS.items():
        model_name = model_str.split("/")[-1]
        filepath = DATA_DIR / f"judge_game_{game_folder}_{model_name}_belief.json"
        if not filepath.exists():
            print(f"  ⚠ Missing belief judge file: {filepath}")
            continue
        with open(filepath) as f:
            raw = json.load(f)
        players = {}
        for player_key, player_data in raw.items():
            if isinstance(player_data, dict):
                if "error" in player_data and len(player_data) == 1:
                    print(f"  ⚠ Skipping {player_key}: belief evaluation error — {player_data['error']}")
                    continue
                players[player_key] = player_data
            else:
                print(f"  ⚠ Unexpected belief data type for {player_key}: {type(player_data)}, skipping.")
        if player_model_map:
            players = {player_model_map.get(k, k): v for k, v in players.items()}
        all_judges[model_name] = players
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


def aggregate_judge_results_language(all_judges: dict, threshold: float = 0.5) -> dict:
    """Majority vote across judges for language rubric results.

    Like aggregate_judge_results but also aggregates the 'frequency' field
    using the median across judges.
    """
    all_player_names = set()
    for judge_data in all_judges.values():
        all_player_names.update(judge_data.keys())

    final = {}
    for player_name in sorted(all_player_names):
        best_behaviors = []
        for judge_data in all_judges.values():
            player_behaviors = judge_data.get(player_name, [])
            if isinstance(player_behaviors, list) and len(player_behaviors) > len(best_behaviors):
                best_behaviors = player_behaviors

        if not best_behaviors:
            print(f"  ⚠ No valid language judge data for {player_name}, skipping.")
            continue

        final[player_name] = []
        for beh in best_behaviors:
            beh_name = normalize_behavior(_get_behavior_name(beh))
            votes = []
            justifications = []
            frequencies = []

            for model_name, judge_data in all_judges.items():
                player_data = judge_data.get(player_name, [])
                if not isinstance(player_data, list):
                    continue
                for b in player_data:
                    if normalize_behavior(_get_behavior_name(b)) == beh_name:
                        votes.append(1 if b.get("present") else 0)
                        justifications.append(f"[{model_name}]: {b.get('justification', '')}")
                        raw_freq = b.get("frequency", 0)
                        try:
                            frequencies.append(int(raw_freq))
                        except (TypeError, ValueError):
                            print(f"  ⚠ Non-numeric frequency '{raw_freq}' for {beh_name}, treating as 0")
                            frequencies.append(0)
                        break

            present = (sum(votes) / len(votes)) > threshold if votes else False
            median_freq = int(np.median(frequencies)) if frequencies else 0

            final[player_name].append({
                "behavior": beh_name,
                "present": present,
                "frequency": median_freq,
                "votes": f"{sum(votes)}/{len(votes)}",
                "justifications": justifications,
            })

    return final


def _majority_vote_str(values: list[str]) -> str:
    """Return the most common string value (first in case of tie)."""
    if not values:
        return ""
    counts = Counter(v.lower().strip() for v in values)
    return counts.most_common(1)[0][0]


def _majority_vote_bool(values: list[bool]) -> bool:
    if not values:
        return False
    return sum(1 for v in values if v) > len(values) / 2


def aggregate_belief_turn_by_turn(all_judges: dict, player_name: str) -> list[dict]:
    """Merge turn-by-turn belief entries across judges.

    Groups entries by (turn +-1, subject_player) and majority-votes on
    belief_state, confidence, accuracy, change_from_previous.
    """
    entries_by_key: dict[tuple, list[dict]] = {}

    for model_name, judge_data in all_judges.items():
        player_data = judge_data.get(player_name, {})
        if not isinstance(player_data, dict):
            continue
        for entry in player_data.get("turn_by_turn", []):
            turn = entry.get("turn", 0)
            subject = str(entry.get("subject_player", "")).strip().lower()
            matched = False
            for key in list(entries_by_key.keys()):
                if key[1] == subject and abs(key[0] - turn) <= 1:
                    entries_by_key[key].append(entry)
                    matched = True
                    break
            if not matched:
                entries_by_key[(turn, subject)] = [entry]

    merged = []
    for (turn, subject), entries in sorted(entries_by_key.items()):
        belief_states = [str(e.get("belief_state", "unknown")).lower().strip() for e in entries]
        confidences = [str(e.get("confidence", "low")).lower().strip() for e in entries]
        accuracies = [bool(e.get("accuracy", False)) for e in entries]
        changes = [bool(e.get("change_from_previous", False)) for e in entries]

        voted_state = _majority_vote_str(belief_states)
        best_entry = entries[0]
        for e in entries:
            if str(e.get("belief_state", "")).lower().strip() == voted_state:
                best_entry = e
                break

        merged.append({
            "turn": turn,
            "subject_player": best_entry.get("subject_player", subject),
            "belief_state": voted_state,
            "confidence": _majority_vote_str(confidences),
            "basis": best_entry.get("basis", ""),
            "change_from_previous": _majority_vote_bool(changes),
            "change_trigger": best_entry.get("change_trigger"),
            "accuracy": _majority_vote_bool(accuracies),
            "judge_agreement": f"{len(entries)}/{len(all_judges)}",
        })

    return merged


def aggregate_belief_quality(all_judges: dict, player_name: str) -> dict:
    """Aggregate belief_updating_quality across judges."""
    qualities = []
    for judge_data in all_judges.values():
        player_data = judge_data.get(player_name, {})
        if isinstance(player_data, dict) and "belief_updating_quality" in player_data:
            qualities.append(player_data["belief_updating_quality"])

    if not qualities:
        return {}

    bool_fields = ["responsive_to_evidence", "anchoring_bias", "recency_bias", "social_influence"]
    result = {}
    for field in bool_fields:
        values = [bool(q.get(field, False)) for q in qualities]
        voted = _majority_vote_bool(values)
        result[field] = voted
        examples_key = f"{field}_examples"
        for q in qualities:
            if bool(q.get(field, False)) == voted and q.get(examples_key):
                result[examples_key] = q[examples_key]
                break
        else:
            result[examples_key] = qualities[0].get(examples_key)

    for count_field in ["correct_updates", "incorrect_updates"]:
        counts = []
        for q in qualities:
            try:
                counts.append(int(q.get(count_field, 0)))
            except (TypeError, ValueError):
                counts.append(0)
        result[count_field] = int(np.median(counts)) if counts else 0

    return result


def aggregate_belief_tom(all_judges: dict, player_name: str) -> dict:
    """Aggregate theory_of_mind across judges."""
    toms = []
    for judge_data in all_judges.values():
        player_data = judge_data.get(player_name, {})
        if isinstance(player_data, dict) and "theory_of_mind" in player_data:
            toms.append(player_data["theory_of_mind"])

    if not toms:
        return {}

    levels = []
    for t in toms:
        try:
            levels.append(int(t.get("deepest_level", 0)))
        except (TypeError, ValueError):
            levels.append(0)
    median_level = int(np.median(levels))

    best_example = toms[0].get("deepest_level_example", "")
    for t, lvl in zip(toms, levels):
        if lvl == median_level and t.get("deepest_level_example"):
            best_example = t["deepest_level_example"]
            break

    result = {
        "deepest_level": median_level,
        "deepest_level_example": best_example,
    }
    for i in range(4):
        key = f"level_{i}_present"
        values = [bool(t.get(key, False)) for t in toms]
        result[key] = _majority_vote_bool(values)

    return result


def aggregate_belief_failed_tom(all_judges: dict, player_name: str) -> list[dict]:
    """Aggregate failed_tom_instances: include failures reported by 2+ judges."""
    type_instances: dict[str, list[dict]] = {}
    for judge_data in all_judges.values():
        player_data = judge_data.get(player_name, {})
        if not isinstance(player_data, dict):
            continue
        for inst in player_data.get("failed_tom_instances", []):
            ftype = str(inst.get("failure_type", "")).strip()
            if ftype:
                type_instances.setdefault(ftype, []).append(inst)

    threshold = len(all_judges) / 2
    merged = []
    for ftype, instances in type_instances.items():
        if len(instances) > threshold:
            merged.append({
                "failure_type": ftype,
                "turn": instances[0].get("turn", 0),
                "description": instances[0].get("description", ""),
                "judge_agreement": f"{len(instances)}/{len(all_judges)}",
            })

    return merged


def aggregate_belief_results(all_judges: dict) -> dict:
    """Orchestrate belief tracking aggregation across all judges for all players."""
    all_player_names = set()
    for judge_data in all_judges.values():
        all_player_names.update(judge_data.keys())

    final = {}
    for player_name in sorted(all_player_names):
        has_data = False
        for judge_data in all_judges.values():
            pd_ = judge_data.get(player_name, {})
            if isinstance(pd_, dict) and "turn_by_turn" in pd_:
                has_data = True
                break
        if not has_data:
            print(f"  ⚠ No valid belief judge data for {player_name}, skipping.")
            continue

        final[player_name] = {
            "turn_by_turn": aggregate_belief_turn_by_turn(all_judges, player_name),
            "belief_updating_quality": aggregate_belief_quality(all_judges, player_name),
            "theory_of_mind": aggregate_belief_tom(all_judges, player_name),
            "failed_tom_instances": aggregate_belief_failed_tom(all_judges, player_name),
        }

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
    tqdm.write(f"[R2] Uploaded → s3://{R2_BUCKET}/{key}")


def upload_language_judgement_to_r2(local_path: str, game_folder: str):
    """Upload language rubric results to results/<game_folder>/judged_game_language.json."""
    r2 = get_r2_client()
    key = f"results/{game_folder}/judged_game_language.json"
    with open(local_path, "rb") as f:
        r2.put_object(Bucket=R2_BUCKET, Key=key, Body=f.read(), ContentType="application/json")
    tqdm.write(f"[R2] Uploaded → s3://{R2_BUCKET}/{key}")


def upload_belief_judgement_to_r2(local_path: str, game_folder: str):
    """Upload belief tracking results to results/<game_folder>/judged_game_belief.json."""
    r2 = get_r2_client()
    key = f"results/{game_folder}/judged_game_belief.json"
    with open(local_path, "rb") as f:
        r2.put_object(Bucket=R2_BUCKET, Key=key, Body=f.read(), ContentType="application/json")
    tqdm.write(f"[R2] Uploaded → s3://{R2_BUCKET}/{key}")


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

    outer_iter = games.items()
    outer_bar = None
    if len(games) > 1:
        outer_bar = tqdm(total=len(games), desc="Games", unit="game")

    for game_folder, entries in outer_iter:
        if outer_bar is not None:
            outer_bar.set_postfix_str(game_folder, refresh=True)
        tqdm.write(f"\n=== Evaluating game: {game_folder} ===")

        parsed = parse_game_logs(entries)
        game_data = create_game_log(
            entries,
            parsed["agent_logs_df"],
            parsed["players"],
            parsed["meeting_transcripts"],
        )

        player_model_map = build_player_model_map(game_data)
        only_mode = args.language_only or args.belief_only
        run_checklist = not only_mode
        run_language = not args.skip_language and not args.belief_only
        run_belief = not args.skip_belief and not args.language_only

        # --- Checklist rubric ---
        if run_checklist:
            tqdm.write(f"\n  --- Checklist rubric ---")
            await evaluate_all(game_data, game_folder, client)

            all_judges = load_current_judge_files(game_folder, player_model_map)
            if not all_judges:
                tqdm.write(f"  No checklist judge files found for {game_folder}, skipping.")
            else:
                final_judgement = aggregate_judge_results(all_judges)
                local_path = RESULTS_DIR / f"judge_game_{game_folder}_final.json"
                with open(local_path, "w", encoding="utf-8") as f:
                    json.dump(final_judgement, f, indent=2, ensure_ascii=False)
                tqdm.write(f"  Saved local copy → {local_path}")
                if not args.local:
                    upload_judgement_to_r2(str(local_path), game_folder)

        # --- Language rubric ---
        if run_language:
            tqdm.write(f"\n  --- Language rubric ---")
            await evaluate_all_language(game_data, game_folder, client)

            all_judges_lang = load_current_judge_files_language(game_folder, player_model_map)
            if not all_judges_lang:
                tqdm.write(f"  No language judge files found for {game_folder}, skipping.")
            else:
                final_lang = aggregate_judge_results_language(all_judges_lang)
                local_lang_path = RESULTS_DIR / f"judge_game_{game_folder}_final_language.json"
                with open(local_lang_path, "w", encoding="utf-8") as f:
                    json.dump(final_lang, f, indent=2, ensure_ascii=False)
                tqdm.write(f"  Saved local copy → {local_lang_path}")
                if not args.local:
                    upload_language_judgement_to_r2(str(local_lang_path), game_folder)

        # --- Belief tracking rubric ---
        if run_belief:
            tqdm.write(f"\n  --- Belief tracking rubric ---")
            await evaluate_all_beliefs(game_data, game_folder, client)

            all_judges_belief = load_current_belief_judge_files(game_folder, player_model_map)
            if not all_judges_belief:
                tqdm.write(f"  No belief judge files found for {game_folder}, skipping.")
            else:
                final_belief = aggregate_belief_results(all_judges_belief)
                local_belief_path = RESULTS_DIR / f"judge_game_{game_folder}_final_belief.json"
                with open(local_belief_path, "w", encoding="utf-8") as f:
                    json.dump(final_belief, f, indent=2, ensure_ascii=False)
                tqdm.write(f"  Saved local copy → {local_belief_path}")
                if not args.local:
                    upload_belief_judgement_to_r2(str(local_belief_path), game_folder)

        if args.game_folder is None:
            mark_game_processed(game_folder)

        if outer_bar is not None:
            outer_bar.update(1)

    if outer_bar is not None:
        outer_bar.close()


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
    parser.add_argument(
        "--skip-language", action="store_true",
        help="Skip the language data rubric evaluation pass.",
    )
    parser.add_argument(
        "--language-only", action="store_true",
        help="Run only the language rubric (skip checklist and belief). Useful for backfilling.",
    )
    parser.add_argument(
        "--skip-belief", action="store_true",
        help="Skip the belief tracking rubric evaluation pass.",
    )
    parser.add_argument(
        "--belief-only", action="store_true",
        help="Run only the belief tracking rubric (skip checklist and language). Useful for backfilling.",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Store results locally only — skip uploading to R2.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
