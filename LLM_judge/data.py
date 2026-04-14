"""
R2 data loader for AmongLLMs game logs.

Games are stored in R2 as:
    <bucket>/game_1_<timestamp>/agent-logs.json
    <bucket>/game_3_<timestamp>/agent-logs.json   # gaps are normal
    ...

Usage:
    # Load only games not yet processed (incremental)
    games = load_new_games(bucket="amongus-leaderboard")

    # Load every game in the bucket (re-run from scratch)
    games = load_all_games(bucket="amongus-leaderboard")
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Optional
import pprint


import boto3
from dotenv import load_dotenv

load_dotenv()
from botocore.exceptions import ClientError


# ---------------------------------------------------------------------------
# R2 client
# ---------------------------------------------------------------------------

def get_r2_client():
    """Return a boto3 S3 client pointed at Cloudflare R2.

    Reads from the same env vars used by human_trials/r2.py:
        S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY, S3_REGION
    """
    endpoint_url = os.environ.get("S3_ENDPOINT_URL")
    access_key = os.environ.get("S3_ACCESS_KEY")
    secret_key = os.environ.get("S3_SECRET_KEY")
    region = os.environ.get("S3_REGION", "auto")

    if not all([endpoint_url, access_key, secret_key]):
        raise ValueError(
            "S3_ENDPOINT_URL, S3_ACCESS_KEY, and S3_SECRET_KEY env vars must be set. "
            "Copy .env.example to .env and fill in your R2 credentials."
        )

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )


# ---------------------------------------------------------------------------
# Listing games in R2
# ---------------------------------------------------------------------------

# Matches keys like "game_1_2026-04-06_18-19-19/agent-logs.json"
_GAME_LOG_RE = re.compile(r"^(game_\d+_[^/]+)/agent-logs\.json$")


def list_game_keys(client, bucket: str) -> list[str]:
    """Return a sorted list of game folder names that have an agent-logs.json.

    Example return value:
        ["game_1_2026-04-06_18-19-19", "game_3_2026-04-06_19-00-00", ...]

    Games are sorted by their numeric game index so gaps (missing game_2, etc.)
    are handled naturally — we just skip them.
    """
    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket)

    game_keys = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            match = _GAME_LOG_RE.match(key)
            if match:
                game_keys.append(match.group(1))

    # Sort by the numeric game index embedded in the folder name
    def _game_index(folder: str) -> int:
        m = re.match(r"game_(\d+)_", folder)
        return int(m.group(1)) if m else 0

    return sorted(set(game_keys), key=_game_index)


# ---------------------------------------------------------------------------
# Downloading a single game
# ---------------------------------------------------------------------------

def _parse_multi_json(text: str) -> list[dict]:
    """Parse a file containing multiple concatenated JSON objects (not an array).

    agent-logs.json stores one JSON object per log entry, back-to-back, so
    json.loads() fails with 'Extra data'. json.JSONDecoder.raw_decode() handles
    this by repeatedly consuming one object at a time from the string.
    """
    decoder = json.JSONDecoder()
    objects = []
    pos = 0
    text = text.strip()
    while pos < len(text):
        # Skip whitespace between objects
        while pos < len(text) and text[pos] in " \t\n\r":
            pos += 1
        if pos >= len(text):
            break
        obj, pos = decoder.raw_decode(text, pos)
        objects.append(obj)
    return objects


def fetch_game_logs(client, bucket: str, game_folder: str) -> Optional[list[dict]]:
    """Download and parse agent-logs.json for one game folder.

    The file is multi-document JSON (one object per log entry, concatenated).
    Returns a list of log entry dicts, each annotated with 'game_id'.
    Returns None if the object is missing or unparseable.
    """
    key = f"{game_folder}/agent-logs.json"
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        raw = response["Body"].read().decode("utf-8")
        entries = _parse_multi_json(raw)
        for entry in entries:
            entry.setdefault("game_id", game_folder)
        return entries
    except ClientError as e:
        code = e.response["Error"]["Code"]
        print(f"[R2] Could not fetch {key}: {code}")
        return None
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[R2] Invalid JSON in {key}: {e}")
        return None


# ---------------------------------------------------------------------------
# Manifest — tracks which games have already been processed
# ---------------------------------------------------------------------------

_DEFAULT_MANIFEST = ".game_manifest.json"


def _load_manifest(manifest_path: str) -> set[str]:
    if not os.path.exists(manifest_path):
        return set()
    with open(manifest_path) as f:
        data = json.load(f)
    return set(data.get("processed", []))


def _save_manifest(manifest_path: str, processed: set[str]) -> None:
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "processed": sorted(processed),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_new_games(
    bucket: str = "amongus-leaderboard",
    manifest_path: str = _DEFAULT_MANIFEST,
    client=None,
) -> dict[str, list[dict]]:
    """Download only games that haven't been processed yet.

    On each call:
      1. Lists all game folders in R2.
      2. Skips folders already recorded in the manifest.
      3. Downloads the new ones, updates the manifest, and returns the data.

    This means repeated calls are cheap — only new games are fetched.

    Args:
        bucket:        R2 bucket name.
        manifest_path: Path to the local JSON manifest file.
        client:        Optional pre-built boto3 client (created automatically if omitted).

    Returns:
        Dict mapping game folder name → list of log entry dicts.
        Empty dict if nothing is new.
    """
    if client is None:
        client = get_r2_client()

    all_keys = list_game_keys(client, bucket)
    processed = _load_manifest(manifest_path)

    new_keys = [k for k in all_keys if k not in processed]
    if not new_keys:
        print(f"[R2] No new games found (checked {len(all_keys)} total).")
        return {}

    print(f"[R2] {len(new_keys)} new game(s) out of {len(all_keys)} total.")

    games: dict[str, list[dict]] = {}
    for folder in new_keys:
        entries = fetch_game_logs(client, bucket, folder)
        if entries is not None:
            games[folder] = entries
            processed.add(folder)
            print(f"[R2]   ✓ {folder} ({len(entries)} log entries)")

    _save_manifest(manifest_path, processed)
    print(f"[R2] Manifest updated → {manifest_path}")
    return games


def load_all_games(
    bucket: str = "amongus-leaderboard",
    client=None,
) -> dict[str, list[dict]]:
    """Download every game in the bucket, ignoring any manifest.

    Use this when you want a full re-run from scratch (e.g. if parsing
    logic changed and you need to reprocess everything).

    Args:
        bucket: R2 bucket name.
        client: Optional pre-built boto3 client.

    Returns:
        Dict mapping game folder name → list of log entry dicts.
    """
    if client is None:
        client = get_r2_client()

    all_keys = list_game_keys(client, bucket)
    print(f"[R2] Loading all {len(all_keys)} game(s) from '{bucket}'...")

    games: dict[str, list[dict]] = {}
    for folder in all_keys:
        entries = fetch_game_logs(client, bucket, folder)
        if entries is not None:
            games[folder] = entries
            print(f"[R2]   ✓ {folder} ({len(entries)} log entries)")

    print(f"[R2] Done — loaded {len(games)}/{len(all_keys)} games.")
    return games

'''games = load_all_games()
first_folder = next(iter(games))

if __name__ == "__main__":
    games = load_all_games()
    if games:
        first_folder = next(iter(games))
        print(f"\n--- First game: {first_folder} ({len(games[first_folder])} entries)---")
        print(json.dumps(games[first_folder], indent=2))'''

