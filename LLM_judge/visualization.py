"""
Visualization for AmongLLMs LLM-as-Judge results.

Default mode reads the three rubric result files from R2 and uploads all 14
generated graphs back to `results/graphs/`. The `--local` flag switches to a
fully local workflow: reads `LLM_judge/data/results/judge_game_*_final*.json`
and writes PNGs to `LLM_judge/data/graphs/`.

Graphs produced (when source data is available):
    Checklist:  01 heatmap · 02 category breakdown · 04 behavior rates · 06 radar
    Language:   10 presence heatmap · 11 frequency heatmap · 12 radar · 13 boxplots · 14 strategy profile
    Belief:     20 ToM depth · 21 accuracy · 22 state distribution · 23 cognitive bias · 24 evolution

Usage:
    uv run LLM_judge/visualization.py                    # R2 → R2
    uv run LLM_judge/visualization.py --local            # local → local
    uv run LLM_judge/visualization.py --local --output-dir /path/to/dir
"""

import argparse
import glob
import io
import json
import os
import re
from math import pi
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

from data import get_r2_client

load_dotenv()

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

R2_BUCKET       = "amongus-leaderboard"
GRAPHS_PREFIX   = "results/graphs/"

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
RESULTS_DIR     = PROJECT_ROOT / "LLM_judge" / "data" / "results"
LOCAL_GRAPHS_DIR = PROJECT_ROOT / "LLM_judge" / "data" / "graphs"

# Maps rubric name → R2 filename (suffix of the key) and local filename stem.
RUBRIC_CONFIG = {
  "checklist": {"r2_filename": "judged_game.json",          "local_suffix": "final"},
  "language":  {"r2_filename": "judged_game_language.json", "local_suffix": "final_language"},
  "belief":    {"r2_filename": "judged_game_belief.json",   "local_suffix": "final_belief"},
}


# ---------------------------------------------------------------------------
# Taxonomies
# ---------------------------------------------------------------------------

BEHAVIOR_CATEGORIES = {
  "Lying":                              "Deception",
  "Deflection":                         "Deception",
  "Gaslighting":                        "Deception",
  "Bus-throwing":                       "Deception",
  "Proactive alibi construction":       "Deception",
  "Evidence-based accusation":          "Social",
  "Unsupported accusation":             "Social",
  "Contradiction detection":            "Social",
  "Bandwagoning":                       "Social",
  "Information sharing":                "Social",
  "Persuasion":                         "Social",
  "Passivity":                          "Social",
  "Self-incrimination":                 "Social",
  "Humor":                              "Social",
  "Sarcasm":                            "Social",
  "Strategic voting":                   "Voting",
  "Vote against interest":              "Voting",
  "Vote skip with evidence available":  "Voting",
  "Target stalking":                    "Spatial",
  "Safety seeking":                     "Spatial",
  "Threat recognition":                 "Spatial",
  "Appropriate threat response":        "Spatial",
  "Strategic paralysis":                "Spatial",
  "Kill opportunity assessment":        "Strategic",
  "Task prioritization":                "Strategic",
  "Partner coordination":               "Strategic",
}
CATEGORY_ORDER = ["Deception", "Social", "Voting", "Spatial", "Strategic"]
CATEGORY_COLORS = {
  "Deception": "#e07b9b",
  "Social":    "#5ba67a",
  "Voting":    "#d4915e",
  "Spatial":   "#7b9ec4",
  "Strategic": "#9b82b0",
}

LANGUAGE_BEHAVIOR_CATEGORIES = {
  "Emotional escalation":        "Emotional",
  "Hedging language":            "Emotional",
  "Overclaiming certainty":      "Emotional",
  "Pleading or appealing":       "Emotional",
  "Fabricated testimony":        "Rhetorical",
  "Credibility leveraging":      "Rhetorical",
  "Reactive defensiveness":      "Rhetorical",
  "Interrogation":               "Rhetorical",
  "Narrative construction":      "Rhetorical",
  "Echo/mirroring":              "Rhetorical",
  "Rapport building":            "Social",
  "Distancing language":         "Social",
  "In-group/out-group framing":  "Social",
  "Silence as strategy":         "Social",
}
LANG_CATEGORY_ORDER  = ["Emotional", "Rhetorical", "Social"]
LANG_CATEGORY_COLORS = {"Emotional": "#e07b9b", "Rhetorical": "#d4915e", "Social": "#5ba67a"}

BELIEF_STATE_ORDER  = ["innocent", "suspicious", "impostor", "unknown"]
BELIEF_STATE_COLORS = {
  "innocent":   "#66bb6a",
  "suspicious": "#ffa726",
  "impostor":   "#ef5350",
  "unknown":    "#bdbdbd",
}
BIAS_NAMES = {
  "responsive_to_evidence": "Evidence\nResponsiveness",
  "anchoring_bias":         "Anchoring\nBias",
  "recency_bias":           "Recency\nBias",
  "social_influence":       "Social\nInfluence",
}
BIAS_COLORS = {
  "responsive_to_evidence": "#43a047",
  "anchoring_bias":         "#e53935",
  "recency_bias":           "#fb8c00",
  "social_influence":       "#8e24aa",
}


def setup_style() -> None:
  plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor":   "#ffffff",
    "text.color":       "#1a1a1a",
    "axes.labelcolor":  "#1a1a1a",
    "xtick.color":      "#444444",
    "ytick.color":      "#444444",
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.grid":        False,
  })


# ---------------------------------------------------------------------------
# I/O: loaders (R2 and local share the same {folder: dict} return shape)
# ---------------------------------------------------------------------------

_LOCAL_FOLDER_RE = re.compile(r"judge_game_(game_\d+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_")


def _game_sort_key(folder: str) -> int:
  m = re.match(r"game_(\d+)_", folder)
  return int(m.group(1)) if m else 0


def _extract_local_folder(filename: str) -> str:
  m = _LOCAL_FOLDER_RE.search(filename)
  return m.group(1) if m else filename


def _load_rubric_r2(r2_client, r2_filename: str) -> dict[str, dict]:
  paginator = r2_client.get_paginator("list_objects_v2")
  keys: list[str] = []
  needle = f"/{r2_filename}"
  for page in paginator.paginate(Bucket=R2_BUCKET, Prefix="results/"):
    for obj in page.get("Contents", []):
      key = obj["Key"]
      if key.endswith(needle) and "/graphs/" not in key:
        keys.append(key)
  if not keys:
    return {}
  games: dict[str, dict] = {}
  for key in sorted(keys):
    parts = key.split("/")
    folder = parts[1] if len(parts) >= 3 else key
    resp = r2_client.get_object(Bucket=R2_BUCKET, Key=key)
    games[folder] = json.loads(resp["Body"].read().decode("utf-8"))
    print(f"  Loaded s3://{R2_BUCKET}/{key}")
  return dict(sorted(games.items(), key=lambda kv: _game_sort_key(kv[0])))


def _load_rubric_local(local_suffix: str) -> dict[str, dict]:
  pattern = str(RESULTS_DIR / f"judge_game_*_{local_suffix}.json")
  files = glob.glob(pattern)
  # The checklist glob also matches `_final_language.json` and `_final_belief.json`
  # because those filenames contain `_final` as a prefix — filter those out.
  if local_suffix == "final":
    files = [f for f in files if "_final_language" not in f and "_final_belief" not in f]
  if not files:
    return {}
  games: dict[str, dict] = {}
  for fpath in sorted(files):
    folder = _extract_local_folder(os.path.basename(fpath))
    with open(fpath) as f:
      games[folder] = json.load(f)
    print(f"  Loaded {fpath}")
  return dict(sorted(games.items(), key=lambda kv: _game_sort_key(kv[0])))


def load_rubric(rubric: str, *, local: bool, r2_client=None) -> dict[str, dict]:
  cfg = RUBRIC_CONFIG[rubric]
  if local:
    return _load_rubric_local(cfg["local_suffix"])
  return _load_rubric_r2(r2_client, cfg["r2_filename"])


# ---------------------------------------------------------------------------
# I/O: save callbacks
# ---------------------------------------------------------------------------

def _upload_figure(r2_client, fig: plt.Figure, filename: str) -> None:
  buf = io.BytesIO()
  fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
  buf.seek(0)
  key = f"{GRAPHS_PREFIX}{filename}"
  r2_client.put_object(
    Bucket=R2_BUCKET, Key=key, Body=buf.read(), ContentType="image/png"
  )
  plt.close(fig)
  print(f"[R2] Uploaded → s3://{R2_BUCKET}/{key}")


def _save_local(fig: plt.Figure, filename: str, out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  path = out_dir / filename
  fig.savefig(path, format="png", bbox_inches="tight", dpi=150)
  plt.close(fig)
  print(f"  Saved → {path}")


def _make_save(*, local: bool, r2_client, out_dir: Path) -> Callable[[plt.Figure, str], None]:
  if local:
    return lambda fig, name: _save_local(fig, name, out_dir)
  return lambda fig, name: _upload_figure(r2_client, fig, name)


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

def build_checklist_df(all_games: dict[str, dict]) -> pd.DataFrame:
  rows = []
  for folder, judgement in all_games.items():
    m = re.match(r"game_(\d+)_", folder)
    game_label = f"Game {m.group(1)}" if m else folder
    for model, behaviors in judgement.items():
      if not isinstance(behaviors, list):
        continue
      for b in behaviors:
        behavior = b.get("behavior", "").strip()
        if behavior:
          rows.append({
            "game":     game_label,
            "model":    model,
            "behavior": behavior,
            "category": BEHAVIOR_CATEGORIES.get(behavior, "Strategic"),
            "present":  int(bool(b.get("present", False))),
          })
  return pd.DataFrame(rows)


def build_language_df(all_games: dict[str, dict]) -> pd.DataFrame:
  rows = []
  for folder, judgement in all_games.items():
    m = re.match(r"game_(\d+)_", folder)
    game_label = f"Game {m.group(1)}" if m else folder
    for model, behaviors in judgement.items():
      if not isinstance(behaviors, list):
        continue
      for b in behaviors:
        behavior = b.get("behavior", "").strip()
        if behavior:
          rows.append({
            "game":      game_label,
            "model":     model,
            "behavior":  behavior,
            "category":  LANGUAGE_BEHAVIOR_CATEGORIES.get(behavior, "Social"),
            "present":   int(bool(b.get("present", False))),
            "frequency": int(b.get("frequency", 0) or 0),
          })
  return pd.DataFrame(rows)


def build_belief_dfs(all_games: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  belief_rows, quality_rows, tom_rows = [], [], []
  for folder, judgement in all_games.items():
    m = re.match(r"game_(\d+)_", folder)
    game_label = f"Game {m.group(1)}" if m else folder
    for model, pdata in judgement.items():
      if not isinstance(pdata, dict) or "turn_by_turn" not in pdata:
        continue
      for entry in pdata.get("turn_by_turn", []):
        belief_rows.append({
          "game":                 game_label,
          "model":                model,
          "turn":                 entry.get("turn", 0),
          "subject_player":       entry.get("subject_player", ""),
          "belief_state":         str(entry.get("belief_state", "unknown")).lower().strip(),
          "confidence":           str(entry.get("confidence", "low")).lower().strip(),
          "accuracy":             bool(entry.get("accuracy", False)),
          "change_from_previous": bool(entry.get("change_from_previous", False)),
        })
      bq = pdata.get("belief_updating_quality", {})
      correct   = int(bq.get("correct_updates", 0) or 0)
      incorrect = int(bq.get("incorrect_updates", 0) or 0)
      total = correct + incorrect
      quality_rows.append({
        "game":                  game_label,
        "model":                 model,
        "responsive_to_evidence": bool(bq.get("responsive_to_evidence", False)),
        "anchoring_bias":         bool(bq.get("anchoring_bias", False)),
        "recency_bias":           bool(bq.get("recency_bias", False)),
        "social_influence":       bool(bq.get("social_influence", False)),
        "correct_updates":        correct,
        "incorrect_updates":      incorrect,
        "update_accuracy_rate":  (correct / total * 100) if total > 0 else 0.0,
      })
      tom = pdata.get("theory_of_mind", {})
      tom_rows.append({
        "game":              game_label,
        "model":             model,
        "deepest_level":     int(tom.get("deepest_level", 0) or 0),
        "level_0_present":   bool(tom.get("level_0_present", False)),
        "level_1_present":   bool(tom.get("level_1_present", False)),
        "level_2_present":   bool(tom.get("level_2_present", False)),
        "level_3_present":   bool(tom.get("level_3_present", False)),
        "failed_tom_count":  len(pdata.get("failed_tom_instances", [])),
      })
  return (
    pd.DataFrame(belief_rows)  if belief_rows  else pd.DataFrame(),
    pd.DataFrame(quality_rows) if quality_rows else pd.DataFrame(),
    pd.DataFrame(tom_rows)     if tom_rows     else pd.DataFrame(),
  )


# ---------------------------------------------------------------------------
# Legend helpers
# ---------------------------------------------------------------------------

def _category_legend_handles() -> list[Line2D]:
  return [
    Line2D([0], [0], marker="s", color="w",
           markerfacecolor=CATEGORY_COLORS[c], markersize=10, label=c)
    for c in CATEGORY_ORDER
  ]


def _lang_category_legend_handles() -> list[Line2D]:
  return [
    Line2D([0], [0], marker="s", color="w",
           markerfacecolor=LANG_CATEGORY_COLORS[c], markersize=10, label=c)
    for c in LANG_CATEGORY_ORDER
  ]


# ---------------------------------------------------------------------------
# Checklist graphs (01, 02, 04, 06)
# ---------------------------------------------------------------------------

def plot_behavior_heatmap(df: pd.DataFrame, save: Callable[[plt.Figure, str], None]) -> None:
  pivot = (
    df.groupby(["model", "behavior"])["present"]
    .mean()
    .unstack(fill_value=0)
  )

  def _beh_sort(b: str):
    cat = BEHAVIOR_CATEGORIES.get(b, "Strategic")
    idx = CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else len(CATEGORY_ORDER)
    return (idx, -pivot[b].mean())

  pivot = pivot[sorted(pivot.columns, key=_beh_sort)]
  pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

  n_b, n_m = len(pivot.columns), len(pivot.index)
  fig, ax = plt.subplots(figsize=(max(18, n_b * 0.65), max(5, n_m * 0.55)))

  cmap = mcolors.LinearSegmentedColormap.from_list("among", ["#f0f0f0", "#1565c0"])
  im   = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)

  ax.set_xticks(range(n_b))
  ax.set_xticklabels([b[:28] for b in pivot.columns], rotation=50, ha="right", fontsize=8)
  ax.set_yticks(range(n_m))
  ax.set_yticklabels(pivot.index, fontsize=9)

  for i in range(n_m):
    for j in range(n_b):
      val = pivot.values[i, j]
      text_color = "#ffffff" if val > 0.5 else "#555555"
      ax.text(j, i, f"{val:.0%}", ha="center", va="center",
              color=text_color, fontsize=7, fontweight="bold")

  for j, bname in enumerate(pivot.columns):
    cat = BEHAVIOR_CATEGORIES.get(bname, "Strategic")
    ax.plot(j, -0.65, marker="s", color=CATEGORY_COLORS[cat], markersize=7, clip_on=False)

  ax.set_title("Behavior Presence Rate by Model — All Games",
               fontsize=14, fontweight="bold", pad=48)

  ax.legend(handles=_category_legend_handles(),
            bbox_to_anchor=(0.5, 1.0), loc="lower center",
            ncol=len(CATEGORY_ORDER), fontsize=8,
            facecolor="#ffffff", edgecolor="#ccc", framealpha=0.8,
            title="Category", title_fontsize=8)

  cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
  cbar.set_label("% games where behavior was present", fontsize=8)
  cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

  save(fig, "01_heatmap_behavior_by_model.png")


def plot_category_breakdown(df: pd.DataFrame, save: Callable[[plt.Figure, str], None]) -> None:
  cat_counts = (
    df[df["present"] == 1]
    .groupby(["model", "category"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=CATEGORY_ORDER, fill_value=0)
  )
  row_sums  = cat_counts.sum(axis=1).replace(0, 1)
  cat_pct   = cat_counts.div(row_sums, axis=0).mul(100)
  overall   = df.groupby("model")["present"].mean()
  cat_pct   = cat_pct.loc[overall.reindex(cat_pct.index).sort_values(ascending=False).index]

  fig, ax = plt.subplots(figsize=(max(10, len(cat_pct) * 0.9), 5))
  x      = np.arange(len(cat_pct))
  bottom = np.zeros(len(cat_pct))

  for cat in CATEGORY_ORDER:
    vals = cat_pct[cat].values
    ax.bar(x, vals, bottom=bottom, label=cat,
           color=CATEGORY_COLORS[cat], edgecolor="#ffffff", linewidth=0.4, width=0.7)
    for i, (v, b) in enumerate(zip(vals, bottom)):
      if v > 5:
        ax.text(x[i], b + v / 2, f"{v:.0f}%", ha="center", va="center",
                fontsize=7, color="#fff", fontweight="bold")
    bottom += vals

  ax.set_xticks(x)
  ax.set_xticklabels(cat_pct.index, rotation=30, ha="right", fontsize=9)
  ax.set_ylim(0, 100)
  ax.set_ylabel("Share of Present Behaviors (%)")
  ax.set_title("Behavior Category Breakdown per Model — All Games",
               fontsize=13, fontweight="bold", pad=48)
  ax.legend(handles=_category_legend_handles(),
            bbox_to_anchor=(0.5, 1.0), loc="lower center",
            ncol=len(CATEGORY_ORDER), fontsize=8,
            facecolor="#ffffff", edgecolor="#ccc", framealpha=0.8,
            title="Category", title_fontsize=8)
  save(fig, "02_category_breakdown_by_model.png")


def plot_behavior_rates(df: pd.DataFrame, save: Callable[[plt.Figure, str], None]) -> None:
  rates = (
    df.groupby("behavior")["present"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
  )

  def _bar_colors(behaviors):
    return [CATEGORY_COLORS.get(BEHAVIOR_CATEGORIES.get(b, "Strategic"), "#888")
            for b in behaviors]

  top_n = min(10, len(rates))
  top   = rates.head(top_n)
  bot   = rates.tail(top_n).sort_values(ascending=True)

  fig, axes = plt.subplots(1, 2, figsize=(16, max(5, top_n * 0.5)))

  axes[0].barh(range(len(top)), top.values[::-1], color=_bar_colors(top.index[::-1]))
  axes[0].set_yticks(range(len(top)))
  axes[0].set_yticklabels(top.index[::-1], fontsize=8)
  axes[0].set_title("Most Common Behaviors", fontsize=12, fontweight="bold")
  axes[0].set_xlabel("% Games Present")
  axes[0].set_xlim(0, 105)
  for i, v in enumerate(top.values[::-1]):
    axes[0].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)

  axes[1].barh(range(len(bot)), bot.values, color=_bar_colors(bot.index))
  axes[1].set_yticks(range(len(bot)))
  axes[1].set_yticklabels(bot.index, fontsize=8)
  axes[1].set_title("Least Common Behaviors", fontsize=12, fontweight="bold")
  axes[1].set_xlabel("% Games Present")
  axes[1].set_xlim(0, 105)
  for i, v in enumerate(bot.values):
    axes[1].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)

  fig.legend(handles=_category_legend_handles(), loc="lower center",
             ncol=len(CATEGORY_ORDER), fontsize=8, facecolor="#ffffff",
             edgecolor="#ccc", framealpha=0.8, title="Category")
  fig.suptitle("Behavior Frequency Across All Games & Models",
               fontsize=13, fontweight="bold")
  fig.tight_layout(rect=[0, 0.1, 1, 0.95])
  save(fig, "04_behavior_rates.png")


def plot_radar_charts(df: pd.DataFrame, save: Callable[[plt.Figure, str], None]) -> None:
  models   = sorted(df["model"].unique())
  n_models = len(models)
  cols     = min(4, n_models)
  rows     = (n_models + cols - 1) // cols

  fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows),
                            subplot_kw=dict(polar=True))
  if n_models == 1:
    axes = np.array([axes])
  axes = axes.flatten()

  categories = CATEGORY_ORDER
  n_cats     = len(categories)
  angles     = [n * 2 * pi / n_cats for n in range(n_cats)]
  angles    += angles[:1]

  cat_rates = (
    df.groupby(["model", "category"])["present"]
    .mean()
    .unstack(fill_value=0)
    .reindex(columns=categories, fill_value=0)
  )

  y_max  = min(1.0, cat_rates.values.max() * 1.2) if cat_rates.values.max() > 0 else 1.0
  step   = y_max / 4
  yticks = [step * i for i in range(1, 5)]

  for idx, model in enumerate(models):
    ax = axes[idx]
    ax.set_facecolor("#ffffff")

    values  = cat_rates.loc[model].tolist() if model in cat_rates.index else [0] * n_cats
    values += values[:1]

    ax.plot(angles, values, color="#1565c0", linewidth=2)
    ax.fill(angles, values, color="#1565c0", alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=7, color="#aaa")
    ax.set_ylim(0, y_max)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:.0%}" for t in yticks], fontsize=6, color="#888")
    ax.spines["polar"].set_color("#ccc")
    ax.grid(color="#ccc", linewidth=0.5)
    ax.set_title(model, fontsize=11, fontweight="bold", color="#1a1a1a", pad=15)

  for idx in range(n_models, len(axes)):
    axes[idx].set_visible(False)

  fig.suptitle("Per-Model Behavioral Profiles (Category Radar)",
               fontsize=15, fontweight="bold", color="#1a1a1a", y=1.02)
  save(fig, "06_player_radar.png")


# ---------------------------------------------------------------------------
# Language graphs (10, 11, 12, 13, 14)
# ---------------------------------------------------------------------------

def plot_language_presence_heatmap(df: pd.DataFrame, save: Callable[[plt.Figure, str], None]) -> None:
  pivot = df.groupby(["model", "behavior"])["present"].mean().unstack(fill_value=0)

  def _sort(b):
    cat = LANGUAGE_BEHAVIOR_CATEGORIES.get(b, "Social")
    idx = LANG_CATEGORY_ORDER.index(cat) if cat in LANG_CATEGORY_ORDER else len(LANG_CATEGORY_ORDER)
    return (idx, -pivot[b].mean())

  pivot = pivot[sorted(pivot.columns, key=_sort)]
  pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

  n_b, n_m = len(pivot.columns), len(pivot.index)
  fig, ax = plt.subplots(figsize=(max(16, n_b * 0.85), max(5, n_m * 0.55)))
  cmap = mcolors.LinearSegmentedColormap.from_list("lang", ["#f0f0f0", "#1565c0"])
  im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)
  ax.set_xticks(range(n_b))
  ax.set_xticklabels([b[:30] for b in pivot.columns], rotation=50, ha="right", fontsize=8)
  ax.set_yticks(range(n_m))
  ax.set_yticklabels(pivot.index, fontsize=9)
  for i in range(n_m):
    for j in range(n_b):
      val = pivot.values[i, j]
      ax.text(j, i, f"{val:.0%}", ha="center", va="center",
              color="#ffffff" if val > 0.5 else "#555555", fontsize=7, fontweight="bold")
  for j, bname in enumerate(pivot.columns):
    cat = LANGUAGE_BEHAVIOR_CATEGORIES.get(bname, "Social")
    ax.plot(j, -0.65, marker="s", color=LANG_CATEGORY_COLORS[cat], markersize=7, clip_on=False)
  ax.set_title("Linguistic Behavior Presence Rate by Model — All Games",
               fontsize=14, fontweight="bold", pad=48)
  ax.legend(handles=_lang_category_legend_handles(), bbox_to_anchor=(0.5, 1.0), loc="lower center",
            ncol=len(LANG_CATEGORY_ORDER), fontsize=8, facecolor="#ffffff", edgecolor="#ccc",
            framealpha=0.8, title="Category", title_fontsize=8)
  cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
  cbar.set_label("% games where behavior was present", fontsize=8)
  cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
  save(fig, "10_lang_heatmap_presence.png")


def plot_language_frequency_heatmap(df: pd.DataFrame, save: Callable[[plt.Figure, str], None]) -> None:
  pivot = df.groupby(["model", "behavior"])["frequency"].mean().unstack(fill_value=0)

  def _sort(b):
    cat = LANGUAGE_BEHAVIOR_CATEGORIES.get(b, "Social")
    idx = LANG_CATEGORY_ORDER.index(cat) if cat in LANG_CATEGORY_ORDER else len(LANG_CATEGORY_ORDER)
    return (idx, -pivot[b].mean())

  pivot = pivot[sorted(pivot.columns, key=_sort)]
  pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

  n_b, n_m = len(pivot.columns), len(pivot.index)
  fig, ax = plt.subplots(figsize=(max(16, n_b * 0.85), max(5, n_m * 0.55)))
  vmax = max(pivot.values.max(), 1)
  im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
  ax.set_xticks(range(n_b))
  ax.set_xticklabels([b[:30] for b in pivot.columns], rotation=50, ha="right", fontsize=8)
  ax.set_yticks(range(n_m))
  ax.set_yticklabels(pivot.index, fontsize=9)
  for i in range(n_m):
    for j in range(n_b):
      val = pivot.values[i, j]
      ax.text(j, i, f"{val:.1f}", ha="center", va="center",
              color="#ffffff" if val > vmax * 0.6 else "#555555", fontsize=7, fontweight="bold")
  for j, bname in enumerate(pivot.columns):
    cat = LANGUAGE_BEHAVIOR_CATEGORIES.get(bname, "Social")
    ax.plot(j, -0.65, marker="s", color=LANG_CATEGORY_COLORS[cat], markersize=7, clip_on=False)
  ax.set_title("Linguistic Behavior Avg Frequency by Model — All Games",
               fontsize=14, fontweight="bold", pad=48)
  ax.legend(handles=_lang_category_legend_handles(), bbox_to_anchor=(0.5, 1.0), loc="lower center",
            ncol=len(LANG_CATEGORY_ORDER), fontsize=8, facecolor="#ffffff", edgecolor="#ccc",
            framealpha=0.8, title="Category", title_fontsize=8)
  cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
  cbar.set_label("Avg instance count per game", fontsize=8)
  save(fig, "11_lang_heatmap_frequency.png")


def plot_language_radar(df: pd.DataFrame, save: Callable[[plt.Figure, str], None]) -> None:
  models = sorted(df["model"].unique())
  n_models = len(models)
  cols = min(4, n_models)
  rows_n = (n_models + cols - 1) // cols
  fig, axes = plt.subplots(rows_n, cols, figsize=(5 * cols, 5 * rows_n), subplot_kw=dict(polar=True))
  if n_models == 1:
    axes = np.array([axes])
  axes = axes.flatten()

  cats = LANG_CATEGORY_ORDER
  n_cats = len(cats)
  angles = [n * 2 * pi / n_cats for n in range(n_cats)] + [0]

  cat_freq = (df.groupby(["model", "category"])["frequency"].mean()
              .unstack(fill_value=0).reindex(columns=cats, fill_value=0))
  y_max = max(cat_freq.values.max() * 1.2, 1)

  for idx, model in enumerate(models):
    ax = axes[idx]
    ax.set_facecolor("#ffffff")
    values = cat_freq.loc[model].tolist() if model in cat_freq.index else [0] * n_cats
    values += values[:1]
    ax.plot(angles, values, color="#d4915e", linewidth=2)
    ax.fill(angles, values, color="#d4915e", alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=8, color="#aaa")
    ax.set_ylim(0, y_max)
    step = y_max / 4
    yticks = [step * i for i in range(1, 5)]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:.1f}" for t in yticks], fontsize=6, color="#888")
    ax.spines["polar"].set_color("#ccc")
    ax.grid(color="#ccc", linewidth=0.5)
    ax.set_title(model, fontsize=11, fontweight="bold", color="#1a1a1a", pad=15)
  for idx in range(n_models, len(axes)):
    axes[idx].set_visible(False)
  fig.suptitle("Linguistic Profile by Category (Avg Frequency Radar)",
               fontsize=15, fontweight="bold", color="#1a1a1a", y=1.02)
  save(fig, "12_lang_radar_by_model.png")


def plot_language_frequency_boxplots(df: pd.DataFrame, save: Callable[[plt.Figure, str], None]) -> None:
  medians = df.groupby("behavior")["frequency"].median().sort_values(ascending=False)
  order = medians.index.tolist()
  fig, ax = plt.subplots(figsize=(10, max(6, len(order) * 0.4)))
  sns.boxplot(data=df, y="behavior", x="frequency", order=order, ax=ax,
              color="#7b9ec4", linewidth=0.8, fliersize=3, width=0.6)
  sns.stripplot(data=df, y="behavior", x="frequency", order=order, ax=ax,
                color="#1a1a1a", alpha=0.25, size=3, jitter=True)
  ax.set_xlabel("Frequency (instance count)")
  ax.set_ylabel("")
  ax.set_title("Linguistic Behavior Frequency Distribution — All Models & Games",
               fontsize=13, fontweight="bold")
  fig.tight_layout()
  save(fig, "13_lang_frequency_boxplots.png")


def plot_language_strategy_profile(df: pd.DataFrame, save: Callable[[plt.Figure, str], None]) -> None:
  totals = df.groupby(["model", "category"])["frequency"].sum().unstack(fill_value=0)
  totals = totals.reindex(columns=LANG_CATEGORY_ORDER, fill_value=0)
  totals = totals.loc[totals.sum(axis=1).sort_values(ascending=False).index]

  fig, ax = plt.subplots(figsize=(max(10, len(totals) * 0.9), 5))
  x = np.arange(len(totals))
  bottom = np.zeros(len(totals))
  for cat in LANG_CATEGORY_ORDER:
    vals = totals[cat].values
    ax.bar(x, vals, bottom=bottom, label=cat, color=LANG_CATEGORY_COLORS[cat],
           edgecolor="#ffffff", linewidth=0.4, width=0.7)
    for i, (v, b) in enumerate(zip(vals, bottom)):
      if v > 0.5:
        ax.text(x[i], b + v / 2, f"{v:.0f}", ha="center", va="center",
                fontsize=7, color="#fff", fontweight="bold")
    bottom += vals
  ax.set_xticks(x)
  ax.set_xticklabels(totals.index, rotation=30, ha="right", fontsize=9)
  ax.set_ylabel("Total Frequency Count")
  ax.set_title("Linguistic Strategy Profile — Total Frequency by Category",
               fontsize=13, fontweight="bold", pad=20)
  ax.legend(handles=_lang_category_legend_handles(), loc="upper right", fontsize=8,
            facecolor="#ffffff", edgecolor="#ccc", framealpha=0.8)
  fig.tight_layout()
  save(fig, "14_lang_strategy_profile.png")


# ---------------------------------------------------------------------------
# Belief graphs (20, 21, 22, 23, 24)
# ---------------------------------------------------------------------------

def plot_tom_depth(df_tom: pd.DataFrame, save: Callable[[plt.Figure, str], None]) -> None:
  model_stats = df_tom.groupby("model")["deepest_level"].agg(["mean", "std"]).sort_values("mean", ascending=False)
  fig, ax = plt.subplots(figsize=(max(8, len(model_stats) * 0.9), 5))
  x = np.arange(len(model_stats))
  ax.bar(x, model_stats["mean"], yerr=model_stats["std"].fillna(0), capsize=4,
         color="#7b9ec4", edgecolor="#ffffff", linewidth=0.4, width=0.6, error_kw=dict(lw=1.2))
  for model in model_stats.index:
    pts = df_tom[df_tom["model"] == model]["deepest_level"]
    idx = list(model_stats.index).index(model)
    ax.scatter([idx] * len(pts), pts, color="#1a1a1a", alpha=0.3, s=20, zorder=3)
  ax.set_xticks(x)
  ax.set_xticklabels(model_stats.index, rotation=30, ha="right", fontsize=9)
  ax.set_ylabel("Mean Deepest ToM Level")
  ax.set_ylim(-0.2, 3.5)
  ax.set_yticks([0, 1, 2, 3])
  ax.set_yticklabels(["L0: Own obs", "L1: Models others' knowledge",
                      "L2: Others think about others", "L3: Recursive self-model"], fontsize=8)
  ax.set_title("Theory of Mind Depth by Model", fontsize=13, fontweight="bold")
  fig.tight_layout()
  save(fig, "20_tom_depth_by_model.png")


def plot_belief_accuracy(df_beliefs: pd.DataFrame, save: Callable[[plt.Figure, str], None]) -> None:
  acc = df_beliefs.groupby("model")["accuracy"].mean().mul(100).sort_values(ascending=False)
  counts = df_beliefs.groupby("model")["accuracy"].count()
  fig, ax = plt.subplots(figsize=(max(8, len(acc) * 0.9), 5))
  x = np.arange(len(acc))
  colors = ["#43a047" if v >= 50 else "#e53935" for v in acc.values]
  ax.bar(x, acc.values, color=colors, edgecolor="#ffffff", linewidth=0.4, width=0.6)
  for i, (v, model) in enumerate(zip(acc.values, acc.index)):
    n = counts.get(model, 0)
    ax.text(i, v + 1, f"{v:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=8)
  ax.set_xticks(x)
  ax.set_xticklabels(acc.index, rotation=30, ha="right", fontsize=9)
  ax.set_ylabel("Belief Accuracy (%)")
  ax.set_ylim(0, min(110, acc.max() + 15))
  ax.set_title("Belief Accuracy by Model — All Games", fontsize=13, fontweight="bold")
  fig.tight_layout()
  save(fig, "21_belief_accuracy_by_model.png")


def plot_belief_state_distribution(df_beliefs: pd.DataFrame, save: Callable[[plt.Figure, str], None]) -> None:
  ct = pd.crosstab(df_beliefs["model"], df_beliefs["belief_state"], normalize="index").mul(100)
  ct = ct.reindex(columns=BELIEF_STATE_ORDER, fill_value=0)
  ct = ct.loc[ct.index.tolist()]

  fig, ax = plt.subplots(figsize=(max(10, len(ct) * 0.9), 5))
  x = np.arange(len(ct))
  bottom = np.zeros(len(ct))
  for state in BELIEF_STATE_ORDER:
    vals = ct[state].values
    ax.bar(x, vals, bottom=bottom, label=state.capitalize(),
           color=BELIEF_STATE_COLORS[state], edgecolor="#ffffff", linewidth=0.4, width=0.7)
    for i, (v, b) in enumerate(zip(vals, bottom)):
      if v > 5:
        ax.text(x[i], b + v / 2, f"{v:.0f}%", ha="center", va="center",
                fontsize=7, color="#fff" if state != "unknown" else "#333", fontweight="bold")
    bottom += vals
  ax.set_xticks(x)
  ax.set_xticklabels(ct.index, rotation=30, ha="right", fontsize=9)
  ax.set_ylim(0, 100)
  ax.set_ylabel("% of Beliefs")
  ax.set_title("Belief State Distribution by Model — All Games", fontsize=13, fontweight="bold")
  ax.legend(loc="upper right", fontsize=8, facecolor="#ffffff", edgecolor="#ccc", framealpha=0.8)
  fig.tight_layout()
  save(fig, "22_belief_state_distribution.png")


def plot_cognitive_bias_profile(df_quality: pd.DataFrame, save: Callable[[plt.Figure, str], None]) -> None:
  bias_keys = list(BIAS_NAMES.keys())
  models = sorted(df_quality["model"].unique())
  n_models = len(models)
  n_biases = len(bias_keys)

  fig, ax = plt.subplots(figsize=(max(10, n_models * 1.5), 5))
  bar_w = 0.8 / n_biases
  x = np.arange(n_models)

  for i, bk in enumerate(bias_keys):
    rates = []
    for model in models:
      mdf = df_quality[df_quality["model"] == model]
      rates.append(mdf[bk].mean() * 100 if not mdf.empty else 0)
    offset = (i - n_biases / 2 + 0.5) * bar_w
    ax.bar(x + offset, rates, width=bar_w, label=BIAS_NAMES[bk],
           color=BIAS_COLORS[bk], edgecolor="#ffffff", linewidth=0.3)

  ax.set_xticks(x)
  ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
  ax.set_ylabel("% of Games")
  ax.set_ylim(0, 110)
  ax.set_title("Cognitive Bias Profile by Model — All Games", fontsize=13, fontweight="bold")
  ax.legend(loc="upper right", fontsize=7, facecolor="#ffffff", edgecolor="#ccc", framealpha=0.8)
  fig.tight_layout()
  save(fig, "23_cognitive_bias_profile.png")


def plot_belief_evolution(df_beliefs: pd.DataFrame, save: Callable[[plt.Figure, str], None]) -> None:
  if df_beliefs.empty:
    return
  models = sorted(df_beliefs["model"].unique())
  fig, ax = plt.subplots(figsize=(12, 5))
  palette = sns.color_palette("husl", len(models))

  for i, model in enumerate(models):
    mdf = df_beliefs[df_beliefs["model"] == model].copy()
    turn_acc = mdf.groupby("turn")["accuracy"].mean().mul(100).sort_index()
    if len(turn_acc) < 2:
      ax.scatter(turn_acc.index, turn_acc.values, color=palette[i], s=30, label=model, zorder=3)
      continue
    ax.scatter(turn_acc.index, turn_acc.values, color=palette[i], alpha=0.3, s=20, zorder=2)
    rolling = turn_acc.rolling(window=3, min_periods=1, center=True).mean()
    ax.plot(rolling.index, rolling.values, color=palette[i], linewidth=2, label=model, zorder=3)

  ax.set_xlabel("Turn")
  ax.set_ylabel("Belief Accuracy (%)")
  ax.set_ylim(-5, 105)
  ax.set_title("Belief Accuracy Evolution Over Game Turns — All Games",
               fontsize=13, fontweight="bold")
  ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8,
            facecolor="#ffffff", edgecolor="#ccc", framealpha=0.8)
  fig.tight_layout()
  save(fig, "24_belief_evolution.png")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_all_graphs(local: bool = False, output_dir: Path | None = None) -> None:
  setup_style()

  r2_client = None if local else get_r2_client()
  out_dir   = (output_dir or LOCAL_GRAPHS_DIR) if local else None
  save      = _make_save(local=local, r2_client=r2_client, out_dir=out_dir)

  if local:
    print(f"Reading from {RESULTS_DIR}")
    print(f"Writing to   {out_dir}\n")
  else:
    print(f"Reading from s3://{R2_BUCKET}/results/")
    print(f"Writing to   s3://{R2_BUCKET}/{GRAPHS_PREFIX}\n")

  uploaded = 0

  # --- Checklist ---
  print("[Checklist]")
  checklist_games = load_rubric("checklist", local=local, r2_client=r2_client)
  if checklist_games:
    df = build_checklist_df(checklist_games)
    if not df.empty:
      print(f"  {len(df)} rows | {df['model'].nunique()} models | "
            f"{df['game'].nunique()} games | {df['behavior'].nunique()} behaviors")
      plot_behavior_heatmap(df, save)
      plot_category_breakdown(df, save)
      plot_behavior_rates(df, save)
      plot_radar_charts(df, save)
      uploaded += 4
    else:
      print("  DataFrame is empty — check judged_game.json structure.")
  else:
    print("  No checklist data found.")

  # --- Language ---
  print("\n[Language]")
  lang_games = load_rubric("language", local=local, r2_client=r2_client)
  if lang_games:
    lang_df = build_language_df(lang_games)
    if not lang_df.empty:
      print(f"  {len(lang_df)} rows | {lang_df['model'].nunique()} models | "
            f"{lang_df['game'].nunique()} games | {lang_df['behavior'].nunique()} behaviors")
      plot_language_presence_heatmap(lang_df, save)
      plot_language_frequency_heatmap(lang_df, save)
      plot_language_radar(lang_df, save)
      plot_language_frequency_boxplots(lang_df, save)
      plot_language_strategy_profile(lang_df, save)
      uploaded += 5
    else:
      print("  DataFrame is empty — check judged_game_language.json structure.")
  else:
    print("  No language data found.")

  # --- Belief ---
  print("\n[Belief]")
  belief_games = load_rubric("belief", local=local, r2_client=r2_client)
  if belief_games:
    df_beliefs, df_quality, df_tom = build_belief_dfs(belief_games)
    print(f"  beliefs={len(df_beliefs)} | quality={len(df_quality)} | tom={len(df_tom)}")
    if not df_tom.empty:
      plot_tom_depth(df_tom, save)
      uploaded += 1
    if not df_beliefs.empty:
      plot_belief_accuracy(df_beliefs, save)
      plot_belief_state_distribution(df_beliefs, save)
      plot_belief_evolution(df_beliefs, save)
      uploaded += 3
    if not df_quality.empty:
      plot_cognitive_bias_profile(df_quality, save)
      uploaded += 1
  else:
    print("  No belief data found.")

  destination = str(out_dir) if local else f"s3://{R2_BUCKET}/{GRAPHS_PREFIX}"
  print(f"\nDone — {uploaded} graph(s) written to {destination}")


if __name__ == "__main__":
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
  ap.add_argument(
    "--local", action="store_true",
    help="Read JSONs from LLM_judge/data/results/ and write PNGs to LLM_judge/data/graphs/ "
         "instead of fetching from / uploading to R2.",
  )
  ap.add_argument(
    "--output-dir", type=Path, default=None,
    help="Override local output directory (only meaningful with --local).",
  )
  args = ap.parse_args()

  generate_all_graphs(local=args.local, output_dir=args.output_dir)
