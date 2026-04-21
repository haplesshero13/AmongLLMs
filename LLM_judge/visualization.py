"""
Visualization for AmongLLMs LLM-as-Judge results.

Loads every results/<game>/judged_game.json from R2, aggregates behavior
data across all games, generates 5 analysis graphs, and uploads them to
results/graphs/ — overwriting the previous versions each run.

Usage:
    python visualization.py
"""

import io
import json
import re
from math import pi

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

R2_BUCKET        = "amongus-leaderboard"
GRAPHS_PREFIX    = "results/graphs/"

# ---------------------------------------------------------------------------
# Behavior taxonomy (kept in sync with prompts.py checklist)
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
# R2 helpers
# ---------------------------------------------------------------------------

def list_judged_game_keys(r2_client) -> list[str]:
  """Return all keys matching results/*/judged_game.json (skips graphs/)."""
  paginator = r2_client.get_paginator("list_objects_v2")
  keys = []
  for page in paginator.paginate(Bucket=R2_BUCKET, Prefix="results/"):
    for obj in page.get("Contents", []):
      key = obj["Key"]
      if key.endswith("/judged_game.json") and "/graphs/" not in key:
        keys.append(key)
  return sorted(keys)


def _game_sort_key(folder: str) -> int:
  m = re.match(r"game_(\d+)_", folder)
  return int(m.group(1)) if m else 0


def load_all_judged_games(r2_client) -> dict[str, dict]:
  """Return {game_folder: judged_game_dict} sorted by game number."""
  keys = list_judged_game_keys(r2_client)
  if not keys:
    print("No judged game results found in R2 under results/.")
    return {}
  print(f"Found {len(keys)} judged game(s).")
  games: dict[str, dict] = {}
  for key in keys:
    parts = key.split("/")
    folder = parts[1] if len(parts) >= 3 else key
    resp = r2_client.get_object(Bucket=R2_BUCKET, Key=key)
    games[folder] = json.loads(resp["Body"].read().decode("utf-8"))
    print(f"  Loaded {key}")
  return dict(sorted(games.items(), key=lambda kv: _game_sort_key(kv[0])))


def _upload_figure(r2_client, fig: plt.Figure, filename: str) -> None:
  """Serialise figure to PNG and upload to R2, overwriting any previous file."""
  buf = io.BytesIO()
  fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
  buf.seek(0)
  key = f"{GRAPHS_PREFIX}{filename}"
  r2_client.put_object(
    Bucket=R2_BUCKET, Key=key, Body=buf.read(), ContentType="image/png"
  )
  plt.close(fig)
  print(f"[R2] Uploaded → s3://{R2_BUCKET}/{key}")


# ---------------------------------------------------------------------------
# Build tidy DataFrame
# ---------------------------------------------------------------------------

def build_dataframe(all_games: dict[str, dict]) -> pd.DataFrame:
  """Flatten all game judgements into a tidy DataFrame."""
  rows = []
  for game_folder, judgement in all_games.items():
    m = re.match(r"game_(\d+)_", game_folder)
    game_label = f"Game {m.group(1)}" if m else game_folder
    for model_name, behaviors in judgement.items():
      if not isinstance(behaviors, list):
        continue
      for b in behaviors:
        behavior = b.get("behavior", "").strip()
        present  = b.get("present", False)
        if behavior:
          rows.append({
            "game":         game_label,
            "game_folder":  game_folder,
            "model":        model_name,
            "behavior":     behavior,
            "category":     BEHAVIOR_CATEGORIES.get(behavior, "Strategic"),
            "present":      int(bool(present)),
          })
  return pd.DataFrame(rows)



def _category_legend_handles() -> list[Line2D]:
  return [
    Line2D([0], [0], marker="s", color="w",
           markerfacecolor=CATEGORY_COLORS[c], markersize=10, label=c)
    for c in CATEGORY_ORDER
  ]


# ---------------------------------------------------------------------------
# Graph 1 — Behavior heatmap  (model × behavior, color = % games present)
# ---------------------------------------------------------------------------

def plot_behavior_heatmap(df: pd.DataFrame, r2_client) -> None:
  pivot = (
    df.groupby(["model", "behavior"])["present"]
    .mean()
    .unstack(fill_value=0)
  )

  # Sort behaviors: grouped by category, then by descending presence rate
  def _beh_sort(b: str):
    cat = BEHAVIOR_CATEGORIES.get(b, "Strategic")
    idx = CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else len(CATEGORY_ORDER)
    return (idx, -pivot[b].mean())

  pivot = pivot[sorted(pivot.columns, key=_beh_sort)]
  # Sort models by total score (highest first)
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

  # Category colour dots above column labels
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

  _upload_figure(r2_client, fig, "01_heatmap_behavior_by_model.png")


# ---------------------------------------------------------------------------
# Graph 2 — Category stacked bar per model
# ---------------------------------------------------------------------------

def plot_category_breakdown(df: pd.DataFrame, r2_client) -> None:
  """Stacked bar per model: share of present behaviors per category, sums to 100%."""
  # Count present behaviors per (model, category)
  cat_counts = (
    df[df["present"] == 1]
    .groupby(["model", "category"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=CATEGORY_ORDER, fill_value=0)
  )
  # Normalise each model's row to 100 %
  row_sums  = cat_counts.sum(axis=1).replace(0, 1)
  cat_pct   = cat_counts.div(row_sums, axis=0).mul(100)
  # Sort by overall presence rate so the ordering remains meaningful
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
  _upload_figure(r2_client, fig, "02_category_breakdown_by_model.png")


# ---------------------------------------------------------------------------
# Graph 3 — Most & least common behaviors (horizontal bar)
# ---------------------------------------------------------------------------

def plot_behavior_rates(df: pd.DataFrame, r2_client) -> None:
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
  _upload_figure(r2_client, fig, "04_behavior_rates.png")


# ---------------------------------------------------------------------------
# Graph 5 — Per-model radar charts (category-level presence rates)
# ---------------------------------------------------------------------------

def plot_radar_charts(df: pd.DataFrame, r2_client) -> None:
  """One radar per model showing category-level presence rates."""
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

  # Scale to actual data range so the polygon fills the chart
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
  _upload_figure(r2_client, fig, "06_player_radar.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all_graphs() -> None:
  r2_client = get_r2_client()

  all_games = load_all_judged_games(r2_client)
  if not all_games:
    print("Nothing to visualize — run evaluation.py first.")
    return

  df = build_dataframe(all_games)
  if df.empty:
    print("DataFrame is empty — check judged_game.json structure.")
    return

  print(
    f"\nBuilt DataFrame: {len(df)} rows | "
    f"{df['model'].nunique()} models | "
    f"{df['game'].nunique()} games | "
    f"{df['behavior'].nunique()} behaviors"
  )

  setup_style()
  print("\nGenerating and uploading graphs...")
  plot_behavior_heatmap(df, r2_client)
  plot_category_breakdown(df, r2_client)
  plot_behavior_rates(df, r2_client)
  plot_radar_charts(df, r2_client)
  print("\nDone — all 4 graphs uploaded to R2 under results/graphs/.")


if __name__ == "__main__":
  generate_all_graphs()
