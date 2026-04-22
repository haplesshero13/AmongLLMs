"""
Standalone script to generate all graphs from local final*.json files.
No R2 connection needed. Reads from LLM_judge/data/results/, saves PNGs
to LLM_judge/data/graphs/.

Usage:
    uv run python generate_local_graphs.py
    uv run python generate_local_graphs.py --output-dir /path/to/output
"""

import glob
import json
import os
import re
import sys
from math import pi
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "LLM_judge" / "data" / "results"
DEFAULT_OUTPUT = str(PROJECT_ROOT / "LLM_judge" / "data" / "graphs")

# ═══════════════════════════════════════════════════════════════════════════
# Taxonomies
# ═══════════════════════════════════════════════════════════════════════════

BEHAVIOR_CATEGORIES = {
    "Lying": "Deception", "Deflection": "Deception", "Gaslighting": "Deception",
    "Bus-throwing": "Deception", "Proactive alibi construction": "Deception",
    "Evidence-based accusation": "Social", "Unsupported accusation": "Social",
    "Contradiction detection": "Social", "Bandwagoning": "Social",
    "Information sharing": "Social", "Persuasion": "Social", "Passivity": "Social",
    "Self-incrimination": "Social", "Humor": "Social", "Sarcasm": "Social",
    "Strategic voting": "Voting", "Vote against interest": "Voting",
    "Vote skip with evidence available": "Voting",
    "Target stalking": "Spatial", "Safety seeking": "Spatial",
    "Threat recognition": "Spatial", "Appropriate threat response": "Spatial",
    "Strategic paralysis": "Spatial",
    "Kill opportunity assessment": "Strategic", "Task prioritization": "Strategic",
    "Partner coordination": "Strategic",
}
CATEGORY_ORDER = ["Deception", "Social", "Voting", "Spatial", "Strategic"]
CATEGORY_COLORS = {
    "Deception": "#e07b9b", "Social": "#5ba67a", "Voting": "#d4915e",
    "Spatial": "#7b9ec4", "Strategic": "#9b82b0",
}

LANGUAGE_BEHAVIOR_CATEGORIES = {
    "Emotional escalation": "Emotional", "Hedging language": "Emotional",
    "Overclaiming certainty": "Emotional", "Pleading or appealing": "Emotional",
    "Fabricated testimony": "Rhetorical", "Credibility leveraging": "Rhetorical",
    "Reactive defensiveness": "Rhetorical", "Interrogation": "Rhetorical",
    "Narrative construction": "Rhetorical", "Echo/mirroring": "Rhetorical",
    "Rapport building": "Social", "Distancing language": "Social",
    "In-group/out-group framing": "Social", "Silence as strategy": "Social",
}
LANG_CATEGORY_ORDER = ["Emotional", "Rhetorical", "Social"]
LANG_CATEGORY_COLORS = {"Emotional": "#e07b9b", "Rhetorical": "#d4915e", "Social": "#5ba67a"}

BELIEF_STATE_ORDER = ["innocent", "suspicious", "impostor", "unknown"]
BELIEF_STATE_COLORS = {
    "innocent": "#66bb6a", "suspicious": "#ffa726", "impostor": "#ef5350", "unknown": "#bdbdbd",
}
BIAS_NAMES = {
    "responsive_to_evidence": "Evidence\nResponsiveness",
    "anchoring_bias": "Anchoring\nBias",
    "recency_bias": "Recency\nBias",
    "social_influence": "Social\nInfluence",
}
BIAS_COLORS = {
    "responsive_to_evidence": "#43a047", "anchoring_bias": "#e53935",
    "recency_bias": "#fb8c00", "social_influence": "#8e24aa",
}


# ═══════════════════════════════════════════════════════════════════════════
# Style + save helpers
# ═══════════════════════════════════════════════════════════════════════════

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
        "text.color": "#1a1a1a", "axes.labelcolor": "#1a1a1a",
        "xtick.color": "#444444", "ytick.color": "#444444",
        "font.family": "sans-serif", "font.size": 10, "axes.grid": False,
    })


def save_fig(fig, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Local file loaders
# ═══════════════════════════════════════════════════════════════════════════

_FOLDER_RE = re.compile(r"judge_game_(game_\d+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_")


def _game_sort_key(folder):
    m = re.match(r"game_(\d+)_", folder)
    return int(m.group(1)) if m else 0


def _extract_folder(filename):
    m = _FOLDER_RE.search(filename)
    return m.group(1) if m else filename


def _load_finals(suffix):
    pattern = str(RESULTS_DIR / f"judge_game_*_{suffix}.json")
    files = glob.glob(pattern)
    if suffix == "final":
        files = [f for f in files if "_final_language" not in f and "_final_belief" not in f]
    games = {}
    for fpath in sorted(files):
        folder = _extract_folder(os.path.basename(fpath))
        with open(fpath) as f:
            games[folder] = json.load(f)
    return dict(sorted(games.items(), key=lambda kv: _game_sort_key(kv[0])))


# ═══════════════════════════════════════════════════════════════════════════
# DataFrame builders
# ═══════════════════════════════════════════════════════════════════════════

def build_checklist_df(all_games):
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
                        "game": game_label, "model": model, "behavior": behavior,
                        "category": BEHAVIOR_CATEGORIES.get(behavior, "Strategic"),
                        "present": int(bool(b.get("present", False))),
                    })
    return pd.DataFrame(rows)


def build_language_df(all_games):
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
                        "game": game_label, "model": model, "behavior": behavior,
                        "category": LANGUAGE_BEHAVIOR_CATEGORIES.get(behavior, "Social"),
                        "present": int(bool(b.get("present", False))),
                        "frequency": int(b.get("frequency", 0) or 0),
                    })
    return pd.DataFrame(rows)


def build_belief_dfs(all_games):
    belief_rows, quality_rows, tom_rows = [], [], []
    for folder, judgement in all_games.items():
        m = re.match(r"game_(\d+)_", folder)
        game_label = f"Game {m.group(1)}" if m else folder
        for model, pdata in judgement.items():
            if not isinstance(pdata, dict) or "turn_by_turn" not in pdata:
                continue
            for entry in pdata.get("turn_by_turn", []):
                belief_rows.append({
                    "game": game_label, "model": model,
                    "turn": entry.get("turn", 0),
                    "subject_player": entry.get("subject_player", ""),
                    "belief_state": str(entry.get("belief_state", "unknown")).lower().strip(),
                    "confidence": str(entry.get("confidence", "low")).lower().strip(),
                    "accuracy": bool(entry.get("accuracy", False)),
                    "change_from_previous": bool(entry.get("change_from_previous", False)),
                })
            bq = pdata.get("belief_updating_quality", {})
            correct = int(bq.get("correct_updates", 0) or 0)
            incorrect = int(bq.get("incorrect_updates", 0) or 0)
            total = correct + incorrect
            quality_rows.append({
                "game": game_label, "model": model,
                "responsive_to_evidence": bool(bq.get("responsive_to_evidence", False)),
                "anchoring_bias": bool(bq.get("anchoring_bias", False)),
                "recency_bias": bool(bq.get("recency_bias", False)),
                "social_influence": bool(bq.get("social_influence", False)),
                "correct_updates": correct, "incorrect_updates": incorrect,
                "update_accuracy_rate": (correct / total * 100) if total > 0 else 0.0,
            })
            tom = pdata.get("theory_of_mind", {})
            tom_rows.append({
                "game": game_label, "model": model,
                "deepest_level": int(tom.get("deepest_level", 0) or 0),
                "level_0_present": bool(tom.get("level_0_present", False)),
                "level_1_present": bool(tom.get("level_1_present", False)),
                "level_2_present": bool(tom.get("level_2_present", False)),
                "level_3_present": bool(tom.get("level_3_present", False)),
                "failed_tom_count": len(pdata.get("failed_tom_instances", [])),
            })
    return (
        pd.DataFrame(belief_rows) if belief_rows else pd.DataFrame(),
        pd.DataFrame(quality_rows) if quality_rows else pd.DataFrame(),
        pd.DataFrame(tom_rows) if tom_rows else pd.DataFrame(),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Legend helpers
# ═══════════════════════════════════════════════════════════════════════════

def _cat_legend():
    return [Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=CATEGORY_COLORS[c], markersize=10, label=c)
            for c in CATEGORY_ORDER]

def _lang_cat_legend():
    return [Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=LANG_CATEGORY_COLORS[c], markersize=10, label=c)
            for c in LANG_CATEGORY_ORDER]


# ═══════════════════════════════════════════════════════════════════════════
# CHECKLIST GRAPHS (01-06)
# ═══════════════════════════════════════════════════════════════════════════

def plot_behavior_heatmap(df, out):
    pivot = df.groupby(["model", "behavior"])["present"].mean().unstack(fill_value=0)
    def _sort(b):
        cat = BEHAVIOR_CATEGORIES.get(b, "Strategic")
        idx = CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else len(CATEGORY_ORDER)
        return (idx, -pivot[b].mean())
    pivot = pivot[sorted(pivot.columns, key=_sort)]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    n_b, n_m = len(pivot.columns), len(pivot.index)
    fig, ax = plt.subplots(figsize=(max(18, n_b * 0.65), max(5, n_m * 0.55)))
    cmap = mcolors.LinearSegmentedColormap.from_list("among", ["#f0f0f0", "#1565c0"])
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n_b))
    ax.set_xticklabels([b[:28] for b in pivot.columns], rotation=50, ha="right", fontsize=8)
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(n_m):
        for j in range(n_b):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    color="#ffffff" if val > 0.5 else "#555555", fontsize=7, fontweight="bold")
    for j, bname in enumerate(pivot.columns):
        cat = BEHAVIOR_CATEGORIES.get(bname, "Strategic")
        ax.plot(j, -0.65, marker="s", color=CATEGORY_COLORS[cat], markersize=7, clip_on=False)
    ax.set_title("Behavior Presence Rate by Model — All Games", fontsize=14, fontweight="bold", pad=48)
    ax.legend(handles=_cat_legend(), bbox_to_anchor=(0.5, 1.0), loc="lower center",
              ncol=len(CATEGORY_ORDER), fontsize=8, facecolor="#ffffff", edgecolor="#ccc",
              framealpha=0.8, title="Category", title_fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("% games where behavior was present", fontsize=8)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    save_fig(fig, "01_heatmap_behavior_by_model.png", out)


def plot_category_breakdown(df, out):
    cat_counts = (df[df["present"] == 1].groupby(["model", "category"]).size()
                  .unstack(fill_value=0).reindex(columns=CATEGORY_ORDER, fill_value=0))
    row_sums = cat_counts.sum(axis=1).replace(0, 1)
    cat_pct = cat_counts.div(row_sums, axis=0).mul(100)
    overall = df.groupby("model")["present"].mean()
    cat_pct = cat_pct.loc[overall.reindex(cat_pct.index).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(10, len(cat_pct) * 0.9), 5))
    x = np.arange(len(cat_pct))
    bottom = np.zeros(len(cat_pct))
    for cat in CATEGORY_ORDER:
        vals = cat_pct[cat].values
        ax.bar(x, vals, bottom=bottom, label=cat, color=CATEGORY_COLORS[cat],
               edgecolor="#ffffff", linewidth=0.4, width=0.7)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 5:
                ax.text(x[i], b + v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=7, color="#fff", fontweight="bold")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(cat_pct.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of Present Behaviors (%)")
    ax.set_title("Behavior Category Breakdown per Model — All Games", fontsize=13, fontweight="bold", pad=48)
    ax.legend(handles=_cat_legend(), bbox_to_anchor=(0.5, 1.0), loc="lower center",
              ncol=len(CATEGORY_ORDER), fontsize=8, facecolor="#ffffff", edgecolor="#ccc",
              framealpha=0.8, title="Category", title_fontsize=8)
    save_fig(fig, "02_category_breakdown_by_model.png", out)


def plot_behavior_rates(df, out):
    rates = df.groupby("behavior")["present"].mean().mul(100).sort_values(ascending=False)
    def _colors(behaviors):
        return [CATEGORY_COLORS.get(BEHAVIOR_CATEGORIES.get(b, "Strategic"), "#888") for b in behaviors]
    top_n = min(10, len(rates))
    top, bot = rates.head(top_n), rates.tail(top_n).sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, top_n * 0.5)))
    axes[0].barh(range(len(top)), top.values[::-1], color=_colors(top.index[::-1]))
    axes[0].set_yticks(range(len(top)))
    axes[0].set_yticklabels(top.index[::-1], fontsize=8)
    axes[0].set_title("Most Common Behaviors", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("% Games Present")
    axes[0].set_xlim(0, 105)
    for i, v in enumerate(top.values[::-1]):
        axes[0].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    axes[1].barh(range(len(bot)), bot.values, color=_colors(bot.index))
    axes[1].set_yticks(range(len(bot)))
    axes[1].set_yticklabels(bot.index, fontsize=8)
    axes[1].set_title("Least Common Behaviors", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("% Games Present")
    axes[1].set_xlim(0, 105)
    for i, v in enumerate(bot.values):
        axes[1].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    fig.legend(handles=_cat_legend(), loc="lower center", ncol=len(CATEGORY_ORDER),
               fontsize=8, facecolor="#ffffff", edgecolor="#ccc", framealpha=0.8, title="Category")
    fig.suptitle("Behavior Frequency Across All Games & Models", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.1, 1, 0.95])
    save_fig(fig, "04_behavior_rates.png", out)


def plot_radar_charts(df, out):
    models = sorted(df["model"].unique())
    n_models = len(models)
    cols = min(4, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), subplot_kw=dict(polar=True))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    categories = CATEGORY_ORDER
    n_cats = len(categories)
    angles = [n * 2 * pi / n_cats for n in range(n_cats)] + [0]
    cat_rates = (df.groupby(["model", "category"])["present"].mean()
                 .unstack(fill_value=0).reindex(columns=categories, fill_value=0))
    y_max = min(1.0, cat_rates.values.max() * 1.2) if cat_rates.values.max() > 0 else 1.0
    step = y_max / 4
    yticks = [step * i for i in range(1, 5)]
    for idx, model in enumerate(models):
        ax = axes[idx]
        ax.set_facecolor("#ffffff")
        values = cat_rates.loc[model].tolist() if model in cat_rates.index else [0] * n_cats
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
    save_fig(fig, "06_player_radar.png", out)


# ═══════════════════════════════════════════════════════════════════════════
# LANGUAGE GRAPHS (10-14)
# ═══════════════════════════════════════════════════════════════════════════

def plot_language_presence_heatmap(df, out):
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
    ax.legend(handles=_lang_cat_legend(), bbox_to_anchor=(0.5, 1.0), loc="lower center",
              ncol=len(LANG_CATEGORY_ORDER), fontsize=8, facecolor="#ffffff", edgecolor="#ccc",
              framealpha=0.8, title="Category", title_fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("% games where behavior was present", fontsize=8)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    save_fig(fig, "10_lang_heatmap_presence.png", out)


def plot_language_frequency_heatmap(df, out):
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
    ax.legend(handles=_lang_cat_legend(), bbox_to_anchor=(0.5, 1.0), loc="lower center",
              ncol=len(LANG_CATEGORY_ORDER), fontsize=8, facecolor="#ffffff", edgecolor="#ccc",
              framealpha=0.8, title="Category", title_fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Avg instance count per game", fontsize=8)
    save_fig(fig, "11_lang_heatmap_frequency.png", out)


def plot_language_radar(df, out):
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
        color = LANG_CATEGORY_COLORS.get(cats[0], "#1565c0")
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
    save_fig(fig, "12_lang_radar_by_model.png", out)


def plot_language_frequency_boxplots(df, out):
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
    save_fig(fig, "13_lang_frequency_boxplots.png", out)


def plot_language_strategy_profile(df, out):
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
    ax.legend(handles=_lang_cat_legend(), loc="upper right", fontsize=8,
              facecolor="#ffffff", edgecolor="#ccc", framealpha=0.8)
    fig.tight_layout()
    save_fig(fig, "14_lang_strategy_profile.png", out)


# ═══════════════════════════════════════════════════════════════════════════
# BELIEF TRACKING GRAPHS (20-24)
# ═══════════════════════════════════════════════════════════════════════════

def plot_tom_depth(df_tom, out):
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
    save_fig(fig, "20_tom_depth_by_model.png", out)


def plot_belief_accuracy(df_beliefs, out):
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
    save_fig(fig, "21_belief_accuracy_by_model.png", out)


def plot_belief_state_distribution(df_beliefs, out):
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
    save_fig(fig, "22_belief_state_distribution.png", out)


def plot_cognitive_bias_profile(df_quality, out):
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
    save_fig(fig, "23_cognitive_bias_profile.png", out)


def plot_belief_evolution(df_beliefs, out):
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
    save_fig(fig, "24_belief_evolution.png", out)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    out = DEFAULT_OUTPUT
    for arg in sys.argv[1:]:
        if arg.startswith("--output-dir="):
            out = arg.split("=", 1)[1]
        elif arg == "--output-dir":
            idx = sys.argv.index(arg)
            if idx + 1 < len(sys.argv):
                out = sys.argv[idx + 1]

    setup_style()
    print(f"Output directory: {out}\n")

    # --- Checklist ---
    checklist_games = _load_finals("final")
    if checklist_games:
        df = build_checklist_df(checklist_games)
        if not df.empty:
            print(f"[Checklist] {len(df)} rows | {df['model'].nunique()} models | "
                  f"{df['game'].nunique()} games | {df['behavior'].nunique()} behaviors")
            plot_behavior_heatmap(df, out)
            plot_category_breakdown(df, out)
            plot_behavior_rates(df, out)
            plot_radar_charts(df, out)
    else:
        print("No checklist data found.")

    # --- Language ---
    lang_games = _load_finals("final_language")
    if lang_games:
        lang_df = build_language_df(lang_games)
        if not lang_df.empty:
            print(f"\n[Language] {len(lang_df)} rows | {lang_df['model'].nunique()} models | "
                  f"{lang_df['game'].nunique()} games | {lang_df['behavior'].nunique()} behaviors")
            plot_language_presence_heatmap(lang_df, out)
            plot_language_frequency_heatmap(lang_df, out)
            plot_language_radar(lang_df, out)
            plot_language_frequency_boxplots(lang_df, out)
            plot_language_strategy_profile(lang_df, out)
    else:
        print("No language data found.")

    # --- Belief ---
    belief_games = _load_finals("final_belief")
    if belief_games:
        df_beliefs, df_quality, df_tom = build_belief_dfs(belief_games)
        print(f"\n[Belief] beliefs={len(df_beliefs)} | quality={len(df_quality)} | tom={len(df_tom)}")
        if not df_tom.empty:
            plot_tom_depth(df_tom, out)
        if not df_beliefs.empty:
            plot_belief_accuracy(df_beliefs, out)
            plot_belief_state_distribution(df_beliefs, out)
            plot_belief_evolution(df_beliefs, out)
        if not df_quality.empty:
            plot_cognitive_bias_profile(df_quality, out)
    else:
        print("No belief data found.")

    print(f"\nDone. All graphs saved to {out}/")


if __name__ == "__main__":
    main()
