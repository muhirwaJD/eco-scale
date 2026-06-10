"""
select_champion.py — Save the champion as a deployable file + headline figure.

Finds the champion automatically (best agent across all sweeps, see agents.py),
copies its model to models/eco_scale_best.zip for deployment, writes
champion_metadata.json, and renders the per-run reward figure for that agent.

Run AFTER training:
    python training/select_champion.py
"""

import os
import sys
import json
import shutil
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from agents import find_champion, ALGORITHMS

CHAMPION_PATH = os.path.join(ROOT, "models", "eco_scale_best.zip")
METADATA_PATH = os.path.join(ROOT, "models", "champion_metadata.json")
FIGURE_PATH = os.path.join(ROOT, "outputs", "champion_mean_reward.png")

# Approximate reference lines for the figure (from the diagnostics, for context).
RANDOM_FLOOR = -470.0          # a random agent scores around here
IDEAL_CEILING = -346.0         # perfect demand-tracking scores around here

# Columns in a results CSV that are NOT hyperparameters.
_NON_HYPERPARAMS = {"Run", "Mean Reward", "Std Reward", "Notes"}


def main():
    champion = find_champion()
    table = pd.read_csv(os.path.join(ROOT, ALGORITHMS[champion.algorithm]["results"]))
    best_row = table.loc[table["Mean Reward"].idxmax()]

    # 1. copy the champion model to a single deployable file
    shutil.copyfile(champion.model_path, CHAMPION_PATH)

    # 2. write metadata (hyperparameters picked up generically per algorithm)
    hyperparams = {c: _to_plain(best_row[c]) for c in table.columns
                   if c not in _NON_HYPERPARAMS}
    metadata = {
        "algorithm": champion.algorithm,
        "run": champion.run,
        "mean_reward": round(champion.reward, 2),
        "std_reward": round(float(best_row["Std Reward"]), 2),
        "notes": str(best_row["Notes"]),
        "hyperparameters": hyperparams,
        "trained_on": "8-trace train split (real Alibaba 2018, multi-trace)",
        "source_model": os.path.relpath(champion.model_path, ROOT),
        "selected_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "note": ("Champion = best run across all algorithms by sweep mean reward. "
                 "Reward scale is specific to this env; compare against the HPA "
                 "baseline and the held-out test results, not the old -12.21 figure."),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # 3. per-run reward figure for the champion's algorithm
    _plot_runs(table, champion)

    print(f"✅ Champion: {champion.algorithm} run {champion.run} "
          f"→ mean reward {champion.reward:.2f}")
    print(f"   model    : {os.path.relpath(CHAMPION_PATH, ROOT)}")
    print(f"   metadata : {os.path.relpath(METADATA_PATH, ROOT)}")
    print(f"   figure   : {os.path.relpath(FIGURE_PATH, ROOT)}")


def _to_plain(value):
    """Turn a numpy value into a plain int/float/str for JSON."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _plot_runs(table, champion):
    best_idx = table["Mean Reward"].idxmax()
    fig, ax = plt.subplots(figsize=(11, 5.5))

    bar_colors = ["#FFD700" if i == best_idx else "#2196F3" for i in range(len(table))]
    ax.bar(table["Run"], table["Mean Reward"], yerr=table["Std Reward"],
           color=bar_colors, alpha=0.88, edgecolor="white", linewidth=0.8,
           error_kw=dict(ecolor="black", alpha=0.5, capsize=3))

    ax.annotate(f"★ Champion (Run {champion.run})\n{champion.reward:.2f}",
                xy=(champion.run, champion.reward),
                xytext=(champion.run, champion.reward + abs(champion.reward) * 0.12),
                ha="center", fontsize=10, color="#B8860B", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#B8860B", lw=1.2))

    ax.axhline(RANDOM_FLOOR, color="#C62828", linestyle=":", lw=1.2)
    ax.text(table["Run"].max(), RANDOM_FLOOR, " random floor",
            ha="right", va="bottom", fontsize=8, color="#C62828", style="italic")
    ax.axhline(IDEAL_CEILING, color="#2E7D32", linestyle="--", lw=1.2)
    ax.text(table["Run"].min(), IDEAL_CEILING, " demand-tracking ceiling",
            ha="left", va="bottom", fontsize=8, color="#2E7D32", style="italic")

    ax.set_title(f"{champion.algorithm} on Real Alibaba Traces — Mean Reward per Run",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Run #", fontsize=11)
    ax.set_ylabel("Mean episode reward", fontsize=11)
    ax.set_xticks(table["Run"])
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_facecolor("#F8F9FA")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
