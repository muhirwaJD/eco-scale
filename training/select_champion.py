"""
select_champion.py — Pick the best DQN run and produce the headline figure.

Reads the sweep results (outputs/dqn_results.csv), selects the run with the
highest mean episode reward, copies its model to models/eco_scale_dqn_best.zip,
writes champion metadata, and renders outputs/dqn_mean_reward.png.

Run AFTER training/dqn_training.py:
    python training/select_champion.py
"""

import os
import json
import shutil
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_PATH = os.path.join(ROOT, "outputs", "dqn_results.csv")
RUN_MODEL_DIR = os.path.join(ROOT, "models", "dqn")          # eco_scale_dqn_run_{i}.zip
CHAMPION_PATH = os.path.join(ROOT, "models", "eco_scale_dqn_best.zip")
METADATA_PATH = os.path.join(ROOT, "models", "champion_metadata.json")
FIGURE_PATH = os.path.join(ROOT, "outputs", "dqn_mean_reward.png")

# Random-agent reference on the real traces (per data-integration note).
RANDOM_BASELINE_RANGE = (-315.0, -127.0)


def main():
    df = pd.read_csv(RESULTS_PATH)
    best_idx = df["Mean Reward"].idxmax()
    best = df.loc[best_idx]
    best_run = int(best["Run"])

    # 1. locate + copy the champion model (final model the CSV reward refers to)
    src = os.path.join(RUN_MODEL_DIR, f"eco_scale_dqn_run_{best_run}.zip")
    if not os.path.exists(src):
        raise FileNotFoundError(f"Champion model not found: {src}")
    shutil.copyfile(src, CHAMPION_PATH)

    # 2. write metadata for reproducibility
    metadata = {
        "champion_run": best_run,
        "notes": str(best["Notes"]),
        "mean_reward": float(best["Mean Reward"]),
        "std_reward": float(best["Std Reward"]),
        "hyperparameters": {
            k: (float(best[k]) if k in ("learning_rate", "gamma") else int(best[k]))
            for k in ["learning_rate", "gamma", "buffer_size", "batch_size",
                      "exploration_fraction", "exploration_final_eps",
                      "target_update_interval"]
        },
        "trace": "cyclical (real Alibaba 2018)",
        "source_model": os.path.relpath(src, ROOT),
        "selected_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "reward_scale_note": (
            "Trained on real Alibaba traces with the rewritten reward (fixed-length "
            "288-step episodes, no termination). Scale is NOT comparable to the "
            "summative synthetic-env figure of -12.21. Random-agent reference on "
            f"these traces: {RANDOM_BASELINE_RANGE[0]:.0f} to {RANDOM_BASELINE_RANGE[1]:.0f}."
        ),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # 3. headline mean-reward figure (10 runs, champion highlighted)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#FFD700" if i == best_idx else "#2196F3" for i in range(len(df))]
    ax.bar(df["Run"], df["Mean Reward"], yerr=df["Std Reward"],
           color=colors, alpha=0.88, edgecolor="white", linewidth=0.8,
           error_kw=dict(ecolor="black", alpha=0.5, capsize=3))

    ax.annotate(f"★ Champion (Run {best_run})\n{best['Mean Reward']:.2f}",
                xy=(best_run, best["Mean Reward"]),
                xytext=(best_run, best["Mean Reward"] + abs(best["Mean Reward"]) * 0.12),
                ha="center", fontsize=10, color="#B8860B", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#B8860B", lw=1.2))

    # random-agent reference band
    ax.axhspan(*RANDOM_BASELINE_RANGE, color="#E57373", alpha=0.12, zorder=0)
    ax.text(df["Run"].max(), np.mean(RANDOM_BASELINE_RANGE),
            "random-agent range", ha="right", va="center",
            fontsize=8, color="#C62828", style="italic")

    ax.set_title("DQN on Real Alibaba Traces — Mean Episode Reward per Run",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Run #", fontsize=11)
    ax.set_ylabel("Mean Episode Reward (10 eval episodes)", fontsize=11)
    ax.set_xticks(df["Run"])
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_facecolor("#F8F9FA")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ Champion: Run {best_run} ({best['Notes']}) "
          f"→ mean reward {best['Mean Reward']:.2f} ± {best['Std Reward']:.2f}")
    print(f"   model    : {os.path.relpath(CHAMPION_PATH, ROOT)}")
    print(f"   metadata : {os.path.relpath(METADATA_PATH, ROOT)}")
    print(f"   figure   : {os.path.relpath(FIGURE_PATH, ROOT)}")


if __name__ == "__main__":
    main()
