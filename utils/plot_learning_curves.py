"""
plot_learning_curves.py — Training learning curves (eval reward vs timesteps).

The sweep figures in generate_plots.py summarize the FINAL score of each run.
This script plots the missing piece: how the agents actually *converged* during
training, read from the EvalCallback logs (`logs/<algo>/run_<n>/evaluations.npz`,
5 eval episodes every 20k steps).

Outputs:
  outputs/training/learning_curves.png   — best DQN vs best PPO, mean ± std band,
                                            with all runs drawn faintly behind
  outputs/training/learning_curves.csv    — the plotted curve data (so the figure
                                            stays reproducible even though logs/
                                            is gitignored)

REINFORCE is a custom (non-SB3) implementation without EvalCallback logs, so it
does not appear here — this is the on-policy-vs-off-policy SB3 comparison.

Run:  python utils/plot_learning_curves.py
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")
OUT_DIR = os.path.join(ROOT, "outputs", "training")

# Best run per algorithm, taken from the sweep result tables (max mean reward).
COLORS = {"DQN": "#E8743B", "PPO": "#19A979"}


def best_run(algo: str) -> int:
    """Run number with the highest sweep mean reward for this algorithm."""
    df = pd.read_csv(os.path.join(OUT_DIR, f"{algo.lower()}_results.csv"))
    return int(df.loc[df["Mean Reward"].idxmax(), "Run"])


def curve(algo: str, run: int):
    """(timesteps, mean, std) of eval reward for one run, or None if missing."""
    path = os.path.join(LOG_DIR, algo.lower(), f"run_{run}", "evaluations.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return d["timesteps"], d["results"].mean(axis=1), d["results"].std(axis=1)


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    rows = []

    for algo in ("DQN", "PPO"):
        color = COLORS[algo]
        best = best_run(algo)

        # Faint lines for every run — shows the whole sweep's convergence spread.
        for npz in sorted(glob.glob(os.path.join(LOG_DIR, algo.lower(), "run_*", "evaluations.npz"))):
            d = np.load(npz)
            ax.plot(d["timesteps"], d["results"].mean(axis=1),
                    color=color, alpha=0.15, linewidth=1)

        # Bold line + std band for the champion/best run.
        c = curve(algo, best)
        if c is None:
            print(f"! no eval log for {algo} run {best}")
            continue
        ts, mean, std = c
        ax.plot(ts, mean, color=color, linewidth=2.5, label=f"{algo} (best — run {best})")
        ax.fill_between(ts, mean - std, mean + std, color=color, alpha=0.18)
        for t, m, s in zip(ts, mean, std):
            rows.append({"algorithm": algo, "run": best, "timesteps": int(t),
                         "eval_reward_mean": round(float(m), 2), "eval_reward_std": round(float(s), 2)})

    ax.set_title("Training Learning Curves — evaluation reward vs timesteps",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Training timesteps", fontsize=11)
    ax.set_ylabel("Mean eval reward (5 episodes)", fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_facecolor("#F8F9FA")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(0.5, -0.02,
             "Bold = best run per algorithm (mean ± std); faint = all 10 sweep runs. "
             "REINFORCE is a custom non-SB3 agent without eval logs.",
             ha="center", fontsize=8, color="#666")

    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, "learning_curves.png")
    csv = os.path.join(OUT_DIR, "learning_curves.csv")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"wrote {png}")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
