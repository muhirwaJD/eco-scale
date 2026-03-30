"""
generate_plots.py — Generate all comparison plots from real training results.
Run: python3.14 generate_plots.py
Outputs saved to outputs/
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUTPUT_DIR = "/home/muhirwa/alu/eco-scale/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load all CSVs ──────────────────────────────────────────────────
dqn = pd.read_csv(f"{OUTPUT_DIR}/dqn_results.csv")
ppo = pd.read_csv(f"{OUTPUT_DIR}/ppo_results.csv")
rf  = pd.read_csv(f"{OUTPUT_DIR}/reinforce_results.csv")

ALGO_COLORS = {
    "DQN":       "#2196F3",
    "PPO":       "#4CAF50",
    "REINFORCE": "#FF9800",
}

# ══════════════════════════════════════════════════════════
# PLOT 1: Cumulative Rewards — subplots per algorithm
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle("Hyperparameter Tuning: Mean Episode Reward per Run",
             fontsize=15, fontweight="bold", y=1.02)

for ax, df, algo in zip(axes, [dqn, ppo, rf], ["DQN", "PPO", "REINFORCE"]):
    color = ALGO_COLORS[algo]
    best_idx = df["Mean Reward"].idxmax()
    bar_colors = [color if i != best_idx else "#FFD700" for i in range(len(df))]
    bars = ax.bar(df["Run"], df["Mean Reward"], color=bar_colors,
                  alpha=0.85, edgecolor="white", linewidth=0.8, width=0.7)
    ax.errorbar(df["Run"], df["Mean Reward"], yerr=df["Std Reward"],
                fmt="none", ecolor="black", alpha=0.5, capsize=3)

    # Annotate best
    best = df.loc[best_idx]
    ax.annotate(f"★ Best\n{best['Mean Reward']:.2f}",
                xy=(best["Run"], best["Mean Reward"]),
                xytext=(best["Run"], best["Mean Reward"] + 3),
                ha="center", fontsize=8, color="#B8860B", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#B8860B", lw=1))

    ax.set_title(algo, fontsize=13, fontweight="bold", color=color)
    ax.set_xlabel("Run #", fontsize=10)
    ax.set_xticks(df["Run"])
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_facecolor("#F8F9FA")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Mean Episode Reward", fontsize=11)
plt.tight_layout()
path = f"{OUTPUT_DIR}/cumulative_rewards.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Plot 1: {path}")

# ══════════════════════════════════════════════════════════
# PLOT 2: Convergence Comparison — all algorithms side by side
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Algorithm Convergence Comparison", fontsize=14, fontweight="bold")

# Left: Best run per algorithm
ax = axes[0]
algos = ["DQN", "PPO", "REINFORCE"]
dfs   = [dqn, ppo, rf]
bests = [df.loc[df["Mean Reward"].idxmax()] for df in dfs]
best_rewards = [b["Mean Reward"] for b in bests]
best_stds    = [b["Std Reward"]  for b in bests]

bars = ax.bar(algos, best_rewards, color=[ALGO_COLORS[a] for a in algos],
              alpha=0.85, edgecolor="white", width=0.5)
ax.errorbar(algos, best_rewards, yerr=best_stds,
            fmt="none", ecolor="black", capsize=6, linewidth=2)
for bar, r, s in zip(bars, best_rewards, best_stds):
    ax.text(bar.get_x() + bar.get_width()/2, r + s + 0.3,
            f"{r:.2f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Mean Episode Reward (best run)", fontsize=11)
ax.set_title("Best Run per Algorithm", fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_facecolor("#F8F9FA")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Right: Full reward distributions as box-style bar
ax2 = axes[1]
width = 0.25
x = np.arange(10)

for idx, (algo, df) in enumerate(zip(algos, dfs)):
    offset = (idx - 1) * width
    ax2.bar(x + offset, df["Mean Reward"], width=width,
            color=ALGO_COLORS[algo], alpha=0.75, label=algo, edgecolor="white")

ax2.set_xlabel("Run #", fontsize=10)
ax2.set_ylabel("Mean Episode Reward", fontsize=11)
ax2.set_title("All Runs — All Algorithms", fontsize=12, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels([f"R{i}" for i in range(1, 11)], fontsize=9)
ax2.legend(fontsize=11)
ax2.grid(axis="y", alpha=0.3, linestyle="--")
ax2.set_facecolor("#F8F9FA")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
path = f"{OUTPUT_DIR}/convergence_comparison.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Plot 2: {path}")

# ══════════════════════════════════════════════════════════
# PLOT 3: Sensitivity Analysis — effect of key hyperparams
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Hyperparameter Sensitivity Analysis", fontsize=14, fontweight="bold")

# DQN: gamma sensitivity
ax = axes[0]
ax.scatter(dqn["gamma"], dqn["Mean Reward"],
           c=ALGO_COLORS["DQN"], s=120, alpha=0.9, edgecolors="white", zorder=3)
for _, row in dqn.iterrows():
    ax.annotate(f"R{int(row['Run'])}", (row["gamma"], row["Mean Reward"]),
                textcoords="offset points", xytext=(5, 2), fontsize=7)
ax.set_xlabel("Gamma (discount factor)", fontsize=11)
ax.set_ylabel("Mean Episode Reward", fontsize=11)
ax.set_title("DQN — Gamma Effect", fontweight="bold", color=ALGO_COLORS["DQN"])
ax.grid(alpha=0.3, linestyle="--")
ax.set_facecolor("#F8F9FA")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# PPO: entropy coefficient sensitivity
ax = axes[1]
ax.scatter(ppo["ent_coef"], ppo["Mean Reward"],
           c=ALGO_COLORS["PPO"], s=120, alpha=0.9, edgecolors="white", zorder=3)
for _, row in ppo.iterrows():
    ax.annotate(f"R{int(row['Run'])}", (row["ent_coef"], row["Mean Reward"]),
                textcoords="offset points", xytext=(5, 2), fontsize=7)
ax.set_xlabel("Entropy Coefficient", fontsize=11)
ax.set_ylabel("Mean Episode Reward", fontsize=11)
ax.set_title("PPO — Entropy Effect", fontweight="bold", color=ALGO_COLORS["PPO"])
ax.grid(alpha=0.3, linestyle="--")
ax.set_facecolor("#F8F9FA")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# REINFORCE: network size effect
ax = axes[2]
colors_rf = [ALGO_COLORS["REINFORCE"] if b else "#aaa" for b in rf["use_baseline"]]
sc = ax.scatter(rf["hidden"], rf["Mean Reward"],
                c=colors_rf, s=120, alpha=0.9, edgecolors="white", zorder=3)
for _, row in rf.iterrows():
    ax.annotate(f"R{int(row['Run'])}", (row["hidden"], row["Mean Reward"]),
                textcoords="offset points", xytext=(5, 2), fontsize=7)
ax.set_xlabel("Hidden Layer Size", fontsize=11)
ax.set_ylabel("Mean Episode Reward", fontsize=11)
ax.set_title("REINFORCE — Network Size Effect", fontweight="bold", color=ALGO_COLORS["REINFORCE"])
legend_els = [mpatches.Patch(color=ALGO_COLORS["REINFORCE"], label="With baseline"),
              mpatches.Patch(color="#aaa", label="No baseline")]
ax.legend(handles=legend_els, fontsize=9)
ax.grid(alpha=0.3, linestyle="--")
ax.set_facecolor("#F8F9FA")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
path = f"{OUTPUT_DIR}/sensitivity_analysis.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Plot 3: {path}")

# ══════════════════════════════════════════════════════════
# PLOT 4: Stability comparison (std reward across all runs)
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Training Stability — Std Reward Across All Runs", fontsize=13, fontweight="bold")

width = 0.25
x = np.arange(10)
for idx, (algo, df) in enumerate(zip(algos, dfs)):
    offset = (idx - 1) * width
    ax.bar(x + offset, df["Std Reward"], width=width,
           color=ALGO_COLORS[algo], alpha=0.8, label=algo, edgecolor="white")

ax.set_xlabel("Run #", fontsize=11)
ax.set_ylabel("Std of Episode Reward (lower = more stable)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels([f"R{i}" for i in range(1, 11)], fontsize=9)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_facecolor("#F8F9FA")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
path = f"{OUTPUT_DIR}/stability_comparison.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Plot 4: {path}")

# ══════════════════════════════════════════════════════════
# PLOT 5: Summary table figure (for report)
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 2.5))
ax.axis("off")

summary_data = []
for algo, df in zip(algos, [dqn, ppo, rf]):
    best = df.loc[df["Mean Reward"].idxmax()]
    worst = df.loc[df["Mean Reward"].idxmin()]
    summary_data.append([
        algo,
        f"{best['Mean Reward']:.2f} ± {best['Std Reward']:.2f}",
        best["Notes"],
        f"{worst['Mean Reward']:.2f}",
        worst["Notes"],
        f"{df['Mean Reward'].mean():.2f}",
    ])

cols = ["Algorithm", "Best Reward", "Best Config", "Worst Reward", "Worst Config", "Avg Reward"]
table = ax.table(cellText=summary_data, colLabels=cols,
                 cellLoc="center", loc="center",
                 colColours=["#1565C0"]*6)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2.0)

# Style header
for j in range(len(cols)):
    table[0, j].set_text_props(color="white", fontweight="bold")
    table[0, j].set_facecolor("#1565C0")

# Highlight best row (DQN, row 1)
for j in range(len(cols)):
    table[1, j].set_facecolor("#E3F2FD")

fig.suptitle("Eco-Scale RL — Algorithm Summary", fontsize=13, fontweight="bold", y=1.05)
plt.tight_layout()
path = f"{OUTPUT_DIR}/summary_table.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Plot 5: {path}")

print("\n" + "="*55)
print("  All plots generated successfully!")
print(f"  Saved to: {OUTPUT_DIR}")
print("="*55)

# Print summary stats
print("\n📊 Final Summary:")
for algo, df in zip(algos, [dqn, ppo, rf]):
    best = df.loc[df["Mean Reward"].idxmax()]
    print(f"  {algo:10s} — Best: {best['Mean Reward']:.2f} ± {best['Std Reward']:.2f}  ({best['Notes']})")
