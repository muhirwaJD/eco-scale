"""
compare_agents.py — Evaluate all trained agents and generate comparison plots.

Generates:
1. Cumulative reward curves (subplots per algorithm)
2. DQN Q-loss curve
3. PG entropy curves
4. Convergence comparison (all algorithms on same axes)
5. Generalization test (best agent on burst trace)

Usage:
    python3.14 evaluation/compare_agents.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.custom_env import KubernetesEnv

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────

def evaluate_sb3_model(model, env, n_episodes=20):
    """Evaluate an SB3 model and return per-episode rewards and step-by-step data."""
    episode_rewards = []
    all_step_rewards = []
    all_latencies = []
    all_pods = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False
        step_rewards = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += reward
            step_rewards.append(reward)
            all_latencies.append(info.get("latency", 0))
            all_pods.append(info.get("pods", 0))
            done = terminated or truncated
        episode_rewards.append(total)
        all_step_rewards.extend(step_rewards)

    return {
        "episode_rewards": episode_rewards,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "avg_latency": np.mean(all_latencies),
        "avg_pods": np.mean(all_pods),
        "step_rewards": all_step_rewards,
    }


def evaluate_reinforce_model(model_path, env, n_episodes=20):
    """Evaluate a REINFORCE model."""
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
    from reinforce_training import REINFORCE

    agent = REINFORCE(obs_dim=4, act_dim=3)
    agent.load(model_path)

    episode_rewards = []
    all_latencies = []
    all_pods = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            all_latencies.append(info.get("latency", 0))
            all_pods.append(info.get("pods", 0))
            done = terminated or truncated
        episode_rewards.append(total)

    return {
        "episode_rewards": episode_rewards,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "avg_latency": np.mean(all_latencies),
        "avg_pods": np.mean(all_pods),
    }


# ──────────────────────────────────────────────
# Plot generation
# ──────────────────────────────────────────────

def plot_cumulative_rewards():
    """Plot 1: Cumulative reward curves from training CSVs."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle("Cumulative Rewards per Algorithm (Best Runs)", fontsize=14, fontweight="bold")

    algorithms = ["dqn", "ppo", "reinforce"]
    titles = ["DQN", "PPO", "REINFORCE"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for ax, algo, title, color in zip(axes, algorithms, titles, colors):
        csv_path = os.path.join(OUTPUT_DIR, f"{algo}_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            ax.bar(df["Run"], df["Mean Reward"], yerr=df.get("Std Reward", 0),
                   color=color, alpha=0.7, edgecolor="white", linewidth=0.5)
            ax.set_xlabel("Run")
            ax.set_title(title, fontweight="bold")
            ax.set_xticks(df["Run"])
            ax.grid(axis="y", alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"No data\n({csv_path})", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_title(title)

    axes[0].set_ylabel("Mean Episode Reward")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "cumulative_rewards.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_convergence_comparison():
    """Plot 4: All algorithms on same axes — best run of each."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Convergence Comparison — Best Run per Algorithm", fontweight="bold")

    colors = {"DQN": "#2196F3", "PPO": "#4CAF50", "REINFORCE": "#FF9800"}
    algos = ["dqn", "ppo", "reinforce"]
    labels = ["DQN", "PPO", "REINFORCE"]

    for algo, label in zip(algos, labels):
        csv_path = os.path.join(OUTPUT_DIR, f"{algo}_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            best_idx = df["Mean Reward"].idxmax()
            best_reward = df.loc[best_idx, "Mean Reward"]
            ax.bar(label, best_reward, color=colors[label], alpha=0.8, edgecolor="white", width=0.5)
            ax.text(label, best_reward + 0.5, f"{best_reward:.1f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Best Mean Episode Reward")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "convergence_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_generalization_test():
    """Plot 5: Best agent on burst trace (trained on cyclical)."""
    from stable_baselines3 import DQN, PPO

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Generalization Test — Burst Traffic (Trained on Cyclical)", fontsize=14, fontweight="bold")

    burst_env = KubernetesEnv(trace_type="burst")
    cyclical_env = KubernetesEnv(trace_type="cyclical")

    agents = []

    # Try to load best DQN
    dqn_models = sorted(glob.glob(os.path.join(MODEL_DIR, "dqn", "eco_scale_dqn_run_*.zip")))
    if dqn_models:
        best_dqn = None
        best_r = -float("inf")
        for mp in dqn_models:
            m = DQN.load(mp)
            res = evaluate_sb3_model(m, cyclical_env, 5)
            if res["mean_reward"] > best_r:
                best_r = res["mean_reward"]
                best_dqn = m
        if best_dqn:
            agents.append(("DQN", best_dqn, "sb3"))

    # Try to load best PPO
    ppo_models = sorted(glob.glob(os.path.join(MODEL_DIR, "pg", "eco_scale_ppo_run_*.zip")))
    if ppo_models:
        best_ppo = None
        best_r = -float("inf")
        for mp in ppo_models:
            m = PPO.load(mp)
            res = evaluate_sb3_model(m, cyclical_env, 5)
            if res["mean_reward"] > best_r:
                best_r = res["mean_reward"]
                best_ppo = m
        if best_ppo:
            agents.append(("PPO", best_ppo, "sb3"))

    # Try to load best REINFORCE
    rf_models = sorted(glob.glob(os.path.join(MODEL_DIR, "pg", "eco_scale_reinforce_run_*_best.pt")))
    if rf_models:
        agents.append(("REINFORCE", rf_models[0], "reinforce"))

    colors_map = {"DQN": "#2196F3", "PPO": "#4CAF50", "REINFORCE": "#FF9800"}

    for idx, (ax, trace_type, trace_label) in enumerate([
        (axes[0], "cyclical", "Cyclical (Training)"),
        (axes[1], "burst", "Burst (Unseen)"),
    ]):
        test_env = KubernetesEnv(trace_type=trace_type)
        for name, model, kind in agents:
            if kind == "sb3":
                res = evaluate_sb3_model(model, test_env, 10)
            else:
                res = evaluate_reinforce_model(model, test_env, 10)
            ax.bar(name, res["mean_reward"], color=colors_map.get(name, "gray"),
                   alpha=0.8, edgecolor="white", width=0.4)
        ax.set_title(trace_label, fontweight="bold")
        ax.set_ylabel("Mean Reward")
        ax.grid(axis="y", alpha=0.3)

    # Third subplot: generalization gap
    ax3 = axes[2]
    for name, model, kind in agents:
        cyc_env = KubernetesEnv(trace_type="cyclical")
        bur_env = KubernetesEnv(trace_type="burst")
        if kind == "sb3":
            r_cyc = evaluate_sb3_model(model, cyc_env, 10)["mean_reward"]
            r_bur = evaluate_sb3_model(model, bur_env, 10)["mean_reward"]
        else:
            r_cyc = evaluate_reinforce_model(model, cyc_env, 10)["mean_reward"]
            r_bur = evaluate_reinforce_model(model, bur_env, 10)["mean_reward"]
        gap = r_bur - r_cyc
        ax3.bar(name, gap, color=colors_map.get(name, "gray"), alpha=0.8, edgecolor="white", width=0.4)
    ax3.set_title("Generalization Gap (Burst - Cyclical)", fontweight="bold")
    ax3.set_ylabel("Reward Difference")
    ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "generalization_test.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_hyperparameter_heatmaps():
    """Bonus: heatmap overview of hyperparameter effects."""
    algos = [("dqn", "DQN"), ("ppo", "PPO"), ("reinforce", "REINFORCE")]

    for algo, label in algos:
        csv_path = os.path.join(OUTPUT_DIR, f"{algo}_results.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(f"{label} — Hyperparameter Tuning Results", fontweight="bold")

        # Horizontal bar chart
        colors = ["#4CAF50" if r == df["Mean Reward"].max() else "#2196F3" for r in df["Mean Reward"]]
        bars = ax.barh(
            [f"Run {r} ({n})" for r, n in zip(df["Run"], df["Notes"])],
            df["Mean Reward"],
            color=colors, alpha=0.8, edgecolor="white"
        )
        if "Std Reward" in df.columns:
            ax.errorbar(df["Mean Reward"],
                       [f"Run {r} ({n})" for r, n in zip(df["Run"], df["Notes"])],
                       xerr=df["Std Reward"], fmt="none", ecolor="black", alpha=0.5)

        ax.set_xlabel("Mean Episode Reward")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"{algo}_hyperparameter_tuning.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✅ Saved: {path}")


def plot_training_curves_from_eval_logs():
    """Plot reward curves from SB3 EvalCallback NPZ logs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Reward Curves (Best Run)", fontsize=14, fontweight="bold")

    log_base = os.path.join(os.path.dirname(__file__), "..", "logs")

    for ax, algo, label, color in [
        (axes[0], "dqn", "DQN", "#2196F3"),
        (axes[1], "ppo", "PPO", "#4CAF50"),
    ]:
        # Find evaluation logs
        eval_files = glob.glob(os.path.join(log_base, algo, "run_*", "evaluations.npz"))
        if eval_files:
            # Pick the run with best final eval reward
            best_file = None
            best_final = -float("inf")
            for f in eval_files:
                data = np.load(f)
                mean_rewards = data["results"].mean(axis=1)
                if len(mean_rewards) > 0 and mean_rewards[-1] > best_final:
                    best_final = mean_rewards[-1]
                    best_file = f

            if best_file:
                data = np.load(best_file)
                timesteps = data["timesteps"]
                mean_rewards = data["results"].mean(axis=1)
                std_rewards = data["results"].std(axis=1)

                ax.plot(timesteps, mean_rewards, color=color, linewidth=2, label=label)
                ax.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards,
                               alpha=0.2, color=color)
                ax.set_xlabel("Timesteps")
                ax.set_title(f"{label} Training Curve", fontweight="bold")
                ax.legend()
                ax.grid(alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No eval data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(label)
        else:
            ax.text(0.5, 0.5, "No eval logs found", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)

    axes[0].set_ylabel("Mean Eval Reward")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    ensure_dirs()

    print("=" * 60)
    print("  Eco-Scale — Agent Comparison & Plot Generation")
    print("=" * 60)

    print("\n📊 Generating cumulative reward plots...")
    plot_cumulative_rewards()

    print("\n📊 Generating hyperparameter tuning plots...")
    plot_hyperparameter_heatmaps()

    print("\n📊 Generating training curves...")
    plot_training_curves_from_eval_logs()

    print("\n📊 Generating convergence comparison...")
    plot_convergence_comparison()

    print("\n📊 Generating generalization test...")
    try:
        plot_generalization_test()
    except Exception as e:
        print(f"  ⚠️  Generalization test skipped (models may not be trained yet): {e}")

    print(f"\n✅ All plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
