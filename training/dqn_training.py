"""
dqn_training.py — Train DQN agent on KubernetesEnv with 10 hyperparameter runs.

Usage:
    python3.14 training/dqn_training.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from environment.custom_env import KubernetesEnv

# ──────────────────────────────────────────────
# 10 Hyperparameter configurations
# ──────────────────────────────────────────────
CONFIGS = [
    # Run 1 — Baseline
    dict(learning_rate=1e-4, gamma=0.99, buffer_size=10000, batch_size=64,
         exploration_fraction=0.3, exploration_final_eps=0.05, target_update_interval=100,
         notes="Baseline"),
    # Run 2 — Higher LR
    dict(learning_rate=1e-3, gamma=0.99, buffer_size=10000, batch_size=64,
         exploration_fraction=0.3, exploration_final_eps=0.05, target_update_interval=100,
         notes="Higher LR"),
    # Run 3 — Lower gamma
    dict(learning_rate=1e-4, gamma=0.95, buffer_size=10000, batch_size=64,
         exploration_fraction=0.3, exploration_final_eps=0.05, target_update_interval=100,
         notes="Lower gamma"),
    # Run 4 — Larger buffer
    dict(learning_rate=1e-4, gamma=0.99, buffer_size=50000, batch_size=64,
         exploration_fraction=0.3, exploration_final_eps=0.05, target_update_interval=100,
         notes="Larger buffer"),
    # Run 5 — Larger batch
    dict(learning_rate=1e-4, gamma=0.99, buffer_size=10000, batch_size=128,
         exploration_fraction=0.3, exploration_final_eps=0.05, target_update_interval=100,
         notes="Larger batch"),
    # Run 6 — More exploration
    dict(learning_rate=1e-4, gamma=0.99, buffer_size=10000, batch_size=64,
         exploration_fraction=0.5, exploration_final_eps=0.05, target_update_interval=100,
         notes="More exploration"),
    # Run 7 — Less exploration
    dict(learning_rate=1e-4, gamma=0.99, buffer_size=10000, batch_size=64,
         exploration_fraction=0.1, exploration_final_eps=0.05, target_update_interval=100,
         notes="Less exploration"),
    # Run 8 — Lower final eps
    dict(learning_rate=1e-4, gamma=0.99, buffer_size=10000, batch_size=64,
         exploration_fraction=0.3, exploration_final_eps=0.01, target_update_interval=100,
         notes="Lower final eps"),
    # Run 9 — Slower target update
    dict(learning_rate=1e-4, gamma=0.99, buffer_size=10000, batch_size=64,
         exploration_fraction=0.3, exploration_final_eps=0.05, target_update_interval=500,
         notes="Slower target update"),
    # Run 10 — Combined changes
    dict(learning_rate=5e-5, gamma=0.999, buffer_size=20000, batch_size=32,
         exploration_fraction=0.4, exploration_final_eps=0.02, target_update_interval=200,
         notes="Combined changes"),
]

TOTAL_TIMESTEPS = 100_000
EVAL_EPISODES = 10
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "dqn")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs", "dqn")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "dqn_results.csv")


def evaluate_model(model, env, n_episodes=10):
    """Run n episodes and return mean reward."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total += reward
            done = terminated or truncated
        rewards.append(total)
    return float(np.mean(rewards)), float(np.std(rewards))


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    results = []

    for i, cfg in enumerate(CONFIGS, start=1):
        notes = cfg.pop("notes")
        print(f"\n{'='*60}")
        print(f"  DQN Run {i}/10 — {notes}")
        print(f"  Config: {cfg}")
        print(f"{'='*60}\n")

        env = KubernetesEnv(trace_type="cyclical")
        eval_env = KubernetesEnv(trace_type="cyclical")

        run_log = os.path.join(LOG_DIR, f"run_{i}")
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(MODEL_DIR, f"run_{i}_best"),
            log_path=run_log,
            eval_freq=5000,
            n_eval_episodes=5,
            deterministic=True,
            verbose=0,
        )

        model = DQN(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=run_log,
            **cfg,
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_cb)

        # Save model
        save_path = os.path.join(MODEL_DIR, f"eco_scale_dqn_run_{i}")
        model.save(save_path)

        # Evaluate
        mean_r, std_r = evaluate_model(model, eval_env, EVAL_EPISODES)
        print(f"  → Mean Reward: {mean_r:.2f} ± {std_r:.2f}")

        results.append({
            "Run": i,
            "learning_rate": cfg["learning_rate"],
            "gamma": cfg["gamma"],
            "buffer_size": cfg["buffer_size"],
            "batch_size": cfg["batch_size"],
            "exploration_fraction": cfg["exploration_fraction"],
            "exploration_final_eps": cfg["exploration_final_eps"],
            "target_update_interval": cfg["target_update_interval"],
            "Mean Reward": round(mean_r, 2),
            "Std Reward": round(std_r, 2),
            "Notes": notes,
        })

        # Put notes back for potential reuse
        cfg["notes"] = notes

    # Save results table
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\n✅ DQN results saved to {RESULTS_PATH}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
