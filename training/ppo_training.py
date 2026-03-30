"""
ppo_training.py — Train PPO agent on KubernetesEnv with 10 hyperparameter runs.

Usage:
    python3.14 training/ppo_training.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from environment.custom_env import KubernetesEnv

# ──────────────────────────────────────────────
# 10 Hyperparameter configurations
# ──────────────────────────────────────────────
CONFIGS = [
    # Run 1 — Baseline
    dict(learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=10, ent_coef=0.01, clip_range=0.2, notes="Baseline"),
    # Run 2 — Higher LR
    dict(learning_rate=1e-3, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=10, ent_coef=0.01, clip_range=0.2, notes="Higher LR"),
    # Run 3 — Lower LR
    dict(learning_rate=1e-4, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=10, ent_coef=0.01, clip_range=0.2, notes="Lower LR"),
    # Run 4 — Lower gamma
    dict(learning_rate=3e-4, gamma=0.95, n_steps=2048, batch_size=64,
         n_epochs=10, ent_coef=0.01, clip_range=0.2, notes="Lower gamma"),
    # Run 5 — Shorter rollouts
    dict(learning_rate=3e-4, gamma=0.99, n_steps=512, batch_size=64,
         n_epochs=10, ent_coef=0.01, clip_range=0.2, notes="Shorter rollouts"),
    # Run 6 — Larger batch
    dict(learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=128,
         n_epochs=10, ent_coef=0.01, clip_range=0.2, notes="Larger batch"),
    # Run 7 — More entropy
    dict(learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=10, ent_coef=0.05, clip_range=0.2, notes="More entropy"),
    # Run 8 — Wider clip range
    dict(learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=10, ent_coef=0.01, clip_range=0.3, notes="Wider clip range"),
    # Run 9 — Fewer epochs
    dict(learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=4, ent_coef=0.01, clip_range=0.2, notes="Fewer epochs"),
    # Run 10 — Combined changes
    dict(learning_rate=5e-4, gamma=0.98, n_steps=1024, batch_size=128,
         n_epochs=5, ent_coef=0.02, clip_range=0.25, notes="Combined changes"),
]

TOTAL_TIMESTEPS = 100_000
EVAL_EPISODES = 10
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "pg")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs", "ppo")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "ppo_results.csv")


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
        print(f"  PPO Run {i}/10 — {notes}")
        print(f"  Config: {cfg}")
        print(f"{'='*60}\n")

        env = make_vec_env(lambda: KubernetesEnv(trace_type="cyclical"), n_envs=4)
        eval_env = KubernetesEnv(trace_type="cyclical")

        run_log = os.path.join(LOG_DIR, f"run_{i}")
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(MODEL_DIR, f"ppo_run_{i}_best"),
            log_path=run_log,
            eval_freq=5000,
            n_eval_episodes=5,
            deterministic=True,
            verbose=0,
        )

        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=run_log,
            **cfg,
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_cb)

        # Save model
        save_path = os.path.join(MODEL_DIR, f"eco_scale_ppo_run_{i}")
        model.save(save_path)

        # Evaluate
        mean_r, std_r = evaluate_model(model, eval_env, EVAL_EPISODES)
        print(f"  → Mean Reward: {mean_r:.2f} ± {std_r:.2f}")

        results.append({
            "Run": i,
            "learning_rate": cfg["learning_rate"],
            "gamma": cfg["gamma"],
            "n_steps": cfg["n_steps"],
            "batch_size": cfg["batch_size"],
            "n_epochs": cfg["n_epochs"],
            "ent_coef": cfg["ent_coef"],
            "clip_range": cfg["clip_range"],
            "Mean Reward": round(mean_r, 2),
            "Std Reward": round(std_r, 2),
            "Notes": notes,
        })

        cfg["notes"] = notes

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\n✅ PPO results saved to {RESULTS_PATH}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
