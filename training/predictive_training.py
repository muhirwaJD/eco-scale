"""
predictive_training.py — Train PPO on the PREDICTIVE env variants.

Extends the champion with one look-ahead feature so it can anticipate load
instead of only reacting. Trains three variants (see environment/custom_env.py):

    oracle   — perfect foresight of the next-H-step peak (upper bound; sim-only)
    trend    — causal recent slope of load (deployable)
    forecast — causal Holt (level+trend) projection H steps ahead (deployable)

Uses the CHAMPION hyperparameters (PPO run 6) so any gain is attributable to the
predictive feature, not to retuning. Each variant is trained with a few seeds and
the best (by train-trace eval reward) is kept.

SAFETY: writes only NEW files —
    models/eco_scale_<variant>.zip + models/<variant>_metadata.json
    models/predictive/<variant>_seed<k>.zip   (per-seed record)
    logs/predictive/<variant>_seed<k>/          (learning-curve logs)
It never touches models/eco_scale_best.zip or models/champion_metadata.json.

Run:
    python training/predictive_training.py                # all variants
    python training/predictive_training.py trend forecast # a subset
Env overrides (handy for a quick smoke run):
    PREDICTIVE_TIMESTEPS=20000 PREDICTIVE_SEEDS=1 python training/predictive_training.py oracle
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from environment.custom_env import KubernetesEnv

ROOT = os.path.join(os.path.dirname(__file__), "..")

# Champion config (PPO run 6, "Larger batch") — held fixed across variants.
CHAMPION_CFG = dict(learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=128,
                    n_epochs=10, ent_coef=0.01, clip_range=0.2)

VARIANTS = ["oracle", "trend", "forecast"]
TIMESTEPS = int(os.environ.get("PREDICTIVE_TIMESTEPS", 150_000))   # match champion budget
SEEDS = int(os.environ.get("PREDICTIVE_SEEDS", 3))
EVAL_EPISODES = 20

with open(os.path.join(ROOT, "data", "split.json")) as _f:
    TRAIN_PATHS = [os.path.join(ROOT, p) for p in json.load(_f)["train"]]

MODEL_DIR = os.path.join(ROOT, "models")
SEED_DIR = os.path.join(ROOT, "models", "predictive")
LOG_DIR = os.path.join(ROOT, "logs", "predictive")


def evaluate(model, env, n_episodes=EVAL_EPISODES):
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(action))
            total += r
            done = term or trunc
        rewards.append(total)
    return float(np.mean(rewards)), float(np.std(rewards))


def train_variant(variant):
    print(f"\n{'='*64}\n  PREDICTIVE VARIANT: {variant}  ({SEEDS} seeds x {TIMESTEPS:,} steps)\n{'='*64}")
    best = None  # (mean, std, seed, model)
    for seed in range(SEEDS):
        env = make_vec_env(lambda: KubernetesEnv(trace_paths=TRAIN_PATHS, predictive=variant), n_envs=4)
        eval_env = KubernetesEnv(trace_paths=TRAIN_PATHS, predictive=variant)
        run_log = os.path.join(LOG_DIR, f"{variant}_seed{seed}")
        eval_cb = EvalCallback(eval_env, log_path=run_log, eval_freq=5000,
                               n_eval_episodes=5, deterministic=True, verbose=0)
        model = PPO("MlpPolicy", env, verbose=0, seed=seed,
                    tensorboard_log=run_log, **CHAMPION_CFG)
        model.learn(total_timesteps=TIMESTEPS, callback=eval_cb)

        os.makedirs(SEED_DIR, exist_ok=True)
        model.save(os.path.join(SEED_DIR, f"{variant}_seed{seed}"))
        mean_r, std_r = evaluate(model, eval_env)
        print(f"  seed {seed}: eval reward {mean_r:.2f} ± {std_r:.2f}")
        if best is None or mean_r > best[0]:
            best = (mean_r, std_r, seed, model)

    mean_r, std_r, seed, model = best
    # promote best seed to the variant's deployable slot (NEW file, not the champion)
    model.save(os.path.join(MODEL_DIR, f"eco_scale_{variant}"))
    meta = {
        "algorithm": "PPO", "predictive": variant, "best_seed": seed,
        "mean_reward_train": round(mean_r, 2), "std_reward_train": round(std_r, 2),
        "horizon": KubernetesEnv.HORIZON, "trend_window": KubernetesEnv.TREND_WINDOW,
        "timesteps": TIMESTEPS, "hyperparameters": CHAMPION_CFG,
        "trained_on": "8-trace train split (real Alibaba 2018, multi-trace)",
        "note": f"Predictive variant '{variant}' — champion PPO config + one look-ahead feature. "
                f"{'Oracle uses known-future load (sim/upper-bound only).' if variant=='oracle' else 'Causal feature — deployable live.'}",
    }
    with open(os.path.join(MODEL_DIR, f"{variant}_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ best seed {seed} → models/eco_scale_{variant}.zip  (train eval {mean_r:.2f})")
    return {"variant": variant, "best_seed": seed,
            "mean_reward_train": round(mean_r, 2), "std_reward_train": round(std_r, 2)}


def main():
    requested = [v for v in sys.argv[1:] if v in VARIANTS] or VARIANTS
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    rows = [train_variant(v) for v in requested]
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "outputs", "training", "predictive_results.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n✅ Saved {out}\n{df.to_string(index=False)}")
    print("\nNOTE: train-trace eval only. Held-out comparison → evaluation/evaluate_predictive.py")


if __name__ == "__main__":
    main()
