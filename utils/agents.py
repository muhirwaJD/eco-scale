"""
agents.py — find and load the best trained agent (the "champion").

The champion is simply whichever training run scored the highest mean reward in
the hyperparameter sweeps. We read that straight from the result tables
(outputs/<algo>_results.csv), so nothing is hardcoded: whatever you trained
last is what gets picked up.

Typical use:
    from utils.agents import find_champion, load_agent

    champ = find_champion()              # e.g. PPO, run 6
    model = load_agent(champ)            # ready-to-use, has .predict(obs)
    print(champ.algorithm, champ.run, champ.reward)
"""

import os
import pandas as pd

# Repo root = parent of the utils/ folder this file lives in.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# For each algorithm: where its result table is, where its models are saved,
# and how the per-run model file is named. Change these in one place only.
ALGORITHMS = {
    "DQN": {
        "results": "outputs/training/dqn_results.csv",
        "model_folder": "models/dqn",
        "model_name": "eco_scale_dqn_run_{run}.zip",
    },
    "PPO": {
        "results": "outputs/training/ppo_results.csv",
        "model_folder": "models/pg",
        "model_name": "eco_scale_ppo_run_{run}.zip",
    },
    "REINFORCE": {
        "results": "outputs/training/reinforce_results.csv",
        "model_folder": "models/pg",
        "model_name": "eco_scale_reinforce_run_{run}.pt",
    },
}


class Champion:
    """A small record describing the best run of one algorithm."""

    def __init__(self, algorithm, run, reward, model_path):
        self.algorithm = algorithm      # "DQN" / "PPO" / "REINFORCE"
        self.run = run                  # which run number won (1-10)
        self.reward = reward            # its mean reward in the sweep
        self.model_path = model_path    # path to the saved model file

    def __repr__(self):
        return (f"Champion({self.algorithm}, run {self.run}, "
                f"reward {self.reward:.2f})")


def _best_run_of(algorithm):
    """Return the best Champion for one algorithm, or None if it wasn't trained."""
    info = ALGORITHMS[algorithm]
    results_file = os.path.join(ROOT, info["results"])
    if not os.path.exists(results_file):
        return None

    table = pd.read_csv(results_file)
    best_row = table.loc[table["Mean Reward"].idxmax()]
    run = int(best_row["Run"])
    model_path = os.path.join(ROOT, info["model_folder"],
                              info["model_name"].format(run=run))
    return Champion(algorithm, run, float(best_row["Mean Reward"]), model_path)


def find_champion(algorithms=("DQN", "PPO", "REINFORCE")):
    """Return the best agent across the given algorithms (highest mean reward).

    With no arguments it considers all three. Pass e.g. ["DQN"] to get the best
    DQN run only.
    """
    candidates = [_best_run_of(a) for a in algorithms]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        raise FileNotFoundError(
            "No result tables found in outputs/. Train the agents first "
            "(e.g. python training/ppo_training.py).")
    return max(candidates, key=lambda c: c.reward)


def load_agent(champion):
    """Load a trained agent from a Champion. Returns an object with .predict(obs).

    DQN and PPO use stable-baselines3; REINFORCE uses the project's own class.
    """
    if not os.path.exists(champion.model_path):
        raise FileNotFoundError(
            f"Model file missing: {champion.model_path}\n"
            f"Re-run training for {champion.algorithm} to regenerate it.")

    if champion.algorithm == "DQN":
        from stable_baselines3 import DQN
        return DQN.load(champion.model_path)
    if champion.algorithm == "PPO":
        from stable_baselines3 import PPO
        return PPO.load(champion.model_path)
    if champion.algorithm == "REINFORCE":
        import sys
        sys.path.insert(0, ROOT)
        from training.reinforce_training import REINFORCE
        agent = REINFORCE(obs_dim=4, act_dim=3)
        agent.load(champion.model_path)
        return agent
    raise ValueError(f"Unknown algorithm: {champion.algorithm}")


if __name__ == "__main__":
    champ = find_champion()
    print(f"Champion: {champ.algorithm} run {champ.run} "
          f"(mean reward {champ.reward:.2f})")
    print(f"Model: {os.path.relpath(champ.model_path, ROOT)}")
