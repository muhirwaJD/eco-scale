"""
inference_engine.py — Load the deployed champion and turn live metrics into a
scaling decision.

It loads models/eco_scale_best.zip (produced by training/select_champion.py) and
reads which algorithm it is from champion_metadata.json — nothing is hardcoded.
The raw metrics are normalized exactly as the training environment does, so the
agent sees inputs in the same form it was trained on.
"""

import os
import json
from environment.custom_env import KubernetesEnv

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(ROOT, "models", "eco_scale_best.zip")
METADATA_PATH = os.path.join(ROOT, "models", "champion_metadata.json")

ACTION_NAMES = {0: "scale_down", 1: "maintain", 2: "scale_up"}


class InferenceEngine:
    """Wraps the champion model and exposes a single decide() call."""

    def __init__(self, model_path=MODEL_PATH, metadata_path=METADATA_PATH):
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        self.algorithm = self.metadata["algorithm"]

        if self.algorithm == "DQN":
            from stable_baselines3 import DQN
            self.model = DQN.load(model_path)
        elif self.algorithm == "PPO":
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path)
        else:
            raise ValueError(f"Serving does not support algorithm: {self.algorithm}")

    def _build_observation(self, cpu_util, pods, queue_depth, day_progress):
        """Normalize raw metrics the same way KubernetesEnv does."""
        return [
            float(cpu_util),                                        # already 0..1
            pods / KubernetesEnv.MAX_PODS,                          # pods -> 0..1
            min(queue_depth / KubernetesEnv.QUEUE_SCALE, 1.0),      # queue -> 0..1
            float(day_progress),                                    # already 0..1
        ]

    def decide(self, cpu_util, pods, queue_depth, day_progress):
        """Return the scaling decision for the current cluster state."""
        obs = self._build_observation(cpu_util, pods, queue_depth, day_progress)
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)
        return {"action": action, "action_name": ACTION_NAMES[action]}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ROOT)
    engine = InferenceEngine()
    print(f"Loaded champion: {engine.algorithm} (run {engine.metadata['run']})")
    # high load, few pods -> should recommend scaling up
    print("high load / few pods :", engine.decide(cpu_util=0.9, pods=3,
                                                   queue_depth=900, day_progress=0.5))
    # low load, many pods -> should recommend scaling down
    print("low load / many pods :", engine.decide(cpu_util=0.2, pods=15,
                                                   queue_depth=200, day_progress=0.5))
