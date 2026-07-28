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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from environment.custom_env import KubernetesEnv


MODEL_PATH = ROOT / "models" / "eco_scale_best.zip"
METADATA_PATH = ROOT / "models" / "champion_metadata.json"

ACTION_NAMES = {0: "scale_down", 1: "maintain", 2: "scale_up"}


class InferenceEngine:
    """Wraps the champion model and exposes a single decide() call."""

    def __init__(self, model_path=MODEL_PATH, metadata_path=METADATA_PATH):
        # 1. Load and display metadata cleanly
        self.metadata = self._load_metadata(metadata_path)

        # 2. Extract algorithm type
        self.algorithm = self.metadata.get("algorithm")

        # 3. Load the corresponding reinforcement learning model
        self.model = self._load_model(model_path)

    def _load_metadata(self, path):
        """Loads JSON metadata and prints it with nice formatting."""
        try:
            with open(path, "r") as f:
                data = json.load(f)

            # Pretty-print JSON with indentation and clean headers
            print("\n" + "=" * 40)
            print("🚀 LOADING CHAMPION METADATA")
            print("=" * 40)
            print(json.dumps(data, indent=1))
            print("=" * 40 + "\n")

            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata file missing at: {path}")

    def _load_model(self, path):
        """Dynamically imports and loads the correct Stable-Baselines3 model."""
        print(f"📦 Initializing {self.algorithm} model loading sequence...")

        if self.algorithm == "DQN":
            from stable_baselines3 import DQN # pyright: ignore [reportMissingImports]

            return DQN.load(path)
        elif self.algorithm == "PPO":
            from stable_baselines3 import PPO # pyright: ignore [reportMissingImports]

            return PPO.load(path)
        else:
            raise ValueError(
                f"Serving does not support algorithm: {self.algorithm}"
            )
    
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
    engine = InferenceEngine()
    print(f"Loaded champion: {engine.algorithm} (run {engine.metadata['run']})")
    # high load, few pods -> should recommend scaling up
    print("high load / few pods :", engine.decide(cpu_util=0.9, pods=3,
                                                   queue_depth=900, day_progress=0.5))
    # low load, many pods -> should recommend scaling down
    print("low load / many pods :", engine.decide(cpu_util=0.2, pods=15,
                                                   queue_depth=200, day_progress=0.5))
