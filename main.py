"""
main.py — Run the best trained DQN agent with pygame visualization.

Usage:
    python3 main.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from environment.custom_env import KubernetesEnv
from environment.rendering import init_pygame, render_frame

BEST_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.zip")
BEST_MODEL_TYPE = "DQN"
BEST_MODEL_NAME = "best_model.zip"
BEST_MEAN_REWARD = -12.21


def load_best_agent():
    """Load the single best trained DQN agent."""
    from stable_baselines3 import DQN
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"\n❌ Model not found at: {BEST_MODEL_PATH}")
        print("   Please ensure models/best_model.zip exists.")
        sys.exit(1)
    model = DQN.load(BEST_MODEL_PATH)
    return model


def run_demo(model, trace_type="cyclical"):
    """Run the model with pygame visualization."""
    import pygame

    env = KubernetesEnv(trace_type=trace_type, render_mode="human")
    screen = init_pygame()
    clock = pygame.time.Clock()

    obs, _ = env.reset()
    total_reward = 0.0
    episode = 1

    print(f"\n🎮 Running {BEST_MODEL_TYPE} agent on {trace_type} traffic...")
    print("   Close the pygame window to exit.\n")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        render_frame(screen, info, action, reward, total_reward)
        clock.tick(4)

        step = info.get("step", 0)
        pods = info.get("pods", 0)
        latency = info.get("latency", 0)
        action_names = {0: "SCALE DOWN", 1: "HOLD", 2: "SCALE UP"}
        print(f"  Step {step:3d} | Pods: {pods:2d} | Latency: {latency:.3f} | "
              f"Action: {action_names.get(action, '?'):10s} | Reward: {reward:+.3f} | "
              f"Total: {total_reward:+.2f}")

        if terminated or truncated:
            print(f"\n  ══ Episode {episode} complete. Total reward: {total_reward:.2f} ══\n")
            obs, _ = env.reset()
            total_reward = 0.0
            episode += 1

    pygame.quit()
    print("\n👋 Demo finished.")


def main():
    print("=" * 60)
    print("  ECO-SCALE — Best Agent Demo")
    print("=" * 60)

    model = load_best_agent()

    print(f"\n✅ Loaded: {BEST_MODEL_NAME} ({BEST_MODEL_TYPE})")
    print(f"   Mean reward: {BEST_MEAN_REWARD} | Run: 6 (More exploration, exploration_fraction=0.5)")

    run_demo(model, trace_type="cyclical")


if __name__ == "__main__":
    main()
