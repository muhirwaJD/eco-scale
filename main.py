"""
main.py — Run the best performing trained agent with pygame visualization.

Usage:
    python3.14 main.py

The script automatically finds the best model across all algorithms
and runs it in the KubernetesEnv with real-time pygame GUI at 4 FPS.
"""

import os
import sys
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from environment.custom_env import KubernetesEnv
from environment.rendering import init_pygame, render_frame


def find_best_model():
    """Find the best performing model across all algorithms."""
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    best_model = None
    best_reward = -float("inf")
    best_name = ""
    best_type = ""

    # Check DQN models
    dqn_models = glob.glob(os.path.join(model_dir, "dqn", "eco_scale_dqn_run_*.zip"))
    if dqn_models:
        from stable_baselines3 import DQN
        eval_env = KubernetesEnv(trace_type="cyclical")
        for mp in dqn_models:
            try:
                m = DQN.load(mp)
                rewards = []
                for _ in range(5):
                    obs, _ = eval_env.reset()
                    total = 0.0
                    done = False
                    while not done:
                        action, _ = m.predict(obs, deterministic=True)
                        obs, r, term, trunc, _ = eval_env.step(int(action))
                        total += r
                        done = term or trunc
                    rewards.append(total)
                mean_r = np.mean(rewards)
                if mean_r > best_reward:
                    best_reward = mean_r
                    best_model = m
                    best_name = os.path.basename(mp)
                    best_type = "DQN"
            except Exception as e:
                print(f"  ⚠️  Failed to load {mp}: {e}")

    # Check PPO models
    ppo_models = glob.glob(os.path.join(model_dir, "pg", "eco_scale_ppo_run_*.zip"))
    if ppo_models:
        from stable_baselines3 import PPO
        eval_env = KubernetesEnv(trace_type="cyclical")
        for mp in ppo_models:
            try:
                m = PPO.load(mp)
                rewards = []
                for _ in range(5):
                    obs, _ = eval_env.reset()
                    total = 0.0
                    done = False
                    while not done:
                        action, _ = m.predict(obs, deterministic=True)
                        obs, r, term, trunc, _ = eval_env.step(int(action))
                        total += r
                        done = term or trunc
                    rewards.append(total)
                mean_r = np.mean(rewards)
                if mean_r > best_reward:
                    best_reward = mean_r
                    best_model = m
                    best_name = os.path.basename(mp)
                    best_type = "PPO"
            except Exception as e:
                print(f"  ⚠️  Failed to load {mp}: {e}")

    # Check REINFORCE models
    rf_models = glob.glob(os.path.join(model_dir, "pg", "eco_scale_reinforce_run_*_best.pt"))
    if rf_models:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "training"))
        from reinforce_training import REINFORCE as RFAgent
        eval_env = KubernetesEnv(trace_type="cyclical")
        for mp in rf_models:
            try:
                agent = RFAgent(obs_dim=4, act_dim=3)
                agent.load(mp)
                rewards = []
                for _ in range(5):
                    obs, _ = eval_env.reset()
                    total = 0.0
                    done = False
                    while not done:
                        action, _ = agent.predict(obs, deterministic=True)
                        obs, r, term, trunc, _ = eval_env.step(action)
                        total += r
                        done = term or trunc
                    rewards.append(total)
                mean_r = np.mean(rewards)
                if mean_r > best_reward:
                    best_reward = mean_r
                    best_model = agent
                    best_name = os.path.basename(mp)
                    best_type = "REINFORCE"
            except Exception as e:
                print(f"  ⚠️  Failed to load {mp}: {e}")

    return best_model, best_name, best_type, best_reward


def run_demo(model, model_type, trace_type="burst"):
    """Run the model with pygame visualization."""
    import pygame

    env = KubernetesEnv(trace_type=trace_type, render_mode="human")
    screen = init_pygame()
    clock = pygame.time.Clock()

    obs, _ = env.reset()
    total_reward = 0.0
    episode = 1

    print(f"\n🎮 Running {model_type} agent on {trace_type} traffic...")
    print("   Close the pygame window to exit.\n")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get action from model
        if model_type == "REINFORCE":
            action, _ = model.predict(obs, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render
        render_frame(screen, info, action, reward, total_reward)
        clock.tick(4)

        # Print step info
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

    model, name, model_type, reward = find_best_model()

    if model is None:
        print("\n⚠️  No trained models found!")
        print("   Please train at least one model first:")
        print("     python3.14 training/dqn_training.py")
        print("     python3.14 training/ppo_training.py")
        print("     python3.14 training/reinforce_training.py")
        return

    print(f"\n🏆 Best model: {name} ({model_type})")
    print(f"   Mean reward: {reward:.2f}")

    run_demo(model, model_type, trace_type="burst")


if __name__ == "__main__":
    main()
