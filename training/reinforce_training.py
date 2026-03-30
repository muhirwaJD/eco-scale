"""
reinforce_training.py — Train REINFORCE (Monte Carlo Policy Gradient) agent
on KubernetesEnv with 10 hyperparameter runs.

REINFORCE is not natively in SB3, so we implement it manually using PyTorch.
Key characteristics:
  - Full episode rollouts (Monte Carlo returns)
  - No value baseline (vanilla) or with baseline
  - Higher variance than PPO/A2C, slower convergence

Usage:
    python3.14 training/reinforce_training.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.custom_env import KubernetesEnv

# ──────────────────────────────────────────────
# Policy Network
# ──────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """Simple MLP policy for discrete action space."""

    def __init__(self, obs_dim=4, act_dim=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):
        logits = self.net(x)
        return Categorical(logits=logits)


class ValueNetwork(nn.Module):
    """Simple baseline network for variance reduction."""

    def __init__(self, obs_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ──────────────────────────────────────────────
# REINFORCE Algorithm
# ──────────────────────────────────────────────

class REINFORCE:
    """Vanilla REINFORCE with optional baseline."""

    def __init__(self, obs_dim=4, act_dim=3, hidden=64,
                 learning_rate=1e-3, gamma=0.99, entropy_coef=0.01,
                 use_baseline=True):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.use_baseline = use_baseline

        self.policy = PolicyNetwork(obs_dim, act_dim, hidden)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        if use_baseline:
            self.value = ValueNetwork(obs_dim, hidden)
            self.value_optimizer = optim.Adam(self.value.parameters(), lr=learning_rate)

    def select_action(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        dist = self.policy(obs_t)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, entropy

    def predict(self, obs, deterministic=False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        dist = self.policy(obs_t)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action.item(), None

    def compute_returns(self, rewards):
        """Compute discounted returns for each timestep."""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self, episode_data):
        """Update policy using REINFORCE with Monte Carlo returns."""
        obs_list, log_probs, entropies, rewards = episode_data

        returns = self.compute_returns(rewards)

        # Baseline
        if self.use_baseline:
            obs_t = torch.FloatTensor(np.array(obs_list))
            values = self.value(obs_t)
            advantages = returns - values.detach()
        else:
            advantages = returns

        # Policy loss
        policy_loss = 0
        entropy_sum = 0
        for log_prob, adv, ent in zip(log_probs, advantages, entropies):
            policy_loss += -log_prob * adv
            entropy_sum += ent

        policy_loss = policy_loss / len(log_probs)
        entropy_bonus = -self.entropy_coef * (entropy_sum / len(entropies))
        total_loss = policy_loss + entropy_bonus

        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.policy_optimizer.step()

        # Baseline update
        if self.use_baseline:
            obs_t = torch.FloatTensor(np.array(obs_list))
            values = self.value(obs_t)
            value_loss = nn.functional.mse_loss(values, returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        return policy_loss.item(), (entropy_sum / len(entropies)).item()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "value_state_dict": self.value.state_dict() if self.use_baseline else None,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        if self.use_baseline and checkpoint["value_state_dict"]:
            self.value.load_state_dict(checkpoint["value_state_dict"])


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────

def train_reinforce(config, run_id, total_episodes=350, eval_episodes=10):
    """Train a single REINFORCE agent and return evaluation results."""
    env = KubernetesEnv(trace_type="cyclical")
    eval_env = KubernetesEnv(trace_type="cyclical")

    agent = REINFORCE(
        obs_dim=4,
        act_dim=3,
        hidden=config.get("hidden", 64),
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        entropy_coef=config["entropy_coef"],
        use_baseline=config.get("use_baseline", True),
    )

    model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "pg")
    os.makedirs(model_dir, exist_ok=True)

    reward_history = []
    best_mean_reward = -float("inf")

    for ep in range(1, total_episodes + 1):
        obs, _ = env.reset()
        obs_list, log_probs, entropies, rewards = [], [], [], []
        done = False

        while not done:
            action, log_prob, entropy = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            obs_list.append(obs)
            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(reward)

            obs = next_obs
            done = terminated or truncated

        episode_reward = sum(rewards)
        reward_history.append(episode_reward)

        # Update policy
        loss, ent = agent.update((obs_list, log_probs, entropies, rewards))

        if ep % 50 == 0:
            recent = np.mean(reward_history[-50:])
            print(f"  Ep {ep:4d} | Reward: {episode_reward:8.2f} | "
                  f"Avg50: {recent:8.2f} | Loss: {loss:.4f} | Entropy: {ent:.4f}")

        # Save best
        if ep >= 50:
            recent_mean = np.mean(reward_history[-50:])
            if recent_mean > best_mean_reward:
                best_mean_reward = recent_mean
                agent.save(os.path.join(model_dir, f"eco_scale_reinforce_run_{run_id}_best.pt"))

    # Final save
    agent.save(os.path.join(model_dir, f"eco_scale_reinforce_run_{run_id}.pt"))

    # Evaluate
    eval_rewards = []
    for _ in range(eval_episodes):
        obs, _ = eval_env.reset()
        total = 0.0
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total += reward
            done = terminated or truncated
        eval_rewards.append(total)

    mean_r = float(np.mean(eval_rewards))
    std_r = float(np.std(eval_rewards))
    return mean_r, std_r, reward_history


# ──────────────────────────────────────────────
# 10 Hyperparameter configurations
# ──────────────────────────────────────────────
CONFIGS = [
    dict(learning_rate=1e-3, gamma=0.99, entropy_coef=0.01, use_baseline=True,
         hidden=64, notes="Baseline"),
    dict(learning_rate=3e-3, gamma=0.99, entropy_coef=0.01, use_baseline=True,
         hidden=64, notes="Higher LR"),
    dict(learning_rate=5e-4, gamma=0.99, entropy_coef=0.01, use_baseline=True,
         hidden=64, notes="Lower LR"),
    dict(learning_rate=1e-3, gamma=0.95, entropy_coef=0.01, use_baseline=True,
         hidden=64, notes="Lower gamma"),
    dict(learning_rate=1e-3, gamma=0.99, entropy_coef=0.05, use_baseline=True,
         hidden=64, notes="More entropy"),
    dict(learning_rate=1e-3, gamma=0.99, entropy_coef=0.001, use_baseline=True,
         hidden=64, notes="Less entropy"),
    dict(learning_rate=1e-3, gamma=0.99, entropy_coef=0.01, use_baseline=False,
         hidden=64, notes="No baseline"),
    dict(learning_rate=1e-3, gamma=0.99, entropy_coef=0.01, use_baseline=True,
         hidden=128, notes="Larger network"),
    dict(learning_rate=1e-3, gamma=0.999, entropy_coef=0.01, use_baseline=True,
         hidden=64, notes="Higher gamma"),
    dict(learning_rate=2e-3, gamma=0.98, entropy_coef=0.02, use_baseline=True,
         hidden=128, notes="Combined changes"),
]


def main():
    results_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "reinforce_results.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    results = []

    for i, cfg in enumerate(CONFIGS, start=1):
        notes = cfg["notes"]
        print(f"\n{'='*60}")
        print(f"  REINFORCE Run {i}/10 — {notes}")
        print(f"  Config: lr={cfg['learning_rate']}, gamma={cfg['gamma']}, "
              f"ent={cfg['entropy_coef']}, baseline={cfg['use_baseline']}, "
              f"hidden={cfg['hidden']}")
        print(f"{'='*60}\n")

        mean_r, std_r, history = train_reinforce(cfg, run_id=i)

        print(f"  → Mean Reward: {mean_r:.2f} ± {std_r:.2f}")

        results.append({
            "Run": i,
            "learning_rate": cfg["learning_rate"],
            "gamma": cfg["gamma"],
            "entropy_coef": cfg["entropy_coef"],
            "use_baseline": cfg["use_baseline"],
            "hidden": cfg["hidden"],
            "Mean Reward": round(mean_r, 2),
            "Std Reward": round(std_r, 2),
            "Notes": notes,
        })

    df = pd.DataFrame(results)
    df.to_csv(results_path, index=False)
    print(f"\n✅ REINFORCE results saved to {results_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
