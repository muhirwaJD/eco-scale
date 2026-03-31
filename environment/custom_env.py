import gymnasium as gym
import numpy as np
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class KubernetesEnv(gym.Env):
    """Custom Gymnasium environment simulating a Kubernetes cluster for pod autoscaling."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, trace_type="cyclical", render_mode=None, max_steps=288):
        super().__init__()
        self.trace_type = trace_type
        self.render_mode = render_mode
        self.max_steps = max_steps

        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(3)

        self.alpha = 0.5   # latency weight
        self.beta = 0.3    # wasted pods weight
        self.gamma_r = 0.05 # scaling cost weight

        self.pod_count = 3
        self.current_step = 0
        self.time_of_day = 0
        self.cpu_util = 0.1
        self.request_queue = 50
        self.latency = 0.0
        self.breach_count = 0
        self.episode_reward = 0.0
        self.latency_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pod_count = 3
        self.current_step = 0
        self.time_of_day = 0
        self.cpu_util = 0.1
        self.request_queue = 50
        self.latency = 0.0
        self.breach_count = 0
        self.episode_reward = 0.0
        self.latency_history = []
        return self._get_obs(), {}

    def step(self, action):
        if action == 0:
            self.pod_count = max(1, self.pod_count - 1)
        elif action == 2:
            self.pod_count = min(20, self.pod_count + 1)

        self.current_step += 1
        self.time_of_day = (self.time_of_day + 1) % 24

        if self.trace_type == "burst":
            self.cpu_util = self._get_burst_traffic(self.current_step)
        else:
            self.cpu_util = self._get_traffic(self.current_step)

        self.request_queue = int(self.cpu_util * 500)
        reward = self._calculate_reward(action)
        self.episode_reward += reward

        terminated = self._check_termination()
        if terminated:
            reward -= 10.0
        truncated = self.current_step >= self.max_steps

        obs = self._get_obs()
        info = {
            "latency": self.latency, "pods": self.pod_count,
            "step": self.current_step, "cpu_util": self.cpu_util,
            "request_queue": self.request_queue, "time_of_day": self.time_of_day,
            "episode_reward": self.episode_reward,
            "wasted_pods": self._get_wasted_pods(),
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return np.array([
            float(self.cpu_util), float(self.pod_count) / 20.0,
            float(self.request_queue) / 1000.0, float(self.time_of_day) / 23.0,
        ], dtype=np.float32)

    def _calculate_reward(self, action):
        self.latency = min(self.request_queue / (self.pod_count * 50.0), 1.0)
        self.latency_history.append(self.latency)
        wasted_pods = self._get_wasted_pods()
        
        # Scaling cost — same for both directions
        scaling_cost = 1.0 if action != 1 else 0.0
        
        prev_latency = self.latency_history[-2] if len(self.latency_history) >= 2 else self.latency
        improvement = max(0.0, prev_latency - self.latency)
        
        # NEW — punish scaling DOWN when latency is already high
        wrong_direction_penalty = 0.0
        if action == 0 and self.latency > 0.5:
            wrong_direction_penalty = -0.5
        
        # NEW — reward scaling UP when latency is high
        right_direction_bonus = 0.0
        if action == 2 and self.latency > 0.5:
            right_direction_bonus = 0.3
            
        return (
            -(self.alpha * self.latency)
            - (self.beta * wasted_pods)
            - (self.gamma_r * scaling_cost)
            + (0.2 * improvement)
            + wrong_direction_penalty
            + right_direction_bonus
        )

    def _get_wasted_pods(self):
        required_pods = max(1, int(np.ceil(self.request_queue / 50.0)))
        return max(0, self.pod_count - required_pods) / 20.0

    def _check_termination(self):
        if self.latency >= 1.0:
            self.breach_count += 1
        else:
            self.breach_count = 0
        return self.breach_count >= 3

    def _get_traffic(self, step):
        hour = step % 24
        if 6 <= hour <= 22:
            base = 0.3 + 0.5 * np.sin((hour - 6) * np.pi / 12)
        else:
            base = 0.1
        noise = self.np_random.normal(0, 0.05)
        return float(np.clip(base + noise, 0.05, 1.0))

    def _get_burst_traffic(self, step):
        base = self._get_traffic(step)
        if self.np_random.random() < 0.1:
            spike = self.np_random.uniform(0.4, 0.7)
            return float(np.clip(base + spike, 0.0, 1.0))
        return base


# Sanity check
env = KubernetesEnv()
obs, _ = env.reset()
print(f"✅ Environment ready. Obs shape: {obs.shape}, Obs: {obs}")
obs, r, t, tr, info = env.step(1)
print(f"   Step OK — Reward: {r:.4f}, Latency: {info['latency']:.3f}, Pods: {info['pods']}")
