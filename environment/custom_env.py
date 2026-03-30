"""
custom_env.py — KubernetesEnv: Custom Gymnasium environment
simulating a Kubernetes cluster for Eco-Scale pod autoscaling.

Observation: [cpu_utilization, pod_count_norm, request_queue_norm, time_of_day_norm]
Actions: 0=scale_down, 1=hold, 2=scale_up
Reward: -(0.5*latency) - (0.3*wasted_pods) - (0.2*scaling_cost)
"""

import gymnasium as gym
import numpy as np


class KubernetesEnv(gym.Env):
    """
    Custom Gymnasium environment simulating a Kubernetes cluster
    for Eco-Scale pod autoscaling research.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, trace_type="cyclical", render_mode=None, max_steps=288):
        super().__init__()
        self.trace_type = trace_type
        self.render_mode = render_mode
        self.max_steps = max_steps  # 288 steps × 5 min = 24 hours

        # --- Observation space: 4 continuous values, all normalized [0, 1] ---
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # --- Action space: 3 discrete actions ---
        # 0 = scale down, 1 = hold, 2 = scale up
        self.action_space = gym.spaces.Discrete(3)

        # --- Reward weights ---
        self.alpha = 0.5  # latency weight (user experience)
        self.beta = 0.3   # wasted pods weight (energy efficiency)
        self.gamma_r = 0.2  # scaling cost weight (operational stability)

        # --- Internal state (set in reset) ---
        self.pod_count = 3
        self.current_step = 0
        self.time_of_day = 0
        self.cpu_util = 0.1
        self.request_queue = 50
        self.latency = 0.0
        self.breach_count = 0

        # --- Tracking for metrics ---
        self.episode_reward = 0.0
        self.latency_history = []

    # ------------------------------------------------------------------
    # Core Gymnasium methods
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
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
        """Execute one timestep in the environment."""
        # 1. Apply action
        if action == 0:
            self.pod_count = max(1, self.pod_count - 1)
        elif action == 2:
            self.pod_count = min(20, self.pod_count + 1)

        # 2. Advance time and get new traffic
        self.current_step += 1
        self.time_of_day = (self.time_of_day + 1) % 24

        if self.trace_type == "burst":
            self.cpu_util = self._get_burst_traffic(self.current_step)
        else:
            self.cpu_util = self._get_traffic(self.current_step)

        self.request_queue = int(self.cpu_util * 500)

        # 3. Calculate reward
        reward = self._calculate_reward(action)
        self.episode_reward += reward

        # 4. Check terminal conditions
        terminated = self._check_termination()
        if terminated:
            reward -= 10.0  # SLA breach penalty
        truncated = self.current_step >= self.max_steps

        # 5. Build observation
        obs = self._get_obs()
        info = {
            "latency": self.latency,
            "pods": self.pod_count,
            "step": self.current_step,
            "cpu_util": self.cpu_util,
            "request_queue": self.request_queue,
            "time_of_day": self.time_of_day,
            "episode_reward": self.episode_reward,
            "wasted_pods": self._get_wasted_pods(),
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self):
        """Return the normalized observation vector."""
        return np.array(
            [
                float(self.cpu_util),
                float(self.pod_count) / 20.0,
                float(self.request_queue) / 1000.0,
                float(self.time_of_day) / 23.0,
            ],
            dtype=np.float32,
        )

    def _calculate_reward(self, action):
        """Compute reward: R = -(α*latency) - (β*wasted) - (γ*scaling_cost)."""
        # Latency proxy
        self.latency = min(self.request_queue / (self.pod_count * 50.0), 1.0)
        self.latency_history.append(self.latency)

        # Wasted pods
        wasted_pods = self._get_wasted_pods()

        # Scaling cost (churn penalty)
        scaling_cost = 1.0 if action != 1 else 0.0

        reward = (
            -(self.alpha * self.latency)
            - (self.beta * wasted_pods)
            - (self.gamma_r * scaling_cost)
        )
        return reward

    def _get_wasted_pods(self):
        """Fraction of idle pods."""
        required_pods = max(1, int(np.ceil(self.request_queue / 50.0)))
        return max(0, self.pod_count - required_pods) / 20.0

    def _check_termination(self):
        """Terminate if latency >= 1.0 for 3 consecutive steps."""
        if self.latency >= 1.0:
            self.breach_count += 1
        else:
            self.breach_count = 0
        return self.breach_count >= 3

    # ------------------------------------------------------------------
    # Traffic simulation
    # ------------------------------------------------------------------

    def _get_traffic(self, step):
        """Cyclical traffic pattern (business hours sinusoidal)."""
        hour = step % 24
        if 6 <= hour <= 22:
            base = 0.3 + 0.5 * np.sin((hour - 6) * np.pi / 12)
        else:
            base = 0.1
        noise = self.np_random.normal(0, 0.05)
        return float(np.clip(base + noise, 0.05, 1.0))

    def _get_burst_traffic(self, step):
        """Burst traffic pattern (flash-sale / spike)."""
        base = self._get_traffic(step)
        if self.np_random.random() < 0.1:
            spike = self.np_random.uniform(0.4, 0.7)
            return float(np.clip(base + spike, 0.0, 1.0))
        return base
