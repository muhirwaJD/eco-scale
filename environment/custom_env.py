import os
import gymnasium as gym
import numpy as np


class KubernetesEnv(gym.Env):
    """
    Pod-autoscaling environment driven by REAL Alibaba cluster traces.

    Traffic is read from a pre-sliced, [0,1]-normalized CPU trace
    (trace_cyclical.npy / trace_burst.npy), one value per timestep.

    Observation (4-D, all scaled to [0,1]):
        [ cpu_util, pods/MAX_PODS, queue/QUEUE_SCALE, day_progress ]
    Action (discrete):
        0 = scale down (-1 pod), 1 = maintain, 2 = scale up (+1 pod)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # --- tunable constants (calibrated so baseline is servable, peaks stress) ---
    MIN_PODS = 1
    MAX_PODS = 20
    START_PODS = 5
    POD_CAPACITY = 0.08      # normalized load one pod serves comfortably
    QUEUE_SCALE = 1000.0     # request_queue normalization for the observation
    BREACH_LIMIT = 5         # consecutive saturated steps before termination

    # reward weights (recalibrated). Energy is charged on the ABSOLUTE pod count
    # so the optimal policy tracks demand at a healthy utilization instead of
    # over-provisioning. Validated offline in training/reward_design.py:
    # demand-tracking beats park-high/park-low/hold under these weights.
    W_LAT = 1.0             # latency (utilization) penalty — service quality
    W_ENERGY = 1.5          # energy cost per running pod (absolute)
    W_SLA = 1.0             # hard penalty for saturation (util >= 1.0)
    W_SCALE = 0.02          # per scaling-action cost
    UTIL_TARGET = 0.70      # healthy utilization the "required" pod count targets

    def __init__(self, trace_type="cyclical", render_mode=None, trace_dir=None,
                 trace_paths=None):
        super().__init__()
        self.trace_type = trace_type
        self.render_mode = render_mode

        # --- load one or many real traces ---
        # Multi-trace mode (trace_paths given): a random trace is sampled each
        # episode so the agent trains across diverse traffic patterns.
        # Single-trace mode (default): backward-compatible trace_type lookup.
        if trace_paths:
            paths = list(trace_paths)
        else:
            trace_dir = trace_dir or os.environ.get("TRACE_DIR", ".")
            fname = "trace_burst.npy" if trace_type == "burst" else "trace_cyclical.npy"
            paths = [os.path.join(trace_dir, fname)]

        self.trace_paths = paths
        self.traces = [np.load(p).astype(np.float32) for p in paths]
        self.trace = self.traces[0]          # active trace; reset() picks one per episode
        self.max_steps = len(self.trace)

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)

        self._init_state()

    def _init_state(self):
        self.pod_count = self.START_PODS
        self.current_step = 0
        self.cpu_util = float(self.trace[0])
        self.request_queue = int(self.cpu_util * self.QUEUE_SCALE)
        self.latency = 0.0
        self.prev_latency = 0.0
        self.breach_count = 0
        self.episode_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # pick the active trace for this episode (random across the pool).
        # single-trace envs (len == 1) stay fully deterministic.
        if len(self.traces) > 1:
            i = int(self.np_random.integers(len(self.traces)))
            self.trace = self.traces[i]
            self.max_steps = len(self.trace)
        self._init_state()
        return self._get_obs(), {}

    def step(self, action):
        # 1. apply action
        if action == 0:
            self.pod_count = max(self.MIN_PODS, self.pod_count - 1)
        elif action == 2:
            self.pod_count = min(self.MAX_PODS, self.pod_count + 1)

        # 2. advance time, read real demand
        self.current_step += 1
        idx = min(self.current_step, self.max_steps - 1)
        self.cpu_util = float(self.trace[idx])
        self.request_queue = int(self.cpu_util * self.QUEUE_SCALE)

        # 3. latency from current load vs capacity
        self.prev_latency = self.latency
        capacity = self.pod_count * self.POD_CAPACITY
        self.latency = float(np.clip(self.cpu_util / capacity, 0.0, 1.0))

        # 4. reward
        reward = self._calculate_reward(action)
        self.episode_reward += reward

        # 5. no hard termination — saturation is handled via the reward penalty.
        #    episodes end only when the fixed-length trace is exhausted.
        terminated = False
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._info()

    def _get_obs(self):
        return np.array([
            self.cpu_util,
            self.pod_count / self.MAX_PODS,
            min(self.request_queue / self.QUEUE_SCALE, 1.0),
            self.current_step / self.max_steps,
        ], dtype=np.float32)

    def _required_pods(self):
        # pods needed to hold utilization at the HEALTHY target (not saturation).
        # This is the "right-sized" reference; wasted pods are counted above it.
        return int(np.clip(np.ceil(self.cpu_util / (self.UTIL_TARGET * self.POD_CAPACITY)),
                           self.MIN_PODS, self.MAX_PODS))

    def _wasted_pods(self):
        return max(0, self.pod_count - self._required_pods()) / self.MAX_PODS

    def _calculate_reward(self, action):
        # latency = clipped utilization (1.0 = saturated); already set in step().
        scaling_cost = 1.0 if action != 1 else 0.0
        breach = 1.0 if self.latency >= 1.0 else 0.0

        # Penalize: poor service (latency), energy use (ABSOLUTE pods), SLA
        # breaches, and churn. Charging energy on absolute pods is what makes
        # tracking demand optimal instead of parking at MAX_PODS.
        return (
            -(self.W_LAT * self.latency)
            - (self.W_ENERGY * self.pod_count / self.MAX_PODS)
            - (self.W_SLA * breach)
            - (self.W_SCALE * scaling_cost)
        )

    def _check_termination(self):
        if self.latency >= 1.0:
            self.breach_count += 1
        else:
            self.breach_count = 0
        return self.breach_count >= self.BREACH_LIMIT

    def _info(self):
        return {
            "latency": self.latency, "pods": self.pod_count,
            "step": self.current_step, "cpu_util": self.cpu_util,
            "request_queue": self.request_queue,
            "required_pods": self._required_pods(),
            "wasted_pods": self._wasted_pods(),
            "episode_reward": self.episode_reward,
        }


if __name__ == "__main__":
    # --- calibration: confirm baseline is servable and peaks stress the system ---
    for tt in ["cyclical", "burst"]:
        env = KubernetesEnv(trace_type=tt)
        lo, hi = float(env.trace.min()), float(env.trace.max())

        # latency at START_PODS for baseline vs peak load (no scaling)
        cap = env.START_PODS * env.POD_CAPACITY
        lat_lo = min(lo / cap, 1.0)
        lat_hi = min(hi / cap, 1.0)
        print(f"[{tt}] cpu {lo:.2f}-{hi:.2f} | "
              f"latency@start pods: baseline {lat_lo:.2f}, peak {lat_hi:.2f} "
              f"| required pods: {int(np.ceil(lo/env.POD_CAPACITY))}-{int(np.ceil(hi/env.POD_CAPACITY))}")

        # random-agent rollout
        env.reset()
        total, term, info = 0.0, False, {}
        for _ in range(env.max_steps):
            _, r, term, trunc, info = env.step(env.action_space.sample())
            total += r
            if term or trunc:
                break
        print(f"        random agent -> ended step {info['step']}, "
              f"reward {total:.1f}, terminated {term}")