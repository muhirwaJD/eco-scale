import os
import glob
import gymnasium as gym
import numpy as np


class KubernetesEnv(gym.Env):
    """
    Pod-autoscaling environment driven by REAL Alibaba cluster traces.

    Traffic is read from pre-sliced, [0,1]-normalized CPU traces
    (data/traces/trace_*.npy), one value per timestep. Pass `trace_paths` to
    train/eval on a specific set; a random trace is sampled each episode when
    more than one is given.

    Observation (4-D, all scaled to [0,1]):
        [ cpu_util, pods/MAX_PODS, queue/QUEUE_SCALE, day_progress ]
    Action (discrete):
        0 = scale down (-1 pod), 1 = maintain, 2 = scale up (+1 pod)

    Predictive variants (optional, opt-in via `predictive=`): append ONE extra
    feature so the agent can anticipate load instead of only reacting. The base
    4-D behaviour is unchanged when predictive is None (default).
        "oracle"   → peak real load over the next `horizon` steps (perfect
                     foresight; only valid on a KNOWN trace — sim/training only).
        "trend"    → causal recent slope of load (Δ over `trend_window` steps);
                     computable live from past readings.
        "forecast" → causal Holt (level+trend) forecast of load `horizon` steps
                     ahead; computable live from past readings.
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

    DEFAULT_TRACE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "traces")

    # predictive feature config
    HORIZON = 6              # look-ahead steps (~30 min at 5-min steps) for oracle/forecast
    TREND_WINDOW = 3         # steps used for the causal slope feature
    _HOLT_ALPHA = 0.5        # forecast: level smoothing
    _HOLT_BETA = 0.3         # forecast: trend smoothing

    def __init__(self, trace_paths=None, render_mode=None, trace_dir=None,
                 predictive=None, horizon=None, trend_window=None):
        super().__init__()
        self.render_mode = render_mode

        # predictive mode (None = classic 4-D obs, fully backward-compatible)
        if predictive not in (None, "oracle", "trend", "forecast"):
            raise ValueError(f"predictive must be None|oracle|trend|forecast, got {predictive!r}")
        self.predictive = predictive
        self.horizon = int(horizon) if horizon else self.HORIZON
        self.trend_window = int(trend_window) if trend_window else self.TREND_WINDOW

        # --- load one or many real traces ---
        # Explicit: pass trace_paths=[...]. Default: every trace_*.npy in
        # data/traces/ (or $TRACE_DIR / the trace_dir arg). When more than one
        # trace is loaded, reset() samples a random one each episode.
        if not trace_paths:
            trace_dir = trace_dir or os.environ.get("TRACE_DIR", self.DEFAULT_TRACE_DIR)
            trace_paths = sorted(glob.glob(os.path.join(trace_dir, "trace_*.npy")))
            if not trace_paths:
                raise FileNotFoundError(
                    f"No traces found in {trace_dir!r}. Pass trace_paths=[...] "
                    f"or run data/make_split.py to populate data/traces/.")

        # Loading pre-cleaned numpy trace files into contiguous float32 arrays
        self.trace_paths = list(trace_paths)
        self.traces = [np.load(p).astype(np.float32) for p in self.trace_paths]
        self.trace = self.traces[0]          # active trace; reset() picks one per episode
        self.max_steps = len(self.trace)

        # base 4-D bounds; predictive adds one feature. "trend" is a signed slope
        # in [-1,1]; "oracle"/"forecast" are load values in [0,1].
        low = [0.0, 0.0, 0.0, 0.0]
        high = [1.0, 1.0, 1.0, 1.0]
        if self.predictive == "trend":
            low.append(-1.0); high.append(1.0)
        elif self.predictive in ("oracle", "forecast"):
            low.append(0.0); high.append(1.0)
        self.observation_space = gym.spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32,
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
        # predictive bookkeeping (causal trend/forecast read only past load)
        self._cpu_history = [self.cpu_util]
        self._holt_level = self.cpu_util
        self._holt_trend = 0.0

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
        self._record_load(self.cpu_util)      # causal history + Holt update (no-op for 4-D obs)

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
        obs = [
            self.cpu_util,
            self.pod_count / self.MAX_PODS,
            min(self.request_queue / self.QUEUE_SCALE, 1.0),
            self.current_step / self.max_steps,
        ]
        if self.predictive:
            obs.append(self._predictive_feature())
        return np.array(obs, dtype=np.float32)

    def _record_load(self, x):
        """Track past load for the causal predictive features (trend/forecast)."""
        self._cpu_history.append(x)
        prev_level = self._holt_level
        self._holt_level = self._HOLT_ALPHA * x + (1 - self._HOLT_ALPHA) * (self._holt_level + self._holt_trend)
        self._holt_trend = self._HOLT_BETA * (self._holt_level - prev_level) + (1 - self._HOLT_BETA) * self._holt_trend

    def _predictive_feature(self):
        """The single extra observation for the active predictive variant."""
        if self.predictive == "oracle":
            # perfect foresight: peak REAL load over the next `horizon` steps.
            future = self.trace[self.current_step + 1: self.current_step + 1 + self.horizon]
            peak = float(future.max()) if len(future) else float(self.trace[-1])
            return min(max(peak, 0.0), 1.0)
        if self.predictive == "trend":
            # causal slope: change in load over the last `trend_window` steps.
            k = self.trend_window
            hist = self._cpu_history
            slope = hist[-1] - (hist[-1 - k] if len(hist) > k else hist[0])
            return float(np.clip(slope, -1.0, 1.0))
        # forecast: causal Holt (level + trend) projection `horizon` steps ahead.
        return float(np.clip(self._holt_level + self.horizon * self._holt_trend, 0.0, 1.0))

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
    import glob
    trace_files = sorted(glob.glob(os.path.join(os.path.dirname(__file__),
                                                 "..", "data", "traces", "trace_*.npy")))
    if not trace_files:
        raise SystemExit("No traces found in data/traces/ — run data/make_split.py first.")

    for tp in trace_files[:3]:                      # sample a few of the real traces
        env = KubernetesEnv(trace_paths=[tp])
        lo, hi = float(env.trace.min()), float(env.trace.max())

        cap = env.START_PODS * env.POD_CAPACITY
        lat_lo, lat_hi = min(lo / cap, 1.0), min(hi / cap, 1.0)
        print(f"[{os.path.basename(tp)}] cpu {lo:.2f}-{hi:.2f} | "
              f"latency@start pods: baseline {lat_lo:.2f}, peak {lat_hi:.2f} "
              f"| required pods (healthy): "
              f"{int(np.ceil(lo/(env.UTIL_TARGET*env.POD_CAPACITY)))}-"
              f"{int(np.ceil(hi/(env.UTIL_TARGET*env.POD_CAPACITY)))}")

        env.reset()
        total, term, info = 0.0, False, {}
        for _ in range(env.max_steps):
            _, r, term, trunc, info = env.step(env.action_space.sample())
            total += r
            if term or trunc:
                break
        print(f"        random agent -> ended step {info['step']}, "
              f"reward {total:.1f}, terminated {term}")