"""
simulation.py — Live RL-vs-HPA simulation for the web control plane.

Runs the REAL champion agent and the REAL HPA baseline side-by-side over the same
real Alibaba trace, one tick at a time, so the web UI can animate "what the agent
did vs what plain HPA would have done" and the savings that result.

All logic lives here (Python), reusing the validated env / agent / HPA — the
frontend is a thin view, so the product never diverges from the tested code.
"""

import os
import json
# pyrefly: ignore [missing-import]
import numpy as np

from environment.custom_env import KubernetesEnv
from baselines.hpa_controller import HPAController
from serving.inference_engine import InferenceEngine

ROOT = os.path.join(os.path.dirname(__file__), "..")

# energy / cost assumptions (same as evaluation/cost_benefit.py)
POWER_PER_POD_W = 50.0
PRICE_FRW_PER_KWH = 175.0
MINUTES_PER_TICK = 5.0

ACTION_NAMES = {0: "scale_down", 1: "maintain", 2: "scale_up"}


def _rationale(action, cpu_util, pods, latency):
    """Plain-English reason for the agent's decision (explainability)."""
    pct = int(round(cpu_util * 100))
    if action == 2:
        return f"Load is high (CPU {pct}%, latency {latency:.0%}) — adding a pod to protect the SLA."
    if action == 0:
        return f"Load is low (CPU {pct}%) for {pods} pods — removing one to save energy."
    return f"Capacity matches demand (CPU {pct}%, latency {latency:.0%}) — holding steady."


class SimulationEngine:
    """Steps the RL agent and HPA over one real trace in lockstep."""

    def __init__(self, trace_path=None, engine=None):
        self.engine = engine or InferenceEngine()
        split = json.load(open(os.path.join(ROOT, "data", "split.json")))
        # default to the first held-out TEST trace (never seen in training)
        self.trace_path = trace_path or os.path.join(ROOT, split["test"][0])
        self.hpa_target = 0.5      # conservative default teams actually deploy
        self.reset()

    # ── lifecycle ────────────────────────────────────────────────
    def reset(self, hpa_target=None):
        if hpa_target is not None:
            self.hpa_target = float(hpa_target)
        self.rl_env = KubernetesEnv(trace_paths=[self.trace_path])
        self.hpa_env = KubernetesEnv(trace_paths=[self.trace_path])
        self.hpa = HPAController(target_util=self.hpa_target)
        self.rl_obs, _ = self.rl_env.reset()
        self.hpa_obs, _ = self.hpa_env.reset()
        self.hpa.reset()
        self.tick = 0
        self.max_ticks = self.rl_env.max_steps
        self._pod_ticks_saved = 0.0
        self._breaches_rl = 0
        self._breaches_hpa = 0
        return self.state(rl_action=1, hpa_action=1)

    # ── one simulation step ──────────────────────────────────────
    def step(self):
        if self.tick >= self.max_ticks - 1:
            return self.state(rl_action=1, hpa_action=1, done=True)

        rl_action = self._agent_action(self.rl_obs)
        hpa_action, _ = self.hpa.predict(self.hpa_obs)

        self.rl_obs, _, _, _, rl_info = self.rl_env.step(int(rl_action))
        self.hpa_obs, _, _, _, hpa_info = self.hpa_env.step(int(hpa_action))

        self.tick += 1
        self._pod_ticks_saved += (hpa_info["pods"] - rl_info["pods"])
        self._breaches_rl += int(rl_info["latency"] >= 1.0)
        self._breaches_hpa += int(hpa_info["latency"] >= 1.0)

        return self.state(rl_action=int(rl_action), hpa_action=int(hpa_action),
                          rl_info=rl_info, hpa_info=hpa_info)

    # ── helpers ──────────────────────────────────────────────────
    def _agent_action(self, obs):
        action, _ = self.engine.model.predict(np.array(obs), deterministic=True)
        return int(action)

    def _action_probs(self, obs):
        """PPO action preferences (down/hold/up) for the 'why not?' panel."""
        try:
            # pyright: ignore [reportMissingImports]
            import torch
            tensor, _ = self.engine.model.policy.obs_to_tensor(np.array(obs, dtype=np.float32))
            dist = self.engine.model.policy.get_distribution(tensor)
            probs = dist.distribution.probs.detach().numpy().ravel()
            return [float(p) for p in probs]
        except Exception:
            return None

    def state(self, rl_action, hpa_action, rl_info=None, hpa_info=None, done=False):
        rl_info = rl_info or self.rl_env._info()
        hpa_info = hpa_info or self.hpa_env._info()

        pod_hours = self._pod_ticks_saved * (MINUTES_PER_TICK / 60.0)
        kwh = pod_hours * POWER_PER_POD_W / 1000.0
        frw = kwh * PRICE_FRW_PER_KWH

        return {
            "tick": self.tick,
            "max_ticks": self.max_ticks,
            "day_progress": round(self.tick / self.max_ticks, 3),
            "cpu": round(float(rl_info["cpu_util"]), 4),
            "rl": {
                "pods": rl_info["pods"],
                "latency": round(float(rl_info["latency"]), 4),
                "required_pods": rl_info["required_pods"],
                "action": rl_action,
                "action_name": ACTION_NAMES[rl_action],
                "observation": [round(float(x), 4) for x in self.rl_obs],
                "probs": self._action_probs(self.rl_obs),
                "rationale": _rationale(rl_action, rl_info["cpu_util"],
                                        rl_info["pods"], rl_info["latency"]),
            },
            "hpa": {
                "pods": hpa_info["pods"],
                "latency": round(float(hpa_info["latency"]), 4),
                "action": hpa_action,
                "action_name": ACTION_NAMES[hpa_action],
                "target": self.hpa_target,
            },
            "savings": {
                "pod_ticks_saved": round(self._pod_ticks_saved, 1),
                "kwh": round(kwh, 3),
                "frw": round(frw, 1),
                "breaches_avoided": self._breaches_hpa - self._breaches_rl,
            },
            "done": done,
        }
