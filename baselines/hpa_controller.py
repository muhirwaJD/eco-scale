"""
hpa_controller.py — Realistic Kubernetes HPA baseline as a drop-in policy.

Mirrors the production Horizontal Pod Autoscaler: it reacts to the CURRENT
observed utilization (no anticipation) and drives replicas toward the standard
HPA target via desired = ceil(pods * util / target_util). Within the Eco-Scale
env it is constrained to ±1 pod/step (same dynamics as the RL agents), and it
applies a scale-DOWN stabilization window (proxy for K8s' ~5-min scaleDown
stabilization); scale-up is immediate.

Exposes the same `.predict(obs, deterministic=True) -> (action, None)` signature
as a stable-baselines3 model, so it plugs straight into the evaluation harness.

Self-test:  python baselines/hpa_controller.py
"""

import os
import sys
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from environment.custom_env import KubernetesEnv


class HPAController:
    """Reactive threshold autoscaler (Kubernetes HPA analogue)."""

    def __init__(self, target_util=None, stabilization_steps=3):
        # Pull dynamics constants from the env so there is one source of truth.
        self.POD_CAPACITY = KubernetesEnv.POD_CAPACITY
        self.MIN_PODS = KubernetesEnv.MIN_PODS
        self.MAX_PODS = KubernetesEnv.MAX_PODS
        # Default to the env's healthy target so it's the fairest single-threshold HPA.
        self.target_util = target_util if target_util is not None else KubernetesEnv.UTIL_TARGET
        self.stabilization_steps = stabilization_steps
        self._below_target_streak = 0

    def reset(self):
        self._below_target_streak = 0

    def _desired_pods(self, cpu_util):
        # Standard HPA: replicas needed to bring utilization to the target.
        return int(np.clip(np.ceil(cpu_util / (self.target_util * self.POD_CAPACITY)),
                           self.MIN_PODS, self.MAX_PODS))

    def predict(self, obs, deterministic=True):
        # obs = [cpu_util, pods/MAX_PODS, queue/QUEUE_SCALE, day_progress]
        cpu_util = float(obs[0])
        pods = int(round(float(obs[1]) * self.MAX_PODS))
        pods = int(np.clip(pods, self.MIN_PODS, self.MAX_PODS))

        desired = self._desired_pods(cpu_util)
        current_util = cpu_util / (pods * self.POD_CAPACITY)

        # track how long we've been under target (for scale-down stabilization)
        if current_util < self.target_util:
            self._below_target_streak += 1
        else:
            self._below_target_streak = 0

        if desired > pods:
            action = 2                      # scale up — immediate
        elif desired < pods:
            # scale down only after sustained low utilization (stabilization window)
            action = 0 if self._below_target_streak >= self.stabilization_steps else 1
        else:
            action = 1                      # maintain
        return action, None


def _self_test():
    import json
    split = json.load(open(os.path.join(ROOT, "data", "split.json")))
    trace = os.path.join(ROOT, split["test"][1])   # a mid-difficulty test trace
    env = KubernetesEnv(trace_paths=[trace])
    hpa = HPAController()

    obs, _ = env.reset(); hpa.reset()
    done = False; R = 0.0; pods = []; breaches = 0; n = 0
    while not done:
        a, _ = hpa.predict(obs)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        R += r; pods.append(info["pods"]); breaches += (info["latency"] >= 1.0); n += 1
    print(f"HPA self-test on {os.path.basename(trace)} (target_util={hpa.target_util})")
    print(f"  reward {R:.2f} | mean pods {np.mean(pods):.2f} | breach {100*breaches/n:.1f}%")
    print("  (sanity: reward should beat random ~-470 and sit near track-ideal ~-346)")


if __name__ == "__main__":
    _self_test()
