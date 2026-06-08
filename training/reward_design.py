"""
reward_design.py — Offline validation of candidate reward weights BEFORE training.

Single source of truth: this imports the REAL KubernetesEnv and only OVERRIDES
the candidate weights on the instance, then drives fixed policies through the
env's own step()/reward. There is NO second copy of the reward formula here, so
this harness can never drift from what the agent actually trains on.

Use it to vet a weight change in ~2s (no training): edit CANDIDATE, run, and
confirm demand-tracking still beats park-high/park-low/hold. Once happy, copy
the same weights into custom_env.py (the source of truth).

Run: python training/reward_design.py
"""

import os
import sys
import glob
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from environment.custom_env import KubernetesEnv

# Candidate weights to test. These mirror the env's current values; change them
# here to trial a tweak, then (if it validates) copy into custom_env.py.
CANDIDATE = dict(W_LAT=1.0, W_ENERGY=1.5, W_SLA=1.0, W_SCALE=0.02, UTIL_TARGET=0.70)


def make_env(trace_path):
    """Real env with the candidate weights overridden on the instance."""
    env = KubernetesEnv(trace_paths=[trace_path])
    for k, v in CANDIDATE.items():
        if not hasattr(env, k):
            raise AttributeError(f"KubernetesEnv has no weight '{k}' — check the name")
        setattr(env, k, v)          # env._calculate_reward() reads these attributes
    return env


# Fixed policies — decided from the env's OWN state/helpers (no duplicated logic).
def track_ideal(env):               # move toward the healthy-utilization pod count
    tgt = env._required_pods()
    return 2 if tgt > env.pod_count else (0 if tgt < env.pod_count else 1)

def park_high(env): return 2        # always scale up -> saturates at MAX_PODS
def park_low(env):  return 0        # always scale down -> MIN_PODS
def hold(env):      return 1        # never act

POLICIES = {"track-ideal": track_ideal, "park-HIGH": park_high,
            "park-LOW": park_low, "hold": hold}


def run(env, policy):
    env.reset()
    R = 0.0; us = []; ps = []; breaches = 0; n = 0
    done = False
    while not done:
        action = policy(env)
        _, r, term, trunc, info = env.step(action)
        done = term or trunc
        R += r
        us.append(info["latency"]); ps.append(info["pods"])
        breaches += (info["latency"] >= 1.0); n += 1
    return dict(R=R, util=np.mean(us), pods=np.mean(ps), breach=breaches / n)


def main():
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "traces", "trace_*.npy")))
    print(f"Candidate weights: {CANDIDATE}")
    print(f"Evaluated on {len(paths)} traces (via the real KubernetesEnv)\n")
    print(f"{'policy':16s} {'reward':>10s} {'mean util':>10s} {'mean pods':>10s} {'breach %':>9s}")
    print("-" * 60)
    rows = {}
    for name, pol in POLICIES.items():
        res = [run(make_env(p), pol) for p in paths]
        R = np.mean([x["R"] for x in res])
        rows[name] = R
        print(f"{name:16s} {R:10.2f} {np.mean([x['util'] for x in res]):10.2f} "
              f"{np.mean([x['pods'] for x in res]):10.2f} "
              f"{100*np.mean([x['breach'] for x in res]):8.1f}%")
    print("-" * 60)
    best = max(rows, key=rows.get)
    print(f"\nBest policy under these weights: {best}")
    print("✅ Reward rewards demand-tracking." if best == "track-ideal"
          else "❌ Reward favours a degenerate policy — retune before training.")
    print(f"   park-HIGH gap vs best: {rows['park-HIGH'] - rows[best]:.1f} "
          f"(should be strongly negative)")


if __name__ == "__main__":
    main()
