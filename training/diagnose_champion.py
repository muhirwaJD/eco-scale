"""
diagnose_champion.py — Behavioral diagnostic for the DQN champion.

A good reward number is not enough; we check WHAT the agent does. On the
held-out TEST traces, compare the champion against:
  - track-ideal : moves toward the healthy-utilization pod count (practical ceiling)
  - random      : the floor

Key signal: `mean|pods-required|` should be SMALL (agent tracks demand), and
`waste` should be at/below track-ideal — not parked high like the old reward.

Run: python training/diagnose_champion.py
"""

import os
import sys
import json
import numpy as np
from collections import Counter

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from stable_baselines3 import DQN
from environment.custom_env import KubernetesEnv
CHAMPION = os.path.join(ROOT, "models", "eco_scale_dqn_best.zip")


def _episode(env, policy):
    obs, _ = env.reset()
    done = False
    pods, req, lats, waste = [], [], [], []
    acts = Counter()
    R = 0.0
    while not done:
        a = policy(obs, env)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        R += r
        acts[a] += 1
        pods.append(info["pods"]); req.append(info["required_pods"])
        lats.append(info["latency"]); waste.append(info["wasted_pods"])
    return dict(R=R, gap=np.mean(np.abs(np.array(pods) - np.array(req))),
                lat=np.mean(lats), waste=np.mean(waste), acts=acts)


def main():
    split = json.load(open(os.path.join(ROOT, "data", "split.json")))
    test_paths = [os.path.join(ROOT, p) for p in split["test"]]
    model = DQN.load(CHAMPION)

    def dqn_pol(obs, env): return int(model.predict(obs, deterministic=True)[0])
    def track_pol(obs, env):
        tgt = env._required_pods()
        return 2 if tgt > env.pod_count else (0 if tgt < env.pod_count else 1)
    def rand_pol(obs, env): return env.action_space.sample()

    print(f"Diagnostic on {len(test_paths)} held-out TEST traces\n")
    print(f"{'policy':14s} {'reward':>9s} {'|pods-req|':>11s} {'latency':>8s} "
          f"{'waste':>7s}  action mix (down/hold/up)")
    print("-" * 72)
    summary = {}
    for name, pol in [("DQN champion", dqn_pol), ("track-ideal", track_pol), ("random", rand_pol)]:
        res = [_episode(KubernetesEnv(trace_paths=[p]), pol) for p in test_paths]
        acts = sum((r["acts"] for r in res), Counter())
        R = np.mean([r["R"] for r in res]); gap = np.mean([r["gap"] for r in res])
        summary[name] = dict(R=R, gap=gap)
        print(f"{name:14s} {R:9.2f} {gap:11.2f} "
              f"{np.mean([r['lat'] for r in res]):8.2f} "
              f"{np.mean([r['waste'] for r in res]):7.3f}  "
              f"{acts.get(0,0)}/{acts.get(1,0)}/{acts.get(2,0)}")
    print("-" * 72)

    d, t, r = summary["DQN champion"], summary["track-ideal"], summary["random"]
    print(f"\nChampion vs random : {d['R'] - r['R']:+.1f}  (want strongly positive)")
    print(f"Champion vs ideal  : {d['R'] - t['R']:+.1f}  (0 = matches practical ceiling)")
    # "tracking" = pod gap far below random AND reward near the ideal ceiling.
    tracks = d["gap"] < 2.5 and d["R"] >= t["R"] - 10
    beats_random = d["R"] > r["R"] + 50
    print("✅ Champion tracks demand (pod gap near ideal, reward at ceiling)." if tracks
          else f"❌ Champion off-target by {d['gap']:.1f} pods on average.")
    print("✅ Champion clearly beats random." if beats_random
          else "❌ Champion not clearly above random.")


if __name__ == "__main__":
    main()
