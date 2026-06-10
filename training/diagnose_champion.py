"""
diagnose_champion.py — Behavioral check on the champion agent.

A good reward number isn't enough; we check WHAT the agent does. On the held-out
test traces, we compare the champion against two reference policies:
  - track-ideal : always moves toward the healthy pod count (the practical best)
  - random      : random actions (the floor)

The champion is found automatically from the training results (see agents.py),
so this always reports on whichever agent actually won — nothing is hardcoded.

Key signal: |pods - required| should be small (the agent tracks demand) and
waste should be low (it isn't just parking lots of idle pods).

Run: python training/diagnose_champion.py
"""

import os
import sys
import json
import numpy as np
from collections import Counter

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from environment.custom_env import KubernetesEnv
from agents import find_champion, load_agent


def run_one_episode(env, choose_action):
    """Run a full episode. `choose_action(obs, env)` returns 0/1/2.

    Returns total reward and per-step lists for the metrics we care about.
    """
    obs, _ = env.reset()
    pods, required, latency, waste = [], [], [], []
    actions = Counter()
    total_reward = 0.0
    done = False
    while not done:
        action = choose_action(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        actions[action] += 1
        pods.append(info["pods"])
        required.append(info["required_pods"])
        latency.append(info["latency"])
        waste.append(info["wasted_pods"])
    pod_gap = np.mean(np.abs(np.array(pods) - np.array(required)))
    return {
        "reward": total_reward,
        "pod_gap": pod_gap,
        "latency": np.mean(latency),
        "waste": np.mean(waste),
        "actions": actions,
    }


def average_over_traces(test_traces, choose_action):
    """Run a policy on every test trace and average the metrics."""
    runs = [run_one_episode(KubernetesEnv(trace_paths=[t]), choose_action)
            for t in test_traces]
    actions = sum((r["actions"] for r in runs), Counter())
    return {
        "reward": np.mean([r["reward"] for r in runs]),
        "pod_gap": np.mean([r["pod_gap"] for r in runs]),
        "latency": np.mean([r["latency"] for r in runs]),
        "waste": np.mean([r["waste"] for r in runs]),
        "actions": actions,
    }


def main():
    # held-out test traces (never seen during training)
    split = json.load(open(os.path.join(ROOT, "data", "split.json")))
    test_traces = [os.path.join(ROOT, p) for p in split["test"]]

    # the real champion, found from the training results
    champion = find_champion()
    model = load_agent(champion)
    champion_label = f"{champion.algorithm} champion"

    # the three policies we compare
    def champion_policy(obs, env):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    def track_ideal_policy(obs, env):
        target = env._required_pods()
        if target > env.pod_count:
            return 2
        if target < env.pod_count:
            return 0
        return 1

    def random_policy(obs, env):
        return env.action_space.sample()

    policies = {
        champion_label: champion_policy,
        "track-ideal": track_ideal_policy,
        "random": random_policy,
    }

    print(f"Diagnostic on {len(test_traces)} held-out TEST traces "
          f"(champion = {champion.algorithm} run {champion.run})\n")
    print(f"{'policy':16s} {'reward':>9s} {'|pods-req|':>11s} {'latency':>8s} "
          f"{'waste':>7s}  action mix (down/hold/up)")
    print("-" * 74)

    results = {}
    for name, policy in policies.items():
        r = average_over_traces(test_traces, policy)
        results[name] = r
        a = r["actions"]
        print(f"{name:16s} {r['reward']:9.2f} {r['pod_gap']:11.2f} "
              f"{r['latency']:8.2f} {r['waste']:7.3f}  "
              f"{a.get(0, 0)}/{a.get(1, 0)}/{a.get(2, 0)}")
    print("-" * 74)

    champ = results[champion_label]
    ideal = results["track-ideal"]
    rand = results["random"]
    print(f"\nChampion vs random : {champ['reward'] - rand['reward']:+.1f}  "
          f"(want strongly positive)")
    print(f"Champion vs ideal  : {champ['reward'] - ideal['reward']:+.1f}  "
          f"(0 = matches the practical ceiling)")

    tracks_demand = champ["pod_gap"] < 2.5 and champ["reward"] >= ideal["reward"] - 10
    beats_random = champ["reward"] > rand["reward"] + 50
    print("✅ Champion tracks demand (pod gap near ideal, reward at ceiling)."
          if tracks_demand else
          f"❌ Champion off-target by {champ['pod_gap']:.1f} pods on average.")
    print("✅ Champion clearly beats random." if beats_random
          else "❌ Champion not clearly above random.")


if __name__ == "__main__":
    main()
