"""
evaluate_predictive.py — Does anticipation beat reaction?

Head-to-head on the 5 HELD-OUT test traces (x offsets for statistical power):
  - current champion (reactive, 4-D obs)
  - predictive agents: oracle / trend / forecast (5-D obs)
  - HPA energy frontier (targets 50/60/70/90%)

All agents share the SAME reward, so rewards are directly comparable. Reports the
multi-metric table, paired t-tests (each predictive vs the champion), and applies
the pre-agreed promotion rule: a predictive agent wins only if it uses fewer
pods / less waste at equal-or-better SLA (breach%) than the champion.

Outputs: outputs/simulation/predictive_comparison.csv + .png

Run: python evaluation/evaluate_predictive.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from stable_baselines3 import PPO
from environment.custom_env import KubernetesEnv
from baselines.hpa_controller import HPAController

OUT_DIR = os.path.join(ROOT, "outputs", "simulation")
MODEL_DIR = os.path.join(ROOT, "models")
N_OFFSETS = 10


def make_env(rolled, predictive):
    """A single-trace env (deterministic) over a rolled copy, in the given obs mode."""
    env = KubernetesEnv(trace_paths=None, predictive=predictive,
                        trace_dir=os.path.join(ROOT, "data", "traces"))
    env.traces = [rolled]; env.trace = rolled; env.max_steps = len(rolled)
    return env


def run_episode(env, kind, agent):
    obs, _ = env.reset()
    if kind == "hpa":
        agent.reset()
    R, lats, pods, waste = 0.0, [], [], []
    done = False
    while not done:
        if kind == "hpa":
            a, _ = agent.predict(obs)
        elif kind == "sb3":
            a, _ = agent.predict(obs, deterministic=True)
        else:  # random
            a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(int(a))
        done = term or trunc
        R += r; lats.append(info["latency"]); pods.append(info["pods"]); waste.append(info["wasted_pods"])
    return R, lats, pods, waste


def build_controllers():
    """name -> (kind, agent, predictive_mode). Skips predictive models not yet trained."""
    ctrls = {}
    ctrls["champion"] = ("sb3", PPO.load(os.path.join(MODEL_DIR, "eco_scale_best.zip")), None)
    for v in ["oracle", "trend", "forecast"]:
        path = os.path.join(MODEL_DIR, f"eco_scale_{v}.zip")
        if os.path.exists(path):
            ctrls[v] = ("sb3", PPO.load(path), v)
        else:
            print(f"  (skip {v}: {path} not found)")
    for t in [0.5, 0.6, 0.7, 0.9]:
        ctrls[f"HPA@{int(t*100)}"] = ("hpa", HPAController(target_util=t), None)
    return ctrls


def evaluate(kind, agent, predictive, test_traces):
    ep_rewards, all_lat, all_pods, all_waste = [], [], [], []
    breaches = steps = 0
    for tpath in test_traces:
        base = np.load(tpath).astype(np.float32)
        stride = len(base) // N_OFFSETS
        for k in range(N_OFFSETS):
            rolled = np.roll(base, -k * stride)
            env = make_env(rolled, predictive)
            R, lats, pods, waste = run_episode(env, kind, agent)
            ep_rewards.append(R)
            all_lat += lats; all_pods += pods; all_waste += waste
            breaches += sum(l >= 1.0 for l in lats); steps += len(lats)
    return dict(rewards=np.array(ep_rewards), mean=float(np.mean(ep_rewards)),
                std=float(np.std(ep_rewards)), p95_lat=float(np.percentile(all_lat, 95)),
                breach=100 * breaches / steps, waste=float(np.mean(all_waste)),
                pods=float(np.mean(all_pods)))


def main():
    split = json.load(open(os.path.join(ROOT, "data", "split.json")))
    test_traces = [os.path.join(ROOT, p) for p in split["test"]]
    print(f"Held-out eval: {len(test_traces)} traces x {N_OFFSETS} offsets "
          f"= {len(test_traces)*N_OFFSETS} paired episodes/controller\n")

    ctrls = build_controllers()
    res = {name: evaluate(k, a, p, test_traces) for name, (k, a, p) in ctrls.items()}

    print(f"{'controller':11s} {'reward':>9s} {'±std':>7s} {'p95lat':>7s} {'breach%':>8s} {'waste':>7s} {'pods':>6s}")
    print("-" * 62)
    rows = []
    for name, m in res.items():
        print(f"{name:11s} {m['mean']:9.2f} {m['std']:7.2f} {m['p95_lat']:7.2f} "
              f"{m['breach']:8.2f} {m['waste']:7.3f} {m['pods']:6.2f}")
        rows.append(dict(controller=name, mean_reward=round(m["mean"], 2), std_reward=round(m["std"], 2),
                         p95_latency=round(m["p95_lat"], 3), breach_pct=round(m["breach"], 3),
                         waste=round(m["waste"], 3), mean_pods=round(m["pods"], 2)))
    print("-" * 62)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "predictive_comparison.csv"), index=False)

    # ── paired t-tests: each predictive variant vs champion ──
    ch = res["champion"]
    print("\nPaired t-tests vs champion (same trace+offset episodes):")
    for v in ["oracle", "trend", "forecast"]:
        if v not in res:
            continue
        a = res[v]["rewards"]
        t, p = stats.ttest_rel(a, ch["rewards"])
        diff = a.mean() - ch["rewards"].mean()
        sig = "significant" if p < 0.05 else "n.s."
        print(f"  {v:9s} reward {a.mean():.2f} vs champion {ch['rewards'].mean():.2f} | "
              f"Δ{diff:+.2f}  t={t:.2f} p={p:.4f} ({sig})")

    # ── promotion decision (deployable variants only; oracle is the ceiling) ──
    print("\nPromotion check (fewer pods & less waste at equal-or-better SLA):")
    verdict = []
    for v in ["trend", "forecast"]:
        if v not in res:
            continue
        m = res[v]
        wins = (m["pods"] <= ch["pods"] + 1e-6 and m["waste"] <= ch["waste"] + 1e-6
                and m["breach"] <= ch["breach"] + 0.05)
        better_reward = m["mean"] > ch["mean"]
        print(f"  {v:9s} pods {m['pods']:.2f} vs {ch['pods']:.2f} | waste {m['waste']:.3f} vs {ch['waste']:.3f} "
              f"| breach {m['breach']:.2f} vs {ch['breach']:.2f} | reward {m['mean']:.2f} vs {ch['mean']:.2f} "
              f"-> {'PROMOTE' if (wins and better_reward) else 'keep champion'}")
        if wins and better_reward:
            verdict.append(v)
    if "oracle" in res:
        o = res["oracle"]
        print(f"  (oracle ceiling: reward {o['mean']:.2f}, pods {o['pods']:.2f}, breach {o['breach']:.2f} "
              f"— upper bound with perfect foresight)")
    print(f"\nVERDICT: {'promote '+', '.join(verdict) if verdict else 'keep the current champion'}")

    _plot(res)
    print("Saved: outputs/simulation/predictive_comparison.csv + .png")


def _plot(res):
    names = list(res.keys())
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Predictive vs Reactive — held-out test traces", fontsize=14, fontweight="bold")

    # energy vs reliability scatter (the frontier view)
    for name, m in res.items():
        marker = "*" if name == "champion" else ("D" if name.startswith("HPA") else "o")
        size = 220 if name in ("champion",) else 90
        ax[0].scatter(m["breach"], m["pods"], s=size, marker=marker, label=name, alpha=0.85)
        ax[0].annotate(name, (m["breach"], m["pods"]), fontsize=8,
                       xytext=(4, 4), textcoords="offset points")
    ax[0].set_xlabel("SLA breach % (lower = safer)"); ax[0].set_ylabel("mean pods (lower = greener)")
    ax[0].set_title("Energy vs Reliability"); ax[0].grid(alpha=0.3)

    # reward bars
    order = [n for n in names]
    vals = [res[n]["mean"] for n in order]
    colors = ["#19A979" if n in ("trend", "forecast", "oracle") else
              ("#F5C518" if n == "champion" else "#9AA7B0") for n in order]
    ax[1].bar(order, vals, color=colors)
    ax[1].set_ylabel("mean reward (higher = better)"); ax[1].set_title("Reward")
    ax[1].tick_params(axis="x", rotation=30); ax[1].grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUT_DIR, "predictive_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
