"""
evaluate_vs_hpa.py — Head-to-head: RL agents vs realistic HPA on held-out traces.

Stage 1 of the HPA comparison (simulated). Runs each controller on the 5 held-out
TEST traces, augmented with random start-offsets for statistical power, and reports
multi-metric results plus a paired t-test (DQN vs HPA).

Metrics (map to the proposal): mean reward, p95 latency, breach rate, waste, pods.

Run: python evaluation/evaluate_vs_hpa.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from environment.custom_env import KubernetesEnv
from baselines.hpa_controller import HPAController
from agents import find_champion, load_agent

OUTPUT_DIR = os.path.join(ROOT, "outputs", "hpa_comparison")
N_OFFSETS = 10                       # start-offsets per trace -> 5 x 10 = 50 paired episodes
RANDOM_REF, IDEAL_REF = -470.0, -346.0


# ── controllers (uniform predict interface) ──────────────────────────
def _sb3_policy(model):
    """Wrap a trained stable-baselines3 model as a choose_action(obs) function."""
    def choose(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return choose


def load_controllers():
    """The RL agents we benchmark (best of each, found from results) + HPA + random."""
    ctrls = {}
    for algorithm in ["PPO", "DQN"]:          # PPO = overall champion, DQN = context
        try:
            champion = find_champion([algorithm])
            ctrls[algorithm] = _sb3_policy(load_agent(champion))
        except FileNotFoundError as e:
            print(f"  ({algorithm} skipped: {e})")

    hpa = HPAController()
    ctrls["HPA"] = ("hpa", hpa)          # special-cased: needs per-episode reset()
    ctrls["random"] = ("random", None)
    return ctrls


def run_episode(env, ctrl):
    """One episode; returns total reward + per-step latency/pods/waste."""
    obs, _ = env.reset()
    kind = ctrl[0] if isinstance(ctrl, tuple) else None
    if kind == "hpa":
        ctrl[1].reset()
    done = False
    R = 0.0; lats = []; pods = []; waste = []
    while not done:
        if kind == "hpa":
            a, _ = ctrl[1].predict(obs)
        elif kind == "random":
            a = env.action_space.sample()
        else:
            a = ctrl(obs)
        obs, r, term, trunc, info = env.step(int(a))
        done = term or trunc
        R += r; lats.append(info["latency"]); pods.append(info["pods"]); waste.append(info["wasted_pods"])
    return R, lats, pods, waste


def evaluate(ctrl, test_traces):
    """Run a controller over all test traces x offsets. Returns per-episode rewards + pooled metrics."""
    ep_rewards = []; all_lat = []; all_pods = []; all_waste = []; breaches = 0; steps = 0
    for tpath in test_traces:
        base = np.load(tpath).astype(np.float32)
        env = KubernetesEnv(trace_paths=[tpath])
        for k in range(N_OFFSETS):
            rolled = np.roll(base, -k * (len(base) // N_OFFSETS))
            env.traces = [rolled]; env.trace = rolled; env.max_steps = len(rolled)
            R, lats, pods, waste = run_episode(env, ctrl)
            ep_rewards.append(R)
            all_lat += lats; all_pods += pods; all_waste += waste
            breaches += sum(l >= 1.0 for l in lats); steps += len(lats)
    return dict(
        rewards=np.array(ep_rewards),
        mean=np.mean(ep_rewards), std=np.std(ep_rewards),
        p95_lat=np.percentile(all_lat, 95), breach=100 * breaches / steps,
        waste=np.mean(all_waste), pods=np.mean(all_pods),
    )


def main():
    split = json.load(open(os.path.join(ROOT, "data", "split.json")))
    test_traces = [os.path.join(ROOT, p) for p in split["test"]]
    print(f"Evaluating on {len(test_traces)} held-out test traces x {N_OFFSETS} offsets "
          f"= {len(test_traces)*N_OFFSETS} episodes/controller\n")

    ctrls = load_controllers()
    res = {name: evaluate(c, test_traces) for name, c in ctrls.items()}

    # ── metric table ──
    print(f"{'controller':10s} {'reward':>9s} {'±std':>7s} {'p95 lat':>8s} "
          f"{'breach%':>8s} {'waste':>7s} {'pods':>6s}")
    print("-" * 60)
    rows = []
    for name, m in res.items():
        print(f"{name:10s} {m['mean']:9.2f} {m['std']:7.2f} {m['p95_lat']:8.2f} "
              f"{m['breach']:8.1f} {m['waste']:7.3f} {m['pods']:6.2f}")
        rows.append(dict(controller=name, mean_reward=round(m["mean"], 2),
                         std_reward=round(m["std"], 2), p95_latency=round(m["p95_lat"], 3),
                         breach_pct=round(m["breach"], 2), waste=round(m["waste"], 3),
                         mean_pods=round(m["pods"], 2)))
    print("-" * 60)
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "dqn_vs_hpa_results.csv"), index=False)

    # ── paired t-tests vs HPA (same trace+offset episodes are paired) ──
    # The overall champion is the headline; the other RL agent is context.
    headline_algo = find_champion().algorithm
    h = res["HPA"]["rewards"]
    headline = None
    for name in ["PPO", "DQN"]:
        if name not in res:
            continue
        a = res[name]["rewards"]
        t, p = stats.ttest_rel(a, h)
        diff = a.mean() - h.mean()
        tag = "HEADLINE" if name == headline_algo else "context "
        print(f"\n[{tag}] Paired t-test  {name} vs HPA  (n={len(a)} episodes)")
        print(f"  {name} {a.mean():.2f} ± {a.std():.2f}  |  HPA {h.mean():.2f} ± {h.std():.2f}")
        print(f"  mean diff {diff:+.2f}  |  t={t:.3f}  p={p:.4f}")
        if p < 0.05:
            print(f"  {'✅' if diff>0 else '❌'} {name} significantly "
                  f"{'BEATS' if diff>0 else 'WORSE than'} HPA (p<0.05).")
        else:
            print(f"  ➖ {name} ≈ HPA (no significant difference, p≥0.05).")
        if name == headline_algo:
            headline = (name, a, t, p, diff)

    if headline is None and "DQN" in res:      # fallback if champion absent
        a = res["DQN"]["rewards"]; t, p = stats.ttest_rel(a, h)
        headline = ("DQN", a, t, p, a.mean() - h.mean())

    name, a, t, p, diff = headline
    _plot(res, name, a, h, t, p)
    print(f"\nSaved: outputs/hpa_comparison/dqn_vs_hpa_results.csv + comparison.png")


def _plot(res, primary, a, h, t, p):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f"RL ({primary}) vs HPA — Held-out Test Traces", fontsize=14, fontweight="bold")

    # paired reward distributions (primary agent vs HPA)
    ax = axes[0]
    ax.boxplot([a, h], labels=[primary, "HPA"], showmeans=True)
    ax.axhline(IDEAL_REF, color="#2E7D32", ls="--", lw=1, label="track-ideal")
    ax.axhline(RANDOM_REF, color="#C62828", ls=":", lw=1, label="random")
    ax.set_ylabel("Episode reward"); ax.set_title(f"Reward (t={t:.2f}, p={p:.3f})")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # p95 latency & breach
    ax = axes[1]
    names = list(res.keys())
    x = np.arange(len(names)); w = 0.4
    ax.bar(x - w/2, [res[n]["p95_lat"] for n in names], w, label="p95 latency", color="#2196F3")
    ax.bar(x + w/2, [res[n]["breach"]/100 for n in names], w, label="breach rate", color="#E57373")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20); ax.set_title("Service quality")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # waste & pods (energy)
    ax = axes[2]
    ax.bar(x - w/2, [res[n]["waste"] for n in names], w, label="waste", color="#FF9800")
    ax2 = ax.twinx()
    ax2.plot(x, [res[n]["pods"] for n in names], "ko-", label="mean pods")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20); ax.set_title("Energy / right-sizing")
    ax.set_ylabel("waste"); ax2.set_ylabel("mean pods"); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "dqn_vs_hpa_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
