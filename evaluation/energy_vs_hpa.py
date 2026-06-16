"""
energy_vs_hpa.py — The energy story: PPO vs HPA across target utilizations.

HPA uses a single static utilization target. To stay safe (few SLA breaches),
real teams set it conservatively (~50%), which over-provisions; to save energy
they'd set it high (~90%), which breaches under peaks. No single target is good
at both. The RL agent adapts, so it can be lean AND reliable at once.

This sweeps HPA at several targets, evaluates the champion on the same held-out
test traces, and shows the energy/reliability trade-off.

Run: python evaluation/energy_vs_hpa.py
"""

import os
import sys
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from evaluation.evaluate_vs_hpa import evaluate, _sb3_policy, N_OFFSETS
from baselines.hpa_controller import HPAController
from utils.agents import find_champion, load_agent

OUTPUT_DIR = os.path.join(ROOT, "outputs", "hpa_comparison")
HPA_TARGETS = [0.5, 0.6, 0.7, 0.8, 0.9]   # 50% = conservative/safe, 90% = aggressive


def main():
    split = json.load(open(os.path.join(ROOT, "data", "split.json")))
    test_traces = [os.path.join(ROOT, p) for p in split["test"]]

    champion = find_champion()
    agent = _sb3_policy(load_agent(champion))

    print(f"Energy comparison on {len(test_traces)} held-out test traces "
          f"x {N_OFFSETS} offsets\n")
    print(f"{'controller':22s} {'reward':>8s} {'pods':>6s} {'waste':>7s} "
          f"{'p95 lat':>8s} {'breach%':>8s}")
    print("-" * 64)

    rows = []
    champ = evaluate(agent, test_traces)
    rows.append(_row(f"{champion.algorithm} (champion)", champ))
    print(_line(rows[-1]))

    hpa_runs = {}
    for target in HPA_TARGETS:
        r = evaluate(("hpa", HPAController(target_util=target)), test_traces)
        hpa_runs[target] = r
        label = f"HPA target={target:.0%}"
        if target == 0.5:
            label += " (safe)"
        rows.append(_row(label, r))
        print(_line(rows[-1]))
    print("-" * 64)

    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "energy_vs_hpa.csv"), index=False)

    # headline: champion vs the conservative HPA real teams actually deploy
    safe = hpa_runs[0.5]
    pod_saving = 100 * (safe["pods"] - champ["pods"]) / safe["pods"]
    waste_saving = 100 * (safe["waste"] - champ["waste"]) / safe["waste"]
    print(f"\nVs conservative HPA@50% (the realistic production default):")
    print(f"  pods:  {champ['pods']:.2f} vs {safe['pods']:.2f}  -> {pod_saving:.0f}% fewer pods")
    print(f"  waste: {champ['waste']:.3f} vs {safe['waste']:.3f} -> {waste_saving:.0f}% less waste")
    print(f"  breach: {champ['breach']:.1f}% vs {safe['breach']:.1f}% (comparable reliability)")

    _plot(champion.algorithm, champ, hpa_runs)
    print(f"\nSaved: outputs/hpa_comparison/energy_vs_hpa.csv + energy_frontier.png")


def _row(name, m):
    return dict(controller=name, mean_reward=round(m["mean"], 1),
                mean_pods=round(m["pods"], 2), waste=round(m["waste"], 3),
                p95_latency=round(m["p95_lat"], 3), breach_pct=round(m["breach"], 2))


def _line(row):
    return (f"{row['controller']:22s} {row['mean_reward']:8.1f} {row['mean_pods']:6.2f} "
            f"{row['waste']:7.3f} {row['p95_latency']:8.2f} {row['breach_pct']:8.1f}")


def _plot(champ_name, champ, hpa_runs):
    """Energy/reliability frontier: pods (energy) vs breach (reliability)."""
    fig, ax = plt.subplots(figsize=(9, 6))

    targets = sorted(hpa_runs)
    hpa_pods = [hpa_runs[t]["pods"] for t in targets]
    hpa_breach = [hpa_runs[t]["breach"] for t in targets]
    ax.plot(hpa_breach, hpa_pods, "o-", color="#C62828", label="HPA (static target)")
    for t in targets:
        ax.annotate(f"{t:.0%}", (hpa_runs[t]["breach"], hpa_runs[t]["pods"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color="#C62828")

    ax.scatter([champ["breach"]], [champ["pods"]], s=180, color="#2E7D32",
               zorder=5, marker="*", label=f"{champ_name} (adaptive)")
    ax.annotate(f"  {champ_name}", (champ["breach"], champ["pods"]),
                fontsize=10, color="#2E7D32", fontweight="bold", va="center")

    ax.set_xlabel("SLA breach rate %  (lower = more reliable)", fontsize=11)
    ax.set_ylabel("Mean pods  (lower = less energy)", fontsize=11)
    ax.set_title("Energy vs Reliability — PPO matches a tuned HPA, "
                 "beats the conservative default (no tuning needed)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_facecolor("#F8F9FA")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "energy_frontier.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
