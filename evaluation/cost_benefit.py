"""
cost_benefit.py — Cost/benefit + variability analysis: PPO vs HPA.

Turns the raw comparison into the numbers a supervisor asked for:
  * energy COST (Rwandan Francs) from the pod-count difference,
  * the DEVIATION (mean +/- std) of latency and pods, i.e. how *consistent*
    each controller is, not just its average.

All cost assumptions are explicit constants below — change them to match your
own infrastructure.

Run: python evaluation/cost_benefit.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from environment.custom_env import KubernetesEnv
from baselines.hpa_controller import HPAController
from agents import find_champion, load_agent
from evaluation.evaluate_vs_hpa import _sb3_policy, N_OFFSETS

# ── cost assumptions (explicit; adjust to your infrastructure) ──
POWER_PER_POD_W = 50.0       # avg power one pod draws under load (watts) — assumption
PRICE_FRW_PER_KWH = 175.0    # Rwanda electricity tariff (RURA 2025)
FRW_PER_USD = 1300.0         # approximate exchange rate
HOURS_PER_EPISODE = 24.0     # 288 steps x 5 min = one day


def collect(controller, test_traces):
    """Run a controller over test traces x offsets; collect PER-EPISODE metrics."""
    ep_latency, ep_pods, ep_p95, ep_breach = [], [], [], []
    is_hpa = isinstance(controller, tuple)
    for tpath in test_traces:
        base = np.load(tpath).astype(np.float32)
        env = KubernetesEnv(trace_paths=[tpath])
        for k in range(N_OFFSETS):
            rolled = np.roll(base, -k * (len(base) // N_OFFSETS))
            env.traces, env.trace, env.max_steps = [rolled], rolled, len(rolled)
            obs, _ = env.reset()
            if is_hpa:
                controller[1].reset()
            lats, pods = [], []
            done = False
            while not done:
                action = controller[1].predict(obs)[0] if is_hpa else controller(obs)
                obs, _, term, trunc, info = env.step(int(action))
                done = term or trunc
                lats.append(info["latency"]); pods.append(info["pods"])
            ep_latency.append(np.mean(lats)); ep_pods.append(np.mean(pods))
            ep_p95.append(np.percentile(lats, 95))
            ep_breach.append(100 * np.mean([l >= 1.0 for l in lats]))
    return {
        "latency_mean": np.mean(ep_latency), "latency_std": np.std(ep_latency),
        "p95_mean": np.mean(ep_p95),
        "pods_mean": np.mean(ep_pods), "pods_std": np.std(ep_pods),
        "breach_mean": np.mean(ep_breach),
    }


def daily_energy_cost_frw(mean_pods):
    kwh = mean_pods * POWER_PER_POD_W / 1000.0 * HOURS_PER_EPISODE
    return kwh * PRICE_FRW_PER_KWH


def main():
    split = json.load(open(os.path.join(ROOT, "data", "split.json")))
    test_traces = [os.path.join(ROOT, p) for p in split["test"]]

    champion = find_champion()
    controllers = {
        f"{champion.algorithm} (champion)": _sb3_policy(load_agent(champion)),
        "HPA@50% (conservative)": ("hpa", HPAController(target_util=0.5)),
        "HPA@70% (tuned)": ("hpa", HPAController(target_util=0.7)),
    }

    print("Assumptions: "
          f"{POWER_PER_POD_W:.0f} W/pod, {PRICE_FRW_PER_KWH:.0f} Frw/kWh, "
          f"{HOURS_PER_EPISODE:.0f} h/day\n")
    print(f"{'controller':24s} {'latency(mean±std)':>20s} {'pods(mean±std)':>17s} "
          f"{'breach%':>8s} {'Frw/day':>9s}")
    print("-" * 84)

    rows = {}
    for name, ctrl in controllers.items():
        m = collect(ctrl, test_traces)
        cost = daily_energy_cost_frw(m["pods_mean"])
        rows[name] = {**m, "frw_per_day": cost}
        print(f"{name:24s} {m['latency_mean']:8.3f} ± {m['latency_std']:.3f}     "
              f"{m['pods_mean']:6.2f} ± {m['pods_std']:.2f}   {m['breach_mean']:7.2f} "
              f"{cost:9.0f}")
    print("-" * 84)

    pd.DataFrame(rows).T.to_csv(os.path.join(ROOT, "outputs", "cost_benefit.csv"))

    champ_name = f"{champion.algorithm} (champion)"
    champ = rows[champ_name]
    print(f"\nCost/benefit of {champ_name} vs each HPA (per namespace):")
    for hpa in ["HPA@50% (conservative)", "HPA@70% (tuned)"]:
        d_frw = rows[hpa]["frw_per_day"] - champ["frw_per_day"]   # +ve = champion cheaper
        d_lat = rows[hpa]["latency_mean"] - champ["latency_mean"]  # +ve = champion faster
        yr = d_frw * 365
        print(f"  vs {hpa}:")
        print(f"    energy : {d_frw:+.0f} Frw/day  ({yr:+,.0f} Frw/yr, "
              f"{yr / FRW_PER_USD:+,.0f} USD/yr)  [+ = champion cheaper]")
        print(f"    latency: {d_lat:+.3f} mean utilization-latency  [+ = champion lower/faster]")
        print(f"    consistency: champion latency std {champ['latency_std']:.3f} "
              f"vs HPA {rows[hpa]['latency_std']:.3f}")


if __name__ == "__main__":
    main()
