"""
compare_realcluster.py — Compare the RL controller vs native HPA on the REAL
cluster, using the per-tick logs from run_experiment.py.

Run after both:
    python deploy/run_experiment.py --mode rl  --duration 240
    python deploy/run_experiment.py --mode hpa --duration 240
    python deploy/compare_realcluster.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "outputs", "realcluster")


def summary(df):
    return {
        "avg_pods": df["replicas"].mean(),
        "max_pods": df["replicas"].max(),
        "avg_cpu_m": df["avg_cpu_m"].mean(),
        "p95_latency_ms": df["p95_latency_s"].mean() * 1000,
        "max_latency_ms": df["p95_latency_s"].max() * 1000,
    }


def main():
    rl_path = os.path.join(OUT, "realcluster_rl.csv")
    hpa_path = os.path.join(OUT, "realcluster_hpa.csv")
    if not (os.path.exists(rl_path) and os.path.exists(hpa_path)):
        raise SystemExit("Need both realcluster_rl.csv and realcluster_hpa.csv "
                         "(run run_experiment.py for each mode first).")

    rl, hpa = pd.read_csv(rl_path), pd.read_csv(hpa_path)
    s_rl, s_hpa = summary(rl), summary(hpa)

    print(f"{'metric':18s} {'RL (PPO)':>12s} {'HPA':>12s}")
    print("-" * 44)
    for k in s_rl:
        print(f"{k:18s} {s_rl[k]:12.1f} {s_hpa[k]:12.1f}")
    print("-" * 44)
    print("\nReal-cluster validation (same load wave on OrbStack Kubernetes).")
    print("Note: agent trained in simulation — expect a sim-to-real transfer gap.")

    pd.DataFrame({"RL_PPO": s_rl, "HPA": s_hpa}).to_csv(
        os.path.join(OUT, "realcluster_comparison.csv"))

    # time series: pods and latency for both controllers
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle("Real-cluster: RL (PPO) vs HPA under the same load wave",
                 fontsize=13, fontweight="bold")
    ax1.plot(rl["t"], rl["replicas"], "o-", color="#2E7D32", label="RL (PPO)")
    ax1.plot(hpa["t"], hpa["replicas"], "s-", color="#C62828", label="HPA")
    ax1.set_ylabel("Pods"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(rl["t"], rl["p95_latency_s"] * 1000, "o-", color="#2E7D32", label="RL (PPO)")
    ax2.plot(hpa["t"], hpa["p95_latency_s"] * 1000, "s-", color="#C62828", label="HPA")
    ax2.set_ylabel("p95 latency (ms)"); ax2.set_xlabel("time (s)")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "realcluster_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: outputs/realcluster/realcluster_comparison.csv + .png")


if __name__ == "__main__":
    main()
