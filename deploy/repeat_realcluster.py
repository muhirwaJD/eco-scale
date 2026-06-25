"""
repeat_realcluster.py — Repeat the real RL-vs-HPA experiment N times for stats.

Runs N independent rounds of (RL phase, then real-HPA phase) under the same load
wave, records avg/max pods and p95 latency for each round, and reports the
mean ± standard deviation across rounds — turning the single-run demo into a
small statistical comparison.

Run:  python deploy/repeat_realcluster.py --runs 3 --duration 120
"""

import os
import sys
import csv
import argparse
import subprocess
import statistics as stats

ROOT = os.path.join(os.path.dirname(__file__), "..")
RC = os.path.join(ROOT, "outputs", "realcluster")


def run_phase(mode, duration):
    subprocess.run([sys.executable, os.path.join(ROOT, "deploy", "run_experiment.py"),
                    "--mode", mode, "--duration", str(duration), "--interval", "15"],
                   cwd=ROOT, check=True)
    rows = list(csv.DictReader(open(os.path.join(RC, f"realcluster_{mode}.csv"))))
    pods = [int(r["replicas"]) for r in rows]
    lat = [float(r["p95_latency_s"]) for r in rows if float(r["p95_latency_s"]) > 0]
    return {
        "avg_pods": sum(pods) / len(pods),
        "max_pods": max(pods),
        "p95_ms": 1000 * sum(lat) / len(lat) if lat else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--duration", type=int, default=120)
    args = ap.parse_args()

    rl_runs, hpa_runs = [], []
    for i in range(args.runs):
        print(f"\n========== ROUND {i + 1}/{args.runs} ==========")
        rl_runs.append(run_phase("rl", args.duration))
        hpa_runs.append(run_phase("hpa", args.duration))
        print(f"  RL  {rl_runs[-1]['avg_pods']:.2f} pods, p95 {rl_runs[-1]['p95_ms']:.0f}ms")
        print(f"  HPA {hpa_runs[-1]['avg_pods']:.2f} pods, p95 {hpa_runs[-1]['p95_ms']:.0f}ms")

    def ms(key, runs):
        vals = [r[key] for r in runs]
        sd = stats.stdev(vals) if len(vals) > 1 else 0.0
        return sum(vals) / len(vals), sd

    rl_pods, rl_pods_sd = ms("avg_pods", rl_runs)
    hpa_pods, hpa_pods_sd = ms("avg_pods", hpa_runs)
    rl_p95, rl_p95_sd = ms("p95_ms", rl_runs)
    hpa_p95, hpa_p95_sd = ms("p95_ms", hpa_runs)
    saving = 100 * (hpa_pods - rl_pods) / hpa_pods

    print("\n" + "=" * 56)
    print(f"AGGREGATE over {args.runs} runs (mean ± std)")
    print("=" * 56)
    print(f"  RL  avg pods : {rl_pods:.2f} ± {rl_pods_sd:.2f}  | p95 {rl_p95:.0f} ± {rl_p95_sd:.0f} ms")
    print(f"  HPA avg pods : {hpa_pods:.2f} ± {hpa_pods_sd:.2f}  | p95 {hpa_p95:.0f} ± {hpa_p95_sd:.0f} ms")
    print(f"  -> RL uses {saving:.0f}% fewer pods at comparable latency")

    out = os.path.join(RC, "realcluster_repeats.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "RL_mean", "RL_std", "HPA_mean", "HPA_std"])
        w.writerow(["avg_pods", round(rl_pods, 2), round(rl_pods_sd, 2),
                    round(hpa_pods, 2), round(hpa_pods_sd, 2)])
        w.writerow(["p95_ms", round(rl_p95, 0), round(rl_p95_sd, 0),
                    round(hpa_p95, 0), round(hpa_p95_sd, 0)])
    print(f"\nSaved {out} ({args.runs} runs)")


if __name__ == "__main__":
    main()
