"""
rl_controller.py — Drive a REAL Kubernetes deployment with the champion agent.

Every tick it reads the real cluster state (replica count + average pod CPU from
metrics-server), maps those into the 4-D observation the agent was trained on,
asks the champion for an action, and applies it with `kubectl scale` (+/-1 pod).

This is the sim-to-real bridge. The observation mapping is necessarily a proxy:
  cpu_util     <- avg pod CPU (millicores) / 1000   (fraction of one core)
  pods         <- current replica count
  queue_depth  <- proxy derived from cpu_util (real app has no sim "queue")
  day_progress <- elapsed time / run duration

Usage:
    python deploy/controller/rl_controller.py --duration 600 --interval 15
Logs per-tick metrics to outputs/realcluster/realcluster_rl.csv.
"""

import os
import sys
import csv
import time
import argparse
import subprocess

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)

from environment.custom_env import KubernetesEnv
from serving.inference_engine import InferenceEngine

DEPLOYMENT = "eco-sample-app"
MIN_PODS, MAX_PODS = 1, 10        # cluster scaling bounds (HPA uses the same)


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def get_replicas():
    out = sh(["kubectl", "get", "deployment", DEPLOYMENT,
              "-o", "jsonpath={.status.readyReplicas}"])
    return int(out) if out.isdigit() else 0


def get_avg_cpu_millicores():
    """Average CPU (millicores) across the app's pods, via metrics-server."""
    out = sh(["kubectl", "top", "pods", "-l", f"app={DEPLOYMENT}", "--no-headers"])
    if not out:
        return 0.0
    cpus = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].endswith("m"):
            cpus.append(float(parts[1][:-1]))
    return sum(cpus) / len(cpus) if cpus else 0.0


def scale_to(n):
    n = max(MIN_PODS, min(MAX_PODS, n))
    sh(["kubectl", "scale", "deployment", DEPLOYMENT, f"--replicas={n}"])
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=600, help="run seconds")
    ap.add_argument("--interval", type=int, default=15, help="decision interval seconds")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "realcluster", "realcluster_rl.csv"))
    args = ap.parse_args()

    engine = InferenceEngine()
    print(f"RL controller using {engine.algorithm} champion on '{DEPLOYMENT}'")

    start = time.perf_counter()
    rows = []
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= args.duration:
            break

        replicas = get_replicas()
        avg_cpu_m = get_avg_cpu_millicores()
        cpu_util = min(avg_cpu_m / 1000.0, 1.0)              # fraction of 1 core
        queue_proxy = cpu_util * KubernetesEnv.QUEUE_SCALE   # proxy (no real queue)
        day_progress = min(elapsed / args.duration, 1.0)

        decision = engine.decide(cpu_util=cpu_util, pods=max(replicas, 1),
                                 queue_depth=queue_proxy, day_progress=day_progress)
        delta = {0: -1, 1: 0, 2: +1}[decision["action"]]
        new_replicas = scale_to(replicas + delta)

        rows.append(dict(t=round(elapsed, 1), replicas=replicas,
                         avg_cpu_m=round(avg_cpu_m, 1), cpu_util=round(cpu_util, 3),
                         action=decision["action_name"], new_replicas=new_replicas))
        print(f"t={elapsed:6.1f}s | pods={replicas} cpu={avg_cpu_m:6.1f}m "
              f"({cpu_util:.2f}) | {decision['action_name']:10s} -> {new_replicas}")
        time.sleep(args.interval)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\n✅ logged {len(rows)} ticks -> {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
