"""
results_service.py — Read pre-computed evaluation outputs and champion metadata.

Keeps the data-loading logic out of api.py so the API file stays a thin
routing layer.
"""

import csv
import json
import os

from environment.custom_env import KubernetesEnv

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _read_csv(path: str) -> list[dict]:
    """Read a CSV file into a list of dicts; returns [] if the file is missing."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def get_results() -> dict:
    """Evaluation results: algorithm sweep + the live-cluster benchmark."""

    # Best run per algorithm (from the training sweeps)
    algos = []
    for algo, fname in [("PPO", "ppo_results.csv"),
                        ("DQN", "dqn_results.csv"),
                        ("REINFORCE", "reinforce_results.csv")]:
        rows = _read_csv(os.path.join(ROOT, "outputs", "training", fname))
        rewards = [float(r["Mean Reward"]) for r in rows if r.get("Mean Reward")]
        if rewards:
            algos.append({
                "algorithm":   algo,
                "runs":        len(rewards),
                "best_reward": round(max(rewards), 2),
                "mean_reward": round(sum(rewards) / len(rewards), 2),
            })

    # Real-cluster benchmark (3-run mean ± std)
    rc = {
        r["metric"]: r
        for r in _read_csv(os.path.join(ROOT, "outputs", "realcluster", "realcluster_repeats.csv"))
    }
    realcluster = None
    if "avg_pods" in rc:
        rl_pods  = float(rc["avg_pods"]["RL_mean"])
        hpa_pods = float(rc["avg_pods"]["HPA_mean"])
        realcluster = {
            "rl_pods":        rl_pods,
            "rl_pods_sd":     float(rc["avg_pods"]["RL_std"]),
            "hpa_pods":       hpa_pods,
            "hpa_pods_sd":    float(rc["avg_pods"]["HPA_std"]),
            "rl_p95":         float(rc.get("p95_ms", {}).get("RL_mean", 0)),
            "hpa_p95":        float(rc.get("p95_ms", {}).get("HPA_mean", 0)),
            "pod_saving_pct": round(100 * (hpa_pods - rl_pods) / hpa_pods) if hpa_pods else 0,
        }

    return {"algorithms": algos, "realcluster": realcluster}


def get_model_info() -> dict:
    """Deployed champion details: metadata + reward design + env constants."""
    meta = {}
    meta_path = os.path.join(ROOT, "models", "champion_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    E = KubernetesEnv
    return {
        "metadata": meta,
        "reward": {
            "latency":     E.W_LAT,
            "energy":      E.W_ENERGY,
            "sla_breach":  E.W_SLA,
            "scaling":     E.W_SCALE,
            "util_target": E.UTIL_TARGET,
        },
        "env": {
            "min_pods":     E.MIN_PODS,
            "max_pods":     E.MAX_PODS,
            "start_pods":   E.START_PODS,
            "pod_capacity": E.POD_CAPACITY,
        },
    }
