"""
benchmark_performance.py — Performance of the product across environments.

Reports the runtime ENVIRONMENT (so the same run on a laptop, in Docker, and in
the Kubernetes cluster gives the "different hardware/software specs") and measures the serving agent's real performance:

  * champion model load time
  * single-decision latency (p50 / p95, milliseconds)
  * decision throughput (decisions/second)

Run:  python tests/benchmark_performance.py
"""

import os
import sys
import time
import platform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def describe_environment():
    """Auto-detected hardware/software specs of wherever this runs."""
    in_docker = os.path.exists("/.dockerenv")
    in_k8s = bool(os.environ.get("KUBERNETES_SERVICE_HOST"))
    where = "Kubernetes pod" if in_k8s else "Docker container" if in_docker else "Local host"
    return {
        "environment": where,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def percentile(values, p):
    s = sorted(values)
    k = int(round((p / 100.0) * (len(s) - 1)))
    return s[k]


def main():
    env = describe_environment()
    print("=" * 56)
    print("ECO-SCALE — PERFORMANCE BENCHMARK")
    print("=" * 56)
    for k, v in env.items():
        print(f"  {k:12s}: {v}")
    print("-" * 56)

    # --- model load time ---
    from serving.inference_engine import InferenceEngine
    t0 = time.perf_counter()
    engine = InferenceEngine()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  agent         : {engine.algorithm}")
    print(f"  model load    : {load_ms:.0f} ms")

    # --- warm up (first prediction compiles graphs / caches) ---
    for _ in range(20):
        engine.decide(cpu_util=0.5, pods=5, queue_depth=500, day_progress=0.5)

    # --- latency + throughput over many decisions, varied inputs ---
    N = 2000
    latencies = []
    t0 = time.perf_counter()
    for i in range(N):
        cpu = (i % 100) / 100.0           # sweep load 0..1 (different data values)
        s = time.perf_counter()
        engine.decide(cpu_util=cpu, pods=1 + (i % 20),
                      queue_depth=int(cpu * 1000), day_progress=(i % 288) / 288.0)
        latencies.append((time.perf_counter() - s) * 1000)
    total_s = time.perf_counter() - t0

    print(f"  decisions     : {N}")
    print(f"  latency p50   : {percentile(latencies, 50):.3f} ms")
    print(f"  latency p95   : {percentile(latencies, 95):.3f} ms")
    print(f"  throughput    : {N / total_s:,.0f} decisions/sec")
    print("=" * 56)
    print("Tip: run this locally AND inside the Docker image to compare specs.")


if __name__ == "__main__":
    main()
