"""
experiment.py — Run the REAL Stage-2 comparison from the web console.

Drives the actual head-to-head on the live cluster, in two sequential phases over
the SAME traffic wave (they can't run at once — two autoscalers would fight):

  1. RL phase  — the champion agent scales the deployment (kubectl scale).
  2. HPA phase — the REAL native Kubernetes HPA (deploy/k8s/hpa.yaml) scales it.

Everything is real: real pods, real CPU (metrics-server), real HPA, real request
latency from the load generator. Runs in a background thread and reports live
progress so the UI can animate it, then a final summary.
"""

import os
import time
import threading

from deploy.controller.rl_controller import (
    get_replicas, get_avg_cpu_millicores, scale_to, DEPLOYMENT, sh,
)
from serving.live_cluster import real_cpu_util, get_cpu_request_millicores

ROOT = os.path.join(os.path.dirname(__file__), "..")
HPA_YAML = os.path.join(ROOT, "deploy", "k8s", "hpa.yaml")


class ExperimentRunner:
    """Runs the real RL-vs-HPA Stage-2 experiment as a background job."""

    def __init__(self, engine, loadgen):
        self.engine = engine
        self.loadgen = loadgen
        self._request_m = get_cpu_request_millicores()         # calibration: pod CPU request
        self.state = "idle"           # idle | running | done | error
        self.phase = None             # rl | hpa | null
        self.duration = 0
        self._t0 = None
        self.series = {"rl": [], "hpa": []}
        self.summary = None
        self.message = ""

    # ── control ──────────────────────────────────────────────────
    def start(self, duration=120, interval=10):
        if self.state == "running":
            return self.status()
        self.duration = max(40, int(duration))
        self.series = {"rl": [], "hpa": []}
        self.summary = None
        self.message = ""
        self.state = "running"
        threading.Thread(target=self._run, args=(self.duration, interval),
                         daemon=True).start()
        return self.status()

    def stop(self):
        self.state = "idle"
        self.phase = None
        self.loadgen.stop()
        sh(["kubectl", "delete", "hpa", DEPLOYMENT, "--ignore-not-found"])
        sh(["kubectl", "scale", "deployment", DEPLOYMENT, "--replicas=1"])
        return self.status()

    # ── phases ───────────────────────────────────────────────────
    def _phase(self, mode, duration, interval):
        self.phase = mode
        self._t0 = time.perf_counter()
        # clean start for this phase
        self.loadgen.stop()
        sh(["kubectl", "scale", "deployment", DEPLOYMENT, "--replicas=1"])
        if mode == "hpa":
            sh(["kubectl", "apply", "-f", HPA_YAML])           # the REAL native HPA
        else:
            sh(["kubectl", "delete", "hpa", DEPLOYMENT, "--ignore-not-found"])
        time.sleep(5)

        self.loadgen.start(duration)
        start = time.perf_counter()
        while self.state == "running":
            elapsed = time.perf_counter() - start
            if elapsed >= duration:
                break
            replicas = max(get_replicas(), 1)
            cpu_m = get_avg_cpu_millicores()
            ls = self.loadgen.status()
            if mode == "rl":
                cu = real_cpu_util(cpu_m, replicas, self._request_m)
                d = self.engine.decide(cpu_util=cu, pods=replicas,
                                       queue_depth=cu * 1000.0,
                                       day_progress=elapsed / duration)
                scale_to(replicas + {0: -1, 1: 0, 2: +1}[d["action"]])
            self.series[mode].append({
                "t": round(elapsed, 1),
                "intensity": ls["intensity"],
                "replicas": replicas,
                "cpu_m": round(cpu_m, 1),
                "p95_ms": ls["p95_ms"],
            })
            time.sleep(interval)
        self.loadgen.stop()

    def _run(self, duration, interval):
        try:
            self._phase("rl", duration, interval)
            if self.state != "running":
                return
            self._phase("hpa", duration, interval)
            sh(["kubectl", "delete", "hpa", DEPLOYMENT, "--ignore-not-found"])
            sh(["kubectl", "scale", "deployment", DEPLOYMENT, "--replicas=1"])
            self.summary = self._summarize()
            self.state = "done"
            self.phase = None
        except Exception as e:                                  # noqa: BLE001
            self.state = "error"
            self.message = str(e)
            self.loadgen.stop()

    def _summarize(self):
        def agg(s):
            if not s:
                return {}
            pods = [x["replicas"] for x in s]
            p95s = [x["p95_ms"] for x in s if x["p95_ms"] > 0]
            return {
                "avg_pods": round(sum(pods) / len(pods), 2),
                "max_pods": max(pods),
                "p95_ms": round(sum(p95s) / len(p95s), 0) if p95s else 0,
            }
        rl, hpa = agg(self.series["rl"]), agg(self.series["hpa"])
        verdict = None
        if rl and hpa:
            pod_diff = 100 * (hpa["avg_pods"] - rl["avg_pods"]) / hpa["avg_pods"]
            verdict = {
                "pod_saving_pct": round(pod_diff, 0),
                "rl_leaner": rl["avg_pods"] < hpa["avg_pods"],
            }
        return {"rl": rl, "hpa": hpa, "verdict": verdict}

    def status(self):
        elapsed = (time.perf_counter() - self._t0) if (self.state == "running" and self._t0) else 0.0
        return {
            "state": self.state,
            "phase": self.phase,
            "elapsed": round(elapsed, 1),
            "duration": self.duration,
            "rl": self.series["rl"],
            "hpa": self.series["hpa"],
            "summary": self.summary,
            "message": self.message,
        }
