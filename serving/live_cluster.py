"""
live_cluster.py — Drive a REAL Kubernetes deployment from the web console.

Same idea as the simulation, but the input is the LIVE cluster instead of a
recorded trace: each step reads the real replica count + average pod CPU
(metrics-server), maps them to the agent's 4-D observation, asks the champion for
an action, and (in autopilot) applies it with `kubectl scale`.

Live mode shows ONLY the agent on the real cluster — there is no HPA line here.
A real RL-vs-HPA comparison can't run on the same deployment simultaneously (two
autoscalers would fight), so that comparison lives in Simulation mode and in the
sequential Stage-2 experiment (deploy/run_experiment.py --mode hpa | rl).
"""

import subprocess

from environment.custom_env import KubernetesEnv
from serving.inference_engine import InferenceEngine

DEPLOYMENT = "eco-sample-app"
MIN_PODS, MAX_PODS = 1, 10           # cluster scaling bounds
DAY_TICKS = 288                      # one "day" cycle for the day-progress feature

ACTION_NAMES = {0: "scale_down", 1: "maintain", 2: "scale_up"}


def _sh(cmd):
    # Returns "" if kubectl is missing or the call fails (e.g. no cluster, or the
    # selected context is unreachable) so the UI degrades cleanly. The timeout
    # stops an unreachable context (e.g. a cloud cluster) from hanging the API.
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=8).stdout.strip()
    except Exception:
        return ""


def kube_contexts():
    """List the real kubectl contexts + the current one."""
    out = _sh(["kubectl", "config", "get-contexts", "-o", "name"])
    names = [l.strip() for l in out.splitlines() if l.strip()]
    return {"current": _sh(["kubectl", "config", "current-context"]), "contexts": names}


def use_context(name):
    """Switch the active kubectl context (only to a known context)."""
    if name not in kube_contexts()["contexts"]:
        return {"ok": False, "error": "unknown context"}
    _sh(["kubectl", "config", "use-context", name])
    return {"ok": True, "current": _sh(["kubectl", "config", "current-context"])}


def cluster_available():
    """True if kubectl can see the target deployment (so the UI can offer live mode)."""
    out = _sh(["kubectl", "get", "deployment", DEPLOYMENT, "-o", "name"])
    return out.endswith(DEPLOYMENT)


def get_cpu_request_millicores():
    """Read the pod's CPU request (e.g. '200m') from the deployment. Default 200."""
    out = _sh(["kubectl", "get", "deployment", DEPLOYMENT, "-o",
               "jsonpath={.spec.template.spec.containers[0].resources.requests.cpu}"])
    if out.endswith("m") and out[:-1].isdigit():
        return float(out[:-1])
    try:
        return float(out) * 1000.0          # e.g. "1" core -> 1000m
    except ValueError:
        return 200.0


def real_cpu_util(avg_cpu_m, replicas, request_m):
    """Convert REAL cluster CPU into the [0,1] load value the agent trained on.

    The agent learned on TOTAL normalized demand, where one pod serves
    POD_CAPACITY of it. So we use the total CPU across all pods
    (avg_cpu_m * replicas), express it in units of the pod's CPU request, and
    rescale by POD_CAPACITY. This makes the agent's "pods needed" match reality
    (e.g. an idle cluster maps to ~0, so the agent stops over-provisioning).
    """
    total_m = avg_cpu_m * max(replicas, 1)
    cu = KubernetesEnv.POD_CAPACITY * total_m / max(request_m, 1.0)
    return float(min(max(cu, 0.0), 1.0))


def cluster_info():
    """Live cluster details for the UI (so nobody needs the terminal)."""
    context = _sh(["kubectl", "config", "current-context"]) or "unknown"
    ns = _sh(["kubectl", "get", "deployment", DEPLOYMENT,
              "-o", "jsonpath={.metadata.namespace}"]) or "default"
    image = _sh(["kubectl", "get", "deployment", DEPLOYMENT,
                 "-o", "jsonpath={.spec.template.spec.containers[0].image}"])
    replicas = _sh(["kubectl", "get", "deployment", DEPLOYMENT,
                    "-o", "jsonpath={.status.readyReplicas}"])
    hpa = _sh(["kubectl", "get", "hpa", DEPLOYMENT, "-o", "name"])

    pods = []
    top = _sh(["kubectl", "top", "pods", "-l", f"app={DEPLOYMENT}", "--no-headers"])
    cpu_by_pod = {}
    for line in top.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            cpu_by_pod[parts[0]] = parts[1]
    listing = _sh(["kubectl", "get", "pods", "-l", f"app={DEPLOYMENT}",
                   "-o", "jsonpath={range .items[*]}{.metadata.name} {.status.phase}{\"\\n\"}{end}"])
    for line in listing.splitlines():
        parts = line.split()
        if parts:
            pods.append({"name": parts[0],
                         "phase": parts[1] if len(parts) > 1 else "?",
                         "cpu": cpu_by_pod.get(parts[0], "—")})
    return {
        "context": context,
        "namespace": ns,
        "deployment": DEPLOYMENT,
        "image": image or "—",
        "replicas": int(replicas) if replicas.isdigit() else 0,
        "native_hpa": bool(hpa),
        "min_pods": MIN_PODS,
        "max_pods": MAX_PODS,
        "pods": pods,
    }


def _rationale(action, cpu_util, pods):
    pct = int(round(cpu_util * 100))
    if action == 2:
        return f"Load is high (CPU {pct}%) — adding a pod to protect the SLA."
    if action == 0:
        return f"Load is low (CPU {pct}%) for {pods} pods — removing one to save energy."
    return f"Capacity matches demand (CPU {pct}%) — holding steady."


class LiveClusterEngine:
    """Reads the live cluster, runs the champion, and (optionally) scales it."""

    def __init__(self, engine=None):
        self.engine = engine or InferenceEngine()
        self.request_m = get_cpu_request_millicores()      # calibration: pod CPU request
        self.reset()

    # ── cluster IO ───────────────────────────────────────────────
    def _get_replicas(self):
        out = _sh(["kubectl", "get", "deployment", DEPLOYMENT,
                   "-o", "jsonpath={.status.readyReplicas}"])
        return int(out) if out.isdigit() else 0

    def _get_avg_cpu_millicores(self):
        out = _sh(["kubectl", "top", "pods", "-l", f"app={DEPLOYMENT}", "--no-headers"])
        cpus = []
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].endswith("m"):
                cpus.append(float(parts[1][:-1]))
        return sum(cpus) / len(cpus) if cpus else 0.0

    def _scale_to(self, n):
        n = max(MIN_PODS, min(MAX_PODS, n))
        _sh(["kubectl", "scale", "deployment", DEPLOYMENT, f"--replicas={n}"])
        return n

    # ── lifecycle ────────────────────────────────────────────────
    def reset(self):
        self.tick = 0
        self._actions = {"up": 0, "hold": 0, "down": 0}
        self._peak_pods = max(self._get_replicas(), MIN_PODS)
        self._replicas = self._peak_pods
        self._avg_cpu_m = self._get_avg_cpu_millicores()
        self._cpu = real_cpu_util(self._avg_cpu_m, self._replicas, self.request_m)
        self._rl = (1, None, "maintain")
        return self._build(applied=False)

    # ── one live step ────────────────────────────────────────────
    def step(self, apply=False):
        replicas = max(self._get_replicas(), MIN_PODS)
        avg_cpu_m = self._get_avg_cpu_millicores()
        cpu_util = real_cpu_util(avg_cpu_m, replicas, self.request_m)
        queue_proxy = cpu_util * KubernetesEnv.QUEUE_SCALE
        day_progress = (self.tick % DAY_TICKS) / DAY_TICKS

        decision = self.engine.decide(cpu_util=cpu_util, pods=replicas,
                                      queue_depth=queue_proxy, day_progress=day_progress)
        action = decision["action"]
        probs = self._probs([cpu_util, replicas / KubernetesEnv.MAX_PODS,
                             min(queue_proxy / KubernetesEnv.QUEUE_SCALE, 1.0), day_progress])

        applied_to = replicas
        if apply:
            applied_to = self._scale_to(replicas + {0: -1, 1: 0, 2: +1}[action])

        self.tick += 1
        self._actions[{0: "down", 1: "hold", 2: "up"}[action]] += 1
        self._replicas = applied_to if apply else replicas
        self._peak_pods = max(self._peak_pods, self._replicas)
        self._cpu = cpu_util
        self._avg_cpu_m = avg_cpu_m
        self._rl = (action, probs, decision["action_name"])
        return self._build(applied=apply)

    def _probs(self, obs):
        try:
            import numpy as np
            tensor, _ = self.engine.model.policy.obs_to_tensor(np.array(obs, dtype=np.float32))
            dist = self.engine.model.policy.get_distribution(tensor)
            return [float(p) for p in dist.distribution.probs.detach().numpy().ravel()]
        except Exception:
            return None

    def _build(self, applied):
        action, probs, name = self._rl
        return {
            "mode": "live",
            "tick": self.tick,
            "max_ticks": DAY_TICKS,
            "day_progress": round((self.tick % DAY_TICKS) / DAY_TICKS, 3),
            "cpu": round(self._cpu, 4),
            "avg_cpu_millicores": round(self._avg_cpu_m, 1),
            "applied": applied,
            "rl": {
                "pods": self._replicas,
                "action": action,
                "action_name": name,
                "observation": [round(self._cpu, 4),
                                round(self._replicas / KubernetesEnv.MAX_PODS, 4),
                                round(min(self._cpu, 1.0), 4),
                                round((self.tick % DAY_TICKS) / DAY_TICKS, 4)],
                "probs": probs,
                "rationale": _rationale(action, self._cpu, self._replicas),
            },
            "stats": {
                "replicas": self._replicas,
                "peak_pods": self._peak_pods,
                "scaling_actions": self._actions["up"] + self._actions["down"],
                "up": self._actions["up"],
                "down": self._actions["down"],
                "hold": self._actions["hold"],
            },
            "done": False,
        }
