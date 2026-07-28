"""
api.py — FastAPI service that serves scaling decisions from the champion agent.

Endpoints are grouped into sections:
  1. Core API        — /info, /health, /config, /predict
  2. Simulation      — /sim/reset, /sim/step
  3. Live cluster    — /live/*, /contexts/*
  4. Load generator  — /live/load/*
  5. Experiment      — /experiment/*
  6. Results & Model — /results, /model

Run locally:
    uvicorn serving.api:app --reload
Then open http://127.0.0.1:8000/docs for the interactive API.
"""

import os
import sys

# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyright: ignore [reportMissingImports]
from fastapi.middleware.cors import CORSMiddleware
# pyright: ignore [reportMissingImports]
from fastapi.staticfiles import StaticFiles
# pyright: ignore [reportMissingImports]
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from environment.custom_env import KubernetesEnv                       # noqa: E402
from serving.experiment import ExperimentRunner                        # noqa: E402
from serving.inference_engine import InferenceEngine                    # noqa: E402
from serving.live_cluster import (                                     # noqa: E402
    LiveClusterEngine, cluster_available, cluster_info,
    kube_contexts, use_context,
)
from serving.load_generator import LoadGenerator                       # noqa: E402
from serving.simulation import SimulationEngine                        # noqa: E402

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Eco-Scale Autoscaler",
    description="Reinforcement-learning scaling decisions for Kubernetes.",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared state (created once at startup, reused by all endpoints)
# ---------------------------------------------------------------------------
engine     = InferenceEngine()
sim        = SimulationEngine(engine=engine)
live       = None                                   # created lazily on first /live call
loadgen    = LoadGenerator()
experiment = ExperimentRunner(engine=engine, loadgen=loadgen)


# ═══════════════════════════════════════════════════════════════════════
# 1. CORE API — service info + single-shot predictions
# ═══════════════════════════════════════════════════════════════════════

class ClusterState(BaseModel):
    """Current cluster metrics (all the agent needs to decide)."""
    cpu_util:     float = Field(..., ge=0, le=1, description="CPU utilization, 0..1")
    pods:         int   = Field(..., ge=1,       description="Current number of running pods")
    queue_depth:  int   = Field(..., ge=0,       description="Pending request queue length")
    day_progress: float = Field(..., ge=0, le=1, description="Fraction of the daily cycle, 0..1")


@app.get("/info")
def info():
    """Basic service metadata."""
    return {
        "service": "Eco-Scale Autoscaler",
        "agent":   engine.algorithm,
        "run":     engine.metadata.get("run"),
        "actions": {0: "scale_down", 1: "maintain", 2: "scale_up"},
    }


@app.get("/health")
def health():
    """Liveness probe."""
    return {"status": "ok", "agent": engine.algorithm}


@app.get("/config")
def config():
    """Environment constants + cost assumptions the UI needs."""
    return {
        "min_pods":     KubernetesEnv.MIN_PODS,
        "max_pods":     KubernetesEnv.MAX_PODS,
        "pod_capacity": KubernetesEnv.POD_CAPACITY,
        "util_target":  KubernetesEnv.UTIL_TARGET,
        "agent":        engine.algorithm,
        "run":          engine.metadata.get("run"),
    }


@app.post("/predict")
def predict(state: ClusterState):
    """Return the recommended scaling action for the given cluster state."""
    decision = engine.decide(
        cpu_util=state.cpu_util,
        pods=state.pods,
        queue_depth=state.queue_depth,
        day_progress=state.day_progress,
    )
    return {"input": state.model_dump(), **decision}


# ═══════════════════════════════════════════════════════════════════════
# 2. SIMULATION — replay a held-out trace: RL vs HPA side-by-side
# ═══════════════════════════════════════════════════════════════════════

class SimConfig(BaseModel):
    hpa_target: float = Field(0.5, ge=0.3, le=0.95,
                              description="HPA target utilization to compare against")


@app.post("/sim/reset")
def sim_reset(cfg: SimConfig | None = None):
    """Restart the simulation; returns the initial state."""
    target = cfg.hpa_target if cfg else None
    return sim.reset(hpa_target=target)


@app.post("/sim/step")
def sim_step():
    """Advance the simulation one tick (5 simulated minutes)."""
    return sim.step()


# ═══════════════════════════════════════════════════════════════════════
# 3. LIVE CLUSTER — read the real Kubernetes cluster and (optionally) scale it
# ═══════════════════════════════════════════════════════════════════════

class LiveStep(BaseModel):
    apply: bool = Field(False, description="If true, actually kubectl-scale the deployment")


class ContextRequest(BaseModel):
    context: str


def _ensure_live() -> LiveClusterEngine | None:
    """Lazily create the live engine, returning None if no cluster is reachable."""
    global live
    if live is None:
        if not cluster_available():
            return None
        live = LiveClusterEngine(engine=engine)
        live.reset()
    return live


@app.get("/live/available")
def live_available():
    """Tell the UI whether a live cluster + target deployment is reachable."""
    return {"available": cluster_available()}


@app.post("/live/reset")
def live_reset():
    """Start a live session; reads the real cluster state (agent only, no HPA)."""
    global live
    if not cluster_available():
        return {"error": "No reachable Kubernetes cluster / deployment."}
    live = LiveClusterEngine(engine=engine)
    return live.reset()


@app.post("/live/step")
def live_step(step: LiveStep | None = None):
    """Read the live cluster, run the agent, and optionally scale it."""
    lce = _ensure_live()
    if lce is None:
        return {"error": "No reachable Kubernetes cluster / deployment."}
    return lce.step(apply=step.apply if step else False)


@app.get("/live/info")
def live_info():
    """Live cluster details (context, namespace, image, pods) for the UI."""
    if not cluster_available():
        return {"error": "No reachable Kubernetes cluster / deployment."}
    return cluster_info()


@app.get("/contexts")
def contexts():
    """Available kubectl contexts + the active one (for the cluster selector)."""
    return kube_contexts()


@app.post("/contexts/use")
def use_ctx(req: ContextRequest):
    """Switch the active kubectl context, then reset the live engine."""
    global live
    result = use_context(req.context)
    live = None
    return result


# ═══════════════════════════════════════════════════════════════════════
# 4. LOAD GENERATOR — UI-controllable traffic wave (scales nothing)
# ═══════════════════════════════════════════════════════════════════════

class LoadConfig(BaseModel):
    duration: int = Field(300, ge=20, le=1800, description="Seconds the traffic wave runs")


@app.post("/live/load/start")
def load_start(cfg: LoadConfig | None = None):
    return loadgen.start(duration=cfg.duration if cfg else 300)


@app.post("/live/load/stop")
def load_stop():
    return loadgen.stop()


@app.get("/live/load/status")
def load_status():
    return loadgen.status()


# ═══════════════════════════════════════════════════════════════════════
# 5. EXPERIMENT — real Stage-2: RL agent vs native Kubernetes HPA
# ═══════════════════════════════════════════════════════════════════════

class ExperimentConfig(BaseModel):
    duration: int = Field(120, ge=40, le=600, description="Seconds per phase")


@app.post("/experiment/start")
def experiment_start(cfg: ExperimentConfig | None = None):
    if not cluster_available():
        return {"error": "No reachable Kubernetes cluster / deployment."}
    return experiment.start(duration=cfg.duration if cfg else 120)


@app.get("/experiment/status")
def experiment_status():
    return experiment.status()


@app.post("/experiment/stop")
def experiment_stop():
    return experiment.stop()


# ═══════════════════════════════════════════════════════════════════════
# 6. RESULTS & MODEL — delegates to results_service.py
# ═══════════════════════════════════════════════════════════════════════

from serving.results_service import get_results, get_model_info        # noqa: E402


@app.get("/results")
def results():
    """Evaluation results: algorithm sweep + the live-cluster benchmark."""
    return get_results()


@app.get("/model")
def model():
    """Deployed champion details: metadata + reward design + env constants."""
    return get_model_info()


# ═══════════════════════════════════════════════════════════════════════
# 7. STATIC FILES — serve the built React console (if present)
# ═══════════════════════════════════════════════════════════════════════
# In production the frontend is built into web/dist and served by this
# same app, so the console + API live behind one URL.  API routes above
# are matched first; this static mount catches everything else (the SPA).

_DIST = os.path.join(os.path.dirname(__file__), "..", "web", "dist")
if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="console")

