"""
api.py — FastAPI service that serves scaling decisions from the champion agent.

Endpoints:
    GET  /         -> info about the deployed agent
    GET  /health   -> liveness check
    POST /predict  -> scaling decision for the current cluster state

Run locally:
    uvicorn serving.api:app --reload
Then open http://127.0.0.1:8000/docs for the interactive API.
"""

import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.inference_engine import InferenceEngine
from serving.simulation import SimulationEngine
from serving.live_cluster import LiveClusterEngine, cluster_available, cluster_info
from serving.load_generator import LoadGenerator
from serving.experiment import ExperimentRunner
from environment.custom_env import KubernetesEnv

app = FastAPI(
    title="Eco-Scale Autoscaler",
    description="Reinforcement-learning scaling decisions for Kubernetes.",
    version="1.0",
)

# Allow the React control-plane (dev server) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the champion once when the service starts, and reuse it everywhere.
engine = InferenceEngine()
sim = SimulationEngine(engine=engine)
live = None      # LiveClusterEngine is created lazily (only if a cluster is present)
loadgen = LoadGenerator()
experiment = ExperimentRunner(engine=engine, loadgen=loadgen)


class ClusterState(BaseModel):
    """Current cluster metrics (all the agent needs to decide)."""
    cpu_util: float = Field(..., ge=0, le=1, description="CPU utilization, 0..1")
    pods: int = Field(..., ge=1, description="Current number of running pods")
    queue_depth: int = Field(..., ge=0, description="Pending request queue length")
    day_progress: float = Field(..., ge=0, le=1,
                                description="Fraction of the daily cycle, 0..1")


@app.get("/")
def info():
    return {
        "service": "Eco-Scale Autoscaler",
        "agent": engine.algorithm,
        "run": engine.metadata.get("run"),
        "actions": {0: "scale_down", 1: "maintain", 2: "scale_up"},
    }


@app.get("/health")
def health():
    return {"status": "ok", "agent": engine.algorithm}


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


# ── control-plane endpoints (used by the React dashboard) ────────────
@app.get("/config")
def config():
    """Environment constants + cost assumptions the UI needs."""
    return {
        "min_pods": KubernetesEnv.MIN_PODS,
        "max_pods": KubernetesEnv.MAX_PODS,
        "pod_capacity": KubernetesEnv.POD_CAPACITY,
        "util_target": KubernetesEnv.UTIL_TARGET,
        "agent": engine.algorithm,
        "run": engine.metadata.get("run"),
    }


class SimConfig(BaseModel):
    hpa_target: float = Field(0.5, ge=0.3, le=0.95,
                              description="HPA target utilization to compare against")


@app.post("/sim/reset")
def sim_reset(cfg: SimConfig | None = None):
    """Restart the live RL-vs-HPA simulation; returns the initial state."""
    target = cfg.hpa_target if cfg else None
    return sim.reset(hpa_target=target)


@app.post("/sim/step")
def sim_step():
    """Advance the simulation one tick (5 simulated minutes)."""
    return sim.step()


# ── LIVE cluster mode (reads the real Kubernetes cluster) ────────────
class LiveStep(BaseModel):
    apply: bool = Field(False, description="If true (autopilot), actually kubectl scale the deployment")


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
    """Read the live cluster, run the agent, and (if apply) scale it."""
    global live
    if live is None:
        if not cluster_available():
            return {"error": "No reachable Kubernetes cluster / deployment."}
        live = LiveClusterEngine(engine=engine)
        live.reset()
    return live.step(apply=step.apply if step else False)


@app.get("/live/info")
def live_info():
    """Live cluster details (context, namespace, image, pods) for the UI."""
    if not cluster_available():
        return {"error": "No reachable Kubernetes cluster / deployment."}
    return cluster_info()


# ── UI-controllable load generator (traffic only; scales nothing) ────
class LoadCfg(BaseModel):
    duration: int = Field(300, ge=20, le=1800, description="seconds the traffic wave runs")


@app.post("/live/load/start")
def load_start(cfg: LoadCfg | None = None):
    return loadgen.start(duration=cfg.duration if cfg else 300)


@app.post("/live/load/stop")
def load_stop():
    return loadgen.stop()


@app.get("/live/load/status")
def load_status():
    return loadgen.status()


# ── real Stage-2 experiment (RL agent vs real native HPA) ────────────
class ExpCfg(BaseModel):
    duration: int = Field(120, ge=40, le=600, description="seconds per phase")


@app.post("/experiment/start")
def experiment_start(cfg: ExpCfg | None = None):
    if not cluster_available():
        return {"error": "No reachable Kubernetes cluster / deployment."}
    return experiment.start(duration=cfg.duration if cfg else 120)


@app.get("/experiment/status")
def experiment_status():
    return experiment.status()


@app.post("/experiment/stop")
def experiment_stop():
    return experiment.stop()
