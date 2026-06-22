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

# Load the champion once when the service starts, and reuse it for the live sim.
engine = InferenceEngine()
sim = SimulationEngine(engine=engine)


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
