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
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.inference_engine import InferenceEngine

app = FastAPI(
    title="Eco-Scale Autoscaler",
    description="Reinforcement-learning scaling decisions for Kubernetes.",
    version="1.0",
)

# Load the champion once when the service starts.
engine = InferenceEngine()


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
