"""
Sample CPU-bound web service for the real-cluster validation.

Each request to /work burns a fixed slice of CPU, so more concurrent traffic
raises pod CPU utilization — which is exactly what both the Kubernetes HPA and
the Eco-Scale RL controller scale on. One worker per pod, so adding pods adds
real capacity (mirrors the simulator's "one pod serves so much load" idea).
"""

import math
import time

from fastapi import FastAPI

app = FastAPI(title="eco-sample-app")

WORK_SECONDS = 0.05   # CPU burned per request (~50 ms)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/work")
def work():
    """Burn ~50 ms of CPU so load translates into measurable utilization."""
    deadline = time.perf_counter() + WORK_SECONDS
    x = 0.0
    while time.perf_counter() < deadline:
        x += math.sqrt(12345.678)
    return {"done": True, "x": x}
