"""
test_api.py — INTEGRATION tests for the FastAPI serving layer.

Strategy: integration testing through the real HTTP interface (FastAPI TestClient).
Exercises every endpoint end-to-end (request -> model -> JSON response) and the
input validation contract, which is what an external user/cluster actually calls.
"""

import pytest
from fastapi.testclient import TestClient

from serving.api import app

client = TestClient(app)


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["agent"] in ("PPO", "DQN")


def test_info_endpoint():
    r = client.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "Eco-Scale Autoscaler"
    assert "agent" in body


def test_predict_returns_decision():
    r = client.post("/predict", json={
        "cpu_util": 0.9, "pods": 3, "queue_depth": 900, "day_progress": 0.5,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["action"] in (0, 1, 2)
    assert body["action_name"] in ("scale_down", "maintain", "scale_up")
    assert body["input"]["pods"] == 3


def test_predict_rejects_out_of_range_cpu():
    """Validation: cpu_util must be 0..1 (HTTP 422 otherwise)."""
    r = client.post("/predict", json={
        "cpu_util": 1.5, "pods": 3, "queue_depth": 900, "day_progress": 0.5,
    })
    assert r.status_code == 422


def test_predict_rejects_zero_pods():
    """Validation: a cluster always has >= 1 pod."""
    r = client.post("/predict", json={
        "cpu_util": 0.5, "pods": 0, "queue_depth": 100, "day_progress": 0.5,
    })
    assert r.status_code == 422


def test_predict_rejects_missing_field():
    r = client.post("/predict", json={"cpu_util": 0.5, "pods": 3})
    assert r.status_code == 422


@pytest.mark.parametrize("cpu,pods", [(0.1, 18), (0.5, 8), (0.95, 2)])
def test_predict_across_data_values(cpu, pods):
    """DIFFERENT DATA VALUES: a spread of cluster states all return valid actions."""
    r = client.post("/predict", json={
        "cpu_util": cpu, "pods": pods, "queue_depth": int(cpu * 1000),
        "day_progress": 0.5,
    })
    assert r.status_code == 200
    assert r.json()["action"] in (0, 1, 2)
