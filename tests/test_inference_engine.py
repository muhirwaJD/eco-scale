"""
test_inference_engine.py — UNIT tests for the serving inference engine.

Strategy: unit testing of the deployed model wrapper. Confirms the champion
loads, the observation is normalized exactly like the training env, and the
agent makes sensible decisions on clear-cut inputs (different data values).
"""

import pytest

from serving.inference_engine import InferenceEngine
from environment.custom_env import KubernetesEnv


@pytest.fixture(scope="module")
def engine():
    return InferenceEngine()


def test_champion_loads_with_known_algorithm(engine):
    assert engine.algorithm in ("PPO", "DQN")
    assert engine.model is not None


def test_observation_normalization_matches_env(engine):
    """The engine must normalize raw metrics the same way KubernetesEnv does."""
    obs = engine._build_observation(cpu_util=0.5, pods=10, queue_depth=500,
                                    day_progress=0.5)
    assert obs[0] == pytest.approx(0.5)
    assert obs[1] == pytest.approx(10 / KubernetesEnv.MAX_PODS)
    assert obs[2] == pytest.approx(min(500 / KubernetesEnv.QUEUE_SCALE, 1.0))
    assert obs[3] == pytest.approx(0.5)
    assert all(0.0 <= v <= 1.0 for v in obs)


def test_decide_returns_valid_action(engine):
    out = engine.decide(cpu_util=0.5, pods=5, queue_depth=500, day_progress=0.5)
    assert out["action"] in (0, 1, 2)
    assert out["action_name"] in ("scale_down", "maintain", "scale_up")


def test_high_load_few_pods_does_not_scale_down(engine):
    """DIFFERENT DATA VALUES: saturated cluster must never recommend scaling down."""
    out = engine.decide(cpu_util=0.95, pods=2, queue_depth=950, day_progress=0.5)
    assert out["action_name"] != "scale_down"


def test_low_load_many_pods_does_not_scale_up(engine):
    """DIFFERENT DATA VALUES: idle over-provisioned cluster must never scale up."""
    out = engine.decide(cpu_util=0.1, pods=18, queue_depth=100, day_progress=0.5)
    assert out["action_name"] != "scale_up"


def test_decisions_are_deterministic(engine):
    """Same input -> same decision (deterministic=True serving)."""
    a = engine.decide(cpu_util=0.6, pods=7, queue_depth=600, day_progress=0.3)
    b = engine.decide(cpu_util=0.6, pods=7, queue_depth=600, day_progress=0.3)
    assert a == b
