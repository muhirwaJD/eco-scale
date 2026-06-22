"""
test_hpa_controller.py — UNIT tests for the HPA baseline policy.

Strategy: unit testing. The HPA controller is the comparison baseline, so its
logic must be correct and trustworthy: it scales UP immediately under high load,
scales DOWN only after a stabilization window, keeps replicas in bounds, and
exposes the SB3-compatible predict() signature used by the evaluation harness.
"""

import numpy as np
import pytest

from baselines.hpa_controller import HPAController
from environment.custom_env import KubernetesEnv

MAX = KubernetesEnv.MAX_PODS


def obs_for(cpu_util, pods, queue=0.0, day=0.5):
    """Build an observation in the env's normalized format."""
    return np.array([cpu_util, pods / MAX, queue, day], dtype=np.float32)


def test_predict_returns_sb3_signature():
    hpa = HPAController()
    action, state = hpa.predict(obs_for(0.5, 5))
    assert action in (0, 1, 2)
    assert state is None


def test_scales_up_immediately_under_high_load():
    hpa = HPAController(target_util=0.7)
    hpa.reset()
    # very high utilization with few pods -> must scale up at once
    action, _ = hpa.predict(obs_for(cpu_util=0.9, pods=2))
    assert action == 2


def test_scale_down_waits_for_stabilization():
    hpa = HPAController(target_util=0.7, stabilization_steps=3)
    hpa.reset()
    low = obs_for(cpu_util=0.05, pods=15)  # hugely over-provisioned
    # first two under-target ticks: hold (stabilization not yet satisfied)
    assert hpa.predict(low)[0] == 1
    assert hpa.predict(low)[0] == 1
    # third tick: now allowed to scale down
    assert hpa.predict(low)[0] == 0


def test_maintains_when_at_target():
    hpa = HPAController(target_util=0.7)
    hpa.reset()
    # pick load so desired == current pods
    pods = 7
    cpu = 0.7 * KubernetesEnv.POD_CAPACITY * pods  # util lands exactly on target
    action, _ = hpa.predict(obs_for(cpu_util=cpu, pods=pods))
    assert action == 1


def test_desired_pods_respects_bounds():
    hpa = HPAController()
    assert hpa._desired_pods(0.0) >= KubernetesEnv.MIN_PODS
    assert hpa._desired_pods(5.0) <= KubernetesEnv.MAX_PODS  # absurd load clamps to MAX


@pytest.mark.parametrize("target", [0.5, 0.6, 0.7, 0.8, 0.9])
def test_lower_target_provisions_more_pods(target):
    """A more conservative target must never want FEWER pods than an aggressive one."""
    hpa_safe = HPAController(target_util=0.5)
    hpa_aggr = HPAController(target_util=0.9)
    cpu = 0.4
    assert hpa_safe._desired_pods(cpu) >= hpa_aggr._desired_pods(cpu)


def test_beats_random_on_a_real_trace(test_traces):
    """SYSTEM check: over a full real trace, HPA must clearly beat a random policy."""
    trace = test_traces[1]
    env = KubernetesEnv(trace_paths=[trace])
    hpa = HPAController()

    # HPA episode
    obs, _ = env.reset(); hpa.reset()
    r_hpa, done = 0.0, False
    while not done:
        a, _ = hpa.predict(obs)
        obs, r, term, trunc, _ = env.step(a)
        r_hpa += r; done = term or trunc

    # random episode (same trace)
    env2 = KubernetesEnv(trace_paths=[trace])
    obs, _ = env2.reset()
    r_rand, done = 0.0, False
    while not done:
        obs, r, term, trunc, _ = env2.step(env2.action_space.sample())
        r_rand += r; done = term or trunc

    assert r_hpa > r_rand + 50, f"HPA ({r_hpa:.0f}) should clearly beat random ({r_rand:.0f})"
