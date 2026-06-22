"""
test_environment.py — UNIT tests for the KubernetesEnv simulator.

Strategy: unit testing. Verifies the core mechanics the whole project rests on —
observation shape/bounds, the action effects, pod limits, reward sign, the
right-sizing reference, and that episodes are the fixed trace length. These are
deterministic, so they pin the environment's contract.
"""

import numpy as np
import pytest

from environment.custom_env import KubernetesEnv


@pytest.fixture
def env(test_traces):
    # single trace -> fully deterministic episode
    return KubernetesEnv(trace_paths=[test_traces[0]])


def test_observation_shape_and_bounds(env):
    obs, info = env.reset()
    assert obs.shape == (4,)
    assert obs.dtype == np.float32
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0), "observation must be in [0,1]"


def test_action_space_is_three_discrete(env):
    assert env.action_space.n == 3


def test_scale_up_adds_one_pod(env):
    env.reset()
    start = env.pod_count
    env.step(2)  # scale up
    assert env.pod_count == start + 1


def test_scale_down_removes_one_pod(env):
    env.reset()
    start = env.pod_count
    env.step(0)  # scale down
    assert env.pod_count == start - 1


def test_maintain_keeps_pod_count(env):
    env.reset()
    start = env.pod_count
    env.step(1)  # maintain
    assert env.pod_count == start


def test_pods_never_exceed_max(env):
    env.reset()
    for _ in range(KubernetesEnv.MAX_PODS + 10):
        env.step(2)  # keep scaling up
    assert env.pod_count == KubernetesEnv.MAX_PODS


def test_pods_never_below_min(env):
    env.reset()
    for _ in range(KubernetesEnv.MAX_PODS + 10):
        env.step(0)  # keep scaling down
    assert env.pod_count == KubernetesEnv.MIN_PODS


def test_reward_is_negative(env):
    """Reward is a sum of penalties, so every step's reward is <= 0."""
    env.reset()
    _, reward, _, _, _ = env.step(1)
    assert reward <= 0.0


def test_more_pods_costs_more_energy(env):
    """At identical latency, holding more pods must score worse (energy penalty).

    Latency is fixed so the only difference is the absolute pod count — this
    isolates the energy term that fixed the over-provisioning bug.
    """
    env.reset()
    env.latency = 0.5          # fix service quality so only energy varies
    env.pod_count = 4
    r_few = env._calculate_reward(action=1)
    env.pod_count = 16
    r_many = env._calculate_reward(action=1)
    assert r_many < r_few, "running more idle pods should be penalised more"


def test_required_pods_within_bounds(env):
    env.reset()
    req = env._required_pods()
    assert KubernetesEnv.MIN_PODS <= req <= KubernetesEnv.MAX_PODS


def test_episode_runs_full_trace_length(env):
    obs, _ = env.reset()
    steps = 0
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(1)
        done = term or trunc
        steps += 1
    assert steps == env.max_steps


def test_info_dict_has_expected_keys(env):
    env.reset()
    _, _, _, _, info = env.step(1)
    for key in ["latency", "pods", "required_pods", "wasted_pods", "cpu_util"]:
        assert key in info


@pytest.mark.parametrize("trace_idx", [0, 1, 2, 3, 4])
def test_runs_on_every_test_trace(test_traces, trace_idx):
    """DIFFERENT DATA VALUES: the env runs cleanly on each held-out trace."""
    env = KubernetesEnv(trace_paths=[test_traces[trace_idx]])
    obs, _ = env.reset()
    done = False
    while not done:
        _, r, term, trunc, _ = env.step(env.action_space.sample())
        assert np.isfinite(r)
        done = term or trunc
