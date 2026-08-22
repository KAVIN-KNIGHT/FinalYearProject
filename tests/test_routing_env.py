import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import pytest
import numpy as np

import satsim.envs  # Triggers environment registration
from satsim.envs.routing_env import SatelliteRoutingEnv


def test_gymnasium_env_checker():
    env = SatelliteRoutingEnv()
    # Critical Acceptance Criterion: Passes gymnasium env checker
    check_env(env.unwrapped, skip_render_check=True)


def test_random_action_rollout_1000_steps():
    """
    CRITICAL ACCEPTANCE CRITERION:
    A random-action rollout completes without crashing for at least 1000 steps.
    """
    env = SatelliteRoutingEnv()
    obs, info = env.reset(seed=42)

    total_steps = 0
    target_steps = 1000

    while total_steps < target_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1

        assert obs is not None
        assert isinstance(reward, float)

        if terminated or truncated:
            obs, info = env.reset()

    assert total_steps >= 1000


def test_independently_loggable_reward_components():
    """
    CRITICAL ACCEPTANCE CRITERION:
    Reward components (delivery, throughput, latency, congestion, loss, hop count, energy, success)
    are each independently loggable in info['reward_components'], not just summed into one opaque scalar.
    """
    env = SatelliteRoutingEnv()
    obs, info = env.reset(seed=42)

    expected_components = {
        "delivery",
        "throughput",
        "latency",
        "congestion",
        "loss",
        "hop_count",
        "energy",
        "success",
    }

    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert "reward_components" in info, "info dictionary missing 'reward_components'!"
        components = info["reward_components"]
        assert isinstance(components, dict)

        assert set(components.keys()) == expected_components

        for comp_name, comp_val in components.items():
            assert isinstance(comp_val, (float, int)), f"Component '{comp_name}' is not numeric!"
            assert not np.isnan(comp_val), f"Component '{comp_name}' is NaN!"

        # Assert total reward is sum of components
        total_sum = float(sum(components.values()))
        assert abs(reward - total_sum) < 1e-6, f"Reward ({reward}) does not match sum of components ({total_sum})"

        if terminated or truncated:
            obs, info = env.reset()
