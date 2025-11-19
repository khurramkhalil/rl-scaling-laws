"""Pytest configuration and fixtures."""

import gymnasium as gym
import numpy as np
import pytest
import torch


@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cpu")


@pytest.fixture
def seed():
    """Default seed for tests."""
    return 42


@pytest.fixture
def discrete_env():
    """Simple discrete action environment."""
    return gym.make("CartPole-v1")


@pytest.fixture
def continuous_env():
    """Simple continuous action environment."""
    return gym.make("Pendulum-v1")


@pytest.fixture
def image_env():
    """Image observation environment (mocked)."""
    # Create a simple wrapper that returns image observations
    env = gym.make("CartPole-v1")

    class ImageWrapper(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(4, 84, 84), dtype=np.uint8
            )

        def observation(self, obs):
            # Return random image for testing
            return np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)

    return ImageWrapper(env)


@pytest.fixture
def sample_observations(discrete_env):
    """Sample observations from discrete environment."""
    obs, _ = discrete_env.reset()
    return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)


@pytest.fixture
def sample_batch():
    """Sample batch for testing."""
    batch_size = 32
    obs_dim = 4
    action_dim = 2

    return {
        "observations": torch.randn(batch_size, obs_dim),
        "actions": torch.randn(batch_size, action_dim),
        "rewards": torch.randn(batch_size, 1),
        "next_observations": torch.randn(batch_size, obs_dim),
        "dones": torch.zeros(batch_size, 1),
    }
