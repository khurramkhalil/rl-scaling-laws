"""Environment wrappers and utilities for RL domains."""

from rl_scaling_laws.environments.factory import make_env, make_vec_env
from rl_scaling_laws.environments.wrappers import (
    FrameStack,
    NormalizeObservation,
    NormalizeReward,
    RecordEpisodeStatistics,
)
from rl_scaling_laws.environments.registry import EnvironmentRegistry, get_env_config

__all__ = [
    "make_env",
    "make_vec_env",
    "FrameStack",
    "NormalizeObservation",
    "NormalizeReward",
    "RecordEpisodeStatistics",
    "EnvironmentRegistry",
    "get_env_config",
]
