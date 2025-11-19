"""RL algorithms with scalable network support."""

from rl_scaling_laws.algorithms.base import BaseAlgorithm
from rl_scaling_laws.algorithms.sac import SAC
from rl_scaling_laws.algorithms.dqn import DQN
from rl_scaling_laws.algorithms.ppo import PPO
from rl_scaling_laws.algorithms.registry import AlgorithmRegistry, get_algorithm, register_algorithm

__all__ = [
    "BaseAlgorithm",
    "SAC",
    "DQN",
    "PPO",
    "AlgorithmRegistry",
    "get_algorithm",
    "register_algorithm",
]
