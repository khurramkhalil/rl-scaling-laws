"""Base class for RL algorithms."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from rl_scaling_laws.utils.device import get_device
from rl_scaling_laws.utils.logging import get_logger

logger = get_logger(__name__)


class BaseAlgorithm(ABC):
    """Abstract base class for RL algorithms."""

    def __init__(
        self,
        observation_space,
        action_space,
        device: Union[str, torch.device] = "auto",
        seed: int = 0,
    ):
        """Initialize the algorithm.

        Args:
            observation_space: Environment observation space.
            action_space: Environment action space.
            device: Device for computation.
            seed: Random seed.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = get_device(device)
        self.seed = seed

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Track training progress
        self.num_timesteps = 0
        self._n_updates = 0

        logger.info(f"Initialized {self.__class__.__name__} on {self.device}")

    @property
    def obs_dim(self) -> int:
        """Observation dimension."""
        if hasattr(self.observation_space, "shape"):
            return int(np.prod(self.observation_space.shape))
        return self.observation_space.n

    @property
    def action_dim(self) -> int:
        """Action dimension."""
        if hasattr(self.action_space, "shape"):
            return int(np.prod(self.action_space.shape))
        return self.action_space.n

    @property
    def is_discrete(self) -> bool:
        """Whether action space is discrete."""
        return not hasattr(self.action_space, "shape")

    @abstractmethod
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Predict action from observation.

        Args:
            observation: Current observation.
            deterministic: Whether to use deterministic policy.

        Returns:
            Action to take.
        """
        pass

    @abstractmethod
    def train(self, batch_size: int) -> Dict[str, float]:
        """Perform one training step.

        Args:
            batch_size: Batch size for training.

        Returns:
            Dictionary of training metrics.
        """
        pass

    @abstractmethod
    def collect_rollouts(
        self,
        env,
        n_steps: int,
    ) -> int:
        """Collect experience from environment.

        Args:
            env: Environment to collect from.
            n_steps: Number of steps to collect.

        Returns:
            Number of transitions collected.
        """
        pass

    def save(self, path: Union[str, Path]) -> None:
        """Save model to file.

        Args:
            path: Path to save to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "num_timesteps": self.num_timesteps,
            "n_updates": self._n_updates,
        }

        # Save model state dicts
        for name, module in self._get_modules().items():
            if module is not None:
                state[f"{name}_state_dict"] = module.state_dict()

        # Save optimizer state dicts
        for name, optimizer in self._get_optimizers().items():
            if optimizer is not None:
                state[f"{name}_optimizer_state_dict"] = optimizer.state_dict()

        torch.save(state, path)
        logger.info(f"Saved model to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load model from file.

        Args:
            path: Path to load from.
        """
        path = Path(path)
        state = torch.load(path, map_location=self.device)

        self.num_timesteps = state.get("num_timesteps", 0)
        self._n_updates = state.get("n_updates", 0)

        # Load model state dicts
        for name, module in self._get_modules().items():
            key = f"{name}_state_dict"
            if key in state and module is not None:
                module.load_state_dict(state[key])

        # Load optimizer state dicts
        for name, optimizer in self._get_optimizers().items():
            key = f"{name}_optimizer_state_dict"
            if key in state and optimizer is not None:
                optimizer.load_state_dict(state[key])

        logger.info(f"Loaded model from {path}")

    def _get_modules(self) -> Dict[str, Optional[nn.Module]]:
        """Get all model modules for saving/loading.

        Returns:
            Dictionary of module names to modules.
        """
        return {}

    def _get_optimizers(self) -> Dict[str, Optional[torch.optim.Optimizer]]:
        """Get all optimizers for saving/loading.

        Returns:
            Dictionary of optimizer names to optimizers.
        """
        return {}

    def get_param_count(self) -> Dict[str, int]:
        """Get parameter counts for all modules.

        Returns:
            Dictionary of module names to parameter counts.
        """
        counts = {}
        for name, module in self._get_modules().items():
            if module is not None:
                counts[name] = sum(p.numel() for p in module.parameters())
        counts["total"] = sum(counts.values())
        return counts

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor on device."""
        return torch.as_tensor(data, device=self.device, dtype=torch.float32)
