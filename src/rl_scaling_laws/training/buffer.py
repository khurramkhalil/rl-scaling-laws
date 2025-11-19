"""Replay buffers for off-policy and on-policy algorithms."""

from typing import Dict, Generator, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch


class ReplayBufferSamples(NamedTuple):
    """Samples from replay buffer."""

    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms."""

    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        device: Union[str, torch.device] = "cpu",
        n_envs: int = 1,
    ):
        """Initialize replay buffer.

        Args:
            buffer_size: Maximum number of transitions to store.
            obs_shape: Shape of observations.
            action_dim: Dimension of actions.
            device: Device to store tensors on.
            n_envs: Number of parallel environments.
        """
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.n_envs = n_envs

        # Allocate buffers
        self.observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_observations[self.pos] = next_obs
        self.dones[self.pos] = done

        self.pos = (self.pos + 1) % self.buffer_size
        self.full = self.full or self.pos == 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Batch of transitions.
        """
        max_idx = self.buffer_size if self.full else self.pos
        indices = np.random.randint(0, max_idx, size=batch_size)

        return ReplayBufferSamples(
            observations=torch.as_tensor(
                self.observations[indices], device=self.device
            ),
            actions=torch.as_tensor(
                self.actions[indices], device=self.device
            ),
            next_observations=torch.as_tensor(
                self.next_observations[indices], device=self.device
            ),
            rewards=torch.as_tensor(
                self.rewards[indices], device=self.device
            ).unsqueeze(-1),
            dones=torch.as_tensor(
                self.dones[indices], device=self.device
            ).unsqueeze(-1),
        )

    def __len__(self) -> int:
        return self.buffer_size if self.full else self.pos

    @property
    def size(self) -> int:
        """Current number of transitions in buffer."""
        return len(self)


class RolloutBufferSamples(NamedTuple):
    """Samples from rollout buffer."""

    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    """Rollout buffer for on-policy algorithms (PPO)."""

    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        device: Union[str, torch.device] = "cpu",
        n_envs: int = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """Initialize rollout buffer.

        Args:
            buffer_size: Number of steps per rollout.
            obs_shape: Shape of observations.
            action_dim: Dimension of actions.
            device: Device to store tensors on.
            n_envs: Number of parallel environments.
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.
        """
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.n_envs = n_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Allocate buffers
        self.observations = np.zeros(
            (buffer_size, n_envs, *obs_shape), dtype=np.float32
        )
        self.actions = np.zeros(
            (buffer_size, n_envs, action_dim), dtype=np.float32
        )
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.advantages = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.returns = np.zeros((buffer_size, n_envs), dtype=np.float32)

        self.pos = 0
        self.full = False

    def reset(self) -> None:
        """Reset buffer for new rollout."""
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
    ) -> None:
        """Add a step to the buffer."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        last_dones: np.ndarray,
    ) -> None:
        """Compute GAE advantages and returns.

        Args:
            last_values: Value estimates for last observation.
            last_dones: Done flags for last step.
        """
        last_gae = 0

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]

            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae

        self.returns = self.advantages + self.values

    def get(
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[RolloutBufferSamples, None, None]:
        """Get batches from the buffer.

        Args:
            batch_size: Size of minibatches. If None, return entire buffer.

        Yields:
            Batches of rollout samples.
        """
        # Flatten buffer
        batch_size = batch_size or self.buffer_size * self.n_envs
        total_size = self.buffer_size * self.n_envs

        indices = np.random.permutation(total_size)

        for start in range(0, total_size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            # Convert to flat indices
            env_indices = batch_indices % self.n_envs
            step_indices = batch_indices // self.n_envs

            yield RolloutBufferSamples(
                observations=torch.as_tensor(
                    self.observations[step_indices, env_indices],
                    device=self.device,
                ),
                actions=torch.as_tensor(
                    self.actions[step_indices, env_indices],
                    device=self.device,
                ),
                old_values=torch.as_tensor(
                    self.values[step_indices, env_indices],
                    device=self.device,
                ),
                old_log_probs=torch.as_tensor(
                    self.log_probs[step_indices, env_indices],
                    device=self.device,
                ),
                advantages=torch.as_tensor(
                    self.advantages[step_indices, env_indices],
                    device=self.device,
                ),
                returns=torch.as_tensor(
                    self.returns[step_indices, env_indices],
                    device=self.device,
                ),
            )
