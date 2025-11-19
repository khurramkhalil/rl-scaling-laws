"""DQN (Deep Q-Network) algorithm implementation."""

from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_scaling_laws.algorithms.base import BaseAlgorithm
from rl_scaling_laws.algorithms.registry import register_algorithm
from rl_scaling_laws.models.builder import NetworkBuilder
from rl_scaling_laws.training.buffer import ReplayBuffer
from rl_scaling_laws.utils.logging import get_logger
from rl_scaling_laws.utils.schedule import linear_schedule

logger = get_logger(__name__)


@register_algorithm("dqn")
class DQN(BaseAlgorithm):
    """Deep Q-Network algorithm.

    Reference: Mnih et al. (2015) "Human-level control through deep
    reinforcement learning"
    """

    def __init__(
        self,
        observation_space,
        action_space,
        # Network configuration
        architecture: str = "mlp",
        target_params: int = 10_000_000,
        # Learning rate
        lr: float = 1e-4,
        # Algorithm parameters
        gamma: float = 0.99,
        target_update_freq: int = 10_000,
        # Exploration
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.01,
        exploration_fraction: float = 0.1,
        # Extensions
        double_q: bool = True,
        dueling: bool = False,
        # Buffer
        buffer_size: int = 1_000_000,
        # Training
        utd_ratio: int = 1,
        batch_size: int = 32,
        learning_starts: int = 50_000,
        train_freq: int = 4,
        # Device
        device: Union[str, torch.device] = "auto",
        seed: int = 0,
        # Total timesteps for exploration schedule
        total_timesteps: int = 1_000_000,
    ):
        """Initialize DQN.

        Args:
            observation_space: Environment observation space.
            action_space: Environment action space.
            architecture: Network architecture type.
            target_params: Target parameter count.
            lr: Learning rate.
            gamma: Discount factor.
            target_update_freq: Steps between target network updates.
            exploration_initial_eps: Initial exploration rate.
            exploration_final_eps: Final exploration rate.
            exploration_fraction: Fraction of training for exploration decay.
            double_q: Use Double DQN.
            dueling: Use Dueling DQN.
            buffer_size: Replay buffer size.
            utd_ratio: Updates per train call.
            batch_size: Batch size for training.
            learning_starts: Steps before starting training.
            train_freq: Environment steps between training.
            device: Device for computation.
            seed: Random seed.
            total_timesteps: Total training timesteps.
        """
        super().__init__(observation_space, action_space, device, seed)

        assert self.is_discrete, "DQN only supports discrete action spaces"

        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.double_q = double_q
        self.utd_ratio = utd_ratio
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq

        # Exploration schedule
        self.exploration_schedule = linear_schedule(
            exploration_initial_eps,
            exploration_final_eps,
        )
        self.exploration_fraction = exploration_fraction
        self.total_timesteps = total_timesteps

        # Build Q-networks
        builder = NetworkBuilder(architecture, target_params // 2)

        self.q_network = builder.build_discrete_q_network(
            self.obs_dim, self.action_dim
        ).to(self.device)

        self.target_network = builder.build_discrete_q_network(
            self.obs_dim, self.action_dim
        ).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        obs_shape = observation_space.shape
        self.buffer = ReplayBuffer(
            buffer_size, obs_shape, 1, self.device  # action_dim=1 for discrete
        )

        # Track last observation
        self._last_obs = None

        logger.info(f"DQN initialized: q_network={self.q_network.num_parameters}")

    @property
    def exploration_rate(self) -> float:
        """Current exploration rate."""
        progress = min(1.0, self.num_timesteps / (self.total_timesteps * self.exploration_fraction))
        return self.exploration_schedule(progress)

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Predict action from observation."""
        # Epsilon-greedy exploration
        if not deterministic and np.random.random() < self.exploration_rate:
            return np.array(self.action_space.sample())

        with torch.no_grad():
            obs = self._to_tensor(observation)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)

            q_values = self.q_network(obs)
            action = q_values.argmax(dim=-1).cpu().numpy().squeeze()

        return action

    def collect_rollouts(
        self,
        env,
        n_steps: int,
    ) -> int:
        """Collect experience from environment."""
        if self._last_obs is None:
            self._last_obs, _ = env.reset()

        collected = 0

        for _ in range(n_steps):
            # Select action with exploration
            action = self.predict(self._last_obs, deterministic=False)

            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            self.buffer.add(
                self._last_obs,
                np.array([action]),
                reward,
                next_obs,
                float(done),
            )

            self._last_obs = next_obs
            self.num_timesteps += 1
            collected += 1

            if done:
                self._last_obs, _ = env.reset()

        return collected

    def train(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """Perform training step."""
        if len(self.buffer) < self.learning_starts:
            return {}

        batch_size = batch_size or self.batch_size
        metrics = {}

        for _ in range(self.utd_ratio):
            # Sample from buffer
            batch = self.buffer.sample(batch_size)

            # Compute target Q values
            with torch.no_grad():
                if self.double_q:
                    # Double DQN: select action with online network
                    next_actions = self.q_network(batch.next_observations).argmax(dim=-1, keepdim=True)
                    # Evaluate with target network
                    next_q = self.target_network(batch.next_observations).gather(1, next_actions)
                else:
                    # Standard DQN
                    next_q = self.target_network(batch.next_observations).max(dim=-1, keepdim=True)[0]

                target_q = batch.rewards + (1 - batch.dones) * self.gamma * next_q

            # Current Q values
            current_q = self.q_network(batch.observations).gather(
                1, batch.actions.long()
            )

            # Loss
            loss = F.smooth_l1_loss(current_q, target_q)

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
            self.optimizer.step()

            self._n_updates += 1

            # Update target network
            if self._n_updates % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

        metrics["loss"] = loss.item()
        metrics["q_value"] = current_q.mean().item()
        metrics["exploration_rate"] = self.exploration_rate

        return metrics

    def _get_modules(self) -> Dict[str, nn.Module]:
        return {
            "q_network": self.q_network,
            "target_network": self.target_network,
        }

    def _get_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return {"optimizer": self.optimizer}
