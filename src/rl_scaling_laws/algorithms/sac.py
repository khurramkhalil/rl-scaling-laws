"""SAC (Soft Actor-Critic) algorithm implementation."""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rl_scaling_laws.algorithms.base import BaseAlgorithm
from rl_scaling_laws.algorithms.registry import register_algorithm
from rl_scaling_laws.models.builder import NetworkBuilder
from rl_scaling_laws.training.buffer import ReplayBuffer
from rl_scaling_laws.utils.logging import get_logger

logger = get_logger(__name__)


class GaussianActor(nn.Module):
    """Gaussian policy for continuous actions."""

    def __init__(
        self,
        backbone: nn.Module,
        action_dim: int,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get mean and log_std from backbone output."""
        output = self.backbone(obs)
        mean, log_std = output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability.

        Args:
            obs: Observation tensor.
            deterministic: If True, return mean action.

        Returns:
            Tuple of (action, log_prob).
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        if deterministic:
            action = mean
            log_prob = torch.zeros(obs.size(0), 1, device=obs.device)
        else:
            # Reparameterization trick
            normal = Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)

            # Log probability with squashing correction
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


@register_algorithm("sac")
class SAC(BaseAlgorithm):
    """Soft Actor-Critic algorithm.

    Reference: Haarnoja et al. (2018) "Soft Actor-Critic: Off-Policy Maximum
    Entropy Deep Reinforcement Learning with a Stochastic Actor"
    """

    def __init__(
        self,
        observation_space,
        action_space,
        # Network configuration
        architecture: str = "mlp",
        target_params: int = 10_000_000,
        # Learning rates
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        # Algorithm parameters
        gamma: float = 0.99,
        tau: float = 0.005,
        init_alpha: float = 0.2,
        auto_entropy_tuning: bool = True,
        target_entropy: Optional[float] = None,
        # Buffer
        buffer_size: int = 1_000_000,
        # Training
        utd_ratio: int = 1,
        batch_size: int = 256,
        learning_starts: int = 10_000,
        # Device
        device: Union[str, torch.device] = "auto",
        seed: int = 0,
    ):
        """Initialize SAC.

        Args:
            observation_space: Environment observation space.
            action_space: Environment action space.
            architecture: Network architecture type.
            target_params: Target parameter count.
            actor_lr: Actor learning rate.
            critic_lr: Critic learning rate.
            alpha_lr: Entropy coefficient learning rate.
            gamma: Discount factor.
            tau: Soft update coefficient.
            init_alpha: Initial entropy coefficient.
            auto_entropy_tuning: Whether to automatically tune entropy.
            target_entropy: Target entropy (default: -action_dim).
            buffer_size: Replay buffer size.
            utd_ratio: Updates per environment step.
            batch_size: Batch size for training.
            learning_starts: Steps before starting training.
            device: Device for computation.
            seed: Random seed.
        """
        super().__init__(observation_space, action_space, device, seed)

        self.gamma = gamma
        self.tau = tau
        self.utd_ratio = utd_ratio
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.auto_entropy_tuning = auto_entropy_tuning

        # Build networks
        builder = NetworkBuilder(architecture, target_params // 3)

        # Actor outputs mean and log_std
        actor_backbone = builder.build_actor(self.obs_dim, self.action_dim * 2)
        self.actor = GaussianActor(actor_backbone, self.action_dim).to(self.device)

        # Twin critics
        critic_builder = NetworkBuilder(architecture, target_params // 3)
        self.critic1 = critic_builder.build_q_network(
            self.obs_dim, self.action_dim
        ).to(self.device)
        self.critic2 = critic_builder.build_q_network(
            self.obs_dim, self.action_dim
        ).to(self.device)

        # Target critics
        self.critic1_target = critic_builder.build_q_network(
            self.obs_dim, self.action_dim
        ).to(self.device)
        self.critic2_target = critic_builder.build_q_network(
            self.obs_dim, self.action_dim
        ).to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Entropy coefficient
        self.log_alpha = torch.tensor(
            np.log(init_alpha), device=self.device, requires_grad=True
        )

        if target_entropy is None:
            self.target_entropy = -self.action_dim
        else:
            self.target_entropy = target_entropy

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # Replay buffer
        obs_shape = observation_space.shape
        self.buffer = ReplayBuffer(
            buffer_size, obs_shape, self.action_dim, self.device
        )

        # Track last observation for collecting
        self._last_obs = None

        logger.info(
            f"SAC initialized: actor={self.actor.backbone.num_parameters}, "
            f"critic={self.critic1.num_parameters}"
        )

    @property
    def alpha(self) -> torch.Tensor:
        """Current entropy coefficient."""
        return self.log_alpha.exp()

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Predict action from observation."""
        with torch.no_grad():
            obs = self._to_tensor(observation)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)

            action, _ = self.actor.sample(obs, deterministic=deterministic)
            action = action.cpu().numpy().squeeze(0)

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
            # Select action
            if self.num_timesteps < self.learning_starts:
                action = env.action_space.sample()
            else:
                action = self.predict(self._last_obs, deterministic=False)

            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            self.buffer.add(
                self._last_obs, action, reward, next_obs, float(done)
            )

            self._last_obs = next_obs
            self.num_timesteps += 1
            collected += 1

            if done:
                self._last_obs, _ = env.reset()

        return collected

    def train(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """Perform one training step."""
        if len(self.buffer) < self.learning_starts:
            return {}

        batch_size = batch_size or self.batch_size
        metrics = {}

        for _ in range(self.utd_ratio):
            # Sample from buffer
            batch = self.buffer.sample(batch_size)

            # Update critics
            critic_loss, critic_metrics = self._update_critics(batch)
            metrics.update(critic_metrics)

            # Update actor
            actor_loss, actor_metrics = self._update_actor(batch)
            metrics.update(actor_metrics)

            # Update alpha
            if self.auto_entropy_tuning:
                alpha_loss, alpha_metrics = self._update_alpha(batch)
                metrics.update(alpha_metrics)

            # Soft update targets
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

            self._n_updates += 1

        return metrics

    def _update_critics(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Update critic networks."""
        with torch.no_grad():
            # Sample next action
            next_action, next_log_prob = self.actor.sample(batch.next_observations)

            # Target Q values
            target_q1 = self.critic1_target(
                torch.cat([batch.next_observations, next_action], dim=-1)
            )
            target_q2 = self.critic2_target(
                torch.cat([batch.next_observations, next_action], dim=-1)
            )
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = batch.rewards + (1 - batch.dones) * self.gamma * target_q

        # Current Q values
        current_q1 = self.critic1(
            torch.cat([batch.observations, batch.actions], dim=-1)
        )
        current_q2 = self.critic2(
            torch.cat([batch.observations, batch.actions], dim=-1)
        )

        # Critic losses
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        # Update critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # Update critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        return critic1_loss + critic2_loss, {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "q_value": current_q1.mean().item(),
        }

    def _update_actor(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Update actor network."""
        action, log_prob = self.actor.sample(batch.observations)

        q1 = self.critic1(torch.cat([batch.observations, action], dim=-1))
        q2 = self.critic2(torch.cat([batch.observations, action], dim=-1))
        q = torch.min(q1, q2)

        actor_loss = (self.alpha.detach() * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss, {
            "actor_loss": actor_loss.item(),
            "entropy": -log_prob.mean().item(),
        }

    def _update_alpha(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Update entropy coefficient."""
        with torch.no_grad():
            _, log_prob = self.actor.sample(batch.observations)

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return alpha_loss, {
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Soft update target network."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def _get_modules(self) -> Dict[str, nn.Module]:
        return {
            "actor": self.actor,
            "critic1": self.critic1,
            "critic2": self.critic2,
            "critic1_target": self.critic1_target,
            "critic2_target": self.critic2_target,
        }

    def _get_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return {
            "actor": self.actor_optimizer,
            "critic1": self.critic1_optimizer,
            "critic2": self.critic2_optimizer,
            "alpha": self.alpha_optimizer,
        }
