"""PPO (Proximal Policy Optimization) algorithm implementation."""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from rl_scaling_laws.algorithms.base import BaseAlgorithm
from rl_scaling_laws.algorithms.registry import register_algorithm
from rl_scaling_laws.models.builder import NetworkBuilder
from rl_scaling_laws.training.buffer import RolloutBuffer
from rl_scaling_laws.utils.logging import get_logger

logger = get_logger(__name__)


@register_algorithm("ppo")
class PPO(BaseAlgorithm):
    """Proximal Policy Optimization algorithm.

    Reference: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
    """

    def __init__(
        self,
        observation_space,
        action_space,
        # Network configuration
        architecture: str = "mlp",
        target_params: int = 10_000_000,
        # Learning rate
        lr: float = 3e-4,
        # Algorithm parameters
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        # Rollout
        n_steps: int = 2048,
        n_envs: int = 1,
        # Training
        n_epochs: int = 10,
        batch_size: int = 64,
        target_kl: Optional[float] = None,
        # Device
        device: Union[str, torch.device] = "auto",
        seed: int = 0,
    ):
        """Initialize PPO.

        Args:
            observation_space: Environment observation space.
            action_space: Environment action space.
            architecture: Network architecture type.
            target_params: Target parameter count.
            lr: Learning rate.
            gamma: Discount factor.
            gae_lambda: GAE lambda.
            clip_range: PPO clip range.
            clip_range_vf: Value function clip range (None for no clipping).
            normalize_advantage: Whether to normalize advantages.
            ent_coef: Entropy coefficient.
            vf_coef: Value function coefficient.
            max_grad_norm: Maximum gradient norm.
            n_steps: Steps per rollout.
            n_envs: Number of parallel environments.
            n_epochs: Number of epochs per update.
            batch_size: Minibatch size.
            target_kl: Target KL divergence for early stopping.
            device: Device for computation.
            seed: Random seed.
        """
        super().__init__(observation_space, action_space, device, seed)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl

        # Build networks
        builder = NetworkBuilder(architecture, target_params // 2)

        # Policy network
        if self.is_discrete:
            policy_output = self.action_dim
        else:
            policy_output = self.action_dim * 2  # mean + log_std

        self.policy = builder.build_actor(self.obs_dim, policy_output).to(self.device)

        # Value network
        value_builder = NetworkBuilder(architecture, target_params // 2)
        self.value = value_builder.build_value_network(self.obs_dim).to(self.device)

        # For continuous actions, learnable log_std
        if not self.is_discrete:
            self.log_std = nn.Parameter(
                torch.zeros(self.action_dim, device=self.device)
            )

        # Optimizer
        params = list(self.policy.parameters()) + list(self.value.parameters())
        if not self.is_discrete:
            params.append(self.log_std)
        self.optimizer = torch.optim.Adam(params, lr=lr)

        # Rollout buffer
        obs_shape = observation_space.shape
        action_dim = 1 if self.is_discrete else self.action_dim
        self.buffer = RolloutBuffer(
            n_steps, obs_shape, action_dim, self.device,
            n_envs, gamma, gae_lambda
        )

        # Track last observation and done
        self._last_obs = None
        self._last_dones = None

        logger.info(
            f"PPO initialized: policy={self.policy.num_parameters}, "
            f"value={self.value.num_parameters}"
        )

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

            action, _, _ = self._get_action(obs, deterministic)
            action = action.cpu().numpy().squeeze()

        return action

    def _get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, and entropy."""
        policy_output = self.policy(obs)

        if self.is_discrete:
            dist = Categorical(logits=policy_output)
            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        else:
            mean = policy_output
            std = self.log_std.exp()
            dist = Normal(mean, std)
            if deterministic:
                action = mean
            else:
                action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy

    def collect_rollouts(
        self,
        env,
        n_steps: int,
    ) -> int:
        """Collect rollout experience."""
        if self._last_obs is None:
            self._last_obs, _ = env.reset()
            self._last_dones = np.zeros((self.n_envs,), dtype=bool)

        self.buffer.reset()
        collected = 0

        for _ in range(n_steps):
            with torch.no_grad():
                obs_tensor = self._to_tensor(self._last_obs)
                if obs_tensor.dim() == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)

                action, log_prob, _ = self._get_action(obs_tensor)
                value = self.value(obs_tensor).flatten()

                action = action.cpu().numpy()
                log_prob = log_prob.cpu().numpy()
                value = value.cpu().numpy()

            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(
                action.squeeze() if self.n_envs == 1 else action
            )
            done = terminated or truncated

            # Store transition
            if self.n_envs == 1:
                self.buffer.add(
                    np.expand_dims(self._last_obs, 0),
                    np.expand_dims(action, 0),
                    np.array([reward]),
                    np.array([done]),
                    value,
                    log_prob,
                )
            else:
                self.buffer.add(
                    self._last_obs, action, reward, done, value, log_prob
                )

            self._last_obs = next_obs
            self._last_dones = np.array([done]) if self.n_envs == 1 else done
            self.num_timesteps += self.n_envs
            collected += 1

            if done:
                self._last_obs, _ = env.reset()

        # Compute returns and advantages
        with torch.no_grad():
            obs_tensor = self._to_tensor(self._last_obs)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            last_values = self.value(obs_tensor).flatten().cpu().numpy()

        self.buffer.compute_returns_and_advantages(last_values, self._last_dones)

        return collected * self.n_envs

    def train(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """Perform training step."""
        batch_size = batch_size or self.batch_size

        all_losses = []
        all_pg_losses = []
        all_value_losses = []
        all_entropy_losses = []
        all_clip_fractions = []

        for epoch in range(self.n_epochs):
            for batch in self.buffer.get(batch_size):
                # Get current policy outputs
                action, log_prob, entropy = self._get_action(batch.observations)

                # For discrete actions, use stored actions for log_prob
                if self.is_discrete:
                    policy_output = self.policy(batch.observations)
                    dist = Categorical(logits=policy_output)
                    log_prob = dist.log_prob(batch.actions.squeeze(-1))
                    entropy = dist.entropy()

                # Current value
                values = self.value(batch.observations).flatten()

                # Ratio for PPO
                ratio = (log_prob - batch.old_log_probs).exp()

                # Normalize advantages
                advantages = batch.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(
                    ratio, 1 - self.clip_range, 1 + self.clip_range
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Clip fraction
                clip_fraction = (torch.abs(ratio - 1) > self.clip_range).float().mean()

                # Value loss
                if self.clip_range_vf is not None:
                    values_clipped = batch.old_values + torch.clamp(
                        values - batch.old_values, -self.clip_range_vf, self.clip_range_vf
                    )
                    vf_loss1 = (values - batch.returns).pow(2)
                    vf_loss2 = (values_clipped - batch.returns).pow(2)
                    vf_loss = torch.max(vf_loss1, vf_loss2).mean()
                else:
                    vf_loss = F.mse_loss(values, batch.returns)

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = pg_loss + self.vf_coef * vf_loss + self.ent_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                all_losses.append(loss.item())
                all_pg_losses.append(pg_loss.item())
                all_value_losses.append(vf_loss.item())
                all_entropy_losses.append(entropy_loss.item())
                all_clip_fractions.append(clip_fraction.item())

            # Early stopping based on KL divergence
            if self.target_kl is not None:
                with torch.no_grad():
                    # Approximate KL
                    log_ratio = log_prob - batch.old_log_probs
                    approx_kl = ((ratio - 1) - log_ratio).mean()

                if approx_kl > self.target_kl:
                    break

        self._n_updates += 1

        return {
            "loss": np.mean(all_losses),
            "pg_loss": np.mean(all_pg_losses),
            "value_loss": np.mean(all_value_losses),
            "entropy_loss": np.mean(all_entropy_losses),
            "clip_fraction": np.mean(all_clip_fractions),
        }

    def _get_modules(self) -> Dict[str, nn.Module]:
        return {
            "policy": self.policy,
            "value": self.value,
        }

    def _get_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return {"optimizer": self.optimizer}
