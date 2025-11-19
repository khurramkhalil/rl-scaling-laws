"""Network builder for creating scalable architectures."""

from typing import Any, Dict, Optional, Tuple

import torch.nn as nn

from rl_scaling_laws.models.base import BaseNetwork
from rl_scaling_laws.models.registry import get_model
from rl_scaling_laws.utils.param_count import count_parameters, format_param_count
from rl_scaling_laws.utils.logging import get_logger

logger = get_logger(__name__)


class NetworkBuilder:
    """Builder for creating networks that match target parameter counts."""

    def __init__(
        self,
        architecture: str = "mlp",
        target_params: int = 10_000_000,
        **architecture_kwargs,
    ):
        """Initialize the network builder.

        Args:
            architecture: Architecture type (mlp, cnn, transformer, moe).
            target_params: Target number of parameters.
            **architecture_kwargs: Additional architecture-specific arguments.
        """
        self.architecture = architecture.lower()
        self.target_params = target_params
        self.architecture_kwargs = architecture_kwargs

    def build_actor(
        self,
        input_dim: int,
        output_dim: int,
        **kwargs,
    ) -> BaseNetwork:
        """Build an actor network.

        Args:
            input_dim: Observation dimension.
            output_dim: Action dimension.
            **kwargs: Override default architecture arguments.

        Returns:
            Actor network.
        """
        merged_kwargs = {**self.architecture_kwargs, **kwargs}

        network = get_model(
            self.architecture,
            input_dim=input_dim,
            output_dim=output_dim,
            target_params=self.target_params,
            **merged_kwargs,
        )

        logger.info(
            f"Built actor network: {self.architecture}, "
            f"params={format_param_count(network.num_parameters)}, "
            f"target={format_param_count(self.target_params)}"
        )

        return network

    def build_critic(
        self,
        input_dim: int,
        output_dim: int = 1,
        **kwargs,
    ) -> BaseNetwork:
        """Build a critic network.

        Args:
            input_dim: Observation (+ action) dimension.
            output_dim: Value output dimension.
            **kwargs: Override default architecture arguments.

        Returns:
            Critic network.
        """
        merged_kwargs = {**self.architecture_kwargs, **kwargs}

        network = get_model(
            self.architecture,
            input_dim=input_dim,
            output_dim=output_dim,
            target_params=self.target_params,
            **merged_kwargs,
        )

        logger.info(
            f"Built critic network: {self.architecture}, "
            f"params={format_param_count(network.num_parameters)}"
        )

        return network

    def build_q_network(
        self,
        obs_dim: int,
        action_dim: int,
        **kwargs,
    ) -> BaseNetwork:
        """Build a Q-network for continuous actions.

        Concatenates observations and actions.

        Args:
            obs_dim: Observation dimension.
            action_dim: Action dimension.
            **kwargs: Override default architecture arguments.

        Returns:
            Q-network.
        """
        return self.build_critic(
            input_dim=obs_dim + action_dim,
            output_dim=1,
            **kwargs,
        )

    def build_discrete_q_network(
        self,
        obs_dim: int,
        num_actions: int,
        **kwargs,
    ) -> BaseNetwork:
        """Build a Q-network for discrete actions.

        Outputs Q-value for each action.

        Args:
            obs_dim: Observation dimension.
            num_actions: Number of discrete actions.
            **kwargs: Override default architecture arguments.

        Returns:
            Q-network.
        """
        return self.build_critic(
            input_dim=obs_dim,
            output_dim=num_actions,
            **kwargs,
        )

    def build_value_network(
        self,
        obs_dim: int,
        **kwargs,
    ) -> BaseNetwork:
        """Build a value network.

        Args:
            obs_dim: Observation dimension.
            **kwargs: Override default architecture arguments.

        Returns:
            Value network.
        """
        return self.build_critic(
            input_dim=obs_dim,
            output_dim=1,
            **kwargs,
        )

    @staticmethod
    def verify_param_count(
        network: nn.Module,
        target: int,
        tolerance: float = 0.1,
    ) -> bool:
        """Verify that network parameter count is within tolerance.

        Args:
            network: Neural network.
            target: Target parameter count.
            tolerance: Acceptable relative error (default 10%).

        Returns:
            True if within tolerance.
        """
        actual = count_parameters(network)
        error = abs(actual - target) / target

        if error > tolerance:
            logger.warning(
                f"Parameter count mismatch: actual={format_param_count(actual)}, "
                f"target={format_param_count(target)}, error={error:.1%}"
            )
            return False

        return True

    def get_param_distribution(
        self,
        actor: nn.Module,
        critic: nn.Module,
    ) -> Dict[str, int]:
        """Get parameter distribution between actor and critic.

        Args:
            actor: Actor network.
            critic: Critic network.

        Returns:
            Dictionary with parameter counts.
        """
        actor_params = count_parameters(actor)
        critic_params = count_parameters(critic)
        total = actor_params + critic_params

        return {
            "actor_params": actor_params,
            "critic_params": critic_params,
            "total_params": total,
            "actor_ratio": actor_params / total if total > 0 else 0,
            "critic_ratio": critic_params / total if total > 0 else 0,
        }


def build_network_for_env(
    env_type: str,
    obs_space,
    action_space,
    architecture: str,
    target_params: int,
    **kwargs,
) -> Tuple[BaseNetwork, BaseNetwork]:
    """Build actor and critic networks for a given environment.

    Args:
        env_type: Environment type (atari, dmc, metaworld, procgen).
        obs_space: Observation space.
        action_space: Action space.
        architecture: Architecture type.
        target_params: Target parameters (split between actor/critic).
        **kwargs: Additional architecture arguments.

    Returns:
        Tuple of (actor, critic) networks.
    """
    # Determine dimensions
    if hasattr(obs_space, "shape"):
        if len(obs_space.shape) == 3:
            # Image observation
            input_channels = obs_space.shape[0]
            obs_dim = input_channels  # CNN handles internally
        else:
            obs_dim = obs_space.shape[0]
    else:
        obs_dim = obs_space.n

    # Action dimension
    if hasattr(action_space, "shape"):
        action_dim = action_space.shape[0]
        discrete = False
    else:
        action_dim = action_space.n
        discrete = True

    # Split params between actor and critic (50/50 by default)
    actor_params = target_params // 2
    critic_params = target_params - actor_params

    builder = NetworkBuilder(architecture, actor_params, **kwargs)

    # Build networks
    if discrete:
        actor = builder.build_critic(obs_dim, action_dim)  # Policy logits
        critic = NetworkBuilder(architecture, critic_params, **kwargs).build_critic(
            obs_dim, action_dim  # Q for each action
        )
    else:
        actor = builder.build_actor(obs_dim, action_dim * 2)  # Mean + log_std
        critic = NetworkBuilder(architecture, critic_params, **kwargs).build_q_network(
            obs_dim, action_dim
        )

    return actor, critic
