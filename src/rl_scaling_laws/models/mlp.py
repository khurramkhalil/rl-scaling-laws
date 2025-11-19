"""Scalable MLP (Multi-Layer Perceptron) architectures."""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from rl_scaling_laws.models.base import BaseNetwork
from rl_scaling_laws.models.registry import register_model


class MLP(BaseNetwork):
    """Standard MLP network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int],
        activation: str = "relu",
        output_activation: Optional[str] = None,
        layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim

        layers = []
        prev_dim = input_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(self.get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_size

        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation:
            layers.append(self.get_activation(output_activation))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@register_model("mlp")
class ScalableMLP(BaseNetwork):
    """MLP with automatic scaling to target parameter count.

    This network automatically adjusts width and depth to match
    a target parameter count.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        target_params: int,
        min_layers: int = 2,
        max_layers: int = 20,
        min_width: int = 64,
        max_width: int = 8192,
        activation: str = "relu",
        layer_norm: bool = False,
        dropout: float = 0.0,
        init_method: str = "orthogonal",
    ):
        """Initialize scalable MLP.

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            target_params: Target number of parameters.
            min_layers: Minimum number of hidden layers.
            max_layers: Maximum number of hidden layers.
            min_width: Minimum hidden layer width.
            max_width: Maximum hidden layer width.
            activation: Activation function.
            layer_norm: Whether to use layer normalization.
            dropout: Dropout probability.
            init_method: Weight initialization method.
        """
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.target_params = target_params

        # Calculate optimal architecture
        num_layers, hidden_size = self._compute_architecture(
            input_dim, output_dim, target_params,
            min_layers, max_layers, min_width, max_width
        )

        # Build network
        layers = []
        prev_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_size))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(self.get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_size

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply_init(init_method)

        # Store architecture info
        self.architecture_info = {
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "target_params": target_params,
            "actual_params": self.num_parameters,
        }

    def _compute_architecture(
        self,
        input_dim: int,
        output_dim: int,
        target_params: int,
        min_layers: int,
        max_layers: int,
        min_width: int,
        max_width: int,
    ) -> Tuple[int, int]:
        """Compute number of layers and width to match target params.

        Uses binary search to find optimal configuration.
        """
        best_config = (min_layers, min_width)
        best_diff = float("inf")

        for num_layers in range(min_layers, max_layers + 1):
            # Binary search for width
            lo, hi = min_width, max_width

            while lo <= hi:
                mid = (lo + hi) // 2
                params = self._count_params(input_dim, output_dim, num_layers, mid)

                diff = abs(params - target_params)
                if diff < best_diff:
                    best_diff = diff
                    best_config = (num_layers, mid)

                if params < target_params:
                    lo = mid + 1
                else:
                    hi = mid - 1

        return best_config

    def _count_params(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        hidden_size: int,
    ) -> int:
        """Count parameters for given architecture."""
        if num_layers == 0:
            return input_dim * output_dim + output_dim

        # First layer
        params = input_dim * hidden_size + hidden_size

        # Hidden layers
        params += (num_layers - 1) * (hidden_size * hidden_size + hidden_size)

        # Output layer
        params += hidden_size * output_dim + output_dim

        return params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DuelingMLP(BaseNetwork):
    """Dueling network architecture for DQN.

    Separates value and advantage streams.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int],
        activation: str = "relu",
    ):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim

        # Shared feature extractor
        layers = []
        prev_dim = input_dim
        for hidden_size in hidden_sizes[:-1]:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(self.get_activation(activation))
            prev_dim = hidden_size

        self.features = nn.Sequential(*layers)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_sizes[-1]),
            self.get_activation(activation),
            nn.Linear(hidden_sizes[-1], 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_sizes[-1]),
            self.get_activation(activation),
            nn.Linear(hidden_sizes[-1], output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values
