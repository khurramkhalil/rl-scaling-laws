"""Base neural network class for RL models."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from rl_scaling_laws.utils.param_count import count_parameters, format_param_count


class BaseNetwork(nn.Module, ABC):
    """Abstract base class for scalable neural networks."""

    def __init__(self):
        super().__init__()
        self._input_dim: Optional[int] = None
        self._output_dim: Optional[int] = None

    @property
    def input_dim(self) -> int:
        """Input dimension of the network."""
        if self._input_dim is None:
            raise ValueError("Input dimension not set")
        return self._input_dim

    @property
    def output_dim(self) -> int:
        """Output dimension of the network."""
        if self._output_dim is None:
            raise ValueError("Output dimension not set")
        return self._output_dim

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return count_parameters(self, trainable_only=True)

    @property
    def param_summary(self) -> str:
        """Formatted parameter count."""
        return format_param_count(self.num_parameters)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        pass

    def reset_parameters(self) -> None:
        """Reset all parameters to their initial values."""
        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name.lower()]

    def apply_init(self, method: str = "orthogonal", gain: float = 1.0) -> None:
        """Apply weight initialization to all layers.

        Args:
            method: Initialization method (orthogonal, xavier, kaiming).
            gain: Gain for initialization.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if method == "orthogonal":
                    nn.init.orthogonal_(module.weight, gain=gain)
                elif method == "xavier":
                    nn.init.xavier_uniform_(module.weight, gain=gain)
                elif method == "kaiming":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                else:
                    raise ValueError(f"Unknown init method: {method}")

                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: List[int],
    activation: str = "relu",
    output_activation: Optional[str] = None,
    layer_norm: bool = False,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Create a simple MLP network.

    Args:
        input_dim: Input dimension.
        output_dim: Output dimension.
        hidden_sizes: List of hidden layer sizes.
        activation: Activation function name.
        output_activation: Activation for output layer (None for linear).
        layer_norm: Whether to use layer normalization.
        dropout: Dropout probability.

    Returns:
        Sequential MLP network.
    """
    layers = []
    prev_dim = input_dim

    activation_fn = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }

    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(prev_dim, hidden_size))

        if layer_norm:
            layers.append(nn.LayerNorm(hidden_size))

        layers.append(activation_fn.get(activation, nn.ReLU)())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        prev_dim = hidden_size

    # Output layer
    layers.append(nn.Linear(prev_dim, output_dim))

    if output_activation is not None:
        layers.append(activation_fn.get(output_activation, nn.Identity)())

    return nn.Sequential(*layers)
