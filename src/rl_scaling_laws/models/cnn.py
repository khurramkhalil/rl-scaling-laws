"""Scalable CNN (Convolutional Neural Network) architectures."""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from rl_scaling_laws.models.base import BaseNetwork
from rl_scaling_laws.models.registry import register_model


class CNN(BaseNetwork):
    """Standard CNN for visual observations (e.g., Atari)."""

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        conv_channels: List[int] = [32, 64, 64],
        conv_kernels: List[int] = [8, 4, 3],
        conv_strides: List[int] = [4, 2, 1],
        mlp_hidden_sizes: List[int] = [512],
        activation: str = "relu",
        image_size: int = 84,
    ):
        super().__init__()
        self._input_dim = input_channels
        self._output_dim = output_dim

        # Build convolutional layers
        conv_layers = []
        in_channels = input_channels

        for out_channels, kernel, stride in zip(conv_channels, conv_kernels, conv_strides):
            conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride)
            )
            conv_layers.append(self.get_activation(activation))
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        # Calculate conv output size
        conv_out_size = self._get_conv_output_size(input_channels, image_size)

        # Build MLP head
        mlp_layers = []
        prev_dim = conv_out_size

        for hidden_size in mlp_hidden_sizes:
            mlp_layers.append(nn.Linear(prev_dim, hidden_size))
            mlp_layers.append(self.get_activation(activation))
            prev_dim = hidden_size

        mlp_layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*mlp_layers)

    def _get_conv_output_size(self, channels: int, size: int) -> int:
        """Calculate the output size of conv layers."""
        with torch.no_grad():
            x = torch.zeros(1, channels, size, size)
            x = self.conv(x)
            return x.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize pixel values to [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        conv_out = self.conv(x)
        flat = conv_out.view(conv_out.size(0), -1)
        return self.mlp(flat)


@register_model("cnn")
class ScalableCNN(BaseNetwork):
    """CNN with automatic scaling to target parameter count."""

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        target_params: int,
        image_size: int = 84,
        activation: str = "relu",
        init_method: str = "orthogonal",
    ):
        """Initialize scalable CNN.

        Args:
            input_channels: Number of input channels (e.g., 4 for frame stack).
            output_dim: Output dimension.
            target_params: Target number of parameters.
            image_size: Input image size (assumes square).
            activation: Activation function.
            init_method: Weight initialization method.
        """
        super().__init__()
        self._input_dim = input_channels
        self._output_dim = output_dim
        self.target_params = target_params

        # Calculate scaling multipliers
        channel_mult, mlp_mult = self._compute_multipliers(
            input_channels, output_dim, target_params, image_size
        )

        # Scaled architecture (Nature DQN base)
        base_channels = [32, 64, 64]
        conv_channels = [int(c * channel_mult) for c in base_channels]

        conv_kernels = [8, 4, 3]
        conv_strides = [4, 2, 1]

        # Build conv layers
        conv_layers = []
        in_channels = input_channels

        for out_channels, kernel, stride in zip(conv_channels, conv_kernels, conv_strides):
            conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride)
            )
            conv_layers.append(self.get_activation(activation))
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        # Calculate conv output size
        conv_out_size = self._get_conv_output_size(input_channels, image_size)

        # Scaled MLP
        base_mlp = 512
        mlp_hidden = int(base_mlp * mlp_mult)

        self.mlp = nn.Sequential(
            nn.Linear(conv_out_size, mlp_hidden),
            self.get_activation(activation),
            nn.Linear(mlp_hidden, output_dim),
        )

        # Initialize
        self.apply_init(init_method)

        # Store architecture info
        self.architecture_info = {
            "conv_channels": conv_channels,
            "mlp_hidden": mlp_hidden,
            "channel_multiplier": channel_mult,
            "mlp_multiplier": mlp_mult,
            "target_params": target_params,
            "actual_params": self.num_parameters,
        }

    def _compute_multipliers(
        self,
        input_channels: int,
        output_dim: int,
        target_params: int,
        image_size: int,
    ) -> Tuple[float, float]:
        """Compute scaling multipliers for channels and MLP width."""
        # Start with base architecture
        base_params = self._estimate_params(1.0, 1.0, input_channels, output_dim, image_size)

        # Scale factor needed
        scale = (target_params / base_params) ** 0.5

        # Distribute between conv and MLP
        # Conv parameters scale roughly with square of channel multiplier
        # MLP parameters scale linearly with mlp multiplier

        channel_mult = min(scale ** 0.7, 16.0)  # Limit channel scaling
        mlp_mult = scale ** 1.3

        # Refine with binary search
        for _ in range(10):
            params = self._estimate_params(
                channel_mult, mlp_mult, input_channels, output_dim, image_size
            )
            ratio = target_params / params
            channel_mult *= ratio ** 0.3
            mlp_mult *= ratio ** 0.7

        return channel_mult, mlp_mult

    def _estimate_params(
        self,
        channel_mult: float,
        mlp_mult: float,
        input_channels: int,
        output_dim: int,
        image_size: int,
    ) -> int:
        """Estimate parameter count for given multipliers."""
        channels = [int(c * channel_mult) for c in [32, 64, 64]]
        kernels = [8, 4, 3]
        strides = [4, 2, 1]

        # Conv params
        params = 0
        in_ch = input_channels
        for out_ch, k in zip(channels, kernels):
            params += in_ch * out_ch * k * k + out_ch
            in_ch = out_ch

        # Calculate feature map size after conv
        size = image_size
        for k, s in zip(kernels, strides):
            size = (size - k) // s + 1

        conv_out = channels[-1] * size * size

        # MLP params
        mlp_hidden = int(512 * mlp_mult)
        params += conv_out * mlp_hidden + mlp_hidden
        params += mlp_hidden * output_dim + output_dim

        return params

    def _get_conv_output_size(self, channels: int, size: int) -> int:
        """Calculate the output size of conv layers."""
        with torch.no_grad():
            x = torch.zeros(1, channels, size, size)
            x = self.conv(x)
            return x.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        conv_out = self.conv(x)
        flat = conv_out.view(conv_out.size(0), -1)
        return self.mlp(flat)


class NatureDQN(CNN):
    """Nature DQN architecture (Mnih et al., 2015)."""

    def __init__(self, input_channels: int, output_dim: int):
        super().__init__(
            input_channels=input_channels,
            output_dim=output_dim,
            conv_channels=[32, 64, 64],
            conv_kernels=[8, 4, 3],
            conv_strides=[4, 2, 1],
            mlp_hidden_sizes=[512],
        )


class ImpalaCNN(BaseNetwork):
    """IMPALA-style CNN with residual blocks."""

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        channels: List[int] = [16, 32, 32],
        activation: str = "relu",
    ):
        super().__init__()
        self._input_dim = input_channels
        self._output_dim = output_dim

        # Build residual stacks
        layers = []
        in_channels = input_channels

        for out_channels in channels:
            layers.append(
                self._make_residual_block(in_channels, out_channels, activation)
            )
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)
        self.activation = self.get_activation(activation)

        # Calculate output size (assumes 64x64 input)
        self.fc = nn.LazyLinear(output_dim)

    def _make_residual_block(
        self, in_channels: int, out_channels: int, activation: str
    ) -> nn.Module:
        """Create a residual block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            _ResidualBlock(out_channels, activation),
            _ResidualBlock(out_channels, activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        x = self.conv(x)
        x = self.activation(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class _ResidualBlock(nn.Module):
    """Simple residual block for IMPALA CNN."""

    def __init__(self, channels: int, activation: str):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        if activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x + residual
