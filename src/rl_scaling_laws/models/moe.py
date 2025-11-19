"""Scalable Mixture of Experts (MoE) architectures for RL."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_scaling_laws.models.base import BaseNetwork
from rl_scaling_laws.models.registry import register_model


class Expert(nn.Module):
    """Single expert MLP."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "gelu",
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class TopKRouter(nn.Module):
    """Top-k router for MoE layers."""

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        jitter: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter = jitter

        self.gate = nn.Linear(input_dim, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing weights.

        Returns:
            gates: Routing weights for each expert
            indices: Top-k expert indices
            load: Load on each expert (for balancing loss)
        """
        # Add noise for load balancing during training
        if self.training and self.jitter > 0:
            x = x + torch.randn_like(x) * self.jitter

        # Compute gate logits
        logits = self.gate(x)  # (batch, num_experts)

        # Get top-k experts
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)

        # Softmax over top-k
        gates = F.softmax(top_k_logits, dim=-1)

        # Compute load for balancing loss
        load = F.softmax(logits, dim=-1).mean(dim=0)

        return gates, indices, load


class MoELayer(nn.Module):
    """Mixture of Experts layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        jitter: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Create experts
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim, activation)
            for _ in range(num_experts)
        ])

        # Router
        self.router = TopKRouter(input_dim, num_experts, top_k, jitter)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with routing.

        Returns:
            output: MoE layer output
            load_balance_loss: Auxiliary loss for load balancing
        """
        batch_size = x.size(0)

        # Get routing weights
        gates, indices, load = self.router(x)

        # Compute expert outputs
        # For efficiency, we compute all experts and mask
        # (for small num_experts; use sparse computation for large)

        expert_outputs = torch.stack([
            expert(x) for expert in self.experts
        ], dim=1)  # (batch, num_experts, output_dim)

        # Gather top-k expert outputs
        batch_idx = torch.arange(batch_size, device=x.device).unsqueeze(1)
        selected_outputs = expert_outputs[batch_idx, indices]  # (batch, top_k, output_dim)

        # Weighted sum
        output = (gates.unsqueeze(-1) * selected_outputs).sum(dim=1)

        # Load balancing loss (encourage uniform expert usage)
        ideal_load = 1.0 / self.num_experts
        load_balance_loss = ((load - ideal_load) ** 2).sum() * self.num_experts

        return output, load_balance_loss


class MoEBlock(nn.Module):
    """MoE block with residual connection and layer norm."""

    def __init__(
        self,
        hidden_dim: int,
        expert_hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        jitter: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.moe = MoELayer(
            hidden_dim, expert_hidden_dim, hidden_dim,
            num_experts, top_k, jitter, activation
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.norm(x)
        x, aux_loss = self.moe(x)
        return x + residual, aux_loss


@register_model("moe")
class ScalableMoE(BaseNetwork):
    """MoE network with automatic scaling to target parameter count."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        target_params: int,
        num_experts: int = 8,
        top_k: int = 2,
        jitter: float = 0.0,
        activation: str = "gelu",
        load_balance_weight: float = 0.01,
        init_method: str = "xavier",
    ):
        """Initialize scalable MoE network.

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            target_params: Target number of parameters.
            num_experts: Number of experts.
            top_k: Number of experts to use per token.
            jitter: Noise for load balancing.
            activation: Activation function.
            load_balance_weight: Weight for auxiliary loss.
            init_method: Weight initialization method.
        """
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.target_params = target_params
        self.load_balance_weight = load_balance_weight

        # Compute architecture
        hidden_dim, expert_hidden, num_layers = self._compute_architecture(
            input_dim, output_dim, target_params, num_experts
        )

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # MoE blocks
        self.blocks = nn.ModuleList([
            MoEBlock(
                hidden_dim, expert_hidden, num_experts,
                top_k, jitter, activation
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Initialize
        self.apply_init(init_method)

        # Store architecture info
        self.architecture_info = {
            "hidden_dim": hidden_dim,
            "expert_hidden_dim": expert_hidden,
            "num_layers": num_layers,
            "num_experts": num_experts,
            "top_k": top_k,
            "target_params": target_params,
            "actual_params": self.num_parameters,
        }

    def _compute_architecture(
        self,
        input_dim: int,
        output_dim: int,
        target_params: int,
        num_experts: int,
    ) -> Tuple[int, int, int]:
        """Compute MoE dimensions to match target params."""
        best_config = (256, 256, 2)
        best_diff = float("inf")

        # Search over configurations
        for num_layers in [1, 2, 3, 4, 6, 8]:
            for hidden_dim in [128, 256, 384, 512, 768, 1024]:
                for expert_hidden in [128, 256, 512, 1024, 2048]:
                    params = self._count_params(
                        input_dim, output_dim, hidden_dim,
                        expert_hidden, num_layers, num_experts
                    )

                    diff = abs(params - target_params)
                    if diff < best_diff:
                        best_diff = diff
                        best_config = (hidden_dim, expert_hidden, num_layers)

        return best_config

    def _count_params(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        expert_hidden: int,
        num_layers: int,
        num_experts: int,
    ) -> int:
        """Count parameters for given MoE config."""
        params = 0

        # Input projection
        params += input_dim * hidden_dim + hidden_dim

        # Per-layer params
        for _ in range(num_layers):
            # LayerNorm
            params += 2 * hidden_dim

            # Router
            params += hidden_dim * num_experts

            # Experts
            expert_params = (
                hidden_dim * expert_hidden + expert_hidden +
                expert_hidden * hidden_dim + hidden_dim
            )
            params += num_experts * expert_params

        # Final norm
        params += 2 * hidden_dim

        # Output projection
        params += hidden_dim * output_dim + output_dim

        return params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Note: During training, access self.aux_loss for the load balancing loss.
        """
        x = self.input_proj(x)

        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss += aux_loss

        x = self.norm(x)
        output = self.output_proj(x)

        # Store auxiliary loss for training
        self.aux_loss = total_aux_loss * self.load_balance_weight

        return output

    def get_aux_loss(self) -> torch.Tensor:
        """Get the auxiliary load balancing loss."""
        return getattr(self, "aux_loss", torch.tensor(0.0))
