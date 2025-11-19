"""Scalable Transformer architectures for RL."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_scaling_laws.models.base import BaseNetwork
from rl_scaling_laws.models.registry import register_model


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class TransformerBlock(nn.Module):
    """Single Transformer block with self-attention and MLP."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.pre_norm = pre_norm

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim, num_heads, dropout=attention_dropout
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pre_norm:
            x = x + self.attn(self.norm1(x), mask)
            x = x + self.mlp(self.norm2(x))
        else:
            x = self.norm1(x + self.attn(x, mask))
            x = self.norm2(x + self.mlp(x))
        return x


class TransformerEncoder(BaseNetwork):
    """Transformer encoder for RL observations."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        pre_norm: bool = True,
        pos_encoding: str = "learned",
    ):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Position encoding
        if pos_encoding == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_buffer(
                "pos_embed",
                self._sinusoidal_encoding(max_seq_len, embed_dim)
            )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio,
                dropout, attention_dropout, pre_norm
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, output_dim)

    def _sinusoidal_encoding(self, max_len: int, embed_dim: int) -> torch.Tensor:
        """Generate sinusoidal position encodings."""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )

        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim) or (batch, input_dim)

        # Handle single observation (not sequence)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        B, N, _ = x.shape

        # Project input
        x = self.input_proj(x)

        # Add position encoding
        x = x + self.pos_embed[:, :N]

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)

        # Pool over sequence (take mean or last token)
        x = x.mean(dim=1)

        # Output projection
        return self.output_proj(x)


@register_model("transformer")
class ScalableTransformer(BaseNetwork):
    """Transformer with automatic scaling to target parameter count."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        target_params: int,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        pre_norm: bool = True,
        init_method: str = "xavier",
    ):
        """Initialize scalable Transformer.

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            target_params: Target number of parameters.
            max_seq_len: Maximum sequence length.
            dropout: Dropout probability.
            attention_dropout: Attention dropout probability.
            pre_norm: Use pre-normalization.
            init_method: Weight initialization method.
        """
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.target_params = target_params

        # Compute optimal architecture
        embed_dim, num_heads, num_layers = self._compute_architecture(
            input_dim, output_dim, target_params
        )

        # Build model
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Learned position encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        mlp_ratio = 4.0
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio,
                dropout, attention_dropout, pre_norm
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, output_dim)

        # Initialize
        self.apply_init(init_method)

        # Store architecture info
        self.architecture_info = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "mlp_ratio": mlp_ratio,
            "target_params": target_params,
            "actual_params": self.num_parameters,
        }

    def _compute_architecture(
        self,
        input_dim: int,
        output_dim: int,
        target_params: int,
    ) -> Tuple[int, int, int]:
        """Compute transformer dimensions to match target params."""
        best_config = (256, 4, 4)
        best_diff = float("inf")

        # Search over configurations
        for num_layers in [2, 4, 6, 8, 12, 16, 24]:
            for embed_dim in [128, 256, 384, 512, 768, 1024, 1536, 2048]:
                # Number of heads (must divide embed_dim)
                for num_heads in [4, 8, 12, 16]:
                    if embed_dim % num_heads != 0:
                        continue

                    params = self._count_params(
                        input_dim, output_dim, embed_dim, num_heads, num_layers
                    )

                    diff = abs(params - target_params)
                    if diff < best_diff:
                        best_diff = diff
                        best_config = (embed_dim, num_heads, num_layers)

        return best_config

    def _count_params(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
    ) -> int:
        """Count parameters for given transformer config."""
        params = 0

        # Input projection
        params += input_dim * embed_dim + embed_dim

        # Per-layer params
        mlp_hidden = int(embed_dim * mlp_ratio)

        # Attention: QKV + output projection
        attn_params = 4 * embed_dim * embed_dim + 4 * embed_dim

        # MLP
        mlp_params = 2 * embed_dim * mlp_hidden + embed_dim + mlp_hidden

        # LayerNorms
        ln_params = 4 * embed_dim

        params += num_layers * (attn_params + mlp_params + ln_params)

        # Final norm
        params += 2 * embed_dim

        # Output projection
        params += embed_dim * output_dim + output_dim

        return params

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        B, N, _ = x.shape

        x = self.input_proj(x)
        x = x + self.pos_embed[:, :N]

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)
        x = x.mean(dim=1)

        return self.output_proj(x)
