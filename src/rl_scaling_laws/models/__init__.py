"""Scalable neural network architectures for RL."""

from rl_scaling_laws.models.registry import ModelRegistry, get_model, register_model
from rl_scaling_laws.models.base import BaseNetwork
from rl_scaling_laws.models.mlp import MLP, ScalableMLP
from rl_scaling_laws.models.cnn import CNN, ScalableCNN
from rl_scaling_laws.models.transformer import TransformerEncoder, ScalableTransformer
from rl_scaling_laws.models.moe import MoELayer, ScalableMoE
from rl_scaling_laws.models.builder import NetworkBuilder

__all__ = [
    "ModelRegistry",
    "get_model",
    "register_model",
    "BaseNetwork",
    "MLP",
    "ScalableMLP",
    "CNN",
    "ScalableCNN",
    "TransformerEncoder",
    "ScalableTransformer",
    "MoELayer",
    "ScalableMoE",
    "NetworkBuilder",
]
