"""Tests for scalable model architectures."""

import pytest
import torch

from rl_scaling_laws.models import (
    ScalableMLP,
    ScalableCNN,
    ScalableTransformer,
    ScalableMoE,
    NetworkBuilder,
)
from rl_scaling_laws.utils.param_count import count_parameters


class TestScalableMLP:
    """Tests for ScalableMLP architecture."""

    def test_basic_construction(self):
        """Test basic MLP construction."""
        model = ScalableMLP(
            input_dim=10,
            output_dim=5,
            target_params=10_000,
        )

        assert model is not None
        assert model.input_dim == 10
        assert model.output_dim == 5

    def test_forward_pass(self):
        """Test forward pass shape."""
        model = ScalableMLP(
            input_dim=10,
            output_dim=5,
            target_params=10_000,
        )

        x = torch.randn(32, 10)
        output = model(x)

        assert output.shape == (32, 5)

    def test_parameter_scaling(self):
        """Test that model scales to target parameters."""
        targets = [10_000, 100_000, 1_000_000]

        for target in targets:
            model = ScalableMLP(
                input_dim=10,
                output_dim=5,
                target_params=target,
            )

            actual = count_parameters(model)
            # Allow 10% tolerance
            assert abs(actual - target) / target < 0.1, \
                f"Target: {target}, Actual: {actual}"

    def test_architecture_info(self):
        """Test architecture info is stored."""
        model = ScalableMLP(
            input_dim=10,
            output_dim=5,
            target_params=50_000,
        )

        assert hasattr(model, "architecture_info")
        assert "num_layers" in model.architecture_info
        assert "hidden_size" in model.architecture_info


class TestScalableCNN:
    """Tests for ScalableCNN architecture."""

    def test_basic_construction(self):
        """Test basic CNN construction."""
        model = ScalableCNN(
            input_channels=4,
            output_dim=18,
            target_params=1_000_000,
        )

        assert model is not None

    def test_forward_pass(self):
        """Test forward pass with image input."""
        model = ScalableCNN(
            input_channels=4,
            output_dim=18,
            target_params=1_000_000,
        )

        x = torch.randn(8, 4, 84, 84)
        output = model(x)

        assert output.shape == (8, 18)

    def test_parameter_scaling(self):
        """Test CNN scales to target parameters."""
        model = ScalableCNN(
            input_channels=4,
            output_dim=18,
            target_params=5_000_000,
        )

        actual = count_parameters(model)
        # Allow 20% tolerance for CNN (harder to match exactly)
        assert abs(actual - 5_000_000) / 5_000_000 < 0.2


class TestScalableTransformer:
    """Tests for ScalableTransformer architecture."""

    def test_basic_construction(self):
        """Test basic Transformer construction."""
        model = ScalableTransformer(
            input_dim=10,
            output_dim=5,
            target_params=1_000_000,
        )

        assert model is not None

    def test_forward_pass_2d(self):
        """Test forward pass with 2D input."""
        model = ScalableTransformer(
            input_dim=10,
            output_dim=5,
            target_params=500_000,
        )

        x = torch.randn(8, 10)
        output = model(x)

        assert output.shape == (8, 5)

    def test_forward_pass_3d(self):
        """Test forward pass with sequence input."""
        model = ScalableTransformer(
            input_dim=10,
            output_dim=5,
            target_params=500_000,
        )

        x = torch.randn(8, 16, 10)  # (batch, seq, features)
        output = model(x)

        assert output.shape == (8, 5)


class TestScalableMoE:
    """Tests for ScalableMoE architecture."""

    def test_basic_construction(self):
        """Test basic MoE construction."""
        model = ScalableMoE(
            input_dim=10,
            output_dim=5,
            target_params=1_000_000,
            num_experts=8,
        )

        assert model is not None

    def test_forward_pass(self):
        """Test forward pass."""
        model = ScalableMoE(
            input_dim=10,
            output_dim=5,
            target_params=500_000,
            num_experts=4,
        )

        x = torch.randn(32, 10)
        output = model(x)

        assert output.shape == (32, 5)

    def test_auxiliary_loss(self):
        """Test that auxiliary loss is computed."""
        model = ScalableMoE(
            input_dim=10,
            output_dim=5,
            target_params=500_000,
        )

        x = torch.randn(32, 10)
        _ = model(x)

        aux_loss = model.get_aux_loss()
        assert aux_loss is not None


class TestNetworkBuilder:
    """Tests for NetworkBuilder utility."""

    def test_build_actor(self):
        """Test building actor network."""
        builder = NetworkBuilder("mlp", target_params=50_000)
        actor = builder.build_actor(10, 4)

        assert actor is not None
        assert actor.input_dim == 10
        assert actor.output_dim == 4

    def test_build_critic(self):
        """Test building critic network."""
        builder = NetworkBuilder("mlp", target_params=50_000)
        critic = builder.build_critic(14, 1)  # obs + action

        assert critic is not None

    def test_different_architectures(self):
        """Test building with different architectures."""
        for arch in ["mlp", "transformer", "moe"]:
            builder = NetworkBuilder(arch, target_params=100_000)
            network = builder.build_actor(10, 5)

            assert network is not None
            assert count_parameters(network) > 0
