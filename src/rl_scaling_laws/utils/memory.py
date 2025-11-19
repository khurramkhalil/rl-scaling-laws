"""Mixed precision training and memory optimization utilities."""

from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from rl_scaling_laws.utils.logging import get_logger

logger = get_logger(__name__)


class MixedPrecisionTrainer:
    """Wrapper for mixed precision training with automatic scaling."""

    def __init__(
        self,
        enabled: bool = True,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        """Initialize mixed precision trainer.

        Args:
            enabled: Whether to use mixed precision.
            init_scale: Initial loss scale.
            growth_factor: Factor to increase scale.
            backoff_factor: Factor to decrease scale on overflow.
            growth_interval: Steps between scale increases.
        """
        self.enabled = enabled and torch.cuda.is_available()

        if self.enabled:
            self.scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
            )
            logger.info("Mixed precision training enabled")
        else:
            self.scaler = None

    @contextmanager
    def autocast_context(self):
        """Context manager for automatic mixed precision."""
        if self.enabled:
            with autocast():
                yield
        else:
            yield

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision."""
        if self.enabled:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Perform optimizer step with unscaling."""
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients before clipping."""
        if self.enabled:
            self.scaler.unscale_(optimizer)

    def get_scale(self) -> float:
        """Get current loss scale."""
        if self.enabled:
            return self.scaler.get_scale()
        return 1.0


class GradientCheckpointer:
    """Utilities for gradient checkpointing to save memory."""

    @staticmethod
    def checkpoint_sequential(
        functions: nn.Sequential,
        segments: int,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """Apply gradient checkpointing to sequential module.

        Args:
            functions: Sequential module.
            segments: Number of checkpoint segments.
            input: Input tensor.

        Returns:
            Output tensor.
        """
        from torch.utils.checkpoint import checkpoint_sequential

        return checkpoint_sequential(functions, segments, input)

    @staticmethod
    def make_checkpointed(module: nn.Module, num_segments: int = 2) -> nn.Module:
        """Wrap a module with gradient checkpointing.

        Args:
            module: Module to wrap.
            num_segments: Number of segments for checkpointing.

        Returns:
            Wrapped module.
        """
        class CheckpointedModule(nn.Module):
            def __init__(self, module, segments):
                super().__init__()
                self.module = module
                self.segments = segments

            def forward(self, x):
                if isinstance(self.module, nn.Sequential) and self.training:
                    return GradientCheckpointer.checkpoint_sequential(
                        self.module, self.segments, x
                    )
                return self.module(x)

        return CheckpointedModule(module, num_segments)


class MemoryTracker:
    """Track and manage GPU memory usage."""

    def __init__(self, device: torch.device):
        self.device = device
        self.peak_memory = 0
        self.checkpoints: Dict[str, float] = {}

    def reset_peak(self) -> None:
        """Reset peak memory tracking."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            self.peak_memory = 0

    def get_current(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device) / 1e6
        return 0.0

    def get_peak(self) -> float:
        """Get peak memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(self.device) / 1e6
        return 0.0

    def get_reserved(self) -> float:
        """Get reserved memory in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved(self.device) / 1e6
        return 0.0

    def checkpoint(self, name: str) -> None:
        """Save memory checkpoint."""
        self.checkpoints[name] = self.get_current()

    def get_summary(self) -> Dict[str, float]:
        """Get memory usage summary."""
        return {
            "current_mb": self.get_current(),
            "peak_mb": self.get_peak(),
            "reserved_mb": self.get_reserved(),
        }

    @contextmanager
    def track_memory(self, name: str):
        """Context manager to track memory for a block."""
        start = self.get_current()
        yield
        end = self.get_current()
        self.checkpoints[name] = end - start
        logger.debug(f"Memory for {name}: {end - start:.1f} MB")


def optimize_memory_for_scale(target_params: int) -> Dict[str, Any]:
    """Get recommended memory optimization settings for parameter scale.

    Args:
        target_params: Target number of parameters.

    Returns:
        Dictionary of optimization settings.
    """
    settings = {
        "use_amp": True,
        "gradient_checkpointing": False,
        "batch_size_factor": 1.0,
        "accumulation_steps": 1,
    }

    if target_params >= 1_000_000_000:  # 1B+
        settings.update({
            "gradient_checkpointing": True,
            "batch_size_factor": 0.25,
            "accumulation_steps": 4,
        })
    elif target_params >= 300_000_000:  # 300M+
        settings.update({
            "gradient_checkpointing": True,
            "batch_size_factor": 0.5,
            "accumulation_steps": 2,
        })
    elif target_params >= 100_000_000:  # 100M+
        settings.update({
            "batch_size_factor": 0.75,
        })

    return settings


def estimate_memory_usage(
    model: nn.Module,
    batch_size: int,
    input_shape: Tuple[int, ...],
    optimizer_type: str = "adam",
) -> Dict[str, float]:
    """Estimate memory usage for a model.

    Args:
        model: PyTorch model.
        batch_size: Batch size.
        input_shape: Input tensor shape (without batch).
        optimizer_type: Type of optimizer.

    Returns:
        Dictionary with memory estimates in MB.
    """
    # Parameter memory
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_mb = param_bytes / 1e6

    # Gradient memory (same as parameters)
    grad_mb = param_mb

    # Optimizer state memory
    if optimizer_type == "adam":
        # Adam stores m and v for each parameter
        optim_mb = 2 * param_mb
    elif optimizer_type == "sgd":
        optim_mb = 0
    else:
        optim_mb = param_mb  # Conservative estimate

    # Activation memory (rough estimate)
    # This varies greatly depending on model architecture
    input_bytes = batch_size * torch.prod(torch.tensor(input_shape)).item() * 4
    activation_mb = input_bytes / 1e6 * 10  # Rough multiplier

    total_mb = param_mb + grad_mb + optim_mb + activation_mb

    return {
        "parameters_mb": param_mb,
        "gradients_mb": grad_mb,
        "optimizer_mb": optim_mb,
        "activations_mb": activation_mb,
        "total_mb": total_mb,
    }
