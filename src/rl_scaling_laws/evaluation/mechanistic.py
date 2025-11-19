"""Mechanistic analysis tools for understanding scaling behavior."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from rl_scaling_laws.utils.logging import get_logger

logger = get_logger(__name__)


class ActivationExtractor:
    """Extract and analyze activations from neural networks."""

    def __init__(self, model: nn.Module):
        """Initialize activation extractor.

        Args:
            model: PyTorch model to extract activations from.
        """
        self.model = model
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks: List = []

    def register_hooks(self, layer_names: Optional[List[str]] = None) -> None:
        """Register forward hooks to capture activations.

        Args:
            layer_names: List of layer names to capture. If None, capture all.
        """
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations[name] = output.detach().cpu()
                elif isinstance(output, tuple):
                    self.activations[name] = output[0].detach().cpu()
            return hook

        for name, module in self.model.named_modules():
            if layer_names is None or name in layer_names:
                if len(list(module.children())) == 0:  # Leaf module
                    hook = module.register_forward_hook(make_hook(name))
                    self.hooks.append(hook)

        logger.info(f"Registered {len(self.hooks)} activation hooks")

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get captured activations."""
        return self.activations

    def clear_activations(self) -> None:
        """Clear stored activations."""
        self.activations = {}

    def compute_effective_rank(self, layer_name: str) -> float:
        """Compute effective rank of activations for a layer.

        Effective rank = exp(entropy of singular values)

        Args:
            layer_name: Name of layer.

        Returns:
            Effective rank.
        """
        if layer_name not in self.activations:
            raise ValueError(f"Layer {layer_name} not found")

        act = self.activations[layer_name]

        # Flatten to 2D: (batch, features)
        if act.dim() > 2:
            act = act.view(act.size(0), -1)

        # Center
        act = act - act.mean(dim=0)

        # SVD
        _, s, _ = torch.svd(act)

        # Normalize
        s = s / s.sum()

        # Entropy
        entropy = -(s * torch.log(s + 1e-10)).sum()

        return torch.exp(entropy).item()

    def compute_all_effective_ranks(self) -> Dict[str, float]:
        """Compute effective rank for all captured layers.

        Returns:
            Dictionary mapping layer names to effective ranks.
        """
        ranks = {}
        for name in self.activations:
            try:
                ranks[name] = self.compute_effective_rank(name)
            except Exception as e:
                logger.warning(f"Failed to compute rank for {name}: {e}")
        return ranks

    def compute_dead_neurons(
        self,
        layer_name: str,
        threshold: float = 1e-6,
    ) -> float:
        """Compute fraction of dead neurons.

        Args:
            layer_name: Name of layer.
            threshold: Variance threshold for dead neurons.

        Returns:
            Fraction of dead neurons.
        """
        if layer_name not in self.activations:
            raise ValueError(f"Layer {layer_name} not found")

        act = self.activations[layer_name]

        if act.dim() > 2:
            act = act.view(act.size(0), -1)

        variances = act.var(dim=0)
        dead = (variances < threshold).float().mean()

        return dead.item()

    def compute_feature_correlation(self, layer_name: str) -> float:
        """Compute average correlation between features.

        High correlation indicates feature collapse.

        Args:
            layer_name: Name of layer.

        Returns:
            Average absolute correlation.
        """
        if layer_name not in self.activations:
            raise ValueError(f"Layer {layer_name} not found")

        act = self.activations[layer_name]

        if act.dim() > 2:
            act = act.view(act.size(0), -1)

        # Correlation matrix
        act_centered = act - act.mean(dim=0)
        std = act_centered.std(dim=0) + 1e-8
        act_normalized = act_centered / std

        corr = (act_normalized.T @ act_normalized) / act.size(0)

        # Average off-diagonal correlation
        n = corr.size(0)
        mask = ~torch.eye(n, dtype=bool)
        avg_corr = corr[mask].abs().mean()

        return avg_corr.item()


class TDOverfittingDetector:
    """Detect TD-overfitting during training."""

    def __init__(
        self,
        window_size: int = 100,
        threshold: float = 0.1,
    ):
        """Initialize detector.

        Args:
            window_size: Window for computing statistics.
            threshold: Threshold for detecting overfitting.
        """
        self.window_size = window_size
        self.threshold = threshold

        # Track metrics
        self.train_td_errors: List[float] = []
        self.val_td_errors: List[float] = []
        self.q_values: List[float] = []
        self.q_stds: List[float] = []
        self.timesteps: List[int] = []

    def update(
        self,
        timestep: int,
        train_td_error: float,
        val_td_error: Optional[float] = None,
        q_value: Optional[float] = None,
        q_std: Optional[float] = None,
    ) -> None:
        """Update metrics.

        Args:
            timestep: Current timestep.
            train_td_error: Training TD error.
            val_td_error: Validation TD error.
            q_value: Mean Q value.
            q_std: Q value standard deviation.
        """
        self.timesteps.append(timestep)
        self.train_td_errors.append(train_td_error)

        if val_td_error is not None:
            self.val_td_errors.append(val_td_error)
        if q_value is not None:
            self.q_values.append(q_value)
        if q_std is not None:
            self.q_stds.append(q_std)

    def detect_overfitting(self) -> Dict[str, Any]:
        """Detect TD-overfitting.

        Returns:
            Detection results.
        """
        results = {
            "is_overfitting": False,
            "crossover_timestep": None,
            "train_val_gap": 0.0,
            "q_value_trend": "stable",
        }

        if len(self.val_td_errors) < self.window_size:
            return results

        # Check for train/val divergence
        train_recent = np.mean(self.train_td_errors[-self.window_size:])
        val_recent = np.mean(self.val_td_errors[-self.window_size:])

        gap = val_recent - train_recent
        results["train_val_gap"] = gap

        if gap > self.threshold * train_recent:
            results["is_overfitting"] = True

            # Find crossover point
            for i in range(len(self.val_td_errors)):
                if self.val_td_errors[i] > self.train_td_errors[i]:
                    results["crossover_timestep"] = self.timesteps[i]
                    break

        # Check Q value trend
        if len(self.q_values) >= self.window_size:
            q_start = np.mean(self.q_values[:self.window_size])
            q_end = np.mean(self.q_values[-self.window_size:])

            if q_end > q_start * 1.5:
                results["q_value_trend"] = "increasing"
            elif q_end < q_start * 0.5:
                results["q_value_trend"] = "decreasing"

        return results

    def get_metrics(self) -> Dict[str, List[float]]:
        """Get all tracked metrics.

        Returns:
            Dictionary of metric lists.
        """
        return {
            "timesteps": self.timesteps,
            "train_td_errors": self.train_td_errors,
            "val_td_errors": self.val_td_errors,
            "q_values": self.q_values,
            "q_stds": self.q_stds,
        }


class PlasticityTracker:
    """Track network plasticity during training."""

    def __init__(self, model: nn.Module):
        """Initialize plasticity tracker.

        Args:
            model: Model to track.
        """
        self.model = model
        self.gradient_history: Dict[str, List[float]] = {}
        self.weight_change_history: Dict[str, List[float]] = {}
        self.initial_weights: Dict[str, torch.Tensor] = {}

        # Store initial weights
        for name, param in model.named_parameters():
            self.initial_weights[name] = param.detach().clone()
            self.gradient_history[name] = []
            self.weight_change_history[name] = []

    def update(self) -> Dict[str, float]:
        """Update plasticity metrics after backward pass.

        Returns:
            Current plasticity metrics.
        """
        metrics = {}
        grad_norms = []
        weight_changes = []

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Gradient norm
                grad_norm = param.grad.norm().item()
                self.gradient_history[name].append(grad_norm)
                grad_norms.append(grad_norm)

                # Weight change from initialization
                weight_change = (param - self.initial_weights[name]).norm().item()
                self.weight_change_history[name].append(weight_change)
                weight_changes.append(weight_change)

        metrics["mean_grad_norm"] = np.mean(grad_norms) if grad_norms else 0
        metrics["max_grad_norm"] = np.max(grad_norms) if grad_norms else 0
        metrics["mean_weight_change"] = np.mean(weight_changes) if weight_changes else 0

        return metrics

    def compute_plasticity_loss(self) -> float:
        """Compute plasticity loss metric.

        Low plasticity = gradients become very small = network stops learning.

        Returns:
            Plasticity loss metric (higher = more plasticity loss).
        """
        if not self.gradient_history:
            return 0.0

        # Compare recent gradients to initial
        plasticity_losses = []

        for name in self.gradient_history:
            history = self.gradient_history[name]
            if len(history) < 100:
                continue

            initial_grad = np.mean(history[:50])
            recent_grad = np.mean(history[-50:])

            if initial_grad > 1e-8:
                ratio = recent_grad / initial_grad
                # Loss is high when recent gradients are much smaller
                plasticity_losses.append(1 - min(ratio, 1.0))

        return np.mean(plasticity_losses) if plasticity_losses else 0.0


def analyze_model_representations(
    model: nn.Module,
    data_loader,
    device: torch.device,
    n_batches: int = 10,
) -> Dict[str, Any]:
    """Comprehensive representation analysis.

    Args:
        model: Model to analyze.
        data_loader: Data loader for computing activations.
        device: Device for computation.
        n_batches: Number of batches to analyze.

    Returns:
        Analysis results.
    """
    extractor = ActivationExtractor(model)
    extractor.register_hooks()

    model.eval()
    all_activations: Dict[str, List[torch.Tensor]] = {}

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= n_batches:
                break

            if isinstance(batch, (tuple, list)):
                inputs = batch[0].to(device)
            else:
                inputs = batch.to(device)

            model(inputs)

            for name, act in extractor.get_activations().items():
                if name not in all_activations:
                    all_activations[name] = []
                all_activations[name].append(act)

            extractor.clear_activations()

    extractor.remove_hooks()

    # Analyze concatenated activations
    results = {"layers": {}}

    for name, acts in all_activations.items():
        concat_act = torch.cat(acts, dim=0)
        extractor.activations[name] = concat_act

        results["layers"][name] = {
            "effective_rank": extractor.compute_effective_rank(name),
            "dead_neurons": extractor.compute_dead_neurons(name),
            "feature_correlation": extractor.compute_feature_correlation(name),
            "mean_activation": concat_act.mean().item(),
            "std_activation": concat_act.std().item(),
        }

    # Summary statistics
    ranks = [r["effective_rank"] for r in results["layers"].values()]
    results["summary"] = {
        "mean_effective_rank": np.mean(ranks),
        "min_effective_rank": np.min(ranks),
        "n_layers": len(results["layers"]),
    }

    return results
