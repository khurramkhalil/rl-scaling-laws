"""Metrics and analysis utilities for scaling law research."""

from typing import List, Optional, Tuple

import numpy as np
import torch


def compute_returns(
    rewards: np.ndarray,
    gamma: float = 0.99,
) -> np.ndarray:
    """Compute discounted returns.

    Args:
        rewards: Array of rewards.
        gamma: Discount factor.

    Returns:
        Array of discounted returns.
    """
    returns = np.zeros_like(rewards)
    running_return = 0.0

    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    return returns


def compute_scaling_exponent(
    params: List[int],
    performance: List[float],
) -> Tuple[float, float, float]:
    """Fit power law: performance = A * params^alpha.

    Args:
        params: List of parameter counts.
        performance: List of performance values.

    Returns:
        Tuple of (alpha, A, r_squared).
    """
    log_params = np.log(params)
    log_perf = np.log(performance)

    # Linear regression in log space
    coeffs = np.polyfit(log_params, log_perf, 1)
    alpha = coeffs[0]
    A = np.exp(coeffs[1])

    # R-squared
    predicted = alpha * log_params + coeffs[1]
    ss_res = np.sum((log_perf - predicted) ** 2)
    ss_tot = np.sum((log_perf - np.mean(log_perf)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return alpha, A, r_squared


def compute_effective_rank(
    activations: torch.Tensor,
) -> float:
    """Compute effective rank of activations.

    Effective rank = exp(entropy of singular values)

    Args:
        activations: Tensor of shape (batch, features).

    Returns:
        Effective rank.
    """
    # Center activations
    activations = activations - activations.mean(dim=0)

    # SVD
    _, s, _ = torch.svd(activations)

    # Normalize singular values
    s = s / s.sum()

    # Entropy
    entropy = -(s * torch.log(s + 1e-10)).sum()

    return torch.exp(entropy).item()


def compute_td_error(
    q_values: torch.Tensor,
    rewards: torch.Tensor,
    next_q_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Compute TD error.

    Args:
        q_values: Current Q values.
        rewards: Rewards.
        next_q_values: Next state Q values.
        dones: Done flags.
        gamma: Discount factor.

    Returns:
        TD errors.
    """
    targets = rewards + (1 - dones) * gamma * next_q_values
    return torch.abs(q_values - targets)


def compute_gradient_stats(
    model: torch.nn.Module,
) -> dict:
    """Compute gradient statistics for a model.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary of gradient statistics.
    """
    grad_norms = []
    grad_means = []
    grad_stds = []

    for param in model.parameters():
        if param.grad is not None:
            grad = param.grad.detach().flatten()
            grad_norms.append(grad.norm().item())
            grad_means.append(grad.mean().item())
            grad_stds.append(grad.std().item())

    return {
        "grad_norm_mean": np.mean(grad_norms) if grad_norms else 0,
        "grad_norm_max": np.max(grad_norms) if grad_norms else 0,
        "grad_mean": np.mean(grad_means) if grad_means else 0,
        "grad_std": np.mean(grad_stds) if grad_stds else 0,
    }


def compute_dead_neurons(
    activations: torch.Tensor,
    threshold: float = 1e-6,
) -> float:
    """Compute fraction of dead neurons.

    Args:
        activations: Activation tensor.
        threshold: Variance threshold for dead neurons.

    Returns:
        Fraction of dead neurons.
    """
    variances = activations.var(dim=0)
    dead = (variances < threshold).float().mean()
    return dead.item()


def pareto_frontier(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Pareto frontier (maximize both).

    Args:
        x: First objective values.
        y: Second objective values.

    Returns:
        Tuple of (x_pareto, y_pareto).
    """
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    pareto_x = [x_sorted[0]]
    pareto_y = [y_sorted[0]]
    max_y = y_sorted[0]

    for i in range(1, len(x_sorted)):
        if y_sorted[i] > max_y:
            pareto_x.append(x_sorted[i])
            pareto_y.append(y_sorted[i])
            max_y = y_sorted[i]

    return np.array(pareto_x), np.array(pareto_y)
