"""Learning rate and hyperparameter schedules."""

import math
from typing import Callable


def linear_schedule(
    initial_value: float,
    final_value: float = 0.0,
) -> Callable[[float], float]:
    """Create a linear schedule function.

    Args:
        initial_value: Starting value.
        final_value: Ending value.

    Returns:
        Function that takes progress (0 to 1) and returns scheduled value.
    """
    def schedule(progress: float) -> float:
        return initial_value + progress * (final_value - initial_value)

    return schedule


def cosine_schedule(
    initial_value: float,
    final_value: float = 0.0,
    warmup_fraction: float = 0.0,
) -> Callable[[float], float]:
    """Create a cosine annealing schedule with optional warmup.

    Args:
        initial_value: Starting value (after warmup).
        final_value: Ending value.
        warmup_fraction: Fraction of training for linear warmup.

    Returns:
        Function that takes progress (0 to 1) and returns scheduled value.
    """
    def schedule(progress: float) -> float:
        if progress < warmup_fraction:
            # Linear warmup
            return initial_value * (progress / warmup_fraction)

        # Cosine annealing
        cosine_progress = (progress - warmup_fraction) / (1 - warmup_fraction)
        return final_value + 0.5 * (initial_value - final_value) * (
            1 + math.cos(math.pi * cosine_progress)
        )

    return schedule


def exponential_schedule(
    initial_value: float,
    decay_rate: float,
) -> Callable[[float], float]:
    """Create an exponential decay schedule.

    Args:
        initial_value: Starting value.
        decay_rate: Rate of decay (value at progress=1 is initial_value * decay_rate).

    Returns:
        Function that takes progress (0 to 1) and returns scheduled value.
    """
    def schedule(progress: float) -> float:
        return initial_value * (decay_rate ** progress)

    return schedule


def step_schedule(
    initial_value: float,
    step_size: float,
    gamma: float,
) -> Callable[[int], float]:
    """Create a step decay schedule.

    Args:
        initial_value: Starting value.
        step_size: Number of steps between decays.
        gamma: Multiplicative factor for decay.

    Returns:
        Function that takes step number and returns scheduled value.
    """
    def schedule(step: int) -> float:
        return initial_value * (gamma ** (step // step_size))

    return schedule
