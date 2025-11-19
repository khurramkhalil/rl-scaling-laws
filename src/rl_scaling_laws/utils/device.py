"""Device utilities."""

from typing import Any, Union

import torch


def get_device(device: Union[str, torch.device, None] = None) -> torch.device:
    """Get the appropriate device for computation.

    Args:
        device: Specified device. If None, automatically selects GPU if available.

    Returns:
        PyTorch device.
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(device)


def to_device(
    data: Any,
    device: Union[str, torch.device],
    non_blocking: bool = True,
) -> Any:
    """Move data to specified device.

    Handles tensors, dicts, lists, and tuples recursively.

    Args:
        data: Data to move (tensor, dict, list, or tuple).
        device: Target device.
        non_blocking: Use non-blocking transfer for CUDA.

    Returns:
        Data on the specified device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, dict):
        return {k: to_device(v, device, non_blocking) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device, non_blocking) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_device(v, device, non_blocking) for v in data)
    return data


def get_gpu_memory_usage() -> dict:
    """Get current GPU memory usage.

    Returns:
        Dictionary with memory statistics (in MB).
    """
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "total": 0}

    return {
        "allocated": torch.cuda.memory_allocated() / 1e6,
        "reserved": torch.cuda.memory_reserved() / 1e6,
        "total": torch.cuda.get_device_properties(0).total_memory / 1e6,
    }


def empty_cache() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
