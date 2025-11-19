"""Parameter counting utilities."""

from typing import Dict, Tuple

import torch
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model.

    Args:
        model: PyTorch model.
        trainable_only: If True, only count trainable parameters.

    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_parameters_by_layer(model: nn.Module) -> Dict[str, int]:
    """Count parameters for each named module.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary mapping layer names to parameter counts.
    """
    layer_params = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = sum(p.numel() for p in module.parameters(recurse=False))
            if params > 0:
                layer_params[name] = params
    return layer_params


def format_param_count(count: int) -> str:
    """Format parameter count with appropriate suffix.

    Args:
        count: Number of parameters.

    Returns:
        Formatted string (e.g., "10.5M", "1.2B").
    """
    if count >= 1e9:
        return f"{count / 1e9:.2f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.2f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.2f}K"
    return str(count)


def get_model_size_bytes(model: nn.Module) -> int:
    """Get model size in bytes.

    Args:
        model: PyTorch model.

    Returns:
        Size in bytes.
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return param_size + buffer_size


def compute_flops_per_forward(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cpu",
) -> int:
    """Estimate FLOPs for a single forward pass.

    This is a rough estimate based on linear and conv layers.

    Args:
        model: PyTorch model.
        input_shape: Input tensor shape (without batch dimension).
        device: Device to run computation on.

    Returns:
        Estimated FLOPs.
    """
    total_flops = 0

    def hook_fn(module, input, output):
        nonlocal total_flops

        if isinstance(module, nn.Linear):
            # FLOPs = 2 * input_features * output_features (multiply-add)
            total_flops += 2 * module.in_features * module.out_features

        elif isinstance(module, nn.Conv2d):
            # FLOPs = 2 * Cout * Hout * Wout * Cin * Kh * Kw
            output_size = output.shape[2] * output.shape[3]
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            total_flops += 2 * module.out_channels * output_size * kernel_ops

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            hooks.append(module.register_forward_hook(hook_fn))

    # Run forward pass
    model = model.to(device)
    x = torch.randn(1, *input_shape, device=device)
    with torch.no_grad():
        model(x)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return total_flops
