"""Distributed training utilities for multi-GPU and multi-node training."""

import os
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from rl_scaling_laws.utils.logging import get_logger

logger = get_logger(__name__)


def setup_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
) -> bool:
    """Initialize distributed training.

    Args:
        backend: Distributed backend (nccl, gloo, mpi).
        init_method: Initialization method.

    Returns:
        True if distributed training is initialized.
    """
    if not dist.is_available():
        logger.warning("Distributed package not available")
        return False

    if dist.is_initialized():
        return True

    # Check for environment variables
    if "RANK" not in os.environ:
        logger.info("RANK not set, running in non-distributed mode")
        return False

    try:
        dist.init_process_group(backend=backend, init_method=init_method)

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        logger.info(
            f"Distributed training initialized: rank={rank}, "
            f"world_size={world_size}, local_rank={local_rank}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to initialize distributed training: {e}")
        return False


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main process."""
    return get_rank() == 0


def wrap_model_ddp(
    model: nn.Module,
    device_id: Optional[int] = None,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """Wrap model with DistributedDataParallel.

    Args:
        model: Model to wrap.
        device_id: GPU device ID.
        find_unused_parameters: Find unused parameters in backward.

    Returns:
        Wrapped model.
    """
    if not dist.is_initialized():
        return model

    if device_id is None:
        device_id = int(os.environ.get("LOCAL_RANK", 0))

    model = model.to(device_id)
    model = DDP(
        model,
        device_ids=[device_id],
        find_unused_parameters=find_unused_parameters,
    )

    return model


def all_reduce_dict(
    data: Dict[str, float],
    op: str = "mean",
) -> Dict[str, float]:
    """All-reduce a dictionary of values across processes.

    Args:
        data: Dictionary of values.
        op: Reduction operation (mean, sum, max, min).

    Returns:
        Reduced dictionary.
    """
    if not dist.is_initialized():
        return data

    world_size = get_world_size()
    result = {}

    for key, value in data.items():
        tensor = torch.tensor(value, dtype=torch.float32)
        if torch.cuda.is_available():
            tensor = tensor.cuda()

        if op == "mean":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor = tensor / world_size
        elif op == "sum":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        elif op == "max":
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        elif op == "min":
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)

        result[key] = tensor.item()

    return result


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast object from source process.

    Args:
        obj: Object to broadcast.
        src: Source rank.

    Returns:
        Broadcasted object.
    """
    if not dist.is_initialized():
        return obj

    object_list = [obj] if get_rank() == src else [None]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


class DistributedSampler:
    """Simple distributed sampler for replay buffers."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        """Initialize sampler.

        Args:
            dataset_size: Total dataset size.
            batch_size: Batch size per process.
            shuffle: Whether to shuffle.
            drop_last: Drop incomplete batches.
        """
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.rank = get_rank()
        self.world_size = get_world_size()
        self.num_samples = dataset_size // self.world_size

    def __iter__(self):
        """Generate indices for this process."""
        import numpy as np

        if self.shuffle:
            indices = np.random.permutation(self.dataset_size)
        else:
            indices = np.arange(self.dataset_size)

        # Partition indices
        indices = indices[self.rank:self.dataset_size:self.world_size]

        # Generate batches
        for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            yield indices[i:i + self.batch_size]

    def __len__(self):
        return self.num_samples // self.batch_size


def sync_gradients(model: nn.Module) -> None:
    """Synchronize gradients across processes.

    Args:
        model: Model with gradients.
    """
    if not dist.is_initialized():
        return

    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(get_world_size())
