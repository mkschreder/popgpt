"""Device and system utilities."""

import os
from contextlib import nullcontext
from typing import Any

import torch


def setup_device(device: str, dtype: str) -> tuple[str, str, Any, torch.dtype]:
    """Setup device and precision context.

    Args:
        device: Device string ('cpu', 'cuda', etc.)
        dtype: Data type string ('float32', 'bfloat16', 'float16')

    Returns:
        Tuple of (device, device_type, autocast_context, torch_dtype)
    """
    # Check if CUDA is requested but not available
    if "cuda" in device and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available, falling back to CPU")
        device = "cpu"
        device_type = "cpu"
    else:
        device_type = "cuda" if "cuda" in device else "cpu"

    # Map dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    ptdtype = dtype_map[dtype]

    # Create autocast context
    if device_type == "cpu":
        ctx = nullcontext()
    else:
        ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Enable tf32 for performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    return device, device_type, ctx, ptdtype


def setup_ddp() -> tuple[bool, int, int, int, bool, int]:
    """Setup distributed data parallel training.

    Returns:
        Tuple of (ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, seed_offset)
    """
    ddp = int(os.environ.get("RANK", -1)) != -1

    if ddp:
        from torch.distributed import init_process_group

        init_process_group(backend=os.environ.get("DDP_BACKEND", "nccl"))
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, seed_offset


def cleanup_ddp(ddp: bool) -> None:
    """Cleanup distributed data parallel training.

    Args:
        ddp: Whether DDP was enabled
    """
    if ddp:
        from torch.distributed import destroy_process_group

        destroy_process_group()
