from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta

import torch
from torch import Tensor, nn
from torch.distributed import destroy_process_group, init_process_group, is_initialized


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


@contextmanager
def distributed_context(
    device: str,
    backend: str,
    *,
    nccl_compatibility_mode: bool = False,
) -> Iterator[DistributedContext]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and backend == "nccl" and nccl_compatibility_mode:
        compatibility = {
            "NCCL_P2P_DISABLE": "1",
            "NCCL_SHM_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
            "NCCL_SOCKET_IFNAME": "lo",
        }
        for name, value in compatibility.items():
            os.environ.setdefault(name, value)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA runtime is not available")
        resolved = torch.device("cuda", local_rank if distributed else _single_device_index(device))
        torch.cuda.set_device(resolved)
    else:
        if distributed and backend == "nccl":
            raise ValueError("NCCL requires CUDA")
        resolved = torch.device(device)
    if distributed:
        init_process_group(
            backend=backend,
            init_method="env://",
            device_id=resolved if resolved.type == "cuda" else None,
            timeout=timedelta(hours=1),
        )
    try:
        yield DistributedContext(rank, local_rank, world_size, resolved)
    finally:
        if distributed and is_initialized():
            destroy_process_group()


def _single_device_index(device: str) -> int:
    parsed = torch.device(device)
    return 0 if parsed.index is None else parsed.index


def unwrap(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def reduce_sums(values: dict[str, Tensor]) -> dict[str, Tensor]:
    if not is_initialized():
        return values
    reduced = {key: value.detach().clone() for key, value in values.items()}
    for value in reduced.values():
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
    return reduced
