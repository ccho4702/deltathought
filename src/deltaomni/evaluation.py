from __future__ import annotations

import torch
from torch import Tensor


def cross_label_permutations(labels: Tensor, *, repeats: int, seed: int) -> Tensor:
    """Build reproducible permutations whose source label always differs from the target label.

    The current SSV2 pilots use balanced class splits. Requiring balance makes the negative control
    exact: every example is replaced once, no source is duplicated, and no replacement retains the
    target class. A plain roll is invalid for class-grouped manifests because it preserves the class
    for most examples.
    """

    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("labels must be a non-empty rank-one tensor")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    labels_cpu = labels.detach().to(device="cpu", dtype=torch.long)
    classes = torch.unique(labels_cpu, sorted=True)
    if classes.numel() < 2:
        raise ValueError("cross-label permutations require at least two classes")
    groups = [torch.nonzero(labels_cpu == value, as_tuple=False).flatten() for value in classes]
    group_size = groups[0].numel()
    if group_size == 0 or any(group.numel() != group_size for group in groups):
        raise ValueError("cross-label permutations require balanced class counts")

    generator = torch.Generator().manual_seed(seed)
    permutations = []
    class_count = len(groups)
    for repeat in range(repeats):
        shift = 1 + repeat % (class_count - 1)
        permutation = torch.empty_like(labels_cpu)
        for class_index, targets in enumerate(groups):
            sources = groups[(class_index + shift) % class_count]
            target_order = targets[torch.randperm(group_size, generator=generator)]
            source_order = sources[torch.randperm(group_size, generator=generator)]
            permutation[target_order] = source_order
        if not torch.equal(torch.sort(permutation).values, torch.arange(labels_cpu.numel())):
            raise RuntimeError("constructed indices are not a permutation")
        if torch.any(labels_cpu[permutation] == labels_cpu):
            raise RuntimeError("constructed permutation retained a target class")
        permutations.append(permutation)
    return torch.stack(permutations).to(labels.device)
