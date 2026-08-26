import pytest
import torch

from deltaomni.evaluation import cross_label_permutations


def test_cross_label_permutations_are_bijective_balanced_negatives() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    first = cross_label_permutations(labels, repeats=6, seed=42)
    second = cross_label_permutations(labels, repeats=6, seed=42)

    assert torch.equal(first, second)
    assert first.shape == (6, labels.numel())
    for permutation in first:
        assert torch.equal(torch.sort(permutation).values, torch.arange(labels.numel()))
        assert torch.all(labels[permutation] != labels)


@pytest.mark.parametrize(
    "labels",
    [torch.tensor([]), torch.tensor([0, 0]), torch.tensor([0, 0, 0, 1])],
)
def test_cross_label_permutations_reject_invalid_splits(labels: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        cross_label_permutations(labels, repeats=1, seed=0)
