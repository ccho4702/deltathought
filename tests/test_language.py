import torch

from deltaomni.language import DeltaLanguageProjector


def test_delta_language_prefix_preserves_full_then_delta_order() -> None:
    projector = DeltaLanguageProjector(input_dim=4, language_dim=8)
    anchor = torch.zeros(2, 3, 4)
    delta = torch.ones(2, 2, 4)

    prefix = projector(anchor, delta, modality_index=1)

    assert prefix.shape == (2, 5, 8)
    assert not torch.equal(prefix[:, :3].mean(dim=1), prefix[:, 3:].mean(dim=1))
    prefix.sum().backward()
    assert projector.projection[1].weight.grad is not None

