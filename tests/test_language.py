import torch

from deltaomni.language import ChangeAwareResampler, DeltaLanguageProjector


def test_delta_language_prefix_preserves_full_then_delta_order() -> None:
    projector = DeltaLanguageProjector(input_dim=4, language_dim=8)
    anchor = torch.zeros(2, 3, 4)
    delta = torch.ones(2, 2, 4)

    prefix = projector(anchor, delta, modality_index=1)

    assert prefix.shape == (2, 5, 8)
    assert not torch.equal(prefix[:, :3].mean(dim=1), prefix[:, 3:].mean(dim=1))
    prefix.sum().backward()
    assert projector.projection[1].weight.grad is not None


def test_change_aware_resampler_compresses_full_and_delta_evidence() -> None:
    resampler = ChangeAwareResampler(
        input_dim=4,
        language_dim=8,
        query_tokens=2,
        num_heads=2,
    )
    anchor = torch.randn(3, 7, 4)
    delta = torch.randn(3, 2, 4)

    prefix = resampler(anchor, delta, modality_index=1)

    assert prefix.shape == (3, 2, 8)
    prefix.square().mean().backward()
    assert resampler.queries.grad is not None
