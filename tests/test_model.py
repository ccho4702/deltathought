from pathlib import Path

import torch

from deltaomni.config import load_config
from deltaomni.model import (
    DeltaCodecModel,
    PairDeltaEncoder,
    expand_embedding_delta,
    pool_embedding_delta,
)
from deltaomni.synthetic import SyntheticInterleavedDataset, collate_examples


def test_identity_delta_is_exactly_zero() -> None:
    config = load_config(Path("configs/sanity.yaml"))
    encoder = PairDeltaEncoder(config.model)
    embedding = torch.randn(2, config.model.embedding_tokens, config.model.embedding_dim)

    delta = encoder(embedding, embedding)

    assert delta.shape == (2, config.model.delta_tokens, config.model.embedding_dim)
    assert torch.count_nonzero(delta) == 0


def test_direct_delta_tokens_preserve_different_token_regions() -> None:
    config = load_config(Path("configs/sanity.yaml"))
    encoder = PairDeltaEncoder(config.model)
    anchor = torch.zeros(1, config.model.embedding_tokens, config.model.embedding_dim)
    current = torch.zeros_like(anchor)
    midpoint = config.model.embedding_tokens // 2
    current[:, :midpoint] = 1.0
    current[:, midpoint:] = -1.0

    delta = encoder(anchor, current)

    assert not torch.allclose(delta[:, 0], delta[:, -1])


def test_reconstructor_has_identity_initialized_direct_path() -> None:
    config = load_config(Path("configs/sanity.yaml"))
    model = DeltaCodecModel(config.model, config.modalities)
    reconstructor = model.codecs[config.modalities[0].value].reconstructor

    expected = torch.eye(config.model.embedding_dim)
    assert torch.equal(reconstructor.direct_projection.weight.detach(), expected)


def test_layout_aware_pooling_preserves_cls_and_2d_patch_regions() -> None:
    difference = torch.zeros(1, 17, 2)
    difference[:, 0] = 7.0
    patch_grid = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4, 1)
    difference[:, 1:, :1] = patch_grid.flatten(1, 2)

    pooled = pool_embedding_delta(difference, output_tokens=5)
    expanded = expand_embedding_delta(pooled, output_tokens=17)

    assert pooled.shape == (1, 5, 2)
    assert expanded.shape == difference.shape
    assert torch.equal(pooled[:, 0], difference[:, 0])
    assert torch.equal(expanded[:, 0], difference[:, 0])
    assert pooled[0, 1, 0] < pooled[0, 2, 0] < pooled[0, 4, 0]


def test_single_audio_token_uses_sequence_fallback() -> None:
    difference = torch.randn(2, 1, 4)

    pooled = pool_embedding_delta(difference, output_tokens=1)
    expanded = expand_embedding_delta(pooled, output_tokens=1)

    assert torch.equal(pooled, difference)
    assert torch.equal(expanded, difference)


def test_full_sequence_loss_has_finite_gradients() -> None:
    config = load_config(Path("configs/sanity.yaml"))
    dataset = SyntheticInterleavedDataset(config, 2, split_seed=123)
    batch = collate_examples([dataset[0], dataset[1]])
    model = DeltaCodecModel(config.model, config.modalities)

    losses = model.forward_sequence(**batch, weights=config.loss)
    losses.total.backward()

    assert torch.isfinite(losses.total)
    assert losses.caption.item() > 0
    trainable_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert trainable_gradients
    assert all(torch.isfinite(gradient).all() for gradient in trainable_gradients)
