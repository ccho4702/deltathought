from pathlib import Path

import torch

from deltaomni.config import load_config
from deltaomni.model import DeltaCodecModel, PairDeltaEncoder
from deltaomni.synthetic import SyntheticInterleavedDataset, collate_examples


def test_identity_delta_is_exactly_zero() -> None:
    config = load_config(Path("configs/sanity.yaml"))
    encoder = PairDeltaEncoder(config.model)
    embedding = torch.randn(2, config.model.embedding_tokens, config.model.embedding_dim)

    delta = encoder(embedding, embedding)

    assert delta.shape == (2, config.model.delta_tokens, config.model.embedding_dim)
    assert torch.count_nonzero(delta) == 0


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

