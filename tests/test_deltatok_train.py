from dataclasses import replace
from pathlib import Path

import torch

from deltaomni.deltatok_train import DeltaTok, load_config


def test_deltatok_reconstructs_shape_with_one_token() -> None:
    config = replace(
        load_config(Path("configs/deltatok_video_integration.yaml")),
        input_dim=32,
        model_dim=16,
        tokens_per_frame=8,
        depth=1,
        num_heads=4,
    )
    model = DeltaTok(config)
    previous = torch.randn(2, 8, 32)
    current = torch.randn(2, 8, 32)
    reconstructed, delta = model(previous, current)
    assert reconstructed.shape == current.shape
    assert delta.shape == (2, 1, 16)
