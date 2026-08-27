from pathlib import Path

import torch

from deltaomni.deltatok_scale_train import (
    ModelConfig,
    PairDataset,
    _evaluate_rollout,
    load_config,
)
from deltaomni.deltatok_train import DeltaTok


def test_scale_config_uses_one_delta_token_and_distributed_effective_batch() -> None:
    config = load_config(Path("configs/deltatok_vggsound_video.yaml"))

    assert config.modality == "video"
    assert config.model.delta_tokens == 1
    assert config.model.depth == 12
    assert config.training.max_steps == 50_000
    effective_batch = (
        config.runtime.per_device_batch_size * 4 * config.runtime.gradient_accumulation_steps
    )
    assert effective_batch == 128


def test_pair_dataset_exposes_every_consecutive_pair_and_reuses_cache(tmp_path: Path) -> None:
    path = tmp_path / "blocks.pt"
    embeddings = torch.arange(4 * 3 * 2, dtype=torch.float16).reshape(4, 3, 2)
    torch.save({"embeddings": embeddings}, path)
    manifest = {
        "splits": {
            "train": [{"cache_path": str(path), "blocks": 4}],
            "validation": [],
        }
    }
    data = PairDataset(manifest, "train", cache_entries=1)

    assert len(data) == 3
    previous, current = data.load_batch(torch.tensor([0, 2]))
    assert torch.equal(previous[0], embeddings[0].float())
    assert torch.equal(current[0], embeddings[1].float())
    assert torch.equal(previous[1], embeddings[2].float())
    assert torch.equal(current[1], embeddings[3].float())


def test_rollout_reports_every_accumulated_delta_horizon(tmp_path: Path) -> None:
    path = tmp_path / "blocks.pt"
    torch.save({"embeddings": torch.randn(4, 3, 4, dtype=torch.float16)}, path)
    data = PairDataset(
        {"splits": {"validation": [{"cache_path": str(path), "blocks": 4}]}},
        "validation",
        cache_entries=1,
    )
    model = DeltaTok(
        ModelConfig(
            input_dim=4,
            model_dim=8,
            tokens_per_frame=3,
            delta_tokens=1,
            depth=1,
            num_heads=2,
        )
    )

    result = _evaluate_rollout(model, data, torch.device("cpu"))

    assert set(result["by_horizon"]) == {"1", "2", "3"}
    assert result["clips"] == 1
    assert result["retrieval_candidates"] == 1
