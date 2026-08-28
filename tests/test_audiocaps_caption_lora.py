from pathlib import Path

import torch

from deltaomni.audiocaps_caption_lora import (
    DeltaPrefixAdapter,
    _caption_metrics,
    load_config,
)


def test_caption_lora_targets_only_thinker_text_layers() -> None:
    config = load_config(Path("configs/audiocaps_caption_lora.yaml"))

    assert config.lora.rank == 8
    assert config.interface.delta_updates == 4
    assert config.interface.hidden_width == 3584
    assert config.training.max_steps == 1000

    early = load_config(Path("configs/audiocaps_caption_lora_earlystop.yaml"))
    assert early.training.max_steps == 200
    assert early.training.warmup_steps == config.training.warmup_steps
    assert early.lora == config.lora

    one_second = load_config(Path("configs/audiocaps_caption_lora_1s.yaml"))
    assert one_second.interface.delta_updates == 9
    assert one_second.training.max_steps == 200
    video = load_config(Path("configs/msrvtt_video_caption_smoke.yaml"))
    assert video.modality == "video"
    assert video.interface.delta_updates == 40


def test_delta_prefix_adapter_emits_one_soft_token_per_update() -> None:
    config = load_config(Path("configs/audiocaps_caption_lora_smoke.yaml"))
    adapter = DeltaPrefixAdapter(config.interface)
    anchors, deltas = adapter(torch.randn(2, 50, 3584), torch.randn(2, 4, 1, 768))

    assert anchors.shape == (2, 50, 3584)
    assert deltas.shape == (2, 4, 3584)

    _, shorter = adapter(torch.randn(1, 50, 3584), torch.randn(1, 2, 1, 768))
    assert shorter.shape == (1, 2, 3584)


def test_caption_metrics_use_best_human_reference() -> None:
    metrics = _caption_metrics(
        ["A dog barks loudly"],
        [("Water is running", "A dog barks loudly nearby")],
    )

    assert metrics["exact_match"] == 0.0
    assert metrics["word_f1"] > 0.8
    assert metrics["rouge_l"] > 0.8
