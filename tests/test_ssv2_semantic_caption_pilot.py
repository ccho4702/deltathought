from pathlib import Path

from deltaomni.ssv2_semantic_caption_pilot import load_config


def test_semantic_caption_pilot_uses_only_selected_tokens_and_large_qwen() -> None:
    config = load_config(Path("configs/ssv2_semantic_caption_a6000.yaml"))

    assert config.hard_tokens
    assert config.runtime.precision == "bfloat16"
    assert config.runtime.per_device_batch_size >= 1
    assert config.shuffle_repeats >= 4
    assert config.ranking_loss_weight > config.target_loss_weight
    assert "usage_entropy_high" in config.delta_run_id
