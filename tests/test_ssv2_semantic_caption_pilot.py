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


def test_layout_caption_uses_balanced_grid_delta() -> None:
    config = load_config(Path("configs/ssv2_semantic_caption_layout17_a6000.yaml"))

    assert "layout17" in config.ssv2_config.name
    assert "layout17" in config.semantic_token_config.name


def test_fidelity_caption_uses_65_grid_tokens() -> None:
    config = load_config(Path("configs/ssv2_semantic_caption_layout65_a6000.yaml"))

    assert "layout65" in config.ssv2_config.name
    assert "layout65" in config.semantic_token_config.name
    assert config.learning_rate == 0.0003
    assert config.max_steps == 800


def test_long_caption_uses_fifteen_delta_updates() -> None:
    config = load_config(
        Path("configs/ssv2_semantic_caption_layout65_16frames_a6000.yaml")
    )

    assert "16frames" in config.ssv2_config.name
    assert "16frames" in config.semantic_token_config.name
    assert config.delta_run_id == "layout65-16f-seed42-validation"

    test_config = load_config(
        Path("configs/ssv2_semantic_caption_layout65_16frames_test_a6000.yaml")
    )
    assert test_config.evaluation_split == "test"
