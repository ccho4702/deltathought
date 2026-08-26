from pathlib import Path

from deltaomni.ssv2_resampler_pilot import load_config


def test_resampler_pilot_uses_alignment_before_captioning() -> None:
    config = load_config(Path("configs/ssv2_resampler_pilot.yaml"))

    assert config.alignment_steps > config.caption_steps
    assert config.query_tokens == 8
    assert 0 < config.temperature < 1
    assert config.alignment_batch_size >= 16
    assert config.caption_batch_size >= 4
    assert config.alignment_guard_weight > 0
    assert config.shuffle_repeats >= 4
    assert config.semantic_config.name == "ssv2_semantic_pilot.yaml"
