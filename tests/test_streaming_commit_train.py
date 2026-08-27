from pathlib import Path

from deltaomni.streaming_commit_train import load_config


def test_streaming_commit_poc_uses_three_sections_and_balanced_positive_weight() -> None:
    config = load_config(Path("configs/streaming_commit_poc.yaml"))

    assert config.sections_per_sequence == 3
    assert config.positive_weight == 8.0
    assert config.delta_width == 768
    assert config.threshold == 0.5
