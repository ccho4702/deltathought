from pathlib import Path

from deltaomni.ssv2_semantic_pilot import load_config


def test_semantic_pilot_keeps_reconstruction_guard() -> None:
    config = load_config(Path("configs/ssv2_semantic_pilot.yaml"))

    assert config.semantic_weight > 0
    assert config.reconstruction_weight > 0
    assert config.max_steps <= 200

