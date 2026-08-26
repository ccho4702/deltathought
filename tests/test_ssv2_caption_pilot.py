from pathlib import Path

from deltaomni.ssv2_pilot import load_pilot_config


def test_caption_pilot_is_short_and_uses_scoped_action_targets() -> None:
    config = load_pilot_config(Path("configs/ssv2_pilot.yaml"))

    assert config.caption.max_steps <= 160
    assert config.caption.ranking_weight > 0
    assert not config.caption.train_delta_encoder
    assert len(config.caption.targets) == len(config.classes)
    assert len(set(config.caption.targets)) == len(config.caption.targets)
    assert all("something" not in target for target in config.caption.targets)


def test_medium_caption_pilot_jointly_aligns_delta_with_reconstruction_guard() -> None:
    config = load_pilot_config(Path("configs/ssv2_pilot_medium.yaml"))

    assert config.caption.train_delta_encoder
    assert config.caption.reconstruction_weight > 0
    assert 0 < config.caption.delta_learning_rate < config.caption.learning_rate
