from pathlib import Path

from deltaomni.delta_setting_sweep import _aggregate_candidates, load_config


def test_delta_setting_sweep_is_bounded_and_has_token_baselines() -> None:
    config = load_config(Path("configs/delta_setting_sweep.yaml"))

    assert config.seeds == (42, 43, 44)
    assert config.delta_tokens == (5, 17, 65)
    assert config.semantic_weights == (1.0, 2.0)
    assert config.reconstruction_steps + config.joint_steps <= 250
    assert config.reconstruction_weight > 0


def test_aggregate_candidates_requires_every_seed_to_qualify() -> None:
    common = {
        "delta_tokens": 8,
        "semantic_weight": 0.5,
        "anchor_mse": 2.0,
        "last_delta_mse": 1.5,
        "shuffled_delta_mse": 2.5,
        "raw_pooled_delta_mse": 1.2,
        "semantic": {"normal": 0.6, "zero": 0.2, "last": 0.3, "shuffled": 0.4},
        "semantic_margin": 0.2,
    }
    candidates = [
        {**common, "seed": 1, "validation_mse": 1.0, "qualified": True},
        {**common, "seed": 2, "validation_mse": 1.2, "qualified": False},
    ]

    aggregated = _aggregate_candidates(candidates)[0]

    assert aggregated["validation_mse"] == 1.1
    assert aggregated["qualified_rate"] == 0.5
    assert not aggregated["qualified"]
