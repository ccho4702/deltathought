from pathlib import Path

from deltaomni.ssv2_delta_search import load_config


def test_delta_search_preregisters_multiseed_validation_and_test_selection() -> None:
    config = load_config(Path("configs/ssv2_delta_search_a6000.yaml"))

    assert config.gpu_count == 4
    assert len(config.seeds) >= 3
    assert config.required_pass_fraction == 1.0
    assert len(config.trials) >= 5
    assert len({trial.name for trial in config.trials}) == len(config.trials)
