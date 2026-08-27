from pathlib import Path

from deltaomni.omni_clotho_prefix_cache import load_config


def test_clotho_prefix_config_preserves_six_to_fourteen_delta_updates() -> None:
    config = load_config(Path("configs/omni_clotho_prefix_cache.yaml"))

    assert config.minimum_blocks - 1 == 6
    assert config.maximum_blocks - 1 == 14
    assert config.train_count == 2048
    assert config.validation_count == 1037
    assert config.test_count == 1045

    one_second = load_config(Path("configs/omni_clotho_prefix_cache_1s.yaml"))
    assert one_second.minimum_blocks - 1 == 14
    assert one_second.maximum_blocks - 1 == 29
    assert one_second.expected_audio_tokens == 25
    assert one_second.encoder_batch_size == 10
    assert one_second.runtime.cpu_threads == 16
