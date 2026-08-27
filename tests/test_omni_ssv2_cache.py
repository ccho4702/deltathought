from pathlib import Path

from deltaomni.omni_ssv2_cache import load_config


def test_omni_ssv2_cache_uses_full_native_block_tokens() -> None:
    config = load_config(Path("configs/omni_ssv2_s1_cache.yaml"))

    assert config.expected_tokens_per_block == 128
    assert config.train_per_class == 512
    assert config.validation_per_class == config.test_per_class == 64
    assert len(config.classes) == 4
    assert config.runtime.nccl_compatibility_mode
