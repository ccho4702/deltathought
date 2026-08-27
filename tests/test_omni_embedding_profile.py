from pathlib import Path

from deltaomni.omni_embedding_profile import load_config


def test_omni_embedding_profile_uses_canonical_ssv2_and_four_batches() -> None:
    config = load_config(Path("configs/omni_embedding_profile.yaml"))

    assert "canonical" in config.canonical_manifest.parts
    assert config.batch_sizes == (1, 2, 4, 8)
    assert config.samples >= 16 * max(config.batch_sizes)
    assert config.runtime.nccl_compatibility_mode
