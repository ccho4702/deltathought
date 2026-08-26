from pathlib import Path

from deltaomni.data.preprocess_ssv2 import load_config


def test_ssv2_canonical_preprocessor_covers_official_splits() -> None:
    config = load_config(Path("configs/canonical/ssv2.yaml"))

    assert config.dataset == "something_something_v2"
    assert set(config.annotations) == {"train", "validation", "test"}
    assert "schema-v2" in config.dataset_revision
    assert config.license_record is None
    assert config.chunk_seconds == 2.0
