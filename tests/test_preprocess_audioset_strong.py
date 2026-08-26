from pathlib import Path

from deltaomni.data.preprocess_audioset_strong import load_config


def test_audioset_canonical_preprocessor_tracks_attrition() -> None:
    config = load_config(Path("configs/canonical/audioset_strong.yaml"))

    assert config.dataset == "audioset_strong"
    assert set(config.annotations) == {"train", "validation"}
    assert set(config.media_roots) == {"train", "validation"}
    assert "schema-v2" in config.dataset_revision
    assert config.license_record is None
    assert config.nominal_duration_seconds == 10.0
