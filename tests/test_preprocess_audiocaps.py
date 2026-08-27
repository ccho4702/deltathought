from pathlib import Path

from deltaomni.data.preprocess_audiocaps import _group_rows, _read_rows, load_config


def test_audiocaps_config_uses_official_splits_and_project_raw_root() -> None:
    config = load_config(Path("configs/canonical/audiocaps.yaml"))

    assert config.dataset == "audiocaps"
    assert config.resource_name == "audiocaps_original"
    assert set(config.annotations) == {"train", "validation", "test"}
    assert all("/dataset/deltathought/raw/" in str(path) for path in config.annotations.values())
    assert config.chunk_seconds == 2.0


def test_audiocaps_validation_rows_group_five_references_per_clip() -> None:
    config = load_config(Path("configs/canonical/audiocaps.yaml"))
    grouped = _group_rows(_read_rows(config.annotations["validation"]))

    assert len(grouped) == 495
    assert {len(values) for values in grouped.values()} == {5}
