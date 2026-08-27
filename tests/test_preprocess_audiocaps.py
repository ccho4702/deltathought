from dataclasses import replace
from pathlib import Path

from deltaomni.data.preprocess_audiocaps import _group_rows, _inspect, _read_rows, load_config


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


def test_audiocaps_inspection_quarantines_empty_existing_media(tmp_path: Path) -> None:
    config = replace(
        load_config(Path("configs/canonical/audiocaps.yaml")),
        media_roots=(tmp_path,),
        cache_root=tmp_path / "cache",
    )
    source_id = "youtube_0_10000"
    (tmp_path / f"{source_id}.flac").touch()

    split, inspected_source, media, reason = _inspect(config, ("train", source_id))

    assert (split, inspected_source) == ("train", source_id)
    assert media is None
    assert reason == "empty_file"
