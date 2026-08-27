from pathlib import Path

import pytest

from deltaomni.data.preprocess_vggsound_s1 import _select_available, load_config


def test_vggsound_s1_config_uses_paired_official_splits() -> None:
    config = load_config(Path("configs/canonical/vggsound_s1.yaml"))

    assert config.dataset == "vggsound"
    assert config.resource_name == "vggsound"
    assert config.train_count == 4096
    assert config.validation_count == config.test_count == 256
    assert "schema-v2" in config.dataset_revision
    assert config.excluded_source_ids == {"53UdZyM9MyE_000252"}


def test_vggsound_selection_skips_missing_and_excluded_sources(tmp_path: Path) -> None:
    video_root = tmp_path / "video"
    audio_root = tmp_path / "audio"
    video_root.mkdir()
    audio_root.mkdir()
    names = ["source-one_000001.mp4", "source-two_000001.mp4", "missing_000001.mp4"]
    for name in names[:2]:
        (video_root / name).touch()
        (audio_root / Path(name).with_suffix(".wav")).touch()

    selected, groups, unavailable = _select_available(
        names,
        count=1,
        seed=42,
        video_root=video_root,
        audio_root=audio_root,
        excluded_groups={"youtube:source-one"},
        excluded_source_ids=frozenset({"missing_000001"}),
    )

    assert selected == ["source-two_000001.mp4"]
    assert groups == {"youtube:source-one", "youtube:source-two"}
    assert unavailable in {0, 1}

    with pytest.raises(ValueError, match="requested VGGSound pairs"):
        _select_available(
            ["missing_000001.mp4"],
            count=1,
            seed=42,
            video_root=video_root,
            audio_root=audio_root,
        )
