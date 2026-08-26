import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from deltaomni.data.canonicalize import read_canonical_dataset, write_canonical_dataset
from deltaomni.data.schema import (
    CanonicalEpisode,
    CaptionBundle,
    MediaAsset,
    MediaBundle,
    ProvenanceRecord,
    TextBundle,
    temporal_grid,
)

SHA256 = "c" * 64


def _episode(split: str, source_id: str) -> CanonicalEpisode:
    duration = 2.0
    return CanonicalEpisode(
        episode_id=f"fixture:{split}:{source_id}",
        dataset="fixture",
        dataset_revision="v2-test",
        split=split,
        source_id=source_id,
        source_group_id=f"fixture-source:{source_id}",
        media=MediaBundle(
            image=None,
            video=MediaAsset(Path(f"/immutable/{source_id}.mp4"), SHA256, duration),
            audio=None,
        ),
        duration_seconds=duration,
        temporal_blocks=temporal_grid(duration, 1.0),
        captions=CaptionBundle(image=None, video=None, audio=None, joint=None),
        text=TextBundle(transcript=None, subtitle=None, ocr=None),
        events=None,
        qa=None,
        provenance=ProvenanceRecord("fixture"),
    )


def test_canonical_dataset_atomic_round_trip_and_coverage(tmp_path: Path) -> None:
    episodes = {
        "train": [_episode("train", "train-one")],
        "validation": [_episode("validation", "validation-one")],
        "test": [_episode("test", "test-one")],
    }

    manifest = write_canonical_dataset(
        tmp_path,
        "fixture",
        "v2-test",
        episodes,
        preprocessing_config_sha256=SHA256,
        code_revision="revision",
        source_files=[{"path": "/immutable/source.json", "sha256": SHA256}],
    )
    loaded = read_canonical_dataset(manifest)

    assert loaded == episodes
    manifest_schema = json.loads(
        Path("schemas/deltaomni.dataset_manifest.v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator.check_schema(manifest_schema)
    Draft202012Validator(manifest_schema).validate(
        json.loads(manifest.read_text(encoding="utf-8"))
    )
    with pytest.raises(FileExistsError, match="already exists"):
        write_canonical_dataset(
            tmp_path,
            "fixture",
            "v2-test",
            episodes,
            preprocessing_config_sha256=SHA256,
            code_revision="revision",
            source_files=[],
        )


def test_canonical_dataset_rejects_cross_split_source_overlap(tmp_path: Path) -> None:
    train = _episode("train", "one")
    test = replace(
        _episode("test", "two"),
        source_group_id=train.source_group_id,
    )

    with pytest.raises(ValueError, match="Cross-split"):
        write_canonical_dataset(
            tmp_path,
            "fixture",
            "v2-test",
            {"train": [train], "test": [test]},
            preprocessing_config_sha256=SHA256,
            code_revision="revision",
            source_files=[],
        )
