from pathlib import Path
from types import SimpleNamespace

import torch

from deltaomni.omni_nextqa_joint_cache import _valid_cache, load_config


def test_nextqa_joint_poc_uses_short_complete_audio_video_clips() -> None:
    config = load_config(Path("configs/omni_nextqa_joint_poc.yaml"))

    assert (config.train_count, config.validation_count, config.test_count) == (64, 16, 16)
    assert config.minimum_seconds == 5.0
    assert config.maximum_seconds == 12.0
    assert config.dataset_resource_name == "nextqa_annotations"
    assert config.media_license_record.name == "nextqa_media.accepted.json"


def test_joint_cache_rejects_missing_or_changed_signature(tmp_path: Path) -> None:
    episode = SimpleNamespace(
        source_id="source",
        source_group_id="group",
        media=SimpleNamespace(
            video=SimpleNamespace(sha256="a" * 64),
            audio=SimpleNamespace(sha256="b" * 64),
        ),
        qa=(object(),),
    )
    path = tmp_path / "cache.pt"
    payload = {
        "schema": "deltaomni.omni_nextqa_joint_prefix.v2",
        "cache_signature": "expected",
        "source_id": "source",
        "source_group_id": "group",
        "split": "validation",
        "blocks": 2,
        "video_media_sha256": "a" * 64,
        "audio_media_sha256": "b" * 64,
        "video_first": torch.zeros(2, 3, dtype=torch.float16),
        "video_deltas": torch.zeros(1, 1, 4, dtype=torch.float16),
        "audio_first": torch.zeros(1, 3, dtype=torch.float16),
        "audio_deltas": torch.zeros(1, 1, 4, dtype=torch.float16),
        "qa": [{}],
    }
    torch.save(payload, path)
    kwargs = {
        "episode": episode,
        "split": "validation",
        "blocks": 2,
        "video_tokens": 2,
        "audio_tokens": 1,
        "full_width": 3,
        "delta_width": 4,
    }

    assert _valid_cache(path, cache_signature="expected", **kwargs)
    assert not _valid_cache(path, cache_signature="changed", **kwargs)
