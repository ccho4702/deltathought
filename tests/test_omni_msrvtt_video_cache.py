from pathlib import Path

import torch

from deltaomni.omni_msrvtt_video_cache import _valid, load_config


def test_msrvtt_video_cache_configs_cover_smoke_and_full_training() -> None:
    smoke = load_config(Path("configs/omni_msrvtt_video_cache_smoke.yaml"))
    full = load_config(Path("configs/omni_msrvtt_video_cache.yaml"))

    assert (smoke.train_count, smoke.validation_count, smoke.test_count) == (64, 32, 32)
    assert (full.train_count, full.validation_count, full.test_count) == (6513, 497, 1000)
    assert smoke.block_seconds == full.block_seconds == 1.0


def test_cache_validator_rejects_changed_signature(tmp_path: Path) -> None:
    config = load_config(Path("configs/omni_msrvtt_video_cache_smoke.yaml"))
    episode = __import__("types").SimpleNamespace(
        source_id="video1",
        media=__import__("types").SimpleNamespace(
            video=__import__("types").SimpleNamespace(sha256="a" * 64)
        ),
        captions=__import__("types").SimpleNamespace(video=(object(),) * 20),
    )
    path = tmp_path / "cache.pt"
    torch.save(
        {
            "schema": "deltaomni.omni_msrvtt_video_prefix.v1",
            "cache_signature": "expected",
            "source_id": "video1",
            "media_sha256": "a" * 64,
            "captions": tuple("caption" for _ in range(20)),
            "first_full": torch.zeros(64, 3, dtype=torch.float16),
            "deltas": torch.zeros(4, 1, 2, dtype=torch.float16),
        },
        path,
    )
    kwargs = {
        "episode": episode,
        "blocks": 5,
        "full_width": 3,
        "delta_width": 2,
        "config": config,
    }
    assert _valid(path, signature="expected", **kwargs)
    assert not _valid(path, signature="changed", **kwargs)
