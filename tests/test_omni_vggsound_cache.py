from pathlib import Path

import torch

from deltaomni.omni_vggsound_cache import CACHE_SCHEMA, _valid_cache, load_config


def test_omni_vggsound_cache_covers_both_native_modalities() -> None:
    config = load_config(Path("configs/omni_vggsound_s1_cache.yaml"))

    assert config.expected_video_tokens == 128
    assert config.expected_audio_tokens == 50
    assert config.block_seconds == 2.0
    assert config.runtime.nccl_compatibility_mode

    one_second = load_config(Path("configs/omni_vggsound_s1_cache_1s.yaml"))
    assert one_second.block_seconds == 1.0
    assert one_second.expected_video_tokens == 64
    assert one_second.expected_audio_tokens == 25


def test_cache_validation_rejects_wrong_shape_and_accepts_complete_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cache.pt"
    payload = {
        "schema": CACHE_SCHEMA,
        "modality": "audio",
        "source_id": "source",
        "model_revision": "revision",
        "embeddings": torch.zeros(2, 50, 32, dtype=torch.float16),
    }
    torch.save(payload, path)

    arguments = {
        "modality": "audio",
        "source_id": "source",
        "blocks": 2,
        "tokens": 50,
        "width": 32,
        "model_revision": "revision",
    }
    assert _valid_cache(path, **arguments)
    assert not _valid_cache(path, **{**arguments, "blocks": 3})
    payload["embeddings"][0, 0, 0] = torch.nan
    torch.save(payload, path)
    assert not _valid_cache(path, **arguments)
