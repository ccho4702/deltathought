from pathlib import Path

from deltaomni.nextqa_reconstruction_pilot import load_config


def test_nextqa_reconstruction_pilot_is_bounded_and_read_only() -> None:
    config = load_config(Path("configs/nextqa_reconstruction_pilot.yaml"))

    assert config.validation_clips == 16
    assert config.frames_per_clip == 8
    assert str(config.media_root).startswith("/mnt/nfs_shared_data/dataset/NExT-QA")
    assert config.cache_root == Path.cwd() / "intermediates/cache/nextqa_reconstruction_pilot"
