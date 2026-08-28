from pathlib import Path

from deltaomni.omni_longvideobench_cache import load_config


def test_longvideobench_cache_uses_full_release_and_bounded_windows() -> None:
    smoke = load_config(Path("configs/omni_longvideobench_cache_smoke.yaml"))
    full = load_config(Path("configs/omni_longvideobench_cache.yaml"))

    assert smoke.maximum_videos == smoke.minimum_videos == 4
    assert full.maximum_videos is None
    assert full.minimum_videos == 753
    assert full.window_seconds == smoke.window_seconds == 120.0
    assert full.block_seconds == smoke.block_seconds == 1.0
    assert full.expected_video_tokens == 64
