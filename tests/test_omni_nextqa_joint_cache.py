from pathlib import Path

from deltaomni.omni_nextqa_joint_cache import load_config, select_episodes


def test_nextqa_joint_poc_uses_short_complete_audio_video_clips() -> None:
    config = load_config(Path("configs/omni_nextqa_joint_poc.yaml"))
    selected = select_episodes(config)

    assert {split: len(values) for split, values in selected.items()} == {
        "train": 64,
        "validation": 16,
        "test": 16,
    }
    assert all(
        episode.media.audio is not None and episode.media.video is not None
        for values in selected.values()
        for episode in values
    )
    assert all(
        config.minimum_seconds <= episode.duration_seconds <= config.maximum_seconds
        for values in selected.values()
        for episode in values
    )
