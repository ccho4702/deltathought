from pathlib import Path

from deltaomni.ego4d_dynamic_commits import CommitWindow, DynamicCommit
from deltaomni.omni_ego4d_goalstep_cache import _event_records, load_config


def test_ego4d_cache_pins_dynamic_one_second_video_windows() -> None:
    config = load_config(Path("configs/omni_ego4d_goalstep_cache.yaml"))

    assert config.block_seconds == 1.0
    assert config.sample_fps == 2.0
    assert config.expected_video_tokens == 64
    assert config.minimum_windows == {"train": 1700, "validation": 470}
    assert config.license_record.name == "ego4d.accepted.json"


def test_ego4d_cache_records_exact_commit_delta_ranges() -> None:
    commits = (
        DynamicCommit("a", "first", 0.0, 3.2, 3.2, 0, 3),
        DynamicCommit("b", "second", 4.0, 7.1, 7.1, 3, 7),
    )
    window = CommitWindow("train:video:0000", "video", "group", 0, 7, commits, False)

    records = _event_records(window)

    assert [(record["delta_start"], record["delta_end"]) for record in records] == [
        (0, 3),
        (3, 7),
    ]
    assert [record["text"] for record in records] == ["first", "second"]
