from pathlib import Path

from deltaomni.data.preprocess_ego4d_goalstep import _leaf_segments, load_config


def test_goalstep_uses_deepest_natural_boundaries_without_irrelevant_segments() -> None:
    video = {
        "segments": [
            {
                "start_time": 0,
                "end_time": 10,
                "step_description": "parent summary",
                "is_relevant": "essential",
                "segments": [
                    {
                        "start_time": 1,
                        "end_time": 3,
                        "step_description": "pick up the bowl",
                        "is_relevant": "essential",
                        "segments": [],
                    },
                    {
                        "start_time": 5,
                        "end_time": 7,
                        "step_description": "look away",
                        "is_relevant": "irrelevant",
                        "segments": [],
                    },
                ],
            },
            {
                "start_time": 11,
                "end_time": 15,
                "step_description": "pour in water",
                "is_relevant": "optional",
                "segments": [],
            },
        ]
    }

    segments = _leaf_segments(video, minimum_seconds=0.5)

    assert [segment["text"] for segment in segments] == [
        "pick up the bowl",
        "pour in water",
    ]
    assert [segment["commit_seconds"] for segment in segments] == [3.0, 15.0]
    assert [segment["depth"] for segment in segments] == [1, 0]


def test_goalstep_config_defines_variable_commit_training_source() -> None:
    config = load_config(Path("configs/canonical/ego4d_goalstep.yaml"))

    assert config.dataset == "ego4d_goalstep"
    assert set(config.annotations) == {"train", "validation"}
    assert config.chunk_seconds == 1.0
    assert config.minimum_commits_per_video == 2
    assert config.minimum_available_videos == {"train": 275, "validation": 69}
    assert config.maximum_videos is None

    smoke = load_config(Path("configs/canonical/ego4d_goalstep_smoke.yaml"))
    assert smoke.maximum_videos == {"train": 8, "validation": 4}
    assert config.media_policy.name == "ego4d_media_policy.yaml"
