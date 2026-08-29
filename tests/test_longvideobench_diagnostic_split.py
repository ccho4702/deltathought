from pathlib import Path

from deltaomni.longvideobench_diagnostic_split import load_config, split_videos


def test_longvideobench_diagnostic_split_is_duration_stratified_and_disjoint() -> None:
    videos = {
        f"video-{group}-{index}": {
            "windows": [],
            "questions": [
                {
                    "id": f"q-{group}-{index}",
                    "duration_group": group,
                    "question_category": "SAA",
                }
            ],
        }
        for group in (15, 60, 600, 3600)
        for index in range(5)
    }
    splits = split_videos(videos, seed=42, train_fraction=0.8)

    assert len(splits["train"]) == 16
    assert len(splits["validation"]) == 4
    assert not (splits["train"].keys() & splits["validation"].keys())
    assert {
        value["questions"][0]["duration_group"]
        for value in splits["validation"].values()
    } == {15, 60, 600, 3600}


def test_longvideobench_diagnostic_split_config_marks_full_cache() -> None:
    config = load_config(Path("configs/longvideobench_diagnostic_split.yaml"))

    assert config.train_fraction == 0.8
    assert config.source_manifest.name == "manifest.json"
