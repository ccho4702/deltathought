from pathlib import Path

import pytest

from deltaomni.longvideobench_qa_analysis import AnalysisConfig, analyze, load_config


def _config(tmp_path: Path) -> AnalysisConfig:
    return AnalysisConfig(
        predictions=tmp_path / "predictions.jsonl",
        frozen_validation_manifest=tmp_path / "frozen.json",
        required_arms=("method", "control"),
        comparisons=(("method", "control"),),
        bootstrap_samples=1000,
        seed=42,
        report_path=tmp_path / "report.json",
    )


def _row(arm: str, question_id: str, video: str, parsed: int | None, answer: int) -> dict:
    return {
        "arm": arm,
        "id": question_id,
        "video_id": video,
        "prediction": "A",
        "parsed_choice": parsed,
        "correct_choice": answer,
        "duration_group": 60,
        "question_category": "T2E",
    }


def test_longvideobench_analysis_is_paired_and_video_clustered(tmp_path: Path) -> None:
    frozen = {
        "schema": "deltaomni.longvideobench_frozen_validation.v1",
        "dataset_revision": "revision",
        "annotation_sha256": "a" * 64,
        "questions": [
            {"id": "q1", "video_id": "v1"},
            {"id": "q2", "video_id": "v1"},
            {"id": "q3", "video_id": "v2"},
        ],
    }
    rows = [
        _row("method", "q1", "v1", 0, 0),
        _row("method", "q2", "v1", 0, 0),
        _row("method", "q3", "v2", 0, 0),
        _row("control", "q1", "v1", 1, 0),
        _row("control", "q2", "v1", 0, 0),
        _row("control", "q3", "v2", 1, 0),
    ]

    report = analyze(rows, frozen, _config(tmp_path))

    assert report["arms"]["method"]["accuracy"] == 1.0
    assert report["arms"]["control"]["accuracy"] == pytest.approx(1 / 3)
    comparison = report["comparisons"]["method_minus_control"]
    assert comparison["accuracy_difference"] == pytest.approx(2 / 3)
    assert comparison["left_only_correct"] == 2
    assert report["source_clusters"] == 2


def test_longvideobench_analysis_rejects_incomplete_arms(tmp_path: Path) -> None:
    frozen = {
        "schema": "deltaomni.longvideobench_frozen_validation.v1",
        "dataset_revision": "revision",
        "annotation_sha256": "a" * 64,
        "questions": [{"id": "q1", "video_id": "v1"}],
    }

    with pytest.raises(ValueError, match="missing"):
        analyze([_row("method", "q1", "v1", 0, 0)], frozen, _config(tmp_path))


def test_longvideobench_analysis_config_preregisters_all_controls() -> None:
    config = load_config(Path("configs/longvideobench_qa_analysis.yaml"))

    assert len(config.required_arms) == 10
    assert ("delta_continuous_kv", "delta_zero") in config.comparisons
    assert ("delta_continuous_kv", "caption_memory_shuffled") in config.comparisons
    assert config.bootstrap_samples == 100000
