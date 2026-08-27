import json
from pathlib import Path

import pytest

from deltaomni.omni_vanilla_baseline import (
    _clean_prediction,
    _lexical_choice,
    _msrvtt_items,
    _parse_choice,
    _text_metrics,
    load_config,
)


def test_parse_choice_requires_a_standalone_valid_letter() -> None:
    assert _parse_choice("B", 5) == 1
    assert _parse_choice("Answer: (D).", 5) == 3
    assert _parse_choice("because the person left", 5) is None
    assert _parse_choice("F", 5) is None


def test_freeform_metrics_and_lexical_mapping() -> None:
    prediction = "driving through mud"
    references = ("splashed when going through mud",)
    metrics = _text_metrics(prediction, references)
    assert metrics["exact_match"] == 0.0
    assert 0.49 < metrics["word_f1"] < 1.0
    assert _lexical_choice(prediction, ("snow covered", references[0], "hit a tree")) == 1
    assert _clean_prediction("Answer: driving through mud\nHuman: next") == prediction


def test_config_rejects_invalid_duration_range(tmp_path: Path) -> None:
    config = {
        "seed": 1,
        "omni_config": "omni.yaml",
        "nextqa_manifest": "nextqa.json",
        "nextqa_selection_manifest": "selection.json",
        "msrvtt_metadata": "msrvtt.json",
        "msrvtt_video_root": "videos",
        "msrvtt_count": 1,
        "minimum_seconds": 10.0,
        "maximum_seconds": 5.0,
        "sample_fps": 2.0,
        "frame_width": 224,
        "frame_height": 224,
        "freeform_max_new_tokens": 8,
        "multiple_choice_max_new_tokens": 2,
        "caption_max_new_tokens": 16,
        "runtime": {
            "device": "cuda",
            "backend": "nccl",
            "nccl_compatibility_mode": True,
            "cpu_threads": 1,
        },
        "run_id": "test",
        "output_root": "outputs",
        "log_root": "logs",
        "report_path": "report.json",
        "comparison_report": "comparison.json",
    }
    path = tmp_path / "config.yaml"
    path.write_text(__import__("yaml").safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid vanilla baseline controls"):
        load_config(path)


def test_msrvtt_selection_is_deterministic(monkeypatch, tmp_path: Path) -> None:
    metadata = {f"video{index}": {str(index): f"caption {index}"} for index in range(5)}
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    video_root = tmp_path / "videos"
    video_root.mkdir()
    for source_id in metadata:
        (video_root / f"{source_id}.mp4").touch()
    monkeypatch.setattr("deltaomni.omni_vanilla_baseline._duration_seconds", lambda _: 8.0)
    config = load_config(Path("configs/qwen2_5_omni_vanilla_baseline_poc.yaml"))
    config = __import__("dataclasses").replace(
        config, msrvtt_metadata=metadata_path, msrvtt_video_root=video_root, msrvtt_count=3
    )
    first = _msrvtt_items(config)
    second = _msrvtt_items(config)
    assert [item["source_id"] for item in first] == [item["source_id"] for item in second]
