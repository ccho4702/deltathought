from pathlib import Path

import pytest

from deltaomni.msrvtt_raw_caption_lora import load_config


def test_raw_caption_configs_define_matched_full_video_baseline() -> None:
    smoke = load_config(Path("configs/msrvtt_raw_caption_smoke.yaml"))
    full = load_config(Path("configs/msrvtt_raw_caption.yaml"))

    assert smoke.sample_fps == full.sample_fps == 2.0
    assert smoke.frame_width == smoke.frame_height == 224
    assert full.train_count == 6513
    assert full.validation_count == 497
    assert full.training.max_steps == 1000
    assert full.evaluation.examples == 128
    assert smoke.training.target_modules_regex == full.training.target_modules_regex
    assert smoke.evaluation.split == full.evaluation.split == "validation"

    overfit = load_config(Path("configs/msrvtt_raw_caption_overfit.yaml"))
    assert overfit.train_count == overfit.evaluation.examples == 16
    assert overfit.evaluation.split == "train"
    assert overfit.training.max_steps == 300
    assert overfit.training.caption_sampling == "first"
    assert full.training.caption_sampling == "random"


def test_raw_caption_config_rejects_invalid_resume_mode(tmp_path: Path) -> None:
    source = Path("configs/msrvtt_raw_caption_smoke.yaml").read_text(encoding="utf-8")
    path = tmp_path / "invalid.yaml"
    path.write_text(source.replace("resume: auto", "resume: latest"), encoding="utf-8")

    with pytest.raises(ValueError, match="valid resume mode"):
        load_config(path)
