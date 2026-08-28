from pathlib import Path

from deltaomni.msrvtt_raw_caption_compare import load_config


def test_raw_caption_comparison_uses_completed_matched_baseline() -> None:
    config = load_config(Path("configs/msrvtt_raw_caption_compare.yaml"))

    assert config.training_config.name == "msrvtt_raw_caption.yaml"
    assert config.checkpoint.name == "step-001000.pt"
    assert config.report_path.name == "msrvtt_raw_caption_comparison.json"
