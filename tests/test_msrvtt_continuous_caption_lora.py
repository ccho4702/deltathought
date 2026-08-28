from pathlib import Path

from deltaomni.msrvtt_continuous_caption_lora import load_config


def test_continuous_caption_configs_use_same_kv_multi_section_training() -> None:
    smoke = load_config(Path("configs/msrvtt_continuous_caption_smoke.yaml"))
    full = load_config(Path("configs/msrvtt_continuous_caption.yaml"))

    assert smoke.sections_per_sequence == full.sections_per_sequence == 3
    assert smoke.training.max_steps == 40
    assert full.training.max_steps == 400
    assert full.evaluation.sequences == 64
    assert full.initial_checkpoint_sha256 == smoke.initial_checkpoint_sha256
    assert smoke.training.zero_ranking_margin == full.training.zero_ranking_margin == 0.1
    assert smoke.training.zero_ranking_weight == full.training.zero_ranking_weight == 1.0
