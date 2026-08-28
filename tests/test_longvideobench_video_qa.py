from pathlib import Path

import torch

from deltaomni.longvideobench_video_qa import _match_deltas, _parse_choice, load_config


def test_longvideobench_video_qa_smoke_has_all_causal_controls() -> None:
    config = load_config(Path("configs/longvideobench_video_qa_smoke.yaml"))
    names = {arm.name for arm in config.arms}

    assert config.maximum_questions == 6
    assert {
        "full_commit_ft",
        "delta_continuous_kv",
        "delta_zero",
        "delta_reversed",
        "delta_last_only",
        "delta_cross_video",
        "caption_memory_removed",
    } == names


def test_longvideobench_choice_parser_and_length_matched_donors() -> None:
    assert _parse_choice("Answer: C", 4) == 2
    assert _parse_choice("unknown", 4) is None
    donor = torch.arange(5).view(5, 1, 1)
    matched = _match_deltas(donor, 3)
    assert matched[:, 0, 0].tolist() == [0, 2, 4]
