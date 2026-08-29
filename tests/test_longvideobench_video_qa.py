import json
from pathlib import Path

import torch

from deltaomni.longvideobench_video_qa import (
    _load_resume_rows,
    _match_deltas,
    _parse_choice,
    _select_arms,
    load_config,
)


def test_longvideobench_video_qa_smoke_has_all_causal_controls() -> None:
    config = load_config(Path("configs/longvideobench_video_qa_smoke.yaml"))
    names = {arm.name for arm in config.arms}

    assert config.maximum_questions == 6
    assert config.answer_strategy == "choice_logit"
    assert {
        "vanilla_commit",
        "full_commit_ft",
        "delta_continuous_kv",
        "delta_zero",
        "delta_reversed",
        "delta_last_only",
        "delta_cross_video",
        "caption_memory_removed",
    } == names
    vanilla = next(arm for arm in config.arms if arm.name == "vanilla_commit")
    assert vanilla.weights == "vanilla"
    assert vanilla.checkpoint is None


def test_longvideobench_choice_parser_and_length_matched_donors() -> None:
    assert _parse_choice("Answer: C", 4) == 2
    assert _parse_choice("unknown", 4) is None
    donor = torch.arange(5).view(5, 1, 1)
    matched = _match_deltas(donor, 3)
    assert matched[:, 0, 0].tolist() == [0, 2, 4]


def test_longvideobench_arm_selection_uses_distinct_output_paths() -> None:
    config = load_config(Path("configs/longvideobench_video_qa_smoke.yaml"))
    selected = _select_arms(config, ["delta_zero"], "zero")

    assert [arm.name for arm in selected.arms] == ["delta_zero"]
    assert selected.predictions_path.name == "smoke_video_only_zero.jsonl"
    assert selected.report_path.name == "longvideobench_video_qa_smoke_zero.json"


def test_longvideobench_resume_rows_require_matching_provenance(tmp_path: Path) -> None:
    config = load_config(Path("configs/longvideobench_video_qa_smoke.yaml"))
    arm = config.arms[0]
    path = tmp_path / "partial.jsonl"
    row = {
        "arm": arm.name,
        "id": "question-1",
        "video_id": "video-1",
        "checkpoint_sha256": arm.checkpoint_sha256,
        "code_revision": "abc123",
        "answer_strategy": "choice_logit",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert _load_resume_rows(
        path,
        (arm,),
        code_revision="abc123",
        answer_strategy="choice_logit",
    ) == [row]
