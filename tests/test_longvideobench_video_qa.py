import json
from pathlib import Path

import torch

from deltaomni.longvideobench_video_qa import (
    _load_resume_rows,
    _match_deltas,
    _parse_choice,
    _payload,
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
        "delta_anchor_only",
        "delta_norm_noise",
        "delta_permuted",
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


def test_longvideobench_multineg_diagnostic_uses_completed_ego_checkpoint() -> None:
    config = load_config(Path("configs/longvideobench_video_qa_multineg_diagnostic.yaml"))

    assert [arm.name for arm in config.arms] == ["delta_multineg_diagnostic"]
    assert config.arms[0].checkpoint_sha256 == (
        "05c036baafba32e3e9fb391cc436fa96acd41dbc7155e3fc68f20871a508760c"
    )


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


def test_longvideobench_strong_delta_controls_preserve_intended_information() -> None:
    deltas = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 1, 3) + 1
    window = {
        "window_id": "video:0001",
        "video_id": "video",
        "start_seconds": 0.0,
        "end_seconds": 4.0,
        "first_full": torch.zeros(2, 3),
        "final_full": torch.ones(2, 3),
        "deltas": deltas,
    }

    anchor_only = _payload(window, mode="anchor_only", donor=None)["deltas"]
    permuted = _payload(window, mode="permuted", donor=None)["deltas"]
    noise = _payload(window, mode="norm_noise", donor=None)["deltas"]

    assert anchor_only.shape == (0, 1, 3)
    assert not torch.equal(permuted, deltas)
    assert sorted(permuted[:, 0, 0].tolist()) == sorted(deltas[:, 0, 0].tolist())
    assert torch.allclose(noise.float().norm(dim=-1), deltas.norm(dim=-1), atol=1e-5)
    assert not torch.equal(noise, deltas)
