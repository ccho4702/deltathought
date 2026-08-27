from types import SimpleNamespace

import torch

from deltaomni.nextqa_joint_head_poc import (
    JointQADataset,
    JointQAHead,
    cross_source_donor_indices,
    load_config,
)


def test_joint_qa_head_scores_five_choices_and_supports_controls() -> None:
    config = load_config(__import__("pathlib").Path("configs/nextqa_joint_head_poc.yaml"))
    model = JointQAHead(config.text_buckets, config.hidden_width)
    batch = {
        "video_full": torch.randn(2, 3584),
        "video_delta": torch.randn(2, 768),
        "audio_full": torch.randn(2, 3584),
        "audio_delta": torch.randn(2, 768),
        "questions": [[1, 2], [3]],
        "choices": [[[1], [2], [3], [4], [5]], [[6], [7], [8], [9], [10]]],
    }
    assert model(batch).shape == (2, 5)
    assert model(batch, "delta_zero").shape == (2, 5)


def test_cross_source_control_never_reuses_the_same_video() -> None:
    data = SimpleNamespace(
        source_group_ids=lambda: ["video-a", "video-a", "video-b", "video-b", "video-c"]
    )

    donors = cross_source_donor_indices(data, seed=42)
    groups = data.source_group_ids()

    assert donors.shape == (5,)
    assert all(groups[index] != groups[donor] for index, donor in enumerate(donors.tolist()))


def test_joint_dataset_uses_donor_delta_but_keeps_target_full_state(tmp_path) -> None:
    records = []
    for index in range(2):
        path = tmp_path / f"source-{index}.pt"
        torch.save(
            {
                "source_id": f"source-{index}",
                "video_first": torch.full((2, 3), float(index)),
                "video_deltas": torch.full((2, 1, 4), float(index + 10)),
                "audio_first": torch.full((1, 3), float(index)),
                "audio_deltas": torch.full((2, 1, 4), float(index + 20)),
                "qa": [
                    {
                        "question": "What happened?",
                        "choices": ["one", "two"],
                        "answer_index": 0,
                        "answer": "one",
                    }
                ],
            },
            path,
        )
        records.append(
            {
                "source_id": f"source-{index}",
                "source_group_id": f"group-{index}",
                "cache_path": str(path),
            }
        )
    data = JointQADataset({"splits": {"validation": records}}, "validation", buckets=32)

    batch = data.batch_with_delta_indices(
        torch.tensor([0]), delta_indices=torch.tensor([1])
    )

    assert torch.equal(batch["video_full"], torch.zeros(1, 3))
    assert torch.equal(batch["video_delta"], torch.full((1, 4), 11.0))
    assert batch["source_group_ids"] == ["group-0"]
