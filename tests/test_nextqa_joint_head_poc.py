import torch

from deltaomni.nextqa_joint_head_poc import JointQAHead, load_config


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
