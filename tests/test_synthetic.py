from pathlib import Path

import torch

from deltaomni.config import load_config
from deltaomni.synthetic import EOS_TOKEN_ID, SyntheticInterleavedDataset


def test_synthetic_stream_contains_same_and_different_commit_times() -> None:
    config = load_config(Path("configs/sanity.yaml"))
    example = SyntheticInterleavedDataset(config, 1, split_seed=123)[0]
    audio_times = set(example.commit_targets[:, 0].nonzero().flatten().tolist())
    video_times = set(example.commit_targets[:, 1].nonzero().flatten().tolist())

    assert audio_times != video_times
    assert audio_times & video_times
    assert example.final_qa_targets.shape == (len(config.modalities),)
    assert example.final_qa_targets.min() >= 0
    assert example.final_qa_targets.max() <= 2
    for time_index, modality_index in example.commit_targets.nonzero().tolist():
        length = int(example.caption_lengths[time_index, modality_index])
        assert example.caption_targets[time_index, modality_index, length - 1] == EOS_TOKEN_ID


def test_caption_class_requires_first_and_last_delta_components() -> None:
    config = load_config(Path("configs/sanity.yaml"))
    dataset = SyntheticInterleavedDataset(config, 8, split_seed=123)
    first_step_differences = []
    last_step_differences = []
    classes = []
    for index in range(8):
        example = dataset[index]
        boundary = int(example.commit_targets[:, 0].nonzero()[0])
        first_step_differences.append(example.full_embeddings[1, 0] - example.full_embeddings[0, 0])
        last_step_differences.append(
            example.full_embeddings[boundary, 0] - example.full_embeddings[boundary - 1, 0]
        )
        classes.append(int(example.caption_targets[boundary, 0, 2] - 5))

    assert classes == [0, 1, 2, 3, 0, 1, 2, 3]
    assert torch.allclose(first_step_differences[0], first_step_differences[1])
    assert torch.allclose(last_step_differences[0], last_step_differences[2])
    assert not torch.allclose(first_step_differences[0], first_step_differences[2])
    assert not torch.allclose(last_step_differences[0], last_step_differences[1])
