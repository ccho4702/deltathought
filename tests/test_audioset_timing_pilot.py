from pathlib import Path

import torch

from deltaomni.audioset_timing_pilot import _audio_filename, _targets, load_timing_config
from deltaomni.data.audioset_strong import StrongEvent


def test_audio_filename_matches_existing_audioset_convention() -> None:
    assert _audio_filename("s9d-2nhuJCQ_30000") == "s9d-2nhuJCQ_30000_40000.flac"


def test_event_ends_map_to_causal_one_second_commit_bins() -> None:
    events = (
        StrongEvent("clip", 0.0, 0.5, "ignored-early"),
        StrongEvent("clip", 0.0, 2.627, "one"),
        StrongEvent("clip", 2.0, 9.239, "two"),
    )

    targets = _targets(events, 10)

    assert torch.equal(targets.nonzero().flatten(), torch.tensor([2, 9]))


def test_audio_timing_pilot_is_bounded() -> None:
    config = load_timing_config(Path("configs/audioset_timing_pilot.yaml"))

    assert config.train_clips == 32
    assert config.validation_clips == 16
    assert config.training.max_steps <= 100
    assert config.model.delta_tokens == config.model.embedding_tokens == 1

