from pathlib import Path

import torch

from deltaomni.ego4d_commit_timing import TimingDataset, _event_counts, load_config


def test_ego4d_timing_targets_actual_event_ends() -> None:
    deltas = torch.randn(8, 1, 3)
    values, targets, elapsed = TimingDataset._example(
        {
            "deltas": deltas,
            "events": [{"delta_end": 3}, {"delta_end": 8}],
        }
    )

    assert torch.equal(values, deltas[:, 0])
    assert targets.tolist() == [0, 0, 1, 0, 0, 0, 0, 1]
    assert elapsed.tolist() == [1, 2, 3, 1, 2, 3, 4, 5]


def test_ego4d_timing_tolerance_matches_each_event_once() -> None:
    predicted = torch.tensor([[False, True, True, False, False, True]])
    expected = torch.tensor([[False, False, True, False, False, True]])
    valid = torch.ones_like(predicted)

    assert _event_counts(predicted, expected, valid, tolerance=0) == (2, 1, 0)
    assert _event_counts(predicted, expected, valid, tolerance=1) == (2, 1, 0)


def test_ego4d_timing_configs_use_real_dynamic_cache() -> None:
    smoke = load_config(Path("configs/ego4d_commit_timing_smoke.yaml"))
    full = load_config(Path("configs/ego4d_commit_timing.yaml"))

    assert smoke.max_steps == 40
    assert full.max_steps == 2000
    assert full.evaluation_windows == 481
    assert full.fixed_interval_seconds == 12
