from pathlib import Path

import torch

from deltaomni.ego4d_commit_timing import (
    TimingDataset,
    _event_counts,
    load_config,
    source_dev_split,
)


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

    source_dev = load_config(Path("configs/ego4d_commit_timing_source_dev.yaml"))
    assert source_dev.dev_fraction == 0.1
    assert source_dev.threshold_candidates == (0.5, 0.6, 0.7, 0.8, 0.9)


def test_ego4d_timing_source_dev_split_is_group_disjoint() -> None:
    records = [
        {"source_group_id": f"source-{source}", "window_id": f"{source}-{window}"}
        for source in range(20)
        for window in range(2)
    ]
    fit, dev = source_dev_split(records, seed=42, dev_fraction=0.1)

    assert len(dev) == 4
    assert len(fit) == 36
    fit_sources = {record["source_group_id"] for record in fit}
    dev_sources = {record["source_group_id"] for record in dev}
    assert not (fit_sources & dev_sources)


def test_ego4d_timing_long_run_retains_selection_checkpoints() -> None:
    config = load_config(Path("configs/ego4d_commit_timing_source_dev_long.yaml"))

    assert config.max_steps == 10_000
    assert config.checkpoint_interval_steps == 500
    assert config.keep_last_checkpoints == 20
