from pathlib import Path

import torch

from deltaomni.ego4d_goalstep_caption_lora import (
    WindowCache,
    _expanded_adapter_state,
    _match_delta_length,
    _permutation_indices,
    load_config,
)


def test_goalstep_caption_configs_require_natural_memory_and_delta_gates() -> None:
    smoke = load_config(Path("configs/ego4d_goalstep_caption_smoke.yaml"))
    full = load_config(Path("configs/ego4d_goalstep_caption.yaml"))

    assert smoke.training.max_steps == 40
    assert full.training.max_steps == 800
    assert full.evaluation.windows == 128
    assert full.evaluation.minimum_delta_gap == 0.01
    assert full.evaluation.minimum_memory_gap == 0.005
    assert smoke.input_mode == full.input_mode == "delta"

    full_smoke = load_config(Path("configs/ego4d_goalstep_full_caption_smoke.yaml"))
    full_baseline = load_config(Path("configs/ego4d_goalstep_full_caption.yaml"))
    assert full_smoke.input_mode == full_baseline.input_mode == "full"
    assert full_baseline.training.max_steps == full.training.max_steps
    assert full_baseline.evaluation.windows == full.evaluation.windows

    multineg = load_config(Path("configs/ego4d_goalstep_caption_multineg.yaml"))
    assert multineg.training.cross_ranking_weight == 0.5
    assert multineg.training.order_ranking_weight == 0.5
    assert multineg.evaluation.nll_windows == 64
    assert multineg.evaluation.minimum_cross_gap == 0.005
    assert multineg.evaluation.minimum_order_gap == 0.005
    assert full.training.cross_ranking_weight == 0.0
    assert full.training.order_ranking_weight == 0.0


def test_goalstep_adapter_expands_learned_msrvtt_positions() -> None:
    target = {"weight": torch.zeros(2, 2), "delta_positions": torch.zeros(119, 3)}
    source = {"weight": torch.ones(2, 2), "delta_positions": torch.ones(40, 3)}

    expanded = _expanded_adapter_state(target, source)

    assert torch.equal(expanded["weight"], source["weight"])
    assert torch.equal(expanded["delta_positions"][:40], source["delta_positions"])
    assert torch.equal(expanded["delta_positions"][40:], target["delta_positions"][40:])


def test_goalstep_negative_controls_are_deterministic_and_source_disjoint() -> None:
    cache = WindowCache(
        {
            "splits": {
                "train": [
                    {"source_group_id": "a"},
                    {"source_group_id": "a"},
                    {"source_group_id": "b"},
                ]
            }
        },
        "train",
        1,
    )
    assert cache.source_disjoint_index(0) == 2
    assert cache.source_disjoint_index(2) == 0

    indices = _permutation_indices("window:event", 8)
    assert torch.equal(indices, _permutation_indices("window:event", 8))
    assert not torch.equal(indices, torch.arange(8))
    assert sorted(indices.tolist()) == list(range(8))

    deltas = torch.arange(5).view(5, 1, 1)
    assert _match_delta_length(deltas, 3)[:, 0, 0].tolist() == [0, 2, 4]
