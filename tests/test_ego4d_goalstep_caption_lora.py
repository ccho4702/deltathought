from pathlib import Path

import torch

from deltaomni.ego4d_goalstep_caption_lora import _expanded_adapter_state, load_config


def test_goalstep_caption_configs_require_natural_memory_and_delta_gates() -> None:
    smoke = load_config(Path("configs/ego4d_goalstep_caption_smoke.yaml"))
    full = load_config(Path("configs/ego4d_goalstep_caption.yaml"))

    assert smoke.training.max_steps == 40
    assert full.training.max_steps == 800
    assert full.evaluation.windows == 128
    assert full.evaluation.minimum_delta_gap == 0.01
    assert full.evaluation.minimum_memory_gap == 0.005
    assert full.license_record.name == "ego4d.accepted.json"
    assert smoke.input_mode == full.input_mode == "delta"

    full_smoke = load_config(Path("configs/ego4d_goalstep_full_caption_smoke.yaml"))
    full_baseline = load_config(Path("configs/ego4d_goalstep_full_caption.yaml"))
    assert full_smoke.input_mode == full_baseline.input_mode == "full"
    assert full_baseline.training.max_steps == full.training.max_steps
    assert full_baseline.evaluation.windows == full.evaluation.windows


def test_goalstep_adapter_expands_learned_msrvtt_positions() -> None:
    target = {"weight": torch.zeros(2, 2), "delta_positions": torch.zeros(119, 3)}
    source = {"weight": torch.ones(2, 2), "delta_positions": torch.ones(40, 3)}

    expanded = _expanded_adapter_state(target, source)

    assert torch.equal(expanded["weight"], source["weight"])
    assert torch.equal(expanded["delta_positions"][:40], source["delta_positions"])
    assert torch.equal(expanded["delta_positions"][40:], target["delta_positions"][40:])
