from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import set_peft_model_state_dict

from deltaomni.msrvtt_raw_caption_lora import (
    _load_model,
    _select,
    evaluate,
)
from deltaomni.msrvtt_raw_caption_lora import (
    load_config as load_training_config,
)
from deltaomni.run_integrity import git_revision, git_worktree_is_clean, sha256_file
from deltaomni.train_sanity import _atomic_json, _set_seed


@dataclass(frozen=True)
class CompareConfig:
    training_config: Path
    checkpoint: Path
    report_path: Path


def load_config(path: Path) -> CompareConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    return CompareConfig(
        training_config=resolve(raw["training_config"]),
        checkpoint=resolve(raw["checkpoint"]),
        report_path=resolve(raw["report_path"]),
    )


def _release() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def run(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = config_path.resolve().parent.parent
    if not git_worktree_is_clean(root):
        raise RuntimeError("Raw caption comparison requires a clean source worktree")
    training = load_training_config(config.training_config)
    if not config.checkpoint.is_file():
        raise FileNotFoundError(f"Missing raw caption checkpoint: {config.checkpoint}")
    _set_seed(training.seed)
    torch.set_num_threads(training.runtime.cpu_threads)
    _, validation = _select(training)
    device = torch.device(training.runtime.device)
    started = datetime.now(UTC)

    vanilla_model = _load_model(training, device)
    vanilla = evaluate(vanilla_model, validation, training)
    del vanilla_model
    _release()

    checkpoint = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    required = {"lora", "next_step", "signature", "world_size", "code_revision"}
    if not isinstance(checkpoint, dict) or not required <= checkpoint.keys():
        raise ValueError("Raw caption comparison checkpoint is incomplete")
    if int(checkpoint["next_step"]) <= training.training.max_steps:
        raise ValueError("Raw caption comparison requires a completed training checkpoint")
    fine_tuned_model = _load_model(training, device)
    set_peft_model_state_dict(fine_tuned_model.thinker, checkpoint["lora"])
    fine_tuned = evaluate(fine_tuned_model, validation, training)
    del fine_tuned_model
    _release()

    vanilla_full = vanilla["metrics"]["full_video"]["word_f1"]
    fine_full = fine_tuned["metrics"]["full_video"]["word_f1"]
    fine_first = fine_tuned["metrics"]["first_block"]["word_f1"]
    result = {
        "schema": "deltaomni.msrvtt_raw_caption_comparison.v1",
        "status": "complete",
        "arms": {
            "vanilla": vanilla,
            "full_video_fine_tuned": fine_tuned,
        },
        "checks": {
            "fine_tuning_beats_vanilla": fine_full > vanilla_full,
            "fine_tuned_full_video_beats_first_block": fine_full > fine_first,
        },
        "passed": fine_full > vanilla_full and fine_full > fine_first,
        "checkpoint": str(config.checkpoint),
        "checkpoint_sha256": sha256_file(config.checkpoint),
        "checkpoint_code_revision": checkpoint["code_revision"],
        "evaluation_code_revision": git_revision(root),
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(config.report_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare vanilla and fine-tuned raw-video Qwen captions"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/msrvtt_raw_caption_compare.yaml")
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
