from __future__ import annotations

import argparse
import json
import os
import random
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from deltaomni.run_integrity import (
    git_revision,
    git_worktree_is_clean,
    resolved_input_signature,
)
from deltaomni.streaming_sequence import CommitHead, commit_loss
from deltaomni.train_sanity import _atomic_json, _set_seed


@dataclass(frozen=True)
class TimingConfig:
    seed: int
    cache_manifest: Path
    device: str
    cpu_threads: int
    cache_entries: int
    delta_width: int
    hidden_width: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    positive_weight: float
    threshold: float
    fixed_interval_seconds: int
    max_steps: int
    checkpoint_interval_steps: int
    keep_last_checkpoints: int
    evaluation_windows: int
    minimum_tolerance_f1_gap: float
    resume: str
    output_root: Path
    log_root: Path
    report_path: Path


def load_config(path: Path) -> TimingConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    paths = {"cache_manifest", "output_root", "log_root", "report_path"}
    values = {key: resolve(value) if key in paths else value for key, value in raw.items()}
    config = TimingConfig(**values)
    positive = (
        config.cpu_threads,
        config.cache_entries,
        config.delta_width,
        config.hidden_width,
        config.batch_size,
        config.learning_rate,
        config.positive_weight,
        config.fixed_interval_seconds,
        config.max_steps,
        config.checkpoint_interval_steps,
        config.keep_last_checkpoints,
        config.evaluation_windows,
        config.minimum_tolerance_f1_gap,
    )
    if min(positive) <= 0 or not 0 < config.threshold < 1:
        raise ValueError("Ego4D commit timing controls must be positive")
    if config.resume not in {"auto", "never"}:
        raise ValueError("Invalid Ego4D commit timing resume mode")
    return config


class TimingDataset:
    def __init__(self, records: list[dict[str, Any]], cache_entries: int) -> None:
        self.records = records
        self.cache_entries = cache_entries
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def load(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = str(record["cache_path"])
        value = self.cache.get(path)
        if value is None:
            value = torch.load(path, map_location="cpu", weights_only=False)
            if value["window_id"] != record["window_id"]:
                raise ValueError(f"Ego4D timing cache mismatch: {path}")
            self.cache[path] = value
            while len(self.cache) > self.cache_entries:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(path)
        return value

    def source_disjoint_index(self, index: int) -> int:
        source = self.records[index]["source_group_id"]
        for offset in range(1, len(self.records)):
            candidate = (index + offset) % len(self.records)
            if self.records[candidate]["source_group_id"] != source:
                return candidate
        raise ValueError("Ego4D timing data has no source-disjoint donor")

    @staticmethod
    def _example(payload: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        deltas = payload["deltas"][:, 0].float()
        targets = torch.zeros(len(deltas), dtype=torch.float32)
        for event in payload["events"]:
            end = int(event["delta_end"])
            if end > 0:
                targets[min(end - 1, len(deltas) - 1)] = 1
        elapsed = torch.empty(len(deltas), dtype=torch.float32)
        since_commit = 0
        for index in range(len(deltas)):
            since_commit += 1
            elapsed[index] = since_commit
            if targets[index] == 1:
                since_commit = 0
        return deltas, targets, elapsed

    def batch(self, indices: Tensor) -> dict[str, Tensor]:
        examples = []
        donors = []
        for index in indices.tolist():
            payload = self.load(index)
            examples.append(self._example(payload))
            donor = self._example(self.load(self.source_disjoint_index(index)))[0]
            donors.append(donor)
        width = max(len(value[0]) for value in examples)
        deltas = torch.zeros(len(examples), width, examples[0][0].shape[-1])
        cross = torch.zeros_like(deltas)
        targets = torch.zeros(len(examples), width)
        elapsed = torch.zeros(len(examples), width)
        valid = torch.zeros(len(examples), width, dtype=torch.bool)
        refresh = torch.zeros_like(valid)
        for row, ((value, target, timing), donor) in enumerate(zip(examples, donors, strict=True)):
            length = len(value)
            donor_indices = torch.linspace(0, len(donor) - 1, length).round().long()
            deltas[row, :length] = value
            cross[row, :length] = donor[donor_indices]
            targets[row, :length] = target
            elapsed[row, :length] = timing
            valid[row, :length] = True
            refresh[row, 0] = True
        return {
            "deltas": deltas,
            "cross": cross,
            "targets": targets,
            "elapsed": elapsed,
            "valid": valid,
            "refresh": refresh,
        }


def _event_counts(
    predicted: Tensor, expected: Tensor, valid: Tensor, tolerance: int
) -> tuple[int, int, int]:
    true_positive = false_positive = false_negative = 0
    for row in range(predicted.shape[0]):
        pred = predicted[row].logical_and(valid[row]).nonzero().flatten().tolist()
        target = expected[row].bool().logical_and(valid[row]).nonzero().flatten().tolist()
        unmatched = set(target)
        matched = 0
        for value in pred:
            candidates = [item for item in unmatched if abs(item - value) <= tolerance]
            if candidates:
                selected = min(candidates, key=lambda item: (abs(item - value), item))
                unmatched.remove(selected)
                matched += 1
        true_positive += matched
        false_positive += len(pred) - matched
        false_negative += len(unmatched)
    return true_positive, false_positive, false_negative


def _metrics(counts: tuple[int, int, int]) -> dict[str, float]:
    true_positive, false_positive, false_negative = counts
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


@torch.no_grad()
def evaluate(model: CommitHead, data: TimingDataset, config: TimingConfig) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    count = min(config.evaluation_windows, len(data))
    totals = {
        name: {tolerance: [0, 0, 0] for tolerance in (0, 1, 3)}
        for name in ("normal", "zero", "cross_video", "fixed_interval")
    }
    losses = {name: [] for name in ("normal", "zero", "cross_video")}
    started = time.perf_counter()
    for start in range(0, count, config.batch_size):
        indices = torch.arange(start, min(start + config.batch_size, count))
        batch = {key: value.to(device) for key, value in data.batch(indices).items()}
        predictions = {}
        for name, values in (
            ("normal", batch["deltas"]),
            ("zero", torch.zeros_like(batch["deltas"])),
            ("cross_video", batch["cross"]),
        ):
            logits = model(values, batch["elapsed"], batch["refresh"], batch["valid"])
            losses[name].append(
                float(
                    commit_loss(
                        logits,
                        batch["targets"],
                        batch["valid"],
                        positive_weight=config.positive_weight,
                    )
                )
            )
            predictions[name] = logits.sigmoid() >= config.threshold
        fixed = torch.zeros_like(batch["valid"])
        fixed[:, config.fixed_interval_seconds - 1 :: config.fixed_interval_seconds] = True
        predictions["fixed_interval"] = fixed
        for name, predicted in predictions.items():
            for tolerance in (0, 1, 3):
                counts = _event_counts(
                    predicted, batch["targets"], batch["valid"], tolerance
                )
                totals[name][tolerance] = [
                    left + right
                    for left, right in zip(totals[name][tolerance], counts, strict=True)
                ]
        completed = min(start + config.batch_size, count)
        elapsed = time.perf_counter() - started
        eta = elapsed / completed * (count - completed)
        print(f"timing_eval={completed}/{count} elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)
    return {
        name: {
            "loss": (sum(losses[name]) / len(losses[name]) if name in losses else None),
            "exact": _metrics(tuple(values[0])),
            "tolerance_1s": _metrics(tuple(values[1])),
            "tolerance_3s": _metrics(tuple(values[3])),
        }
        for name, values in totals.items()
    }


def _atomic_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _latest(run_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    for path in sorted((run_dir / "checkpoints").glob("step-*.pt"), reverse=True):
        try:
            value = torch.load(path, map_location="cpu", weights_only=False)
        except (EOFError, OSError, RuntimeError):
            continue
        if {"model", "optimizer", "next_step", "signature", "rng", "initial"} <= value.keys():
            return path, value
    return None


def run(config_path: Path, run_id: str | None, stop_after_step: int | None) -> dict[str, Any]:
    config = load_config(config_path)
    root = config_path.resolve().parent.parent
    if not git_worktree_is_clean(root):
        raise RuntimeError("Ego4D commit timing requires a clean Git worktree")
    _set_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    manifest = json.loads(config.cache_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "deltaomni.omni_ego4d_goalstep_manifest.v2":
        raise ValueError("Unexpected Ego4D commit timing cache")
    train = TimingDataset(manifest["splits"]["train"], config.cache_entries)
    validation = TimingDataset(manifest["splits"]["validation"], config.cache_entries)
    device = torch.device(config.device)
    model = CommitHead(config.delta_width, config.hidden_width).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    signature = resolved_input_signature(config, {"cache_manifest": config.cache_manifest})
    revision = git_revision(root)
    selected = run_id or f"ego4d-timing-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = config.output_root / selected
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "resolved_config.json").exists():
        _atomic_json(run_dir / "resolved_config.json", asdict(config))
        _atomic_json(
            run_dir / "metadata.json",
            {
                "code_revision": revision,
                "gpu": torch.cuda.get_device_name(device),
                "torch_version": torch.__version__,
                "started_at_utc": datetime.now(UTC).isoformat(),
            },
        )
    start_step = 1
    initial = None
    resumed = _latest(run_dir) if config.resume == "auto" else None
    if resumed is not None:
        path, value = resumed
        if value["signature"] != signature or value["code_revision"] != revision:
            raise ValueError("Incompatible Ego4D commit timing checkpoint")
        model.load_state_dict(value["model"])
        optimizer.load_state_dict(value["optimizer"])
        random.setstate(value["rng"]["python"])
        torch.random.set_rng_state(value["rng"]["torch"])
        torch.cuda.set_rng_state(value["rng"]["cuda"], device)
        start_step = int(value["next_step"])
        initial = value["initial"]
        print(f"resume={path} next_step={start_step}", flush=True)
    if initial is None:
        initial = evaluate(model, validation, config)
    final_step = min(config.max_steps, stop_after_step or config.max_steps)
    log_path = config.log_root / selected / "metrics.jsonl"
    started = time.perf_counter()
    for step in range(start_step, final_step + 1):
        generator = torch.Generator().manual_seed(config.seed * 1_000_003 + step)
        indices = torch.randint(0, len(train), (config.batch_size,), generator=generator)
        batch = {key: value.to(device) for key, value in train.batch(indices).items()}
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["deltas"], batch["elapsed"], batch["refresh"], batch["valid"])
        loss = commit_loss(
            logits, batch["targets"], batch["valid"], positive_weight=config.positive_weight
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"step": step, "loss": float(loss)}) + "\n")
        if step % 10 == 0 or step == final_step:
            elapsed = time.perf_counter() - started
            eta = elapsed / (step - start_step + 1) * (final_step - step)
            print(
                f"timing_step={step}/{final_step} loss={float(loss):.5f} eta={eta:.1f}s",
                flush=True,
            )
        if step % config.checkpoint_interval_steps == 0 or step == final_step:
            _atomic_save(
                run_dir / "checkpoints" / f"step-{step:06d}.pt",
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "next_step": step + 1,
                    "signature": signature,
                    "code_revision": revision,
                    "initial": initial,
                    "rng": {
                        "python": random.getstate(),
                        "torch": torch.random.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state(device),
                    },
                },
            )
            checkpoints = sorted((run_dir / "checkpoints").glob("step-*.pt"))
            for path in checkpoints[: -config.keep_last_checkpoints]:
                path.unlink()
    if final_step < config.max_steps:
        result = {"run_id": selected, "status": "interrupted", "step": final_step}
        _atomic_json(run_dir / "status.json", result)
        return result
    final = evaluate(model, validation, config)
    normal = final["normal"]["tolerance_3s"]["f1"]
    checks = {
        "normal_beats_zero": normal
        >= final["zero"]["tolerance_3s"]["f1"] + config.minimum_tolerance_f1_gap,
        "normal_beats_cross_video": normal
        >= final["cross_video"]["tolerance_3s"]["f1"] + config.minimum_tolerance_f1_gap,
        "normal_beats_fixed_interval": normal
        >= final["fixed_interval"]["tolerance_3s"]["f1"]
        + config.minimum_tolerance_f1_gap,
    }
    result = {
        "schema": "deltaomni.ego4d_commit_timing.v1",
        "run_id": selected,
        "status": "complete",
        "initial": initial,
        "final": final,
        "checks": checks,
        "passed": all(checks.values()),
        "training_seconds": time.perf_counter() - started,
        "code_revision": revision,
    }
    _atomic_json(run_dir / "summary.json", result)
    _atomic_json(run_dir / "status.json", result)
    _atomic_json(config.report_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Ego4D one-second commit timing")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/ego4d_commit_timing_smoke.yaml")
    )
    parser.add_argument("--run-id")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.run_id, args.stop_after_step), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
