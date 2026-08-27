from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
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

from deltaomni.run_integrity import git_worktree_is_clean, resolved_input_signature
from deltaomni.streaming_sequence import CommitHead, StreamingSequence, build_sequences, commit_loss
from deltaomni.train_sanity import _atomic_json, _set_seed


@dataclass(frozen=True)
class CommitConfig:
    seed: int
    prefix_manifest: Path
    sections_per_sequence: int
    delta_width: int
    hidden_width: int
    device: str
    cpu_threads: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    positive_weight: float
    threshold: float
    max_steps: int
    checkpoint_interval_steps: int
    keep_last_checkpoints: int
    resume: str
    output_root: Path
    log_root: Path


def load_config(path: Path) -> CommitConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    config = CommitConfig(
        seed=int(raw["seed"]),
        prefix_manifest=resolve(raw["prefix_manifest"]),
        sections_per_sequence=int(raw["sections_per_sequence"]),
        delta_width=int(raw["delta_width"]),
        hidden_width=int(raw["hidden_width"]),
        device=str(raw["device"]),
        cpu_threads=int(raw["cpu_threads"]),
        batch_size=int(raw["batch_size"]),
        learning_rate=float(raw["learning_rate"]),
        weight_decay=float(raw["weight_decay"]),
        positive_weight=float(raw["positive_weight"]),
        threshold=float(raw["threshold"]),
        max_steps=int(raw["max_steps"]),
        checkpoint_interval_steps=int(raw["checkpoint_interval_steps"]),
        keep_last_checkpoints=int(raw["keep_last_checkpoints"]),
        resume=str(raw["resume"]),
        output_root=resolve(raw["output_root"]),
        log_root=resolve(raw["log_root"]),
    )
    positive = (
        config.sections_per_sequence,
        config.delta_width,
        config.hidden_width,
        config.cpu_threads,
        config.batch_size,
        config.learning_rate,
        config.positive_weight,
        config.max_steps,
        config.checkpoint_interval_steps,
        config.keep_last_checkpoints,
    )
    if min(positive) <= 0 or not 0 < config.threshold < 1:
        raise ValueError("Streaming commit controls must be positive")
    if config.resume not in {"auto", "never"}:
        raise ValueError("Invalid streaming commit resume mode")
    return config


class SequenceDataset:
    def __init__(self, sequences: list[StreamingSequence], cache_entries: int = 512) -> None:
        self.sequences = sequences
        self.cache_entries = cache_entries
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.sequences)

    def _load(self, path: Path) -> dict[str, Any]:
        key = str(path)
        value = self.cache.get(key)
        if value is not None:
            self.cache.move_to_end(key)
            return value
        value = torch.load(path, map_location="cpu", weights_only=False)
        self.cache[key] = value
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_entries:
            self.cache.popitem(last=False)
        return value

    def batch(self, indices: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        deltas = []
        commits = []
        refreshes = []
        elapsed_values = []
        for index in indices.tolist():
            sequence = self.sequences[index]
            section_deltas = [
                self._load(section.cache_path)["deltas"][:, 0].float()
                for section in sequence.sections
            ]
            commit, refresh, elapsed, _ = sequence.timeline()
            values = torch.cat(section_deltas)
            if values.shape[0] != commit.shape[0]:
                raise ValueError(f"Streaming delta/timeline mismatch: {sequence.sequence_id}")
            deltas.append(values)
            commits.append(commit)
            refreshes.append(refresh)
            elapsed_values.append(elapsed)
        widths = {value.shape[0] for value in deltas}
        if len(widths) != 1:
            raise ValueError("PoC batch requires fixed-length AudioCaps sections")
        return (
            torch.stack(deltas),
            torch.stack(commits),
            torch.stack(refreshes),
            torch.stack(elapsed_values),
        )


def _atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _latest_checkpoint(run_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    for path in sorted((run_dir / "checkpoints").glob("step-*.pt"), reverse=True):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except (EOFError, OSError, RuntimeError):
            continue
        if {"model", "optimizer", "next_step", "signature", "rng"} <= payload.keys():
            return path, payload
    return None


def _prune(run_dir: Path, keep: int) -> None:
    paths = sorted((run_dir / "checkpoints").glob("step-*.pt"))
    for path in paths[:-keep]:
        path.unlink()


@torch.no_grad()
def evaluate(
    model: CommitHead,
    data: SequenceDataset,
    config: CommitConfig,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    model.eval()
    if len(data) < 2:
        raise ValueError("Commit controls require at least two validation sequences")
    totals = {
        name: {
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "correct": 0,
            "total": 0,
            "loss": 0.0,
            "batches": 0,
        }
        for name in ("normal", "zero", "cross_sequence")
    }
    for start in range(0, len(data), config.batch_size):
        indices = torch.arange(start, min(start + config.batch_size, len(data)))
        deltas, targets, refresh, elapsed = data.batch(indices)
        donor_indices = (indices + 1) % len(data)
        cross_sequence, _, _, _ = data.batch(donor_indices)
        deltas, targets = deltas.to(device), targets.to(device)
        cross_sequence = cross_sequence.to(device)
        refresh, elapsed = refresh.to(device), elapsed.to(device)
        valid = torch.ones_like(refresh)
        for name, values in (
            ("normal", deltas),
            ("zero", torch.zeros_like(deltas)),
            ("cross_sequence", cross_sequence),
        ):
            logits = model(values, elapsed, refresh, valid)
            totals[name]["loss"] += float(
                commit_loss(
                    logits,
                    targets,
                    valid,
                    positive_weight=config.positive_weight,
                )
            )
            predicted = logits.sigmoid() >= config.threshold
            expected = targets.bool()
            totals[name]["true_positive"] += int((predicted & expected).sum())
            totals[name]["false_positive"] += int((predicted & ~expected).sum())
            totals[name]["false_negative"] += int((~predicted & expected).sum())
            totals[name]["correct"] += int(predicted.eq(expected).sum())
            totals[name]["total"] += expected.numel()
            totals[name]["batches"] += 1
    result = {}
    for name, values in totals.items():
        true_positive = values["true_positive"]
        false_positive = values["false_positive"]
        false_negative = values["false_negative"]
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        result[name] = {
            "loss": values["loss"] / values["batches"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": values["correct"] / values["total"],
            "predicted_commits": float(true_positive + false_positive),
            "target_commits": float(true_positive + false_negative),
        }
    return result


def run(config_path: Path, run_id: str | None, stop_after_step: int | None) -> dict[str, Any]:
    config = load_config(config_path)
    project_root = config_path.resolve().parent.parent
    if not git_worktree_is_clean(project_root):
        raise RuntimeError("Streaming commit runs require a clean Git worktree")
    _set_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    device = torch.device(config.device)
    sequences, discarded = build_sequences(
        config.prefix_manifest,
        sections_per_sequence=config.sections_per_sequence,
        seed=config.seed,
    )
    train = SequenceDataset(sequences["train"])
    validation = SequenceDataset(sequences["validation"])
    model = CommitHead(config.delta_width, config.hidden_width).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    signature = resolved_input_signature(config, {"prefix_manifest": config.prefix_manifest})
    selected = run_id or (
        f"streaming-commit-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )
    run_dir = config.output_root / selected
    log_path = config.log_root / selected / "metrics.jsonl"
    if run_dir.exists() and config.resume == "never":
        raise FileExistsError(f"Streaming commit run already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "resolved_config.json").is_file():
        _atomic_json(run_dir / "resolved_config.json", asdict(config))
    code_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=config_path.resolve().parent.parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not (run_dir / "metadata.json").is_file():
        _atomic_json(
            run_dir / "metadata.json",
            {
                "code_revision": code_revision,
                "device": config.device,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "torch_version": torch.__version__,
                "started_at_utc": datetime.now(UTC).isoformat(),
            },
        )
    start_step = 1
    resumed = _latest_checkpoint(run_dir) if config.resume == "auto" else None
    if resumed is not None:
        checkpoint_path, payload = resumed
        if payload["signature"] != signature:
            raise ValueError("Streaming commit checkpoint configuration mismatch")
        if payload.get("code_revision") != code_revision:
            raise ValueError("Exact streaming commit resume requires original code revision")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        random.setstate(payload["rng"]["python"])
        torch.random.set_rng_state(payload["rng"]["torch"])
        if device.type == "cuda" and payload["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state(payload["rng"]["cuda"], device)
        start_step = int(payload["next_step"])
        print(f"resume={checkpoint_path} next_step={start_step}", flush=True)
    final_step = min(config.max_steps, stop_after_step or config.max_steps)
    started = time.perf_counter()
    for step in range(start_step, final_step + 1):
        generator = torch.Generator().manual_seed(config.seed * 1_000_003 + step)
        indices = torch.randint(0, len(train), (config.batch_size,), generator=generator)
        deltas, targets, refresh, elapsed = train.batch(indices)
        deltas, targets = deltas.to(device), targets.to(device)
        refresh, elapsed = refresh.to(device), elapsed.to(device)
        valid = torch.ones_like(refresh)
        optimizer.zero_grad(set_to_none=True)
        logits = model(deltas, elapsed, refresh, valid)
        loss = commit_loss(
            logits,
            targets,
            valid,
            positive_weight=config.positive_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"step": step, "loss": float(loss)}) + "\n")
        if step % 10 == 0 or step == final_step:
            elapsed_time = time.perf_counter() - started
            eta = elapsed_time / (step - start_step + 1) * (final_step - step)
            print(
                f"commit_step={step}/{final_step} loss={float(loss):.5f} "
                f"elapsed={elapsed_time:.1f}s eta={eta:.1f}s",
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
                    "code_revision": code_revision,
                    "rng": {
                        "python": random.getstate(),
                        "torch": torch.random.get_rng_state(),
                        "cuda": (
                            torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                        ),
                    },
                },
            )
            _prune(run_dir, config.keep_last_checkpoints)
    if final_step < config.max_steps:
        result = {"run_id": selected, "status": "interrupted", "step": final_step}
    else:
        metrics = evaluate(model, validation, config, device)
        mechanics_passed = metrics["normal"]["f1"] >= 0.99
        content_specific_passed = (
            metrics["normal"]["f1"] > metrics["cross_sequence"]["f1"]
        )
        result = {
            "schema": "deltaomni.streaming_commit_poc.v2",
            "run_id": selected,
            "status": "complete",
            "metrics": metrics,
            "mechanics_passed": mechanics_passed,
            "content_specific_passed": content_specific_passed,
            "research_passed": False,
            "passed": False,
            "limitations": [
                "Every section has the same fixed commit period.",
                "Natural timing requires variable boundaries and no-event negatives.",
            ],
            "sequences": {split: len(values) for split, values in sequences.items()},
            "discarded_sections": discarded,
            "training_seconds": time.perf_counter() - started,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_json(run_dir / "summary.json", result)
    _atomic_json(run_dir / "status.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Train repeated-section streaming commit PoC")
    parser.add_argument("--config", type=Path, default=Path("configs/streaming_commit_poc.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.run_id, args.stop_after_step), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
