from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
import uuid
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from deltaomni.deltatok_train import DeltaTok
from deltaomni.distributed import distributed_context, reduce_sums, unwrap
from deltaomni.train_sanity import _atomic_json, _set_seed


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    backend: str
    nccl_compatibility_mode: bool
    precision: str
    cpu_threads: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    cache_entries: int


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    model_dim: int
    tokens_per_frame: int
    delta_tokens: int
    depth: int
    num_heads: int


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    checkpoint_interval_steps: int
    keep_last_checkpoints: int
    gradient_clip_norm: float
    resume: str


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int
    retrieval_limit: int
    minimum_mse_relative_improvement: float
    minimum_retrieval_absolute_improvement: float


@dataclass(frozen=True)
class ScaleConfig:
    seed: int
    modality: str
    cache_manifest: Path
    runtime: RuntimeConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    output_root: Path
    log_root: Path


def load_config(path: Path) -> ScaleConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    config = ScaleConfig(
        seed=int(raw["seed"]),
        modality=str(raw["modality"]),
        cache_manifest=resolve(raw["cache_manifest"]),
        runtime=RuntimeConfig(**raw["runtime"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        output_root=resolve(raw["output_root"]),
        log_root=resolve(raw["log_root"]),
    )
    if config.modality not in {"video", "audio"}:
        raise ValueError("DeltaTok modality must be video or audio")
    if config.runtime.precision != "bfloat16":
        raise ValueError("Scale DeltaTok requires bfloat16")
    if config.training.resume not in {"auto", "never"}:
        raise ValueError("Invalid DeltaTok resume mode")
    if config.model.model_dim % config.model.num_heads:
        raise ValueError("DeltaTok model width must divide evenly across heads")
    positive = (
        config.runtime.cpu_threads,
        config.runtime.per_device_batch_size,
        config.runtime.gradient_accumulation_steps,
        config.runtime.cache_entries,
        config.model.input_dim,
        config.model.model_dim,
        config.model.tokens_per_frame,
        config.model.delta_tokens,
        config.model.depth,
        config.model.num_heads,
        config.training.learning_rate,
        config.training.warmup_steps,
        config.training.max_steps,
        config.training.checkpoint_interval_steps,
        config.training.keep_last_checkpoints,
        config.training.gradient_clip_norm,
        config.evaluation.batch_size,
        config.evaluation.retrieval_limit,
        config.evaluation.minimum_mse_relative_improvement,
        config.evaluation.minimum_retrieval_absolute_improvement,
    )
    if min(positive) <= 0:
        raise ValueError("Scale DeltaTok controls must be positive")
    return config


class PairDataset:
    def __init__(self, manifest: dict[str, Any], split: str, cache_entries: int) -> None:
        if split not in manifest["splits"]:
            raise ValueError(f"Missing cache split: {split}")
        self.records = list(manifest["splits"][split])
        self.pairs = [
            (record_index, step)
            for record_index, record in enumerate(self.records)
            for step in range(1, int(record["blocks"]))
        ]
        self.cache_entries = cache_entries
        self.cache: OrderedDict[str, Tensor] = OrderedDict()
        self.lock = Lock()

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_record(self, record_index: int) -> Tensor:
        path = self.records[record_index]["cache_path"]
        with self.lock:
            cached = self.cache.get(path)
            if cached is not None:
                self.cache.move_to_end(path)
                return cached
        payload = torch.load(path, map_location="cpu", weights_only=False)
        values = payload["embeddings"].float()
        if values.ndim != 3 or values.shape[0] != int(self.records[record_index]["blocks"]):
            raise ValueError(f"Invalid cached block tensor: {path}")
        with self.lock:
            self.cache[path] = values
            self.cache.move_to_end(path)
            while len(self.cache) > self.cache_entries:
                self.cache.popitem(last=False)
        return values

    def load_batch(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        values = []
        for index in indices.cpu().tolist():
            record_index, step = self.pairs[index]
            blocks = self._load_record(record_index)
            values.append((blocks[step - 1], blocks[step]))
        return (
            torch.stack([previous for previous, _ in values]),
            torch.stack([current for _, current in values]),
        )

    def load_record(self, record_index: int) -> Tensor:
        return self._load_record(record_index)


def _load_datasets(config: ScaleConfig) -> tuple[PairDataset, PairDataset, dict[str, Any]]:
    manifest = json.loads(config.cache_manifest.read_text(encoding="utf-8"))
    if manifest.get("modality") != config.modality:
        raise ValueError("DeltaTok modality/cache manifest mismatch")
    if int(manifest["tokens_per_block"]) != config.model.tokens_per_frame:
        raise ValueError("DeltaTok token count/cache manifest mismatch")
    if int(manifest["embedding_width"]) != config.model.input_dim:
        raise ValueError("DeltaTok width/cache manifest mismatch")
    return (
        PairDataset(manifest, "train", config.runtime.cache_entries),
        PairDataset(manifest, "validation", config.runtime.cache_entries),
        manifest,
    )


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _checkpoint(path: Path) -> dict[str, Any] | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (EOFError, OSError, RuntimeError):
        return None
    required = {
        "next_step",
        "model",
        "optimizer",
        "config_signature",
        "rng_states",
        "world_size",
    }
    return payload if required <= payload.keys() else None


def _latest_checkpoint(run_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    for path in sorted((run_dir / "checkpoints").glob("step-*.pt"), reverse=True):
        payload = _checkpoint(path)
        if payload is not None:
            return path, payload
    return None


def _rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_rng(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    torch.random.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def _gather_rng_states(context) -> list[dict[str, Any]] | None:
    local = _rng_state(context.device)
    if context.world_size == 1:
        return [local]
    gathered = [None] * context.world_size if context.is_primary else None
    torch.distributed.gather_object(local, gathered, dst=0)
    return gathered


def _broadcast_string(value: str | None, context) -> str:
    values = [value]
    if context.world_size > 1:
        torch.distributed.broadcast_object_list(values, src=0)
    if values[0] is None:
        raise RuntimeError("Primary rank did not select a run ID")
    return str(values[0])


def _select_run_id(config: ScaleConfig, signature: str) -> str:
    active_path = config.output_root / "active_run.json"
    if config.training.resume == "auto" and active_path.is_file():
        active = json.loads(active_path.read_text(encoding="utf-8"))
        if active.get("config_signature") == signature and active.get("status") != "complete":
            return str(active["run_id"])
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"deltatok-{config.modality}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()


def _prune_checkpoints(run_dir: Path, keep: int) -> None:
    checkpoints = sorted((run_dir / "checkpoints").glob("step-*.pt"))
    for path in checkpoints[:-keep]:
        path.unlink()


@torch.no_grad()
def _evaluate_pairs(
    model: DeltaTok,
    data: PairDataset,
    config: ScaleConfig,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    squared = copy_squared = cosine = 0.0
    reconstructed = []
    targets = []
    count = 0
    for start in range(0, len(data), config.evaluation.batch_size):
        indices = torch.arange(start, min(start + config.evaluation.batch_size, len(data)))
        previous, current = data.load_batch(indices)
        previous = previous.to(device)
        current = current.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            predicted, _ = model(previous, current)
        batch = len(indices)
        squared += float((predicted.float() - current).square().mean()) * batch
        copy_squared += float((previous - current).square().mean()) * batch
        cosine += (
            float(F.cosine_similarity(predicted.float().flatten(1), current.flatten(1)).mean())
            * batch
        )
        if count < config.evaluation.retrieval_limit:
            retained = min(batch, config.evaluation.retrieval_limit - count)
            reconstructed.append(predicted[:retained].mean(1).float().cpu())
            targets.append(current[:retained].mean(1).float().cpu())
        count += batch
    rec = torch.cat(reconstructed)
    target = torch.cat(targets)
    similarity = F.normalize(rec) @ F.normalize(target).T
    retrieval = float(similarity.argmax(1).eq(torch.arange(len(rec))).float().mean())
    return {
        "mse": squared / count,
        "copy_previous_mse": copy_squared / count,
        "cosine": cosine / count,
        "retrieval_r1": retrieval,
        "retrieval_candidates": len(rec),
        "pairs": count,
    }


@torch.no_grad()
def _evaluate_rollout(
    model: DeltaTok,
    data: PairDataset,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    by_horizon: dict[int, dict[str, float]] = {}
    final_reconstructed = []
    final_targets = []
    final_anchors = []
    final_reversed = []
    normal_final = anchor_final = reversed_final = 0.0
    for record_index in range(len(data.records)):
        actual = data.load_record(record_index).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            deltas = [
                model.encode(actual[step - 1 : step], actual[step : step + 1])
                for step in range(1, actual.shape[0])
            ]
            reconstructed = actual[0:1]
            for horizon, delta in enumerate(deltas, start=1):
                reconstructed = model.decode(reconstructed, delta)
                target = actual[horizon : horizon + 1]
                teacher = model.decode(actual[horizon - 1 : horizon], delta)
                values = by_horizon.setdefault(
                    horizon,
                    {"rollout_squared": 0.0, "teacher_squared": 0.0, "count": 0.0},
                )
                values["rollout_squared"] += float((reconstructed.float() - target).square().mean())
                values["teacher_squared"] += float((teacher.float() - target).square().mean())
                values["count"] += 1
            reverse = actual[0:1]
            for delta in reversed(deltas):
                reverse = model.decode(reverse, delta)
        final_target = actual[-1:]
        anchor = actual[0:1]
        normal_final += float((reconstructed.float() - final_target).square().mean())
        anchor_final += float((anchor - final_target).square().mean())
        reversed_final += float((reverse.float() - final_target).square().mean())
        final_reconstructed.append(reconstructed.mean(1).float().cpu())
        final_targets.append(final_target.mean(1).float().cpu())
        final_anchors.append(anchor.mean(1).float().cpu())
        final_reversed.append(reverse.mean(1).float().cpu())
    rec = torch.cat(final_reconstructed)
    target = torch.cat(final_targets)
    anchor = torch.cat(final_anchors)
    reverse = torch.cat(final_reversed)

    def retrieval(values: Tensor) -> float:
        similarity = F.normalize(values) @ F.normalize(target).T
        return float(similarity.argmax(1).eq(torch.arange(len(values))).float().mean())

    count = len(data.records)
    return {
        "by_horizon": {
            str(horizon): {
                "rollout_mse": values["rollout_squared"] / values["count"],
                "teacher_forced_mse": values["teacher_squared"] / values["count"],
                "examples": int(values["count"]),
            }
            for horizon, values in sorted(by_horizon.items())
        },
        "final_mse": normal_final / count,
        "anchor_final_mse": anchor_final / count,
        "reversed_delta_final_mse": reversed_final / count,
        "final_retrieval_r1": retrieval(rec),
        "anchor_final_retrieval_r1": retrieval(anchor),
        "reversed_delta_final_retrieval_r1": retrieval(reverse),
        "retrieval_candidates": count,
        "clips": count,
    }


def _metadata(config: ScaleConfig, context, code_revision: str) -> dict[str, Any]:
    return {
        "code_revision": code_revision,
        "cache_manifest": str(config.cache_manifest),
        "cache_manifest_sha256": hashlib.sha256(config.cache_manifest.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "world_size": context.world_size,
        "gpu": (
            torch.cuda.get_device_name(context.device) if context.device.type == "cuda" else None
        ),
        "started_at_utc": datetime.now(UTC).isoformat(),
    }


def run(
    config_path: Path,
    run_id_override: str | None,
    stop_after_step: int | None,
) -> dict[str, Any]:
    config = load_config(config_path)
    torch.set_num_threads(config.runtime.cpu_threads)
    _set_seed(config.seed)
    with distributed_context(
        config.runtime.device,
        config.runtime.backend,
        nccl_compatibility_mode=config.runtime.nccl_compatibility_mode,
    ) as context:
        train, validation, _ = _load_datasets(config)
        model: nn.Module = DeltaTok(config.model).to(context.device)
        if context.world_size > 1:
            model = DistributedDataParallel(model, device_ids=[context.local_rank])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        resolved = asdict(config)
        signature = json.dumps(resolved, sort_keys=True, default=str)
        selected = run_id_override
        if context.is_primary and selected is None:
            selected = _select_run_id(config, signature)
        run_id = _broadcast_string(selected, context)
        run_dir = config.output_root / run_id
        log_path = config.log_root / run_id / "metrics.jsonl"
        code_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config_path.resolve().parent.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if run_dir.exists() and config.training.resume == "never":
            raise FileExistsError(f"DeltaTok run already exists: {run_dir}")
        if context.is_primary:
            run_dir.mkdir(parents=True, exist_ok=True)
            if not (run_dir / "resolved_config.json").is_file():
                _atomic_json(run_dir / "resolved_config.json", resolved)
            if not (run_dir / "metadata.json").is_file():
                _atomic_json(run_dir / "metadata.json", _metadata(config, context, code_revision))
            _atomic_json(
                config.output_root / "active_run.json",
                {"run_id": run_id, "status": "running", "config_signature": signature},
            )
        if context.world_size > 1:
            torch.distributed.barrier()

        start_step = 1
        resumed = _latest_checkpoint(run_dir) if config.training.resume == "auto" else None
        if resumed is not None:
            checkpoint_path, payload = resumed
            if payload["config_signature"] != signature:
                raise ValueError("DeltaTok checkpoint configuration mismatch")
            if int(payload["world_size"]) != context.world_size:
                raise ValueError("Exact resume requires the original world size")
            if payload.get("code_revision") != code_revision:
                raise ValueError("Exact resume requires the original code revision")
            unwrap(model).load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            start_step = int(payload["next_step"])
            _restore_rng(payload["rng_states"][context.rank], context.device)
            if context.is_primary:
                print(f"resume={checkpoint_path} next_step={start_step}", flush=True)
                _append_jsonl(
                    log_path,
                    {"event": "resume", "checkpoint": str(checkpoint_path), "step": start_step},
                )

        final_step = min(config.training.max_steps, stop_after_step or config.training.max_steps)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(context.device)
        for step in range(start_step, final_step + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated = torch.zeros(2, device=context.device)
            for accumulation in range(config.runtime.gradient_accumulation_steps):
                generator = torch.Generator().manual_seed(
                    config.seed * 1_000_003 + step * 101 + accumulation
                )
                global_batch = config.runtime.per_device_batch_size * context.world_size
                indices = torch.randint(0, len(train), (global_batch,), generator=generator)
                local_indices = indices.reshape(context.world_size, -1)[context.rank]
                previous, current = train.load_batch(local_indices)
                previous = previous.to(context.device, non_blocking=True)
                current = current.to(context.device, non_blocking=True)
                synchronize = accumulation == config.runtime.gradient_accumulation_steps - 1
                sync = (
                    nullcontext()
                    if synchronize or not isinstance(model, DistributedDataParallel)
                    else model.no_sync()
                )
                with sync:
                    with torch.autocast(
                        device_type=context.device.type,
                        dtype=torch.bfloat16,
                    ):
                        predicted, _ = model(previous, current)
                        loss = F.mse_loss(predicted.float(), current) / (
                            config.runtime.gradient_accumulation_steps
                        )
                    loss.backward()
                accumulated += torch.stack((loss.detach(), F.mse_loss(previous, current).detach()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            warmup = min(1.0, step / config.training.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = config.training.learning_rate * warmup
            optimizer.step()
            reduced = reduce_sums({"metrics": accumulated})["metrics"] / context.world_size
            if context.is_primary:
                elapsed = time.perf_counter() - started
                completed = step - start_step + 1
                eta = elapsed / completed * (final_step - step)
                global_effective_batch = (
                    config.runtime.per_device_batch_size
                    * context.world_size
                    * config.runtime.gradient_accumulation_steps
                )
                record = {
                    "event": "train",
                    "step": step,
                    "mse": float(reduced[0]),
                    "copy_previous_mse": float(reduced[1])
                    / config.runtime.gradient_accumulation_steps,
                    "learning_rate": config.training.learning_rate * warmup,
                    "global_batch": global_effective_batch,
                }
                _append_jsonl(log_path, record)
                if step % 10 == 0 or step == final_step:
                    peak = torch.cuda.max_memory_reserved(context.device) / 2**30
                    print(
                        f"deltatok_step={step}/{final_step} mse={record['mse']:.6f} "
                        f"global_batch={global_effective_batch} peak_gib={peak:.2f} "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )
            checkpoint_due = (
                step % config.training.checkpoint_interval_steps == 0 or step == final_step
            )
            if checkpoint_due:
                rng_states = _gather_rng_states(context)
                if context.is_primary:
                    _atomic_torch_save(
                        run_dir / "checkpoints" / f"step-{step:06d}.pt",
                        {
                            "next_step": step + 1,
                            "model": unwrap(model).state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "config_signature": signature,
                            "rng_states": rng_states,
                            "world_size": context.world_size,
                            "code_revision": code_revision,
                        },
                    )
                    _prune_checkpoints(run_dir, config.training.keep_last_checkpoints)
        if context.world_size > 1:
            torch.distributed.barrier()

        if final_step < config.training.max_steps:
            result = {
                "schema": "deltaomni.deltatok_scale_training.v1",
                "run_id": run_id,
                "status": "interrupted",
                "step": final_step,
                "training_seconds": time.perf_counter() - started,
            }
        elif context.is_primary:
            teacher = _evaluate_pairs(unwrap(model), validation, config, context.device)
            rollout = _evaluate_rollout(unwrap(model), validation, context.device)
            minimum_relative = config.evaluation.minimum_mse_relative_improvement
            minimum_retrieval = config.evaluation.minimum_retrieval_absolute_improvement
            checks = {
                "teacher_beats_copy_previous": teacher["mse"]
                <= teacher["copy_previous_mse"] * (1 - minimum_relative),
                "rollout_beats_anchor_final": rollout["final_mse"]
                <= rollout["anchor_final_mse"] * (1 - minimum_relative),
                "rollout_beats_reversed_delta": (
                    rollout["final_mse"]
                    <= rollout["reversed_delta_final_mse"] * (1 - minimum_relative)
                ),
                "rollout_retrieval_beats_anchor": (
                    rollout["final_retrieval_r1"]
                    >= rollout["anchor_final_retrieval_r1"] + minimum_retrieval
                ),
                "rollout_retrieval_beats_reversed_delta": (
                    rollout["final_retrieval_r1"]
                    >= rollout["reversed_delta_final_retrieval_r1"] + minimum_retrieval
                ),
            }
            result = {
                "schema": "deltaomni.deltatok_scale_training.v1",
                "run_id": run_id,
                "status": "complete",
                "modality": config.modality,
                "training_seconds": time.perf_counter() - started,
                "teacher_forced": teacher,
                "autoregressive_rollout": rollout,
                "checks": checks,
                "passed": all(checks.values()),
                "completed_at_utc": datetime.now(UTC).isoformat(),
            }
            _atomic_json(run_dir / "summary.json", result)
        else:
            result = {}
        if context.is_primary:
            _atomic_json(run_dir / "status.json", result)
            _atomic_json(
                config.output_root / "active_run.json",
                {
                    "run_id": run_id,
                    "status": result["status"],
                    "config_signature": signature,
                },
            )
        if context.world_size > 1:
            torch.distributed.barrier()
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Train paper-scale DeltaTok with DDP")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/deltatok_vggsound_video.yaml"),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    report = run(args.config, args.run_id, args.stop_after_step)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
