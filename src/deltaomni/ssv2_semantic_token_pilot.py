from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from deltaomni.distributed import DistributedContext, distributed_context, reduce_sums, unwrap
from deltaomni.evaluation import cross_label_permutations
from deltaomni.model import (
    ModalityDeltaCodec,
    PairDeltaEncoder,
    expand_embedding_delta,
    pool_embedding_delta,
)
from deltaomni.semantic_tokens import SemanticTokenBottleneck, assignment_statistics
from deltaomni.ssv2_pilot import load_pilot_config, prepare_embeddings
from deltaomni.ssv2_semantic_pilot import load_config as load_semantic_config
from deltaomni.train_sanity import _atomic_json, _set_seed


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    backend: str
    precision: str
    per_device_batch_size: int
    gradient_accumulation_steps: int
    num_workers: int
    cpu_threads: int
    tf32: bool
    nccl_compatibility_mode: bool


@dataclass(frozen=True)
class TokenConfig:
    hidden_dim: int
    token_count: int
    codebook_size: int
    num_heads: int
    temperature_start: float
    temperature_end: float
    semantic_weight: float
    reconstruction_weight: float
    sample_entropy_weight: float
    usage_entropy_weight: float


@dataclass(frozen=True)
class SemanticTokenPilotConfig:
    seed: int
    initialization: str
    ssv2_config: Path
    semantic_config: Path
    runtime: RuntimeConfig
    token: TokenConfig
    learning_rate: float
    weight_decay: float
    max_steps: int
    checkpoint_interval_steps: int
    evaluation_batch_size: int
    evaluation_split: str
    shuffle_repeats: int
    minimum_accuracy_gap: float
    minimum_effective_codes: float
    resume: str
    output_root: Path
    log_root: Path


def load_config(path: Path) -> SemanticTokenPilotConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    runtime = raw["runtime"]
    token = raw["token"]
    config = SemanticTokenPilotConfig(
        seed=int(raw["seed"]),
        initialization=str(raw.get("initialization", "semantic_checkpoint")),
        ssv2_config=resolve(raw["ssv2_config"]),
        semantic_config=resolve(raw["semantic_config"]),
        runtime=RuntimeConfig(
            device=str(runtime["device"]),
            backend=str(runtime["backend"]),
            precision=str(runtime["precision"]),
            per_device_batch_size=int(runtime["per_device_batch_size"]),
            gradient_accumulation_steps=int(runtime["gradient_accumulation_steps"]),
            num_workers=int(runtime["num_workers"]),
            cpu_threads=int(runtime["cpu_threads"]),
            tf32=bool(runtime["tf32"]),
            nccl_compatibility_mode=bool(runtime.get("nccl_compatibility_mode", False)),
        ),
        token=TokenConfig(
            hidden_dim=int(token["hidden_dim"]),
            token_count=int(token["token_count"]),
            codebook_size=int(token["codebook_size"]),
            num_heads=int(token["num_heads"]),
            temperature_start=float(token["temperature_start"]),
            temperature_end=float(token["temperature_end"]),
            semantic_weight=float(token["semantic_weight"]),
            reconstruction_weight=float(token["reconstruction_weight"]),
            sample_entropy_weight=float(token["sample_entropy_weight"]),
            usage_entropy_weight=float(token["usage_entropy_weight"]),
        ),
        learning_rate=float(raw["learning_rate"]),
        weight_decay=float(raw["weight_decay"]),
        max_steps=int(raw["max_steps"]),
        checkpoint_interval_steps=int(raw["checkpoint_interval_steps"]),
        evaluation_batch_size=int(raw["evaluation_batch_size"]),
        evaluation_split=str(raw.get("evaluation_split", "validation")),
        shuffle_repeats=int(raw["shuffle_repeats"]),
        minimum_accuracy_gap=float(raw["minimum_accuracy_gap"]),
        minimum_effective_codes=float(raw.get("minimum_effective_codes", 3.0)),
        resume=str(raw["resume"]),
        output_root=resolve(raw["output_root"]),
        log_root=resolve(raw["log_root"]),
    )
    if config.runtime.precision not in {"float32", "bfloat16"}:
        raise ValueError("precision must be float32 or bfloat16")
    if config.resume not in {"auto", "never"}:
        raise ValueError("resume must be auto or never")
    if config.runtime.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config.evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation_split must be validation or test")
    if config.initialization not in {"semantic_checkpoint", "random"}:
        raise ValueError("initialization must be semantic_checkpoint or random")
    return config


def _assert_evaluation_checkpoint_compatible(
    checkpoint_signature: str,
    evaluation_signature: str,
) -> None:
    checkpoint_config = json.loads(checkpoint_signature)
    evaluation_config = json.loads(evaluation_signature)
    checkpoint_config.pop("evaluation_split", None)
    evaluation_config.pop("evaluation_split", None)
    if checkpoint_config != evaluation_config:
        raise ValueError("evaluation checkpoint training configuration is incompatible")


class CachedEmbeddingSplit:
    def __init__(self, manifest: dict[str, Any], split: str) -> None:
        self.records = manifest["splits"][split]
        self.labels = torch.tensor(
            [int(record["class_index"]) for record in self.records],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.records)

    def load_batch(
        self,
        indices: Tensor,
        executor: ThreadPoolExecutor | None,
    ) -> tuple[Tensor, Tensor]:
        selected = indices.detach().to(device="cpu", dtype=torch.long).tolist()

        def load(index: int) -> Tensor:
            payload = torch.load(
                self.records[index]["cache_path"],
                map_location="cpu",
                weights_only=False,
            )
            return payload["embeddings"].float()

        embeddings = (
            list(executor.map(load, selected))
            if executor is not None
            else list(map(load, selected))
        )
        return torch.stack(embeddings), self.labels[selected]


class SemanticTokenModel(nn.Module):
    def __init__(self, codec: ModalityDeltaCodec, bottleneck: SemanticTokenBottleneck) -> None:
        super().__init__()
        self.codec = codec
        self.codec.policy.requires_grad_(False)
        self.codec.caption_decoder.requires_grad_(False)
        self.bottleneck = bottleneck

    def condition(self, full: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        anchor = full[:, 0]
        previous = anchor
        slots = torch.zeros(
            full.shape[0],
            self.codec.delta_encoder.queries.shape[0],
            full.shape[-1],
            dtype=full.dtype,
            device=full.device,
        )
        losses = []
        last_delta = None
        for time_index in range(1, full.shape[1]):
            current = full[:, time_index]
            last_delta = self.codec.delta_encoder(previous, current)
            slots = self.codec.accumulator(slots, last_delta)
            reconstructed = self.codec.reconstructor(anchor, slots)
            losses.append(F.smooth_l1_loss(reconstructed, current))
            previous = current
        if last_delta is None:
            raise ValueError("semantic-token pilot requires at least two frames")
        return slots, last_delta, torch.stack(losses).mean()

    def forward(self, full: Tensor, temperature: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        slots, _, reconstruction = self.condition(full)
        output = self.bottleneck(
            slots,
            modality_index=1,
            temperature=temperature,
            hard=True,
        )
        statistics = assignment_statistics(output.assignment_probabilities)
        return (
            output.class_logits,
            reconstruction,
            statistics["sample_entropy"],
            statistics["usage_entropy"],
        )

    def reconstruction_sums(self, full: Tensor) -> dict[str, Tensor]:
        anchor = full[:, 0]
        previous = anchor
        slots = torch.zeros(
            full.shape[0],
            self.codec.delta_encoder.queries.shape[0],
            full.shape[-1],
            dtype=full.dtype,
            device=full.device,
        )
        sums = {
            name: full.new_zeros((), dtype=torch.float64)
            for name in ("learned", "anchor", "last", "raw_pooled", "elements")
        }
        for time_index in range(1, full.shape[1]):
            current = full[:, time_index]
            delta = self.codec.delta_encoder(previous, current)
            slots = self.codec.accumulator(slots, delta)
            learned = self.codec.reconstructor(anchor, slots)
            last = self.codec.reconstructor(anchor, delta)
            pooled = pool_embedding_delta(
                current - anchor,
                self.codec.delta_encoder.queries.shape[0],
            )
            raw_pooled = anchor + expand_embedding_delta(
                pooled,
                current.shape[1],
            )
            sums["learned"] += (learned - current).double().square().sum()
            sums["anchor"] += (anchor - current).double().square().sum()
            sums["last"] += (last - current).double().square().sum()
            sums["raw_pooled"] += (raw_pooled - current).double().square().sum()
            sums["elements"] += current.numel()
            previous = current
        return sums


def _temperature(config: SemanticTokenPilotConfig, step: int) -> float:
    progress = min(1.0, max(0.0, (step - 1) / max(1, config.max_steps - 1)))
    start = math.log(config.token.temperature_start)
    end = math.log(config.token.temperature_end)
    return math.exp(start + progress * (end - start))


def _latest_semantic_checkpoint(config: SemanticTokenPilotConfig) -> tuple[str, Path]:
    semantic = load_semantic_config(config.semantic_config)
    summary = json.loads(
        (semantic.output_root / "latest_summary.json").read_text(encoding="utf-8")
    )
    run_dir = semantic.output_root / summary["run_id"]
    checkpoints = sorted(run_dir.glob("checkpoints/step-*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No semantic checkpoint under {run_dir}")
    return str(summary["run_id"]), checkpoints[-1]


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
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
    for path in sorted(run_dir.glob("checkpoints/step-*.pt"), reverse=True):
        payload = _checkpoint(path)
        if payload is not None:
            return path, payload
    return None


def _rng_state(context: DistributedContext) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(context.device) if context.device.type == "cuda" else None,
    }


def _restore_rng(state: dict[str, Any], context: DistributedContext) -> None:
    random.setstate(state["python"])
    torch.random.set_rng_state(state["torch"])
    if context.device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], context.device)


def _gather_rng_states(context: DistributedContext) -> list[dict[str, Any]] | None:
    local = _rng_state(context)
    if context.world_size == 1:
        return [local]
    gathered: list[dict[str, Any] | None] | None = (
        [None] * context.world_size if context.is_primary else None
    )
    torch.distributed.gather_object(local, gathered, dst=0)
    if gathered is None:
        return None
    return [state for state in gathered if state is not None]


def _broadcast_run_id(value: str | None, context: DistributedContext) -> str:
    values = [value]
    if context.world_size > 1:
        torch.distributed.broadcast_object_list(values, src=0)
    if values[0] is None:
        raise RuntimeError("primary rank did not select a run id")
    return values[0]


def _select_run_id(config: SemanticTokenPilotConfig, signature: str) -> str:
    active_path = config.output_root / "active_run.json"
    if config.resume == "auto" and active_path.is_file():
        active = json.loads(active_path.read_text(encoding="utf-8"))
        if active.get("config_signature") == signature and active.get("status") != "complete":
            return str(active["run_id"])
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"ssv2-semantic-token-{timestamp}-{uuid.uuid4().hex[:8]}"


def _wilson(successes: float, total: float) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - margin, center + margin


@torch.no_grad()
def _evaluate(
    model: SemanticTokenModel,
    validation: CachedEmbeddingSplit,
    config: SemanticTokenPilotConfig,
    context: DistributedContext,
    executor: ThreadPoolExecutor | None,
) -> dict[str, Any]:
    model.eval()
    permutations = cross_label_permutations(
        validation.labels,
        repeats=config.shuffle_repeats,
        seed=config.seed + 50_000,
    )
    conditions = ["normal", "zero", "last"] + [
        f"shuffled_{index}" for index in range(config.shuffle_repeats)
    ]
    modes = ("soft", "hard")
    counts = {
        f"{mode}_{condition}": torch.zeros(2, device=context.device, dtype=torch.float64)
        for mode in modes
        for condition in conditions
    }
    code_counts = torch.zeros(
        config.token.codebook_size,
        device=context.device,
        dtype=torch.float64,
    )
    reconstruction = {
        name: torch.zeros((), device=context.device, dtype=torch.float64)
        for name in ("learned", "anchor", "last", "raw_pooled", "elements")
    }
    local_indices = torch.arange(context.rank, len(validation), context.world_size)
    raw_model = unwrap(model)
    for start in range(0, local_indices.numel(), config.evaluation_batch_size):
        targets = local_indices[start : start + config.evaluation_batch_size]
        full, labels = validation.load_batch(targets, executor)
        if context.device.type == "cuda":
            full = full.pin_memory()
        full = full.to(context.device, non_blocking=True)
        labels = labels.to(context.device, non_blocking=True)
        slots, last, _ = raw_model.condition(full)
        current_reconstruction = raw_model.reconstruction_sums(full)
        for name, value in current_reconstruction.items():
            reconstruction[name] += value
        states = {"normal": slots, "zero": torch.zeros_like(slots), "last": last}
        for repeat, permutation in enumerate(permutations):
            sources = permutation[targets]
            source_full, _ = validation.load_batch(sources, executor)
            if context.device.type == "cuda":
                source_full = source_full.pin_memory()
            source_full = source_full.to(context.device, non_blocking=True)
            source_slots, _, _ = raw_model.condition(source_full)
            states[f"shuffled_{repeat}"] = source_slots
        for condition, state in states.items():
            for mode in modes:
                output = raw_model.bottleneck(
                    state,
                    modality_index=1,
                    temperature=config.token.temperature_end,
                    hard=mode == "hard",
                )
                correct = output.class_logits.argmax(dim=-1).eq(labels).sum()
                counts[f"{mode}_{condition}"] += torch.tensor(
                    [float(correct), float(labels.numel())],
                    device=context.device,
                    dtype=torch.float64,
                )
                if mode == "hard" and condition == "normal":
                    code_counts += torch.bincount(
                        output.code_ids.flatten(),
                        minlength=config.token.codebook_size,
                    )
    reduced = reduce_sums(
        {
            **counts,
            "code_counts": code_counts,
            **{f"reconstruction_{key}": value for key, value in reconstruction.items()},
        }
    )
    metrics: dict[str, Any] = {}
    for mode in modes:
        for condition in ("normal", "zero", "last"):
            correct, total = reduced[f"{mode}_{condition}"].tolist()
            low, high = _wilson(correct, total)
            metrics[f"{mode}_{condition}"] = {
                "accuracy": correct / total,
                "correct": int(correct),
                "count": int(total),
                "wilson_95": [low, high],
            }
        shuffled = []
        for repeat in range(config.shuffle_repeats):
            correct, total = reduced[f"{mode}_shuffled_{repeat}"].tolist()
            shuffled.append(correct / total)
        metrics[f"{mode}_shuffled"] = {
            "accuracy_mean": sum(shuffled) / len(shuffled),
            "accuracy_min": min(shuffled),
            "accuracy_max": max(shuffled),
            "repeats": shuffled,
        }
    usage = reduced["code_counts"]
    active_codes = int((usage > 0).sum())
    probabilities = usage / usage.sum().clamp_min(1)
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * nonzero.log()).sum())
    metrics["code_usage"] = {
        "active_codes": active_codes,
        "effective_codes": math.exp(entropy),
        "counts": [int(value) for value in usage.tolist()],
    }
    elements = float(reduced["reconstruction_elements"])
    metrics["reconstruction"] = {
        name: float(reduced[f"reconstruction_{name}"]) / elements
        for name in ("learned", "anchor", "last", "raw_pooled")
    }
    return metrics


def run(
    config_path: Path,
    backbone_config_path: Path,
    provenance_path: Path,
    run_id_override: str | None = None,
    evaluation_source_run_id: str | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    torch.set_num_threads(config.runtime.cpu_threads)
    with distributed_context(
        config.runtime.device,
        config.runtime.backend,
        nccl_compatibility_mode=config.runtime.nccl_compatibility_mode,
    ) as context:
        if config.runtime.tf32 and context.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if (
            config.runtime.precision == "bfloat16"
            and context.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("Configured bfloat16 is not supported by this CUDA device")
        _set_seed(config.seed + context.rank)
        ssv2 = load_pilot_config(config.ssv2_config)
        manifest_path = ssv2.cache_root / "manifest.json"
        if context.is_primary:
            prepare_embeddings(ssv2, backbone_config_path, provenance_path)
        if context.world_size > 1:
            torch.distributed.barrier()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train = CachedEmbeddingSplit(manifest, "train")
        validation = CachedEmbeddingSplit(manifest, config.evaluation_split)
        if len(torch.unique(train.labels)) != len(ssv2.classes):
            raise ValueError("training manifest does not cover every configured class")

        codec = ModalityDeltaCodec(ssv2.model)
        source_run_id = None
        if config.initialization == "semantic_checkpoint":
            source_run_id, source_checkpoint = _latest_semantic_checkpoint(config)
            source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
            incompatible = codec.load_state_dict(source["codec"], strict=False)
            allowed_missing = {"reconstructor.direct_projection.weight"}
            if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
                raise ValueError(f"Incompatible semantic checkpoint: {incompatible}")
        bottleneck = SemanticTokenBottleneck(
            input_dim=ssv2.model.embedding_dim,
            hidden_dim=config.token.hidden_dim,
            token_count=config.token.token_count,
            codebook_size=config.token.codebook_size,
            classes=len(ssv2.classes),
            num_heads=config.token.num_heads,
        )
        model: nn.Module = SemanticTokenModel(codec, bottleneck).to(context.device)
        if context.world_size > 1:
            model = DistributedDataParallel(model, device_ids=[context.local_rank])
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        resolved = asdict(config)
        signature = json.dumps(resolved, sort_keys=True, default=str)
        selected_run_id = run_id_override
        if context.is_primary and selected_run_id is None:
            selected_run_id = _select_run_id(config, signature)
        run_id = _broadcast_run_id(selected_run_id, context)
        run_dir = config.output_root / run_id
        log_path = config.log_root / run_id / "metrics.jsonl"
        if context.is_primary:
            run_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(
                config.output_root / "active_run.json",
                {"run_id": run_id, "status": "running", "config_signature": signature},
            )
        if context.world_size > 1:
            torch.distributed.barrier()

        start_step = 1
        resumed = _latest_checkpoint(run_dir) if config.resume == "auto" else None
        if evaluation_source_run_id is not None:
            if evaluation_source_run_id == run_id:
                raise ValueError("evaluation source and destination run IDs must differ")
            source_run_dir = config.output_root / evaluation_source_run_id
            source_checkpoint = _latest_checkpoint(source_run_dir)
            if source_checkpoint is None:
                raise FileNotFoundError(f"No checkpoint under {source_run_dir}")
            checkpoint_path, payload = source_checkpoint
            _assert_evaluation_checkpoint_compatible(
                payload["config_signature"],
                signature,
            )
            if int(payload["world_size"]) != context.world_size:
                raise ValueError("evaluation requires the checkpoint's distributed world size")
            unwrap(model).load_state_dict(payload["model"])
            start_step = config.max_steps + 1
            if context.is_primary:
                print(f"evaluation_only={checkpoint_path}", flush=True)
        elif resumed is not None:
            checkpoint_path, payload = resumed
            if payload["config_signature"] != signature:
                raise ValueError("checkpoint configuration is incompatible")
            if int(payload["world_size"]) != context.world_size:
                raise ValueError("exact resume requires the original distributed world size")
            unwrap(model).load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            start_step = int(payload["next_step"])
            _restore_rng(payload["rng_states"][context.rank], context)
            if context.is_primary:
                print(f"resume={checkpoint_path} next_step={start_step}", flush=True)

        executor = (
            ThreadPoolExecutor(max_workers=config.runtime.num_workers)
            if config.runtime.num_workers > 0
            else None
        )
        started = time.perf_counter()
        global_batch = config.runtime.per_device_batch_size * context.world_size
        try:
            for step in range(start_step, config.max_steps + 1):
                optimizer.zero_grad(set_to_none=True)
                accumulated = torch.zeros(4, device=context.device)
                for accumulation in range(config.runtime.gradient_accumulation_steps):
                    generator = torch.Generator().manual_seed(
                        config.seed * 1_000_003 + step * 101 + accumulation
                    )
                    indices = torch.randint(0, len(train), (global_batch,), generator=generator)
                    local = indices.reshape(context.world_size, -1)[context.rank]
                    full, labels = train.load_batch(local, executor)
                    if context.device.type == "cuda":
                        full = full.pin_memory()
                    full = full.to(context.device, non_blocking=True)
                    labels = labels.to(context.device, non_blocking=True)
                    autocast = torch.autocast(
                        device_type=context.device.type,
                        dtype=torch.bfloat16,
                        enabled=config.runtime.precision == "bfloat16",
                    )
                    synchronize = accumulation == config.runtime.gradient_accumulation_steps - 1
                    sync_context = (
                        nullcontext()
                        if synchronize or not isinstance(model, DistributedDataParallel)
                        else model.no_sync()
                    )
                    with sync_context:
                        with autocast:
                            logits, reconstruction, sample_entropy, usage_entropy = model(
                                full,
                                _temperature(config, step),
                            )
                            semantic = F.cross_entropy(logits, labels)
                            loss = (
                                config.token.semantic_weight * semantic
                                + config.token.reconstruction_weight * reconstruction
                                + config.token.sample_entropy_weight * sample_entropy
                                - config.token.usage_entropy_weight * usage_entropy
                            ) / config.runtime.gradient_accumulation_steps
                        loss.backward()
                    accumulated += torch.stack(
                        (
                            loss.detach(),
                            semantic.detach(),
                            reconstruction.detach(),
                            usage_entropy.detach(),
                        )
                    )
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                reduced = reduce_sums({"metrics": accumulated})["metrics"] / context.world_size
                if context.is_primary and (step % 10 == 0 or step == config.max_steps):
                    elapsed = time.perf_counter() - started
                    completed = step - start_step + 1
                    eta = elapsed / completed * (config.max_steps - step)
                    peak = (
                        torch.cuda.max_memory_reserved(context.device) / 2**30
                        if context.device.type == "cuda"
                        else 0.0
                    )
                    print(
                        f"semantic_token_step={step}/{config.max_steps} "
                        f"loss={float(reduced[0]):.5f} semantic={float(reduced[1]):.5f} "
                        f"reconstruction={float(reduced[2]):.5f} "
                        f"global_batch={global_batch * config.runtime.gradient_accumulation_steps} "
                        f"peak_reserved_gib={peak:.2f} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )
                if context.is_primary:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with log_path.open("a", encoding="utf-8") as stream:
                        stream.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "loss": float(reduced[0]),
                                    "semantic": float(reduced[1]),
                                    "reconstruction": float(reduced[2]),
                                    "temperature": _temperature(config, step),
                                }
                            )
                            + "\n"
                        )
                if step % config.checkpoint_interval_steps == 0 or step == config.max_steps:
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
                                "delta_algorithm": PairDeltaEncoder.ALGORITHM_VERSION,
                            },
                        )
            training_seconds = time.perf_counter() - started
            trained_steps = max(0, config.max_steps - start_step + 1)
            metrics = _evaluate(unwrap(model), validation, config, context, executor)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        hard = metrics["hard_normal"]["accuracy"]
        soft = metrics["soft_normal"]["accuracy"]
        chance = 1 / len(ssv2.classes)
        checks = {
            "hard_above_chance": hard > chance,
            "hard_beats_zero": hard > metrics["hard_zero"]["accuracy"],
            "hard_beats_last": hard > metrics["hard_last"]["accuracy"],
            "hard_beats_cross_label_shuffle": (
                hard - metrics["hard_shuffled"]["accuracy_max"]
                >= config.minimum_accuracy_gap
            ),
            "soft_above_chance": soft > chance,
            "soft_beats_cross_label_shuffle": (
                soft - metrics["soft_shuffled"]["accuracy_max"]
                >= config.minimum_accuracy_gap
            ),
            "codebook_not_collapsed": (
                metrics["code_usage"]["active_codes"] >= len(ssv2.classes)
                and metrics["code_usage"]["effective_codes"]
                >= config.minimum_effective_codes
            ),
            "reconstruction_beats_anchor": (
                metrics["reconstruction"]["learned"] < metrics["reconstruction"]["anchor"]
            ),
            "reconstruction_beats_last": (
                metrics["reconstruction"]["learned"] < metrics["reconstruction"]["last"]
            ),
            "reconstruction_beats_raw_pooled": (
                metrics["reconstruction"]["learned"]
                < metrics["reconstruction"]["raw_pooled"]
            ),
        }
        passed = all(checks.values())
        report = {
            "run_id": run_id,
            "evaluation_source_run_id": evaluation_source_run_id,
            "source_semantic_run_id": source_run_id,
            "status": "signal" if passed else "inconclusive",
            "world_size": context.world_size,
            "global_batch_size": global_batch * config.runtime.gradient_accumulation_steps,
            "training_seconds": training_seconds,
            "training_samples_per_second": (
                trained_steps
                * global_batch
                * config.runtime.gradient_accumulation_steps
                / max(training_seconds, 1e-9)
            ),
            "peak_reserved_gib_per_rank": (
                torch.cuda.max_memory_reserved(context.device) / 2**30
                if context.device.type == "cuda"
                else 0.0
            ),
            "evaluation_split": config.evaluation_split,
            "metrics": metrics,
            "checks": checks,
            "passed": passed,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        if context.is_primary:
            _atomic_json(run_dir / "summary.json", report)
            _atomic_json(config.output_root / "latest_summary.json", report)
            _atomic_json(
                config.output_root / "active_run.json",
                {"run_id": run_id, "status": "complete", "config_signature": signature},
            )
        if context.world_size > 1:
            torch.distributed.barrier()
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run scalable SSV2 semantic-token pilot")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ssv2_semantic_token_a6000.yaml"),
    )
    parser.add_argument("--backbones", type=Path, default=Path("configs/backbones.yaml"))
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--evaluation-source-run-id")
    args = parser.parse_args()
    report = run(
        args.config,
        args.backbones,
        args.provenance,
        args.run_id,
        args.evaluation_source_run_id,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
