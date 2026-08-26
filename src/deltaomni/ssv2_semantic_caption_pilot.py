from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from deltaomni.backbones import load_backbone_config
from deltaomni.distributed import DistributedContext, distributed_context, reduce_sums, unwrap
from deltaomni.evaluation import cross_label_permutations
from deltaomni.language import FrozenCausalCaptionBackend, SemanticTokenLanguageAdapter
from deltaomni.model import ModalityDeltaCodec
from deltaomni.provenance import audit as audit_provenance
from deltaomni.semantic_tokens import SemanticTokenBottleneck
from deltaomni.ssv2_pilot import load_pilot_config
from deltaomni.ssv2_semantic_token_pilot import (
    CachedEmbeddingSplit,
    SemanticTokenModel,
    _atomic_torch_save,
    _broadcast_run_id,
    _gather_rng_states,
    _latest_checkpoint,
    _restore_rng,
)
from deltaomni.train_sanity import _atomic_json, _set_seed


@dataclass(frozen=True)
class CaptionRuntimeConfig:
    device: str
    backend: str
    precision: str
    per_device_batch_size: int
    num_workers: int
    cpu_threads: int
    tf32: bool
    nccl_compatibility_mode: bool


@dataclass(frozen=True)
class SemanticCaptionConfig:
    seed: int
    ssv2_config: Path
    semantic_token_config: Path
    delta_run_id: str
    runtime: CaptionRuntimeConfig
    learning_rate: float
    weight_decay: float
    target_loss_weight: float
    ranking_loss_weight: float
    max_steps: int
    checkpoint_interval_steps: int
    evaluation_batch_size: int
    evaluation_split: str
    shuffle_repeats: int
    minimum_accuracy_gap: float
    hard_tokens: bool
    output_root: Path
    log_root: Path


def load_config(path: Path) -> SemanticCaptionConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    runtime = raw["runtime"]
    config = SemanticCaptionConfig(
        seed=int(raw["seed"]),
        ssv2_config=resolve(raw["ssv2_config"]),
        semantic_token_config=resolve(raw["semantic_token_config"]),
        delta_run_id=str(raw["delta_run_id"]),
        runtime=CaptionRuntimeConfig(
            device=str(runtime["device"]),
            backend=str(runtime["backend"]),
            precision=str(runtime["precision"]),
            per_device_batch_size=int(runtime["per_device_batch_size"]),
            num_workers=int(runtime["num_workers"]),
            cpu_threads=int(runtime["cpu_threads"]),
            tf32=bool(runtime["tf32"]),
            nccl_compatibility_mode=bool(runtime.get("nccl_compatibility_mode", False)),
        ),
        learning_rate=float(raw["learning_rate"]),
        weight_decay=float(raw["weight_decay"]),
        target_loss_weight=float(raw.get("target_loss_weight", 1.0)),
        ranking_loss_weight=float(raw.get("ranking_loss_weight", 1.0)),
        max_steps=int(raw["max_steps"]),
        checkpoint_interval_steps=int(raw["checkpoint_interval_steps"]),
        evaluation_batch_size=int(raw["evaluation_batch_size"]),
        evaluation_split=str(raw["evaluation_split"]),
        shuffle_repeats=int(raw["shuffle_repeats"]),
        minimum_accuracy_gap=float(raw["minimum_accuracy_gap"]),
        hard_tokens=bool(raw["hard_tokens"]),
        output_root=resolve(raw["output_root"]),
        log_root=resolve(raw["log_root"]),
    )
    if config.evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation_split must be validation or test")
    if config.runtime.precision != "bfloat16":
        raise ValueError("scaled caption pilot requires bfloat16")
    return config


def _load_delta_model(
    config: SemanticCaptionConfig,
    device: torch.device,
) -> tuple[SemanticTokenModel, Any]:
    token_raw = yaml.safe_load(config.semantic_token_config.read_text(encoding="utf-8"))
    ssv2 = load_pilot_config(config.ssv2_config)
    token = token_raw["token"]
    model = SemanticTokenModel(
        ModalityDeltaCodec(ssv2.model),
        SemanticTokenBottleneck(
            input_dim=ssv2.model.embedding_dim,
            hidden_dim=int(token["hidden_dim"]),
            token_count=int(token["token_count"]),
            codebook_size=int(token["codebook_size"]),
            classes=len(ssv2.classes),
            num_heads=int(token["num_heads"]),
        ),
    )
    token_output_root = Path(token_raw["output_root"])
    if not token_output_root.is_absolute():
        token_output_root = config.semantic_token_config.resolve().parent.parent / token_output_root
    run_dir = token_output_root / config.delta_run_id
    checkpoints = sorted(run_dir.glob("checkpoints/step-*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No delta checkpoint under {run_dir}")
    payload = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.requires_grad_(False)
    return model.to(device).eval(), ssv2


def _tokens(
    model: SemanticTokenModel,
    state: Tensor,
    *,
    hard: bool,
) -> Tensor:
    output = model.bottleneck(
        state,
        modality_index=1,
        temperature=0.2,
        hard=hard,
    )
    return output.tokens.detach()


@torch.no_grad()
def _evaluate(
    adapter: SemanticTokenLanguageAdapter,
    backend: FrozenCausalCaptionBackend,
    delta_model: SemanticTokenModel,
    split: CachedEmbeddingSplit,
    targets: tuple[str, ...],
    prompt: str,
    config: SemanticCaptionConfig,
    context: DistributedContext,
    executor: ThreadPoolExecutor | None,
) -> dict[str, Any]:
    adapter.eval()
    permutations = cross_label_permutations(
        split.labels,
        repeats=config.shuffle_repeats,
        seed=config.seed + 70_000,
    )
    conditions = ["normal", "zero", "last"] + [
        f"shuffled_{repeat}" for repeat in range(config.shuffle_repeats)
    ]
    totals = {
        condition: torch.zeros(3, device=context.device, dtype=torch.float64)
        for condition in conditions
    }
    local_indices = torch.arange(context.rank, len(split), context.world_size)
    for start in range(0, local_indices.numel(), config.evaluation_batch_size):
        indices = local_indices[start : start + config.evaluation_batch_size]
        full, labels = split.load_batch(indices, executor)
        if context.device.type == "cuda":
            full = full.pin_memory()
        full = full.to(context.device, non_blocking=True)
        labels = labels.to(context.device, non_blocking=True)
        slots, last, _ = delta_model.condition(full)
        states = {"normal": slots, "zero": torch.zeros_like(slots), "last": last}
        for repeat, permutation in enumerate(permutations):
            source_full, _ = split.load_batch(permutation[indices], executor)
            if context.device.type == "cuda":
                source_full = source_full.pin_memory()
            source_full = source_full.to(context.device, non_blocking=True)
            source_slots, _, _ = delta_model.condition(source_full)
            states[f"shuffled_{repeat}"] = source_slots
        for condition, state in states.items():
            prefix = adapter(_tokens(delta_model, state, hard=config.hard_tokens), 1)
            losses = backend.candidate_caption_losses(prefix, prompt, targets)
            target_nll = losses.gather(1, labels[:, None]).sum().double()
            correct = losses.argmin(dim=-1).eq(labels).sum().double()
            totals[condition] += torch.stack(
                (correct, target_nll, torch.tensor(labels.numel(), device=context.device))
            )
    reduced = reduce_sums(totals)
    metrics = {}
    for condition in ("normal", "zero", "last"):
        correct, nll, count = reduced[condition].tolist()
        metrics[condition] = {
            "accuracy": correct / count,
            "target_nll": nll / count,
            "count": int(count),
        }
    shuffled = []
    for repeat in range(config.shuffle_repeats):
        correct, nll, count = reduced[f"shuffled_{repeat}"].tolist()
        shuffled.append({"accuracy": correct / count, "target_nll": nll / count})
    metrics["shuffled"] = {
        "accuracy_mean": sum(value["accuracy"] for value in shuffled) / len(shuffled),
        "accuracy_max": max(value["accuracy"] for value in shuffled),
        "target_nll_mean": sum(value["target_nll"] for value in shuffled) / len(shuffled),
        "target_nll_min": min(value["target_nll"] for value in shuffled),
        "repeats": shuffled,
    }
    return metrics


def run(config_path: Path, run_id_override: str | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    torch.set_num_threads(config.runtime.cpu_threads)
    with distributed_context(
        config.runtime.device,
        config.runtime.backend,
        nccl_compatibility_mode=config.runtime.nccl_compatibility_mode,
    ) as context:
        _set_seed(config.seed + context.rank)
        if config.runtime.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        delta_model, ssv2 = _load_delta_model(config, context.device)
        manifest = json.loads((ssv2.cache_root / "manifest.json").read_text(encoding="utf-8"))
        train = CachedEmbeddingSplit(manifest, "train")
        evaluation = CachedEmbeddingSplit(manifest, config.evaluation_split)
        backbone = load_backbone_config(Path("configs/backbones.yaml"))
        backend = FrozenCausalCaptionBackend(
            backbone.language_large,
            backbone.cache_dir,
            context.device,
            audit_provenance(Path("configs/provenance.yaml")),
            dtype=torch.bfloat16,
        )
        token_hidden = delta_model.bottleneck.codebook.shape[1]
        adapter: nn.Module = SemanticTokenLanguageAdapter(token_hidden, backend.hidden_size).to(
            context.device
        )
        if context.world_size > 1:
            adapter = DistributedDataParallel(adapter, device_ids=[context.local_rank])
        optimizer = torch.optim.AdamW(
            adapter.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        signature = json.dumps(asdict(config), sort_keys=True, default=str)
        selected = run_id_override
        if context.is_primary and selected is None:
            selected = (
                f"ssv2-semantic-caption-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
                f"{uuid.uuid4().hex[:8]}"
            )
        run_id = _broadcast_run_id(selected, context)
        run_dir = config.output_root / run_id
        if context.is_primary:
            run_dir.mkdir(parents=True, exist_ok=True)
        if context.world_size > 1:
            torch.distributed.barrier()
        start_step = 1
        resumed = _latest_checkpoint(run_dir)
        if resumed is not None:
            _, payload = resumed
            incompatible = (
                payload["config_signature"] != signature
                or payload["world_size"] != context.world_size
            )
            if incompatible:
                raise ValueError("caption checkpoint is incompatible")
            unwrap(adapter).load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            _restore_rng(payload["rng_states"][context.rank], context)
            start_step = int(payload["next_step"])
        executor = (
            ThreadPoolExecutor(config.runtime.num_workers)
            if config.runtime.num_workers > 0
            else None
        )
        global_batch = config.runtime.per_device_batch_size * context.world_size
        started = time.perf_counter()
        try:
            for step in range(start_step, config.max_steps + 1):
                generator = torch.Generator().manual_seed(config.seed * 3_000_017 + step)
                indices = torch.randint(0, len(train), (global_batch,), generator=generator)
                local = indices.reshape(context.world_size, -1)[context.rank]
                full, labels = train.load_batch(local, executor)
                if context.device.type == "cuda":
                    full = full.pin_memory()
                full = full.to(context.device, non_blocking=True)
                labels = labels.to(context.device, non_blocking=True)
                with torch.no_grad():
                    slots, _, _ = delta_model.condition(full)
                    semantic_tokens = _tokens(delta_model, slots, hard=config.hard_tokens)
                adapter.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prefix = adapter(semantic_tokens, 1)
                    candidate_losses = backend.candidate_caption_losses(
                        prefix,
                        ssv2.caption.prompt,
                        ssv2.caption.targets,
                    )
                    target_loss = candidate_losses.gather(1, labels[:, None]).mean()
                    ranking_loss = F.cross_entropy(-candidate_losses, labels)
                    loss = (
                        config.target_loss_weight * target_loss
                        + config.ranking_loss_weight * ranking_loss
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                if context.is_primary and (step % 10 == 0 or step == config.max_steps):
                    elapsed = time.perf_counter() - started
                    eta = elapsed / (step - start_step + 1) * (config.max_steps - step)
                    print(
                        f"caption_step={step}/{config.max_steps} loss={float(loss):.5f} "
                        f"target={float(target_loss):.5f} rank={float(ranking_loss):.5f} "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )
                if step % config.checkpoint_interval_steps == 0 or step == config.max_steps:
                    rng_states = _gather_rng_states(context)
                    if context.is_primary:
                        _atomic_torch_save(
                            run_dir / "checkpoints" / f"step-{step:06d}.pt",
                            {
                                "next_step": step + 1,
                                "model": unwrap(adapter).state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "config_signature": signature,
                                "rng_states": rng_states,
                                "world_size": context.world_size,
                            },
                        )
            metrics = _evaluate(
                unwrap(adapter),
                backend,
                delta_model,
                evaluation,
                ssv2.caption.targets,
                ssv2.caption.prompt,
                config,
                context,
                executor,
            )
        finally:
            if executor is not None:
                executor.shutdown()
        normal = metrics["normal"]
        checks = {
            "accuracy_above_chance": normal["accuracy"] > 1 / len(ssv2.caption.targets),
            "accuracy_beats_zero": normal["accuracy"] > metrics["zero"]["accuracy"],
            "accuracy_beats_last": normal["accuracy"] > metrics["last"]["accuracy"],
            "accuracy_beats_worst_shuffle": (
                normal["accuracy"] - metrics["shuffled"]["accuracy_max"]
                >= config.minimum_accuracy_gap
            ),
        }
        nll_diagnostics = {
            "nll_beats_zero": normal["target_nll"] < metrics["zero"]["target_nll"],
            "nll_beats_last": normal["target_nll"] < metrics["last"]["target_nll"],
            "nll_beats_shuffle": (
                normal["target_nll"] < metrics["shuffled"]["target_nll_min"]
            ),
        }
        passed = all(checks.values())
        report = {
            "run_id": run_id,
            "delta_run_id": config.delta_run_id,
            "evaluation_split": config.evaluation_split,
            "world_size": context.world_size,
            "training_seconds": time.perf_counter() - started,
            "metrics": metrics,
            "checks": checks,
            "nll_diagnostics": nll_diagnostics,
            "passed": passed,
            "status": "signal" if passed else "inconclusive",
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        if context.is_primary:
            _atomic_json(run_dir / "summary.json", report)
            _atomic_json(config.output_root / "latest_summary.json", report)
        if context.world_size > 1:
            torch.distributed.barrier()
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train semantic-token-only Qwen caption bridge")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ssv2_semantic_caption_a6000.yaml"),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()
    report = run(args.config, args.run_id)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
