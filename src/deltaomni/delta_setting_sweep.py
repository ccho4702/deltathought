from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from torch.nn import functional as F

from deltaomni.model import ModalityDeltaCodec, PairDeltaEncoder
from deltaomni.nextqa_reconstruction_pilot import load_config as load_nextqa_config
from deltaomni.ssv2_pilot import (
    _evaluate_reconstruction,
    _forward_reconstruction,
    _load_embeddings,
    load_pilot_config,
)
from deltaomni.ssv2_semantic_pilot import SemanticHead, _forward, _semantic_metrics
from deltaomni.train_sanity import _atomic_json, _set_seed


@dataclass(frozen=True)
class SweepConfig:
    seeds: tuple[int, ...]
    device: str
    ssv2_config: Path
    nextqa_config: Path
    delta_tokens: tuple[int, ...]
    semantic_weights: tuple[float, ...]
    batch_size: int
    reconstruction_steps: int
    joint_steps: int
    reconstruction_learning_rate: float
    joint_learning_rate: float
    reconstruction_weight: float
    selection_mse_tolerance: float
    output_root: Path


def load_config(path: Path) -> SweepConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    seed_values = raw["seeds"] if "seeds" in raw else [raw["seed"]]
    seeds = tuple(int(value) for value in seed_values)
    if not seeds:
        raise ValueError("At least one sweep seed is required")
    tolerance = float(raw.get("selection_mse_tolerance", 0.10))
    if tolerance < 0:
        raise ValueError("selection_mse_tolerance must be non-negative")
    return SweepConfig(
        seeds=seeds,
        device=str(raw["device"]),
        ssv2_config=resolve(raw["ssv2_config"]),
        nextqa_config=resolve(raw["nextqa_config"]),
        delta_tokens=tuple(int(value) for value in raw["delta_tokens"]),
        semantic_weights=tuple(float(value) for value in raw["semantic_weights"]),
        batch_size=int(raw["batch_size"]),
        reconstruction_steps=int(raw["reconstruction_steps"]),
        joint_steps=int(raw["joint_steps"]),
        reconstruction_learning_rate=float(raw["reconstruction_learning_rate"]),
        joint_learning_rate=float(raw["joint_learning_rate"]),
        reconstruction_weight=float(raw["reconstruction_weight"]),
        selection_mse_tolerance=tolerance,
        output_root=resolve(raw["output_root"]),
    )


def _codec_parameters(codec: ModalityDeltaCodec) -> list[torch.nn.Parameter]:
    return [
        *codec.delta_encoder.parameters(),
        *codec.accumulator.parameters(),
        *codec.reconstructor.parameters(),
    ]


def _batch_indices(count: int, batch_size: int, seed: int, step: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed * 1_000_003 + step)
    return torch.randint(0, count, (batch_size,), generator=generator)


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _train_candidate(
    config: SweepConfig,
    token_count: int,
    semantic_weight: float,
    seed: int,
    train_full: torch.Tensor,
    train_labels: torch.Tensor,
    validation_full: torch.Tensor,
    validation_labels: torch.Tensor,
    classes: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    device = train_full.device
    ssv2 = load_pilot_config(config.ssv2_config)
    model_config = replace(ssv2.model, delta_tokens=token_count)
    _set_seed(seed + token_count * 101 + int(semantic_weight * 10))
    codec = ModalityDeltaCodec(model_config).to(device)
    codec_parameters = _codec_parameters(codec)
    optimizer = torch.optim.AdamW(
        codec_parameters,
        lr=config.reconstruction_learning_rate,
    )
    for step in range(1, config.reconstruction_steps + 1):
        indices = _batch_indices(
            train_full.shape[0],
            config.batch_size,
            seed,
            step,
        ).to(device)
        codec.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = _forward_reconstruction(codec, train_full[indices])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(codec_parameters, 1.0)
        optimizer.step()

    head = SemanticHead(model_config.embedding_dim, classes).to(device)
    parameters = [*codec_parameters, *head.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=config.joint_learning_rate)
    for step in range(1, config.joint_steps + 1):
        indices = _batch_indices(
            train_full.shape[0],
            config.batch_size,
            seed + token_count,
            step,
        ).to(device)
        codec.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _, _, reconstruction_loss = _forward(codec, head, train_full[indices])
        semantic_loss = F.cross_entropy(logits, train_labels[indices])
        loss = config.reconstruction_weight * reconstruction_loss + semantic_weight * semantic_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

    reconstruction = _evaluate_reconstruction(codec, validation_full)
    semantic = _semantic_metrics(codec, head, validation_full, validation_labels)
    control_max = max(semantic["zero"], semantic["last"], semantic["shuffled"])
    semantic_margin = semantic["normal"] - control_max
    metrics = {
        "seed": seed,
        "delta_tokens": token_count,
        "semantic_weight": semantic_weight,
        "validation_mse": reconstruction["mse"],
        "anchor_mse": reconstruction["anchor_mse"],
        "last_delta_mse": reconstruction["last_delta_mse"],
        "shuffled_delta_mse": reconstruction["shuffled_delta_mse"],
        "raw_pooled_delta_mse": reconstruction["raw_pooled_delta_mse"],
        "semantic": semantic,
        "semantic_margin": semantic_margin,
        "qualified": (
            reconstruction["mse"] < reconstruction["anchor_mse"]
            and reconstruction["mse"] < reconstruction["raw_pooled_delta_mse"]
            and semantic_margin > 0
        ),
    }
    return metrics, _cpu_state(codec), _cpu_state(head)


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.pstdev(values)


def _aggregate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for candidate in candidates:
        key = (int(candidate["delta_tokens"]), float(candidate["semantic_weight"]))
        groups.setdefault(key, []).append(candidate)

    aggregated = []
    scalar_metrics = (
        "validation_mse",
        "anchor_mse",
        "last_delta_mse",
        "shuffled_delta_mse",
        "raw_pooled_delta_mse",
        "semantic_margin",
    )
    for (token_count, semantic_weight), runs in groups.items():
        summary: dict[str, Any] = {
            "delta_tokens": token_count,
            "semantic_weight": semantic_weight,
            "seeds": [int(run["seed"]) for run in runs],
            "runs": runs,
        }
        for name in scalar_metrics:
            mean, std = _mean_and_std([float(run[name]) for run in runs])
            summary[name] = mean
            summary[f"{name}_std"] = std
        summary["semantic"] = {}
        summary["semantic_std"] = {}
        for name in ("normal", "zero", "last", "shuffled"):
            mean, std = _mean_and_std([float(run["semantic"][name]) for run in runs])
            summary["semantic"][name] = mean
            summary["semantic_std"][name] = std
        summary["qualified_rate"] = statistics.fmean(float(run["qualified"]) for run in runs)
        summary["qualified"] = all(bool(run["qualified"]) for run in runs)
        aggregated.append(summary)
    return aggregated


def run(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    ssv2 = load_pilot_config(config.ssv2_config)
    manifest = json.loads((ssv2.cache_root / "manifest.json").read_text(encoding="utf-8"))
    train_full, train_labels = _load_embeddings(manifest, "train")
    validation_full, validation_labels = _load_embeddings(manifest, "validation")
    device = torch.device(config.device)
    train_full, train_labels = train_full.to(device), train_labels.to(device)
    validation_full = validation_full.to(device)
    validation_labels = validation_labels.to(device)
    candidates = []
    states: dict[tuple[int, float, int], tuple[dict[str, Tensor], dict[str, Tensor]]] = {}
    started = time.perf_counter()
    total = len(config.delta_tokens) * len(config.semantic_weights) * len(config.seeds)
    completed = 0
    for token_count in config.delta_tokens:
        for semantic_weight in config.semantic_weights:
            for seed in config.seeds:
                metrics, codec_state, head_state = _train_candidate(
                    config,
                    token_count,
                    semantic_weight,
                    seed,
                    train_full,
                    train_labels,
                    validation_full,
                    validation_labels,
                    len(ssv2.classes),
                )
                candidates.append(metrics)
                states[(token_count, semantic_weight, seed)] = (codec_state, head_state)
                completed += 1
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * (total - completed)
                print(
                    f"sweep={completed}/{total} seed={seed} tokens={token_count} "
                    f"semantic={semantic_weight:g} mse={metrics['validation_mse']:.4f} "
                    f"margin={metrics['semantic_margin']:.4f} eta={eta:.1f}s",
                    flush=True,
                )
                torch.cuda.empty_cache()

    aggregated = _aggregate_candidates(candidates)
    if not aggregated:
        raise RuntimeError("Delta sweep produced no candidates")
    fidelity_best = max(
        aggregated,
        key=lambda metrics: (
            bool(metrics["qualified"]),
            float(metrics["qualified_rate"]),
            -float(metrics["validation_mse"]),
            float(metrics["semantic_margin"]),
        ),
    )
    qualified = [candidate for candidate in aggregated if candidate["qualified"]]
    selection_pool = qualified or aggregated
    fidelity_mse = min(float(candidate["validation_mse"]) for candidate in selection_pool)
    balanced_pool = [
        candidate
        for candidate in selection_pool
        if float(candidate["validation_mse"])
        <= fidelity_mse * (1.0 + config.selection_mse_tolerance)
    ]
    best_metric = max(
        balanced_pool,
        key=lambda metrics: (
            -int(metrics["delta_tokens"]),
            float(metrics["semantic_margin"]),
            -float(metrics["validation_mse"]),
        ),
    )
    representative = min(
        best_metric["runs"],
        key=lambda run: abs(float(run["validation_mse"]) - float(best_metric["validation_mse"])),
    )
    representative_seed = int(representative["seed"])
    best_codec_state, best_head_state = states[
        (
            int(best_metric["delta_tokens"]),
            float(best_metric["semantic_weight"]),
            representative_seed,
        )
    ]
    best_model_config = replace(ssv2.model, delta_tokens=int(best_metric["delta_tokens"]))
    best_codec = ModalityDeltaCodec(best_model_config).to(device)
    best_codec.load_state_dict(best_codec_state)
    nextqa = load_nextqa_config(config.nextqa_config)
    nextqa_manifest = json.loads((nextqa.cache_root / "manifest.json").read_text(encoding="utf-8"))
    nextqa_embeddings = []
    for record in nextqa_manifest["records"]:
        payload = torch.load(record["cache_path"], map_location="cpu", weights_only=False)
        nextqa_embeddings.append(payload["embeddings"].float())
    nextqa_full = torch.stack(nextqa_embeddings).to(device)
    nextqa_reconstruction = _evaluate_reconstruction(best_codec, nextqa_full)
    cross_domain = {
        "mse": nextqa_reconstruction["mse"],
        "anchor_mse": nextqa_reconstruction["anchor_mse"],
        "last_delta_mse": nextqa_reconstruction["last_delta_mse"],
        "shuffled_delta_mse": nextqa_reconstruction["shuffled_delta_mse"],
        "raw_pooled_delta_mse": nextqa_reconstruction["raw_pooled_delta_mse"],
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"delta-setting-sweep-{timestamp}-{uuid.uuid4().hex[:8]}"
    run_dir = config.output_root / run_id
    report = {
        "run_id": run_id,
        "delta_algorithm": PairDeltaEncoder.ALGORITHM_VERSION,
        "selection_split": "ssv2_validation",
        "seeds": list(config.seeds),
        "candidate_runs": candidates,
        "candidates": aggregated,
        "selected": best_metric,
        "selected_checkpoint_seed": representative_seed,
        "fidelity_best": fidelity_best,
        "selection_rule": (
            "all seeds qualified; smallest token count within "
            f"{config.selection_mse_tolerance:.1%} of best validation MSE; "
            "then semantic margin and validation MSE"
        ),
        "nextqa_diagnostic": cross_domain,
        "passed": bool(best_metric["qualified"]),
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(run_dir / "summary.json", report)
    _atomic_torch_save(
        run_dir / "best_checkpoint.pt",
        {
            "codec": best_codec_state,
            "semantic_head": best_head_state,
            "selected": best_metric,
            "delta_algorithm": PairDeltaEncoder.ALGORITHM_VERSION,
        },
    )
    _atomic_json(config.output_root / "latest_summary.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep real delta token and semantic settings")
    parser.add_argument("--config", type=Path, default=Path("configs/delta_setting_sweep.yaml"))
    args = parser.parse_args()
    report = run(args.config)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
