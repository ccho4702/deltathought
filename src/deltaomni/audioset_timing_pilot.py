from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml
from torch import Tensor
from torch.nn import functional as F

from deltaomni.backbones import ClapEmbeddingBackend, load_backbone_config
from deltaomni.config import ModelConfig
from deltaomni.data.audioset_strong import StrongEvent, inspect_tsv
from deltaomni.model import ModalityDeltaCodec
from deltaomni.provenance import audit as audit_provenance
from deltaomni.train_sanity import _atomic_json, _set_seed


@dataclass(frozen=True)
class TimingTrainingConfig:
    batch_size: int
    learning_rate: float
    max_steps: int
    checkpoint_interval_steps: int
    max_grad_norm: float


@dataclass(frozen=True)
class TimingPilotConfig:
    seed: int
    device: str
    cpu_threads: int
    sample_rate: int
    chunk_seconds: float
    chunks_per_clip: int
    train_clips: int
    validation_clips: int
    embedding_batch_size: int
    train_annotations: Path
    validation_annotations: Path
    train_audio_roots: tuple[Path, ...]
    validation_audio_roots: tuple[Path, ...]
    cache_root: Path
    output_root: Path
    model: ModelConfig
    training: TimingTrainingConfig


def load_timing_config(path: Path) -> TimingPilotConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    model = raw["model"]
    training = raw["training"]
    return TimingPilotConfig(
        seed=int(raw["seed"]),
        device=str(raw["device"]),
        cpu_threads=int(raw["cpu_threads"]),
        sample_rate=int(raw["sample_rate"]),
        chunk_seconds=float(raw["chunk_seconds"]),
        chunks_per_clip=int(raw["chunks_per_clip"]),
        train_clips=int(raw["train_clips"]),
        validation_clips=int(raw["validation_clips"]),
        embedding_batch_size=int(raw["embedding_batch_size"]),
        train_annotations=resolve(raw["train_annotations"]),
        validation_annotations=resolve(raw["validation_annotations"]),
        train_audio_roots=tuple(resolve(value) for value in raw["train_audio_roots"]),
        validation_audio_roots=tuple(
            resolve(value) for value in raw["validation_audio_roots"]
        ),
        cache_root=resolve(raw["cache_root"]),
        output_root=resolve(raw["output_root"]),
        model=ModelConfig(
            embedding_dim=int(model["embedding_dim"]),
            hidden_dim=int(model["hidden_dim"]),
            embedding_tokens=int(model["embedding_tokens"]),
            delta_tokens=int(model["delta_tokens"]),
            num_heads=int(model["num_heads"]),
            caption_vocab_size=16,
            max_caption_length=8,
        ),
        training=TimingTrainingConfig(
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            max_steps=int(training["max_steps"]),
            checkpoint_interval_steps=int(training["checkpoint_interval_steps"]),
            max_grad_norm=float(training["max_grad_norm"]),
        ),
    )


def _audio_filename(clip_id: str) -> str:
    _, start_text = clip_id.rsplit("_", 1)
    return f"{clip_id}_{int(start_text) + 10_000}.flac"


def _find_audio(clip_id: str, roots: tuple[Path, ...]) -> Path | None:
    filename = _audio_filename(clip_id)
    for root in roots:
        candidate = root / filename
        if candidate.is_file():
            return candidate
    return None


def _targets(events: tuple[StrongEvent, ...], chunks: int) -> Tensor:
    targets = torch.zeros(chunks, dtype=torch.bool)
    for event in events:
        if event.end_seconds <= 1.0:
            continue
        index = min(chunks - 1, max(1, math.ceil(event.end_seconds) - 1))
        targets[index] = True
    return targets


def select_clips(
    annotations: Path,
    roots: tuple[Path, ...],
    count: int,
    seed: int,
    chunks: int,
) -> list[dict[str, Any]]:
    inspected = inspect_tsv(annotations)
    candidates = []
    for clip_id, events in inspected.events.items():
        targets = _targets(events, chunks)
        internal = any(1.0 < event.end_seconds < 9.5 for event in events)
        if not internal or not targets.any() or len(events) > 12:
            continue
        candidates.append((clip_id, events, targets))
    candidates.sort(
        key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest()
    )
    selected = []
    for clip_id, events, targets in candidates:
        media_path = _find_audio(clip_id, roots)
        if media_path is None:
            continue
        selected.append(
            {
                "clip_id": clip_id,
                "events": events,
                "targets": targets,
                "media_path": media_path,
            }
        )
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Found {len(selected)}/{count} AudioSet clips with media")
    return selected


def _read_chunks(path: Path, config: TimingPilotConfig) -> list[np.ndarray]:
    waveform, source_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = waveform.mean(axis=1)
    if source_rate != config.sample_rate:
        mono = librosa.resample(mono, orig_sr=source_rate, target_sr=config.sample_rate)
    total_samples = int(config.sample_rate * config.chunk_seconds * config.chunks_per_clip)
    mono = np.pad(mono, (0, max(0, total_samples - len(mono))))[:total_samples]
    chunk_samples = int(config.sample_rate * config.chunk_seconds)
    return [
        mono[index * chunk_samples : (index + 1) * chunk_samples]
        for index in range(config.chunks_per_clip)
    ]


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def prepare(
    config: TimingPilotConfig,
    backbone_config_path: Path,
    provenance_path: Path,
) -> dict[str, Any]:
    selected = {
        "train": select_clips(
            config.train_annotations,
            config.train_audio_roots,
            config.train_clips,
            config.seed,
            config.chunks_per_clip,
        ),
        "validation": select_clips(
            config.validation_annotations,
            config.validation_audio_roots,
            config.validation_clips,
            config.seed + 1,
            config.chunks_per_clip,
        ),
    }
    pending = []
    for split, records in selected.items():
        for record in records:
            cache_path = config.cache_root / split / f"{record['clip_id']}.pt"
            if not cache_path.is_file():
                pending.append((split, record, _read_chunks(record["media_path"], config)))
    if pending:
        backbone_config = load_backbone_config(backbone_config_path)
        backend = ClapEmbeddingBackend(
            backbone_config.audio,
            backbone_config.cache_dir,
            torch.device(config.device),
            audit_provenance(provenance_path),
        )
        chunks = [chunk for _, _, record_chunks in pending for chunk in record_chunks]
        encoded = []
        for start in range(0, len(chunks), config.embedding_batch_size):
            batch = chunks[start : start + config.embedding_batch_size]
            encoded.append(backend.encode(batch).cpu())
            done = min(start + len(batch), len(chunks))
            print(f"encode_audio_chunks={done}/{len(chunks)}", flush=True)
        embeddings = torch.cat(encoded)
        offset = 0
        for split, record, record_chunks in pending:
            count = len(record_chunks)
            clip_embeddings = embeddings[offset : offset + count]
            offset += count
            _atomic_torch_save(
                config.cache_root / split / f"{record['clip_id']}.pt",
                {
                    "schema": "deltaomni.audioset_timing_embedding.v1",
                    "clip_id": record["clip_id"],
                    "media_path": str(record["media_path"]),
                    "targets": record["targets"],
                    "embeddings": clip_embeddings.to(torch.float16),
                },
            )
    manifest = {
        "schema": "deltaomni.audioset_timing_manifest.v1",
        "splits": {
            split: [
                {
                    "clip_id": record["clip_id"],
                    "cache_path": str(config.cache_root / split / f"{record['clip_id']}.pt"),
                }
                for record in records
            ]
            for split, records in selected.items()
        },
    }
    _atomic_json(config.cache_root / "manifest.json", manifest)
    return manifest


def _load_split(manifest: dict[str, Any], split: str) -> tuple[Tensor, Tensor]:
    embeddings = []
    targets = []
    for record in manifest["splits"][split]:
        payload = torch.load(record["cache_path"], map_location="cpu", weights_only=False)
        embeddings.append(payload["embeddings"].float())
        targets.append(payload["targets"])
    return torch.stack(embeddings), torch.stack(targets)


def _policy_parameters(codec: ModalityDeltaCodec) -> list[torch.nn.Parameter]:
    return [
        *codec.delta_encoder.parameters(),
        *codec.accumulator.parameters(),
        *codec.policy.parameters(),
    ]


def _teacher_loss(
    codec: ModalityDeltaCodec,
    full: Tensor,
    targets: Tensor,
    pos_weight: Tensor,
) -> Tensor:
    anchor = full[:, 0]
    previous = anchor
    slots = torch.zeros(full.shape[0], 1, full.shape[-1], device=full.device)
    load = torch.zeros(full.shape[0], device=full.device)
    losses = []
    for time_index in range(1, full.shape[1]):
        current = full[:, time_index]
        delta = codec.delta_encoder(previous, current)
        slots = codec.accumulator(slots, delta)
        load = load + codec.policy.novelty_score(delta)
        trigger_logits, _ = codec.policy(slots, load)
        target = targets[:, time_index].float()
        losses.append(
            F.binary_cross_entropy_with_logits(
                trigger_logits,
                target,
                pos_weight=pos_weight,
            )
        )
        reset = targets[:, time_index, None, None]
        anchor = torch.where(reset, current, anchor)
        slots = torch.where(reset, torch.zeros_like(slots), slots)
        load = torch.where(target.bool(), torch.zeros_like(load), load)
        previous = current
    return torch.stack(losses).mean()


@torch.no_grad()
def _learned_probabilities(codec: ModalityDeltaCodec, full: Tensor) -> Tensor:
    codec.eval()
    anchor = full[:, 0]
    previous = anchor
    slots = torch.zeros(full.shape[0], 1, full.shape[-1], device=full.device)
    load = torch.zeros(full.shape[0], device=full.device)
    probabilities = [torch.zeros(full.shape[0], device=full.device)]
    for time_index in range(1, full.shape[1]):
        current = full[:, time_index]
        delta = codec.delta_encoder(previous, current)
        slots = codec.accumulator(slots, delta)
        load = load + codec.policy.novelty_score(delta)
        logits, _ = codec.policy(slots, load)
        probability = torch.sigmoid(logits)
        probabilities.append(probability)
        predicted_reset = probability.ge(0.5)
        reset = predicted_reset[:, None, None]
        anchor = torch.where(reset, current, anchor)
        slots = torch.where(reset, torch.zeros_like(slots), slots)
        load = torch.where(predicted_reset, torch.zeros_like(load), load)
        previous = current
    return torch.stack(probabilities, dim=1)


def _binary_metrics(targets: Tensor, predictions: Tensor, tolerance: int = 0) -> dict[str, float]:
    target_rows = targets.bool().cpu()
    prediction_rows = predictions.bool().cpu()
    matched_predictions = 0
    matched_targets = 0
    total_predictions = int(prediction_rows.sum())
    total_targets = int(target_rows.sum())
    for target, prediction in zip(target_rows, prediction_rows, strict=True):
        target_indices = target.nonzero().flatten().tolist()
        prediction_indices = prediction.nonzero().flatten().tolist()
        matched_predictions += sum(
            any(abs(index - target_index) <= tolerance for target_index in target_indices)
            for index in prediction_indices
        )
        matched_targets += sum(
            any(
                abs(index - prediction_index) <= tolerance
                for prediction_index in prediction_indices
            )
            for index in target_indices
        )
    precision = matched_predictions / max(total_predictions, 1)
    recall = matched_targets / max(total_targets, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def _raw_change_scores(full: Tensor) -> Tensor:
    previous = F.normalize(full[:, :-1, 0], dim=-1)
    current = F.normalize(full[:, 1:, 0], dim=-1)
    scores = 1 - (previous * current).sum(dim=-1)
    return torch.cat((torch.zeros(full.shape[0], 1, device=full.device), scores), dim=1)


def _best_threshold(scores: Tensor, targets: Tensor) -> float:
    quantiles = torch.linspace(0.05, 0.95, 37, device=scores.device)
    candidates = torch.quantile(scores.flatten(), quantiles)
    best = max(
        candidates.tolist(),
        key=lambda value: _binary_metrics(targets, scores >= value)["f1"],
    )
    return float(best)


def train_and_evaluate(config: TimingPilotConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    device = torch.device(config.device)
    train_full, train_targets = _load_split(manifest, "train")
    validation_full, validation_targets = _load_split(manifest, "validation")
    train_full, train_targets = train_full.to(device), train_targets.to(device)
    validation_full = validation_full.to(device)
    validation_targets = validation_targets.to(device)
    _set_seed(config.seed)
    codec = ModalityDeltaCodec(config.model).to(device)
    optimizer = torch.optim.AdamW(
        _policy_parameters(codec),
        lr=config.training.learning_rate,
    )
    positives = train_targets[:, 1:].sum().float()
    negatives = train_targets[:, 1:].numel() - positives
    pos_weight = (negatives / positives.clamp_min(1)).to(device)
    initial_probabilities = _learned_probabilities(codec, validation_full)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"audioset-timing-{timestamp}-{uuid.uuid4().hex[:8]}"
    run_dir = config.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(run_dir / "resolved_config.json", asdict(config))
    started = time.perf_counter()
    for step in range(1, config.training.max_steps + 1):
        generator = torch.Generator().manual_seed(config.seed * 1_000_003 + step)
        indices = torch.randint(
            0,
            train_full.shape[0],
            (config.training.batch_size,),
            generator=generator,
        ).to(device)
        codec.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _teacher_loss(codec, train_full[indices], train_targets[indices], pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(_policy_parameters(codec), config.training.max_grad_norm)
        optimizer.step()
        if step % 10 == 0 or step == config.training.max_steps:
            elapsed = time.perf_counter() - started
            eta = elapsed / step * (config.training.max_steps - step)
            print(
                f"timing_step={step}/{config.training.max_steps} loss={float(loss):.5f} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
        if step % config.training.checkpoint_interval_steps == 0:
            _atomic_torch_save(
                run_dir / "checkpoints" / f"step-{step:06d}.pt",
                {"step": step, "model": codec.state_dict(), "optimizer": optimizer.state_dict()},
            )

    learned_train_probabilities = _learned_probabilities(codec, train_full)
    learned_probabilities = _learned_probabilities(codec, validation_full)
    learned_threshold = _best_threshold(learned_train_probabilities, train_targets)
    raw_train_scores = _raw_change_scores(train_full)
    raw_validation_scores = _raw_change_scores(validation_full)
    raw_threshold = _best_threshold(raw_train_scores, train_targets)
    fixed_final = torch.zeros_like(validation_targets)
    fixed_final[:, -1] = True
    metrics = {
        "initial_exact": _binary_metrics(validation_targets, initial_probabilities >= 0.5),
        "learned_exact": _binary_metrics(validation_targets, learned_probabilities >= 0.5),
        "learned_plus_one": _binary_metrics(
            validation_targets, learned_probabilities >= 0.5, tolerance=1
        ),
        "learned_calibrated_exact": _binary_metrics(
            validation_targets, learned_probabilities >= learned_threshold
        ),
        "learned_calibrated_plus_one": _binary_metrics(
            validation_targets,
            learned_probabilities >= learned_threshold,
            tolerance=1,
        ),
        "raw_change_exact": _binary_metrics(
            validation_targets, raw_validation_scores >= raw_threshold
        ),
        "raw_change_plus_one": _binary_metrics(
            validation_targets, raw_validation_scores >= raw_threshold, tolerance=1
        ),
        "fixed_final_exact": _binary_metrics(validation_targets, fixed_final),
        "fixed_final_plus_one": _binary_metrics(validation_targets, fixed_final, tolerance=1),
        "raw_change_threshold": raw_threshold,
        "learned_threshold": learned_threshold,
        "train_positive_rate": float(train_targets[:, 1:].float().mean()),
        "validation_positive_rate": float(validation_targets[:, 1:].float().mean()),
    }
    checks = {
        "learned_f1_positive": metrics["learned_calibrated_exact"]["f1"] > 0,
        "learned_beats_fixed": (
            metrics["learned_calibrated_exact"]["f1"]
            > metrics["fixed_final_exact"]["f1"]
        ),
        "learned_beats_raw_exact": (
            metrics["learned_calibrated_exact"]["f1"]
            > metrics["raw_change_exact"]["f1"]
        ),
        "learned_plus_one_signal": (
            metrics["learned_calibrated_plus_one"]["f1"]
            > metrics["fixed_final_plus_one"]["f1"]
        ),
    }
    passed = all(checks.values())
    report = {
        "run_id": run_id,
        "status": "signal" if passed else "inconclusive",
        "metrics": metrics,
        "checks": checks,
        "passed": passed,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(run_dir / "summary.json", report)
    _atomic_json(config.output_root / "latest_summary.json", report)
    return report


def run(config_path: Path, backbone_config_path: Path, provenance_path: Path) -> dict[str, Any]:
    config = load_timing_config(config_path)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")
    torch.set_num_threads(config.cpu_threads)
    _set_seed(config.seed)
    manifest = prepare(config, backbone_config_path, provenance_path)
    torch.cuda.empty_cache()
    return train_and_evaluate(config, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AudioSet Strong learned timing pilot")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/audioset_timing_pilot.yaml")
    )
    parser.add_argument("--backbones", type=Path, default=Path("configs/backbones.yaml"))
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    args = parser.parse_args()
    report = run(args.config, args.backbones, args.provenance)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
