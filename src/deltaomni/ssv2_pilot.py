from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import av
import torch
import yaml
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F

from deltaomni.backbones import DinoV2EmbeddingBackend, load_backbone_config
from deltaomni.config import ModelConfig
from deltaomni.model import (
    ModalityDeltaCodec,
    PairDeltaEncoder,
    expand_embedding_delta,
    pool_embedding_delta,
)
from deltaomni.provenance import audit as audit_provenance
from deltaomni.train_sanity import _atomic_json, _set_seed


@dataclass(frozen=True)
class PilotTrainingConfig:
    batch_size: int
    learning_rate: float
    max_steps: int
    checkpoint_interval_steps: int
    max_grad_norm: float


@dataclass(frozen=True)
class ProbeConfig:
    steps: int
    learning_rate: float
    hidden_dim: int


@dataclass(frozen=True)
class CaptionPilotConfig:
    prompt: str
    targets: tuple[str, ...]
    batch_size: int
    learning_rate: float
    max_steps: int
    checkpoint_interval_steps: int
    ranking_weight: float
    train_delta_encoder: bool
    delta_learning_rate: float
    reconstruction_weight: float


@dataclass(frozen=True)
class PilotConfig:
    seed: int
    device: str
    cpu_threads: int
    media_dir: Path
    access_mode: str
    train_annotations: Path
    validation_annotations: Path
    classes: tuple[str, ...]
    train_per_class: int
    validation_per_class: int
    test_per_class: int
    frames_per_clip: int
    minimum_decoded_frames: int
    embedding_batch_size: int
    cache_root: Path
    output_root: Path
    model: ModelConfig
    training: PilotTrainingConfig
    probe: ProbeConfig
    caption: CaptionPilotConfig


def load_pilot_config(path: Path) -> PilotConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    model = raw["model"]
    training = raw["training"]
    probe = raw["probe"]
    caption = raw["caption"]
    return PilotConfig(
        seed=int(raw["seed"]),
        device=str(raw["device"]),
        cpu_threads=int(raw["cpu_threads"]),
        media_dir=resolve(raw["media_dir"]),
        access_mode=str(raw["access_mode"]),
        train_annotations=resolve(raw["train_annotations"]),
        validation_annotations=resolve(raw["validation_annotations"]),
        classes=tuple(str(value) for value in raw["classes"]),
        train_per_class=int(raw["train_per_class"]),
        validation_per_class=int(raw["validation_per_class"]),
        test_per_class=int(raw.get("test_per_class", 0)),
        frames_per_clip=int(raw["frames_per_clip"]),
        minimum_decoded_frames=int(raw.get("minimum_decoded_frames", 0)),
        embedding_batch_size=int(raw["embedding_batch_size"]),
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
        training=PilotTrainingConfig(
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            max_steps=int(training["max_steps"]),
            checkpoint_interval_steps=int(training["checkpoint_interval_steps"]),
            max_grad_norm=float(training["max_grad_norm"]),
        ),
        probe=ProbeConfig(
            steps=int(probe["steps"]),
            learning_rate=float(probe["learning_rate"]),
            hidden_dim=int(probe["hidden_dim"]),
        ),
        caption=CaptionPilotConfig(
            prompt=str(caption["prompt"]),
            targets=tuple(str(value) for value in caption["targets"]),
            batch_size=int(caption.get("batch_size", 8)),
            learning_rate=float(caption["learning_rate"]),
            max_steps=int(caption["max_steps"]),
            checkpoint_interval_steps=int(caption["checkpoint_interval_steps"]),
            ranking_weight=float(caption["ranking_weight"]),
            train_delta_encoder=bool(caption["train_delta_encoder"]),
            delta_learning_rate=float(caption["delta_learning_rate"]),
            reconstruction_weight=float(caption["reconstruction_weight"]),
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected annotation list: {path}")
    return value


def select_records(
    records: list[dict[str, Any]],
    classes: tuple[str, ...],
    count_per_class: int,
    seed: int,
    eligibility: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for label_index, template in enumerate(classes):
        candidates = [record for record in records if record.get("template") == template]
        candidates = [
            record for record in candidates if (Path(str(record["id"])).name == str(record["id"]))
        ]
        candidates.sort(
            key=lambda record: hashlib.sha256(f"{seed}:{record['id']}".encode()).hexdigest()
        )
        eligible_candidates = []
        for record in candidates:
            if eligibility is None or eligibility(record):
                eligible_candidates.append(record)
                if len(eligible_candidates) == count_per_class:
                    break
        if len(eligible_candidates) < count_per_class:
            raise ValueError(
                f"Only {len(eligible_candidates)} eligible records found for {template}"
            )
        for record in eligible_candidates:
            selected.append({**record, "class_index": label_index})
    return selected


def _decode_uniform_frames(path: Path, count: int) -> tuple[list[Image.Image], dict[str, Any]]:
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        decoded = [frame.to_image().convert("RGB") for frame in container.decode(stream)]
        duration = (
            float(container.duration / av.time_base)
            if container.duration is not None
            else float(decoded[-1].time or 0)
        )
        average_rate = float(stream.average_rate) if stream.average_rate is not None else None
    if len(decoded) < count:
        raise ValueError(f"Video has {len(decoded)} frames, requires {count}: {path}")
    indices = torch.linspace(0, len(decoded) - 1, count).round().long().tolist()
    images = [decoded[index] for index in indices]
    return images, {
        "decoded_frames": len(decoded),
        "selected_indices": indices,
        "duration_seconds": duration,
        "average_rate": average_rate,
        "width": images[0].width,
        "height": images[0].height,
    }


def prepare_embeddings(
    config: PilotConfig,
    backbone_config_path: Path,
    provenance_path: Path,
) -> dict[str, Any]:
    if config.access_mode != "read_only_existing_shared_copy":
        raise ValueError("SSV2 pilot must use the existing shared copy read-only")
    checked_media = 0

    def eligible(record: dict[str, Any]) -> bool:
        nonlocal checked_media
        if config.minimum_decoded_frames <= 0:
            return True
        media_path = config.media_dir / f"{record['id']}.webm"
        if not media_path.is_file():
            return False
        with av.open(str(media_path), mode="r") as container:
            decoded_frames = sum(1 for _ in container.decode(container.streams.video[0]))
        checked_media += 1
        if checked_media % 100 == 0:
            print(f"eligibility_checked={checked_media}", flush=True)
        return decoded_frames >= config.minimum_decoded_frames

    eligibility = eligible if config.minimum_decoded_frames > 0 else None
    selected_by_split = {
        "train": select_records(
            _records(config.train_annotations),
            config.classes,
            config.train_per_class,
            config.seed,
            eligibility,
        )
    }
    held_out = select_records(
        _records(config.validation_annotations),
        config.classes,
        config.validation_per_class + config.test_per_class,
        config.seed + 1,
        eligibility,
    )
    selected_by_split["validation"] = []
    if config.test_per_class:
        selected_by_split["test"] = []
    for class_index in range(len(config.classes)):
        class_records = [
            record for record in held_out if int(record["class_index"]) == class_index
        ]
        selected_by_split["validation"].extend(class_records[: config.validation_per_class])
        if config.test_per_class:
            selected_by_split["test"].extend(class_records[config.validation_per_class :])
    source_ids = {
        split: {str(record["id"]) for record in records}
        for split, records in selected_by_split.items()
    }
    split_names = tuple(source_ids)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            if source_ids[left] & source_ids[right]:
                raise ValueError(f"SSV2 pilot {left}/{right} IDs overlap")

    pending: list[tuple[str, dict[str, Any], Path]] = []
    started = time.perf_counter()
    total = sum(len(records) for records in selected_by_split.values())
    completed = 0
    for split, records in selected_by_split.items():
        for record in records:
            source_id = str(record["id"])
            cache_path = config.cache_root / split / f"{source_id}.pt"
            if cache_path.is_file():
                completed += 1
                continue
            media_path = config.media_dir / f"{source_id}.webm"
            if not media_path.is_file():
                raise FileNotFoundError(media_path)
            pending.append((split, record, media_path))
            completed += 1
            if completed % 100 == 0 or completed == total:
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * (total - completed)
                print(
                    f"discover={completed}/{total} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )

    if pending:
        backbone_config = load_backbone_config(backbone_config_path)
        if backbone_config.video.model_id != "facebook/dinov2-base":
            raise ValueError("SSV2 pilot is pinned to DINOv2-base")
        backend = DinoV2EmbeddingBackend(
            backbone_config.video,
            backbone_config.cache_dir,
            torch.device(config.device),
            audit_provenance(provenance_path),
        )
        clips_per_batch = max(1, config.embedding_batch_size // config.frames_per_clip)
        encoded_clips = 0
        for start in range(0, len(pending), clips_per_batch):
            clip_batch = pending[start : start + clips_per_batch]
            decoded = [
                (*item, *_decode_uniform_frames(item[2], config.frames_per_clip))
                for item in clip_batch
            ]
            flat_images = [image for *_, images, _ in decoded for image in images]
            encoded = backend.encode(flat_images).cpu()
            offset = 0
            for split, record, media_path, images, media_info in decoded:
                count = len(images)
                clip_embeddings = encoded[offset : offset + count]
                offset += count
                if clip_embeddings.shape[1:] != (
                    config.model.embedding_tokens,
                    config.model.embedding_dim,
                ):
                    raise ValueError(f"Unexpected DINO embedding shape: {clip_embeddings.shape}")
                cache_path = config.cache_root / split / f"{record['id']}.pt"
                _atomic_torch_save(
                    cache_path,
                    {
                        "schema": "deltaomni.ssv2_embedding.v1",
                        "source_id": str(record["id"]),
                        "split": split,
                        "class_index": int(record["class_index"]),
                        "template": record["template"],
                        "label": record["label"],
                        "media_path": str(media_path),
                        "media_sha256": _sha256(media_path),
                        "media": media_info,
                        "model_id": backbone_config.video.model_id,
                        "model_revision": backbone_config.video.revision,
                        "embeddings": clip_embeddings.to(torch.float16),
                    },
                )
                encoded_clips += 1
            progress_interval = clips_per_batch * 8
            if encoded_clips % progress_interval == 0 or encoded_clips == len(pending):
                elapsed = time.perf_counter() - started
                eta = elapsed / encoded_clips * (len(pending) - encoded_clips)
                print(
                    f"encode_clips={encoded_clips}/{len(pending)} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )

    manifest = {
        "schema": "deltaomni.ssv2_pilot_manifest.v1",
        "classes": list(config.classes),
        "splits": {
            split: [
                {
                    "source_id": str(record["id"]),
                    "class_index": int(record["class_index"]),
                    "cache_path": str(config.cache_root / split / f"{record['id']}.pt"),
                }
                for record in records
            ]
            for split, records in selected_by_split.items()
        },
    }
    _atomic_json(config.cache_root / "manifest.json", manifest)
    return manifest


def _load_embeddings(manifest: dict[str, Any], split: str) -> tuple[Tensor, Tensor]:
    embeddings = []
    labels = []
    for record in manifest["splits"][split]:
        payload = torch.load(record["cache_path"], map_location="cpu", weights_only=False)
        embeddings.append(payload["embeddings"].float())
        labels.append(int(record["class_index"]))
    return torch.stack(embeddings), torch.tensor(labels, dtype=torch.long)


def _codec_parameters(codec: ModalityDeltaCodec) -> list[nn.Parameter]:
    return [
        *codec.delta_encoder.parameters(),
        *codec.accumulator.parameters(),
        *codec.reconstructor.parameters(),
    ]


def _forward_reconstruction(codec: ModalityDeltaCodec, full: Tensor) -> tuple[Tensor, Tensor]:
    anchor = full[:, 0]
    previous = anchor
    slots = torch.zeros(
        full.shape[0],
        codec.delta_encoder.queries.shape[0],
        full.shape[-1],
        device=full.device,
    )
    reconstructions = [anchor]
    losses = []
    for time_index in range(1, full.shape[1]):
        current = full[:, time_index]
        delta = codec.delta_encoder(previous, current)
        slots = codec.accumulator(slots, delta)
        step_reconstructed = codec.reconstructor(previous, delta)
        section_reconstructed = codec.reconstructor(anchor, slots)
        losses.append(
            0.5
            * (
                F.smooth_l1_loss(step_reconstructed, current)
                + F.smooth_l1_loss(section_reconstructed, current)
            )
        )
        reconstructions.append(section_reconstructed)
        previous = current
    return torch.stack(reconstructions, dim=1), torch.stack(losses).mean()


@torch.no_grad()
def _evaluate_reconstruction(codec: ModalityDeltaCodec, full: Tensor) -> dict[str, Any]:
    codec.eval()
    anchor = full[:, 0]
    previous = anchor
    slots = torch.zeros(
        full.shape[0],
        codec.delta_encoder.queries.shape[0],
        full.shape[-1],
        device=full.device,
    )
    reconstructed_frames = [anchor]
    learned_errors = []
    anchor_errors = []
    last_errors = []
    shuffled_errors = []
    raw_pooled_errors = []
    final_slots = None
    for time_index in range(1, full.shape[1]):
        current = full[:, time_index]
        delta = codec.delta_encoder(previous, current)
        slots = codec.accumulator(slots, delta)
        reconstructed = codec.reconstructor(anchor, slots)
        last_reconstructed = codec.reconstructor(anchor, delta)
        shuffled = codec.reconstructor(anchor, slots.roll(1, dims=0))
        raw_difference = current - anchor
        pooled = pool_embedding_delta(
            raw_difference,
            codec.delta_encoder.queries.shape[0],
        )
        raw_reconstructed = anchor + expand_embedding_delta(pooled, current.shape[1])
        learned_errors.append((reconstructed - current).square().mean())
        anchor_errors.append((anchor - current).square().mean())
        last_errors.append((last_reconstructed - current).square().mean())
        shuffled_errors.append((shuffled - current).square().mean())
        raw_pooled_errors.append((raw_reconstructed - current).square().mean())
        reconstructed_frames.append(reconstructed)
        previous = current
        final_slots = slots
    return {
        "reconstructed": torch.stack(reconstructed_frames, dim=1),
        "final_slots": final_slots,
        "mse": float(torch.stack(learned_errors).mean()),
        "anchor_mse": float(torch.stack(anchor_errors).mean()),
        "last_delta_mse": float(torch.stack(last_errors).mean()),
        "shuffled_delta_mse": float(torch.stack(shuffled_errors).mean()),
        "raw_pooled_delta_mse": float(torch.stack(raw_pooled_errors).mean()),
        "per_step": {
            "learned_mse": [float(error) for error in learned_errors],
            "anchor_mse": [float(error) for error in anchor_errors],
            "last_delta_mse": [float(error) for error in last_errors],
            "shuffled_delta_mse": [float(error) for error in shuffled_errors],
            "raw_pooled_delta_mse": [float(error) for error in raw_pooled_errors],
        },
    }


def _retrieval_r1(predicted: Tensor, target: Tensor) -> float:
    predicted_flat = F.normalize(predicted[:, 1:].flatten(0, 1).flatten(1), dim=-1)
    target_flat = F.normalize(target[:, 1:].flatten(0, 1).flatten(1), dim=-1)
    similarities = predicted_flat @ target_flat.T
    expected = torch.arange(similarities.shape[0], device=similarities.device)
    return float(similarities.argmax(dim=1).eq(expected).float().mean())


class ActionProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


def _sequence_features(sequence: Tensor) -> Tensor:
    return sequence[:, :, 0].flatten(1)


def _fit_probe(
    train_features: Tensor,
    train_labels: Tensor,
    config: PilotConfig,
) -> ActionProbe:
    probe = ActionProbe(
        train_features.shape[-1],
        config.probe.hidden_dim,
        len(config.classes),
    ).to(train_features.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.probe.learning_rate)
    for _ in range(config.probe.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(train_features), train_labels)
        loss.backward()
        optimizer.step()
    return probe.eval()


@torch.no_grad()
def _accuracy(probe: ActionProbe, features: Tensor, labels: Tensor) -> float:
    return float(probe(features).argmax(dim=-1).eq(labels).float().mean())


def train_and_evaluate(config: PilotConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    device = torch.device(config.device)
    train_full, train_labels = _load_embeddings(manifest, "train")
    validation_full, validation_labels = _load_embeddings(manifest, "validation")
    train_full = train_full.to(device)
    train_labels = train_labels.to(device)
    validation_full = validation_full.to(device)
    validation_labels = validation_labels.to(device)
    codec = ModalityDeltaCodec(config.model).to(device)
    optimizer = torch.optim.AdamW(
        _codec_parameters(codec),
        lr=config.training.learning_rate,
    )
    initial = _evaluate_reconstruction(codec, validation_full)
    run_id = f"ssv2-pilot-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = config.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(run_dir / "resolved_config.json", asdict(config))
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
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
        _, loss = _forward_reconstruction(codec, train_full[indices])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(_codec_parameters(codec), config.training.max_grad_norm)
        optimizer.step()
        if step % 10 == 0 or step == config.training.max_steps:
            elapsed = time.perf_counter() - started
            eta = elapsed / step * (config.training.max_steps - step)
            print(
                f"train_step={step}/{config.training.max_steps} loss={float(loss):.6f} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
        if step % config.training.checkpoint_interval_steps == 0:
            _atomic_torch_save(
                run_dir / "checkpoints" / f"step-{step:06d}.pt",
                {
                    "step": step,
                    "delta_algorithm": PairDeltaEncoder.ALGORITHM_VERSION,
                    "model": codec.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "torch_rng_state": torch.random.get_rng_state(),
                },
            )

    train_evaluation = _evaluate_reconstruction(codec, train_full)
    validation = _evaluate_reconstruction(codec, validation_full)
    train_features = _sequence_features(train_full)
    validation_features = _sequence_features(validation_full)
    reconstructed_validation_features = _sequence_features(validation["reconstructed"])
    anchor_validation = validation_full[:, :1].expand_as(validation_full)
    probe = _fit_probe(train_features, train_labels, config)
    full_accuracy = _accuracy(probe, validation_features, validation_labels)
    reconstructed_accuracy = _accuracy(
        probe,
        reconstructed_validation_features,
        validation_labels,
    )
    anchor_accuracy = _accuracy(probe, _sequence_features(anchor_validation), validation_labels)
    delta_probe = _fit_probe(
        train_evaluation["final_slots"].flatten(1),
        train_labels,
        config,
    )
    delta_accuracy = _accuracy(
        delta_probe,
        validation["final_slots"].flatten(1),
        validation_labels,
    )
    metrics = {
        "initial_validation_mse": initial["mse"],
        "validation_mse": validation["mse"],
        "anchor_mse": validation["anchor_mse"],
        "last_delta_mse": validation["last_delta_mse"],
        "shuffled_delta_mse": validation["shuffled_delta_mse"],
        "raw_pooled_delta_mse": validation["raw_pooled_delta_mse"],
        "retrieval_r1": _retrieval_r1(validation["reconstructed"], validation_full),
        "anchor_retrieval_r1": _retrieval_r1(anchor_validation, validation_full),
        "full_action_accuracy": full_accuracy,
        "reconstructed_action_accuracy": reconstructed_accuracy,
        "anchor_action_accuracy": anchor_accuracy,
        "delta_state_action_accuracy": delta_accuracy,
        "chance_accuracy": 1 / len(config.classes),
        "per_step": validation["per_step"],
    }
    checks = {
        "reconstruction_loss_decreased": (
            metrics["validation_mse"] < metrics["initial_validation_mse"]
        ),
        "learned_beats_anchor": metrics["validation_mse"] < metrics["anchor_mse"],
        "learned_beats_last_delta": metrics["validation_mse"] < metrics["last_delta_mse"],
        "learned_beats_shuffled": metrics["validation_mse"] < metrics["shuffled_delta_mse"],
        "retrieval_improves": metrics["retrieval_r1"] > metrics["anchor_retrieval_r1"],
        "action_probe_above_chance": metrics["full_action_accuracy"] > metrics["chance_accuracy"],
        "action_accuracy_preserved": (
            metrics["reconstructed_action_accuracy"] >= metrics["full_action_accuracy"] - 0.10
        ),
    }
    report = {
        "run_id": run_id,
        "status": "signal" if all(checks.values()) else "inconclusive",
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(run_dir / "summary.json", report)
    _atomic_json(config.output_root / "latest_summary.json", report)
    return report


def run(config_path: Path, backbone_config_path: Path, provenance_path: Path) -> dict[str, Any]:
    config = load_pilot_config(config_path)
    if not torch.cuda.is_available() and config.device.startswith("cuda"):
        raise RuntimeError("CUDA is required by the configured SSV2 pilot")
    torch.set_num_threads(config.cpu_threads)
    _set_seed(config.seed)
    manifest = prepare_embeddings(config, backbone_config_path, provenance_path)
    if config.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return train_and_evaluate(config, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the small real-media SSV2 delta pilot")
    parser.add_argument("--config", type=Path, default=Path("configs/ssv2_pilot.yaml"))
    parser.add_argument("--backbones", type=Path, default=Path("configs/backbones.yaml"))
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    args = parser.parse_args()
    report = run(args.config, args.backbones, args.provenance)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
