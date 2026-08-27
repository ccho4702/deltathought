from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import av
import torch
import yaml
from PIL import Image, ImageOps

from deltaomni.data.schema import CanonicalEpisode, iter_jsonl
from deltaomni.distributed import distributed_context
from deltaomni.omni_backbones import (
    QwenOmniThinkerEmbeddingBackend,
    load_omni_backbone_config,
)
from deltaomni.provenance import audit as audit_provenance
from deltaomni.train_sanity import _atomic_json


@dataclass(frozen=True)
class CacheRuntime:
    device: str
    backend: str
    nccl_compatibility_mode: bool
    cpu_threads: int


@dataclass(frozen=True)
class OmniSSV2CacheConfig:
    seed: int
    canonical_manifest: Path
    omni_config: Path
    classes: tuple[str, ...]
    train_per_class: int
    validation_per_class: int
    test_per_class: int
    sample_fps: float
    block_seconds: float
    frame_width: int
    frame_height: int
    expected_tokens_per_block: int
    runtime: CacheRuntime
    cache_root: Path
    report_path: Path


def load_config(path: Path) -> OmniSSV2CacheConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    runtime = raw["runtime"]
    config = OmniSSV2CacheConfig(
        seed=int(raw["seed"]),
        canonical_manifest=resolve(raw["canonical_manifest"]),
        omni_config=resolve(raw["omni_config"]),
        classes=tuple(str(value) for value in raw["classes"]),
        train_per_class=int(raw["train_per_class"]),
        validation_per_class=int(raw["validation_per_class"]),
        test_per_class=int(raw["test_per_class"]),
        sample_fps=float(raw["sample_fps"]),
        block_seconds=float(raw["block_seconds"]),
        frame_width=int(raw["frame_width"]),
        frame_height=int(raw["frame_height"]),
        expected_tokens_per_block=int(raw["expected_tokens_per_block"]),
        runtime=CacheRuntime(
            device=str(runtime["device"]),
            backend=str(runtime["backend"]),
            nccl_compatibility_mode=bool(runtime["nccl_compatibility_mode"]),
            cpu_threads=int(runtime["cpu_threads"]),
        ),
        cache_root=resolve(raw["cache_root"]),
        report_path=resolve(raw["report_path"]),
    )
    if len(config.classes) < 2 or min(
        config.train_per_class,
        config.validation_per_class,
        config.test_per_class,
        config.frame_width,
        config.frame_height,
        config.expected_tokens_per_block,
        config.block_seconds,
    ) <= 0:
        raise ValueError("Invalid Omni SSV2 cache configuration")
    return config


def _hash(seed: int, source_id: str) -> str:
    return hashlib.sha256(f"{seed}:{source_id}".encode()).hexdigest()


def _canonical_split_path(config: OmniSSV2CacheConfig, split: str) -> Path:
    manifest = json.loads(config.canonical_manifest.read_text(encoding="utf-8"))
    return config.canonical_manifest.parent / manifest["splits"][split]["path"]


def _select(
    episodes: list[CanonicalEpisode],
    classes: tuple[str, ...],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected = []
    for label, template in enumerate(classes):
        candidates = [
            episode for episode in episodes if episode.metadata.get("template") == template
        ]
        candidates.sort(key=lambda episode: _hash(seed, episode.source_id))
        if len(candidates) < count:
            raise ValueError(f"Found {len(candidates)}/{count} episodes for {template}")
        selected.extend(
            {"episode": episode, "label": label} for episode in candidates[:count]
        )
    return selected


def select_splits(config: OmniSSV2CacheConfig) -> dict[str, list[dict[str, Any]]]:
    train = list(iter_jsonl(_canonical_split_path(config, "train")))
    official_validation = list(iter_jsonl(_canonical_split_path(config, "validation")))
    selected_train = _select(
        train,
        config.classes,
        config.train_per_class,
        config.seed,
    )
    held_out = _select(
        official_validation,
        config.classes,
        config.validation_per_class + config.test_per_class,
        config.seed + 1,
    )
    validation = []
    test = []
    for label in range(len(config.classes)):
        values = [record for record in held_out if record["label"] == label]
        validation.extend(values[: config.validation_per_class])
        test.extend(values[config.validation_per_class :])
    return {"train": selected_train, "validation": validation, "test": test}


def _decode_blocks(
    episode: CanonicalEpisode,
    config: OmniSSV2CacheConfig,
) -> list[list[Image.Image]]:
    if episode.media.video is None or episode.temporal_blocks is None:
        raise ValueError(f"Episode has no temporal video: {episode.episode_id}")
    decoded = []
    with av.open(str(episode.media.video.path), mode="r") as container:
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate) if stream.average_rate is not None else None
        for index, frame in enumerate(container.decode(stream)):
            timestamp = (
                float(frame.time)
                if frame.time is not None
                else index / source_fps if source_fps else 0.0
            )
            decoded.append((timestamp, frame.to_image().convert("RGB")))
    if not decoded:
        raise ValueError(f"No frames decoded: {episode.media.video.path}")
    blocks = []
    target_count = max(2, round(config.block_seconds * config.sample_fps))
    if target_count % 2:
        target_count += 1
    for block in episode.temporal_blocks:
        targets = [
            min(block.end_seconds - 1e-6, block.start_seconds + index / config.sample_fps)
            for index in range(target_count)
        ]
        frames = [min(decoded, key=lambda value: abs(value[0] - target))[1] for target in targets]
        blocks.append(
            [
                ImageOps.pad(
                    frame,
                    (config.frame_width, config.frame_height),
                    color=(0, 0, 0),
                )
                for frame in frames
            ]
        )
    return blocks


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def run(config_path: Path, provenance_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    torch.set_num_threads(config.runtime.cpu_threads)
    with distributed_context(
        config.runtime.device,
        config.runtime.backend,
        nccl_compatibility_mode=config.runtime.nccl_compatibility_mode,
    ) as context:
        splits = select_splits(config)
        flat = [
            {"split": split, **record}
            for split, records in splits.items()
            for record in records
        ]
        local = flat[context.rank :: context.world_size]
        backend = QwenOmniThinkerEmbeddingBackend(
            load_omni_backbone_config(config.omni_config),
            context.device,
            audit_provenance(provenance_path),
        )
        started = time.perf_counter()
        processed = 0
        for record in local:
            episode = record["episode"]
            cache_path = config.cache_root / record["split"] / f"{episode.source_id}.pt"
            if cache_path.is_file():
                continue
            blocks = _decode_blocks(episode, config)
            features = []
            metadata = []
            for block in blocks:
                encoded, block_metadata = backend.encode_video_chunks([block])
                if encoded[0].shape[0] != config.expected_tokens_per_block:
                    raise ValueError(f"Unexpected Omni tokens: {encoded[0].shape}")
                features.append(encoded[0].cpu().to(torch.float16))
                metadata.append(block_metadata[0])
            _atomic_torch_save(
                cache_path,
                {
                    "schema": "deltaomni.omni_ssv2_blocks.v1",
                    "source_id": episode.source_id,
                    "split": record["split"],
                    "label": record["label"],
                    "embeddings": torch.stack(features),
                    "blocks": [block.__dict__ for block in episode.temporal_blocks or ()],
                    "encoder_metadata": metadata,
                    "model_revision": backend.config.revision,
                },
            )
            processed += 1
            if context.is_primary and (processed % 25 == 0 or processed == len(local)):
                elapsed = time.perf_counter() - started
                eta = elapsed / processed * (len(local) - processed)
                print(
                    f"omni_cache={processed}/{len(local)} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
        if context.world_size > 1:
            torch.distributed.barrier()
        report = {}
        if context.is_primary:
            code_revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=config_path.resolve().parent.parent,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            manifest = {
                "schema": "deltaomni.omni_ssv2_cache_manifest.v1",
                "model_revision": backend.config.revision,
                "code_revision": code_revision,
                "classes": list(config.classes),
                "splits": {
                    split: [
                        {
                            "source_id": record["episode"].source_id,
                            "label": record["label"],
                            "cache_path": str(
                                config.cache_root / split / f"{record['episode'].source_id}.pt"
                            ),
                        }
                        for record in records
                    ]
                    for split, records in splits.items()
                },
            }
            for records in manifest["splits"].values():
                if not all(Path(record["cache_path"]).is_file() for record in records):
                    raise FileNotFoundError("Omni cache is incomplete")
            _atomic_json(config.cache_root / "manifest.json", manifest)
            report = {
                "schema": "deltaomni.omni_ssv2_cache_summary.v1",
                "model_revision": backend.config.revision,
                "splits": {split: len(records) for split, records in splits.items()},
                "elapsed_seconds": time.perf_counter() - started,
                "completed_at_utc": datetime.now(UTC).isoformat(),
            }
            _atomic_json(config.report_path, report)
        if context.world_size > 1:
            torch.distributed.barrier()
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache native Omni SSV2 block embeddings")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/omni_ssv2_s1_cache.yaml"),
    )
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    args = parser.parse_args()
    report = run(args.config, args.provenance)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
