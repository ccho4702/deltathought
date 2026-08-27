from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import av
import numpy as np
import soundfile as sf
import torch
import yaml
from PIL import Image, ImageOps
from transformers.utils import logging as transformers_logging

from deltaomni.data.schema import CanonicalEpisode, iter_jsonl
from deltaomni.distributed import distributed_context
from deltaomni.omni_backbones import (
    QwenOmniThinkerEmbeddingBackend,
    load_omni_backbone_config,
)
from deltaomni.provenance import audit as audit_provenance
from deltaomni.train_sanity import _atomic_json

CACHE_SCHEMA = "deltaomni.omni_vggsound_blocks.v1"
MANIFEST_SCHEMA = "deltaomni.omni_vggsound_cache_manifest.v1"


@dataclass(frozen=True)
class CacheRuntime:
    device: str
    backend: str
    nccl_compatibility_mode: bool
    cpu_threads: int


@dataclass(frozen=True)
class OmniVGGSoundCacheConfig:
    canonical_manifest: Path
    omni_config: Path
    sample_fps: float
    block_seconds: float
    frame_width: int
    frame_height: int
    expected_video_tokens: int
    expected_audio_tokens: int
    runtime: CacheRuntime
    cache_root: Path
    report_path: Path


def load_config(path: Path) -> OmniVGGSoundCacheConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    runtime = raw["runtime"]
    config = OmniVGGSoundCacheConfig(
        canonical_manifest=resolve(raw["canonical_manifest"]),
        omni_config=resolve(raw["omni_config"]),
        sample_fps=float(raw["sample_fps"]),
        block_seconds=float(raw["block_seconds"]),
        frame_width=int(raw["frame_width"]),
        frame_height=int(raw["frame_height"]),
        expected_video_tokens=int(raw["expected_video_tokens"]),
        expected_audio_tokens=int(raw["expected_audio_tokens"]),
        runtime=CacheRuntime(
            device=str(runtime["device"]),
            backend=str(runtime["backend"]),
            nccl_compatibility_mode=bool(runtime["nccl_compatibility_mode"]),
            cpu_threads=int(runtime["cpu_threads"]),
        ),
        cache_root=resolve(raw["cache_root"]),
        report_path=resolve(raw["report_path"]),
    )
    positive = (
        config.sample_fps,
        config.block_seconds,
        config.frame_width,
        config.frame_height,
        config.expected_video_tokens,
        config.expected_audio_tokens,
        config.runtime.cpu_threads,
    )
    if min(positive) <= 0:
        raise ValueError("VGGSound Omni cache controls must be positive")
    if round(config.sample_fps * config.block_seconds) < 2:
        raise ValueError("Video blocks require at least two sampled frames")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _split_path(manifest_path: Path, split: str) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "vggsound":
        raise ValueError("Omni VGGSound cache requires a canonical VGGSound manifest")
    return manifest_path.parent / manifest["splits"][split]["path"]


def load_episodes(config: OmniVGGSoundCacheConfig) -> dict[str, list[CanonicalEpisode]]:
    episodes = {
        split: list(iter_jsonl(_split_path(config.canonical_manifest, split)))
        for split in ("train", "validation", "test")
    }
    groups = {
        split: {episode.source_group_id for episode in values} for split, values in episodes.items()
    }
    if groups["train"] & groups["validation"] or groups["train"] & groups["test"]:
        raise ValueError("VGGSound source overlap reached the embedding cache")
    if groups["validation"] & groups["test"]:
        raise ValueError("VGGSound validation/test source overlap reached the embedding cache")
    return episodes


def _block_count(episode: CanonicalEpisode, block_seconds: float) -> int:
    aligned = float(episode.metadata.get("aligned_duration_seconds", 0.0))
    count = math.floor((aligned + 1e-6) / block_seconds)
    if count < 2:
        raise ValueError(f"Episode has fewer than two aligned blocks: {episode.episode_id}")
    return count


def _decode_video_blocks(
    episode: CanonicalEpisode,
    config: OmniVGGSoundCacheConfig,
) -> list[list[Image.Image]]:
    if episode.media.video is None:
        raise ValueError(f"Episode has no video: {episode.episode_id}")
    frames_per_block = round(config.sample_fps * config.block_seconds)
    block_count = _block_count(episode, config.block_seconds)
    targets = [index / config.sample_fps for index in range(block_count * frames_per_block)]
    selected: list[Image.Image] = []
    target_index = 0
    previous_frame = None
    previous_timestamp = 0.0
    with av.open(str(episode.media.video.path), mode="r") as container:
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate) if stream.average_rate is not None else None
        for index, frame in enumerate(container.decode(stream)):
            timestamp = (
                float(frame.time)
                if frame.time is not None
                else index / source_fps
                if source_fps
                else 0.0
            )
            if previous_frame is None:
                previous_frame = frame
                previous_timestamp = timestamp
                continue
            if timestamp < previous_timestamp:
                raise ValueError(f"Non-monotonic video timestamps: {episode.media.video.path}")
            while target_index < len(targets) and targets[target_index] <= timestamp:
                target = targets[target_index]
                nearest = (
                    previous_frame
                    if abs(previous_timestamp - target) <= abs(timestamp - target)
                    else frame
                )
                selected.append(nearest.to_image().convert("RGB"))
                target_index += 1
            if target_index == len(targets):
                break
            previous_frame = frame
            previous_timestamp = timestamp
    if previous_frame is None:
        raise ValueError(f"No frames decoded: {episode.media.video.path}")
    while target_index < len(targets):
        selected.append(previous_frame.to_image().convert("RGB"))
        target_index += 1
    result = []
    for block_index in range(block_count):
        start = block_index * frames_per_block
        result.append(
            [
                ImageOps.pad(
                    frame,
                    (config.frame_width, config.frame_height),
                    color=(0, 0, 0),
                )
                for frame in selected[start : start + frames_per_block]
            ]
        )
    return result


def _decode_audio_blocks(
    episode: CanonicalEpisode,
    config: OmniVGGSoundCacheConfig,
    sample_rate: int,
) -> list[np.ndarray]:
    if episode.media.audio is None:
        raise ValueError(f"Episode has no audio: {episode.episode_id}")
    waveform, source_rate = sf.read(
        episode.media.audio.path,
        always_2d=True,
        dtype="float32",
    )
    if source_rate != sample_rate:
        raise ValueError(
            f"Unexpected VGGSound sample rate {source_rate}: {episode.media.audio.path}"
        )
    mono = waveform.mean(axis=1)
    samples_per_block = round(config.block_seconds * sample_rate)
    count = _block_count(episode, config.block_seconds)
    required = count * samples_per_block
    if mono.shape[0] < required:
        raise ValueError(f"Audio shorter than aligned duration: {episode.media.audio.path}")
    return [
        np.ascontiguousarray(mono[start : start + samples_per_block])
        for start in range(0, required, samples_per_block)
    ]


def _cache_path(config: OmniVGGSoundCacheConfig, modality: str, split: str, source_id: str) -> Path:
    return config.cache_root / modality / split / f"{source_id}.pt"


def _valid_cache(
    path: Path,
    *,
    modality: str,
    source_id: str,
    blocks: int,
    tokens: int,
    width: int,
    model_revision: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (EOFError, OSError, RuntimeError):
        return False
    embeddings = payload.get("embeddings")
    return bool(
        payload.get("schema") == CACHE_SCHEMA
        and payload.get("modality") == modality
        and payload.get("source_id") == source_id
        and payload.get("model_revision") == model_revision
        and isinstance(embeddings, torch.Tensor)
        and embeddings.dtype == torch.float16
        and tuple(embeddings.shape) == (blocks, tokens, width)
        and torch.isfinite(embeddings).all()
    )


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _encode_video(
    backend: QwenOmniThinkerEmbeddingBackend,
    episode: CanonicalEpisode,
    split: str,
    config: OmniVGGSoundCacheConfig,
) -> None:
    features = []
    metadata = []
    for block in _decode_video_blocks(episode, config):
        encoded, block_metadata = backend.encode_video_chunks([block])
        if encoded[0].shape != (config.expected_video_tokens, backend.output_dim):
            raise ValueError(f"Unexpected Omni video shape: {tuple(encoded[0].shape)}")
        features.append(encoded[0].cpu().to(torch.float16))
        metadata.append(block_metadata[0])
    _atomic_torch_save(
        _cache_path(config, "video", split, episode.source_id),
        {
            "schema": CACHE_SCHEMA,
            "modality": "video",
            "source_id": episode.source_id,
            "source_group_id": episode.source_group_id,
            "split": split,
            "embeddings": torch.stack(features),
            "encoder_metadata": metadata,
            "model_revision": backend.config.revision,
            "media_sha256": episode.media.video.sha256 if episode.media.video else None,
        },
    )


def _encode_audio(
    backend: QwenOmniThinkerEmbeddingBackend,
    episode: CanonicalEpisode,
    split: str,
    config: OmniVGGSoundCacheConfig,
) -> None:
    features = []
    metadata = []
    for block in _decode_audio_blocks(episode, config, backend.config.sample_rate):
        encoded, block_metadata = backend.encode_audio_chunks([block])
        if encoded[0].shape != (config.expected_audio_tokens, backend.output_dim):
            raise ValueError(f"Unexpected Omni audio shape: {tuple(encoded[0].shape)}")
        features.append(encoded[0].cpu().to(torch.float16))
        metadata.append(block_metadata[0])
    _atomic_torch_save(
        _cache_path(config, "audio", split, episode.source_id),
        {
            "schema": CACHE_SCHEMA,
            "modality": "audio",
            "source_id": episode.source_id,
            "source_group_id": episode.source_group_id,
            "split": split,
            "embeddings": torch.stack(features),
            "encoder_metadata": metadata,
            "model_revision": backend.config.revision,
            "media_sha256": episode.media.audio.sha256 if episode.media.audio else None,
        },
    )


def _records_for_modality(
    config: OmniVGGSoundCacheConfig,
    episodes: dict[str, list[CanonicalEpisode]],
    modality: str,
) -> dict[str, list[dict[str, Any]]]:
    return {
        split: [
            {
                "source_id": episode.source_id,
                "source_group_id": episode.source_group_id,
                "blocks": _block_count(episode, config.block_seconds),
                "cache_path": str(_cache_path(config, modality, split, episode.source_id)),
            }
            for episode in values
        ]
        for split, values in episodes.items()
    }


def _write_manifests(
    config: OmniVGGSoundCacheConfig,
    episodes: dict[str, list[CanonicalEpisode]],
    *,
    model_revision: str,
    code_revision: str,
    width: int,
) -> dict[str, Any]:
    summary = {}
    for modality, tokens in (
        ("video", config.expected_video_tokens),
        ("audio", config.expected_audio_tokens),
    ):
        records = _records_for_modality(config, episodes, modality)
        total_bytes = 0
        total_blocks = 0
        for values in records.values():
            for record in values:
                path = Path(record["cache_path"])
                if not _valid_cache(
                    path,
                    modality=modality,
                    source_id=record["source_id"],
                    blocks=record["blocks"],
                    tokens=tokens,
                    width=width,
                    model_revision=model_revision,
                ):
                    raise ValueError(f"Invalid or incomplete cache: {path}")
                record["bytes"] = path.stat().st_size
                record["sha256"] = _sha256(path)
                total_bytes += record["bytes"]
                total_blocks += record["blocks"]
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "modality": modality,
            "model_revision": model_revision,
            "code_revision": code_revision,
            "canonical_manifest": str(config.canonical_manifest),
            "canonical_manifest_sha256": _sha256(config.canonical_manifest),
            "tokens_per_block": tokens,
            "embedding_width": width,
            "block_seconds": config.block_seconds,
            "splits": records,
        }
        _atomic_json(config.cache_root / f"{modality}_manifest.json", manifest)
        summary[modality] = {"blocks": total_blocks, "bytes": total_bytes}
    return summary


@torch.no_grad()
def run(config_path: Path, provenance_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    transformers_logging.set_verbosity_error()
    torch.set_num_threads(config.runtime.cpu_threads)
    with distributed_context(
        config.runtime.device,
        config.runtime.backend,
        nccl_compatibility_mode=config.runtime.nccl_compatibility_mode,
    ) as context:
        episodes = load_episodes(config)
        flat = [(split, episode) for split, values in episodes.items() for episode in values]
        local = flat[context.rank :: context.world_size]
        backend = QwenOmniThinkerEmbeddingBackend(
            load_omni_backbone_config(config.omni_config),
            context.device,
            audit_provenance(provenance_path),
        )
        torch.cuda.reset_peak_memory_stats(context.device)
        started = time.perf_counter()
        for completed, (split, episode) in enumerate(local, start=1):
            blocks = _block_count(episode, config.block_seconds)
            video_path = _cache_path(config, "video", split, episode.source_id)
            audio_path = _cache_path(config, "audio", split, episode.source_id)
            video_valid = _valid_cache(
                video_path,
                modality="video",
                source_id=episode.source_id,
                blocks=blocks,
                tokens=config.expected_video_tokens,
                width=backend.output_dim,
                model_revision=backend.config.revision,
            )
            audio_valid = _valid_cache(
                audio_path,
                modality="audio",
                source_id=episode.source_id,
                blocks=blocks,
                tokens=config.expected_audio_tokens,
                width=backend.output_dim,
                model_revision=backend.config.revision,
            )
            if not video_valid:
                _encode_video(backend, episode, split, config)
            if not audio_valid:
                _encode_audio(backend, episode, split, config)
            if context.is_primary and (completed % 25 == 0 or completed == len(local)):
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * (len(local) - completed)
                print(
                    f"omni_vggsound_rank0={completed}/{len(local)} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
        if context.world_size > 1:
            torch.distributed.barrier()
        local_peak = torch.tensor(
            float(torch.cuda.max_memory_reserved(context.device)),
            device=context.device,
        )
        if context.world_size > 1:
            torch.distributed.all_reduce(local_peak, op=torch.distributed.ReduceOp.MAX)
        report = {}
        if context.is_primary:
            project_root = config_path.resolve().parent.parent
            code_revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            print("validating_and_hashing_cache", flush=True)
            modality_summary = _write_manifests(
                config,
                episodes,
                model_revision=backend.config.revision,
                code_revision=code_revision,
                width=backend.output_dim,
            )
            report = {
                "schema": "deltaomni.omni_vggsound_cache_summary.v1",
                "model_revision": backend.config.revision,
                "code_revision": code_revision,
                "world_size": context.world_size,
                "splits": {split: len(values) for split, values in episodes.items()},
                "modalities": modality_summary,
                "peak_reserved_bytes_per_rank_max": int(local_peak.item()),
                "elapsed_seconds": time.perf_counter() - started,
                "completed_at_utc": datetime.now(UTC).isoformat(),
            }
            _atomic_json(config.report_path, report)
        if context.world_size > 1:
            torch.distributed.barrier()
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache native Omni VGGSound block embeddings")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/omni_vggsound_s1_cache.yaml"),
    )
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    args = parser.parse_args()
    report = run(args.config, args.provenance)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
