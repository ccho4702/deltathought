from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import torch
import yaml
from PIL import Image

from deltaomni.data.schema import CanonicalEpisode, iter_jsonl
from deltaomni.distributed import distributed_context
from deltaomni.omni_backbones import (
    QwenOmniThinkerEmbeddingBackend,
    load_omni_backbone_config,
)
from deltaomni.provenance import audit as audit_provenance
from deltaomni.train_sanity import _atomic_json


@dataclass(frozen=True)
class ProfileRuntime:
    device: str
    backend: str
    nccl_compatibility_mode: bool
    cpu_threads: int


@dataclass(frozen=True)
class EmbeddingProfileConfig:
    canonical_manifest: Path
    omni_config: Path
    seed: int
    samples: int
    block_seconds: float
    sample_fps: float
    batch_sizes: tuple[int, ...]
    warmup_batches: int
    runtime: ProfileRuntime
    output: Path


def load_config(path: Path) -> EmbeddingProfileConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    runtime = raw["runtime"]
    config = EmbeddingProfileConfig(
        canonical_manifest=resolve(raw["canonical_manifest"]),
        omni_config=resolve(raw["omni_config"]),
        seed=int(raw["seed"]),
        samples=int(raw["samples"]),
        block_seconds=float(raw["block_seconds"]),
        sample_fps=float(raw["sample_fps"]),
        batch_sizes=tuple(int(value) for value in raw["batch_sizes"]),
        warmup_batches=int(raw["warmup_batches"]),
        runtime=ProfileRuntime(
            device=str(runtime["device"]),
            backend=str(runtime["backend"]),
            nccl_compatibility_mode=bool(runtime["nccl_compatibility_mode"]),
            cpu_threads=int(runtime["cpu_threads"]),
        ),
        output=resolve(raw["output"]),
    )
    if min(config.samples, config.warmup_batches + 1) <= 0:
        raise ValueError("Profile sample and warmup counts must be positive")
    if not config.batch_sizes or min(config.batch_sizes) <= 0:
        raise ValueError("Profile batch sizes must be positive")
    if min(config.block_seconds, config.sample_fps) <= 0:
        raise ValueError("Profile temporal controls must be positive")
    return config


def _select_episodes(config: EmbeddingProfileConfig) -> list[CanonicalEpisode]:
    manifest = json.loads(config.canonical_manifest.read_text(encoding="utf-8"))
    train_path = config.canonical_manifest.parent / manifest["splits"]["train"]["path"]
    ranked = []
    for episode in iter_jsonl(train_path):
        key = hashlib.sha256(f"{config.seed}:{episode.source_id}".encode()).hexdigest()
        ranked.append((key, episode))
        if len(ranked) == config.samples * 4:
            break
    ranked.sort(key=lambda value: value[0])
    selected = [episode for _, episode in ranked[: config.samples]]
    if len(selected) != config.samples:
        raise ValueError(f"Found {len(selected)}/{config.samples} canonical episodes")
    return selected


def _sample_first_block(
    path: Path,
    block_seconds: float,
    sample_fps: float,
) -> list[Image.Image]:
    target_count = max(2, round(block_seconds * sample_fps))
    if target_count % 2:
        target_count += 1
    frames = []
    next_time = 0.0
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate) if stream.average_rate is not None else None
        for index, frame in enumerate(container.decode(stream)):
            timestamp = (
                float(frame.time)
                if frame.time is not None
                else index / source_fps if source_fps else 0.0
            )
            if timestamp + 1e-9 < next_time:
                continue
            frames.append(frame.to_image().convert("RGB"))
            next_time += 1.0 / sample_fps
            if len(frames) == target_count:
                break
    if not frames:
        raise ValueError(f"No video frames decoded: {path}")
    while len(frames) < target_count:
        frames.append(frames[-1].copy())
    return frames


@torch.no_grad()
def run(config_path: Path, provenance_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    torch.set_num_threads(config.runtime.cpu_threads)
    with distributed_context(
        config.runtime.device,
        config.runtime.backend,
        nccl_compatibility_mode=config.runtime.nccl_compatibility_mode,
    ) as context:
        selected = _select_episodes(config)
        local = selected[context.rank :: context.world_size]
        videos = []
        for index, episode in enumerate(local, start=1):
            if episode.media.video is None:
                raise ValueError(f"Profile episode has no video: {episode.episode_id}")
            videos.append(
                _sample_first_block(
                    episode.media.video.path,
                    config.block_seconds,
                    config.sample_fps,
                )
            )
            if context.is_primary and (index % 8 == 0 or index == len(local)):
                print(f"profile_decode={index}/{len(local)}", flush=True)
        backend = QwenOmniThinkerEmbeddingBackend(
            load_omni_backbone_config(config.omni_config),
            context.device,
            audit_provenance(provenance_path),
        )
        results = []
        for batch_size in config.batch_sizes:
            usable = len(videos) // batch_size * batch_size
            batches = [videos[start : start + batch_size] for start in range(0, usable, batch_size)]
            if len(batches) <= config.warmup_batches:
                raise ValueError(f"Not enough profile batches for batch_size={batch_size}")
            for batch in batches[: config.warmup_batches]:
                backend.encode_video_chunks(batch)
            torch.cuda.synchronize(context.device)
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            token_counts = []
            measured = batches[config.warmup_batches :]
            for batch in measured:
                _, metadata = backend.encode_video_chunks(batch)
                token_counts.extend(int(value["tokens"]) for value in metadata)
            torch.cuda.synchronize(context.device)
            elapsed = time.perf_counter() - started
            local_stats = torch.tensor(
                [
                    sum(len(batch) for batch in measured),
                    elapsed,
                    torch.cuda.max_memory_reserved(context.device) / 2**30,
                    min(token_counts),
                    max(token_counts),
                ],
                dtype=torch.float64,
                device=context.device,
            )
            total_clips = local_stats[0].clone()
            max_elapsed = local_stats[1].clone()
            max_memory = local_stats[2].clone()
            min_tokens = local_stats[3].clone()
            max_tokens = local_stats[4].clone()
            if context.world_size > 1:
                torch.distributed.all_reduce(total_clips, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(max_elapsed, op=torch.distributed.ReduceOp.MAX)
                torch.distributed.all_reduce(max_memory, op=torch.distributed.ReduceOp.MAX)
                torch.distributed.all_reduce(min_tokens, op=torch.distributed.ReduceOp.MIN)
                torch.distributed.all_reduce(max_tokens, op=torch.distributed.ReduceOp.MAX)
            result = {
                "batch_size_per_rank": batch_size,
                "clips": int(total_clips.item()),
                "elapsed_seconds": float(max_elapsed.item()),
                "clips_per_second": float(total_clips.item() / max_elapsed.item()),
                "peak_reserved_gib_per_rank": float(max_memory.item()),
                "tokens_per_block_min": int(min_tokens.item()),
                "tokens_per_block_max": int(max_tokens.item()),
            }
            results.append(result)
            if context.is_primary:
                print(json.dumps(result), flush=True)
        report = {
            "schema": "deltaomni.qwen2_5_omni_embedding_profile.v1",
            "model_revision": backend.config.revision,
            "canonical_manifest": str(config.canonical_manifest),
            "world_size": context.world_size,
            "samples": config.samples,
            "results": results,
        }
        if context.is_primary:
            _atomic_json(config.output, report)
        if context.world_size > 1:
            torch.distributed.barrier()
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile native Omni video embeddings")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/omni_embedding_profile.yaml"),
    )
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    args = parser.parse_args()
    report = run(args.config, args.provenance)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
