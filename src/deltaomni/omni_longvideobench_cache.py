from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
from transformers.utils import logging as transformers_logging

from deltaomni.data.longvideobench import IndexedLongVideoBenchStore
from deltaomni.deltatok import DeltaTok
from deltaomni.deltatok_scale_train import load_config as load_deltatok_config
from deltaomni.distributed import distributed_context
from deltaomni.omni_backbones import QwenOmniThinkerEmbeddingBackend, load_omni_backbone_config
from deltaomni.omni_nextqa_joint_cache import _encode_batches
from deltaomni.provenance import audit as audit_provenance
from deltaomni.run_integrity import (
    git_revision,
    git_worktree_is_clean,
    require_media_policy,
    resolved_input_signature,
    sha256_file,
)
from deltaomni.train_sanity import _atomic_json

CACHE_SCHEMA = "deltaomni.omni_longvideobench_window.v1"
MANIFEST_SCHEMA = "deltaomni.omni_longvideobench_manifest.v1"


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    backend: str
    nccl_compatibility_mode: bool
    cpu_threads: int


@dataclass(frozen=True)
class CacheConfig:
    seed: int
    dataset_resource_name: str
    annotations: Path
    archive_index: Path
    media_policy: Path
    omni_config: Path
    deltatok_config: Path
    deltatok_checkpoint: Path
    deltatok_sha256: str
    window_seconds: float
    block_seconds: float
    sample_fps: float
    frame_width: int
    frame_height: int
    expected_video_tokens: int
    encoder_batch_size: int
    maximum_videos: int | None
    minimum_videos: int
    runtime: RuntimeConfig
    cache_root: Path
    report_path: Path


@dataclass(frozen=True)
class VideoWindow:
    video_id: str
    video_path: str
    duration_seconds: float
    index: int
    start_seconds: float
    end_seconds: float

    @property
    def window_id(self) -> str:
        return f"{self.video_id}:{self.index:04d}"


def load_config(path: Path) -> CacheConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    paths = {
        "annotations",
        "archive_index",
        "media_policy",
        "omni_config",
        "deltatok_config",
        "deltatok_checkpoint",
        "cache_root",
        "report_path",
    }
    values = {key: resolve(value) if key in paths else value for key, value in raw.items()}
    values["runtime"] = RuntimeConfig(**raw["runtime"])
    config = CacheConfig(**values)
    positive = (
        config.window_seconds,
        config.block_seconds,
        config.sample_fps,
        config.frame_width,
        config.frame_height,
        config.expected_video_tokens,
        config.encoder_batch_size,
        config.minimum_videos,
        config.runtime.cpu_threads,
    )
    if min(positive) <= 0 or len(config.deltatok_sha256) != 64:
        raise ValueError("Invalid LongVideoBench cache controls")
    if config.maximum_videos is not None and config.maximum_videos < config.minimum_videos:
        raise ValueError("LongVideoBench maximum is below its required minimum")
    if not (config.block_seconds * config.sample_fps).is_integer():
        raise ValueError("LongVideoBench block duration and FPS require an integer frame grid")
    if not (config.window_seconds / config.block_seconds).is_integer():
        raise ValueError("LongVideoBench window duration must align to blocks")
    return config


def _select(config: CacheConfig) -> tuple[list[VideoWindow], dict[str, list[dict[str, Any]]]]:
    rows = json.loads(config.annotations.read_text(encoding="utf-8"))
    videos: dict[str, dict[str, Any]] = {}
    questions: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        video_id = str(row["video_id"])
        metadata = {
            "video_id": video_id,
            "video_path": str(row["video_path"]),
            "subtitle_path": str(row["subtitle_path"]),
            "duration": float(row["duration"]),
        }
        if video_id in videos and videos[video_id] != metadata:
            raise ValueError(f"LongVideoBench video metadata mismatch: {video_id}")
        videos[video_id] = metadata
        questions.setdefault(video_id, []).append(row)
    selected_ids = sorted(
        videos,
        key=lambda video_id: hashlib.sha256(f"{config.seed}:{video_id}".encode()).hexdigest(),
    )
    if config.maximum_videos is not None:
        selected_ids = selected_ids[: config.maximum_videos]
    if len(selected_ids) < config.minimum_videos:
        raise ValueError(f"Found {len(selected_ids)}/{config.minimum_videos} LongVideoBench videos")
    windows = []
    for video_id in selected_ids:
        metadata = videos[video_id]
        count = math.ceil(metadata["duration"] / config.window_seconds)
        for index in range(count):
            start = index * config.window_seconds
            end = min(metadata["duration"], start + config.window_seconds)
            if end - start < config.block_seconds:
                continue
            windows.append(
                VideoWindow(
                    video_id,
                    metadata["video_path"],
                    metadata["duration"],
                    index,
                    start,
                    end,
                )
            )
    return windows, {video_id: questions[video_id] for video_id in selected_ids}


def _cache_path(config: CacheConfig, window: VideoWindow) -> Path:
    digest = hashlib.sha256(window.window_id.encode()).hexdigest()
    return config.cache_root / "windows" / digest[:2] / f"{digest}.pt"


def _atomic_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _targets(window: VideoWindow, config: CacheConfig) -> list[float]:
    frames_per_block = round(config.block_seconds * config.sample_fps)
    complete_blocks = max(
        2,
        math.floor((window.end_seconds - window.start_seconds) / config.block_seconds),
    )
    return [
        window.start_seconds + block * config.block_seconds + frame / config.sample_fps
        for block in range(complete_blocks)
        for frame in range(frames_per_block)
    ]


def _video_blocks(
    store: IndexedLongVideoBenchStore,
    window: VideoWindow,
    config: CacheConfig,
) -> list[list[Image.Image]]:
    targets = _targets(window, config)
    frames_per_block = round(config.block_seconds * config.sample_fps)
    selected: list[Image.Image] = []
    target_index = 0
    previous = None
    previous_time = 0.0
    with store.open_video(window.video_path) as media, av.open(media) as container:
        stream = container.streams.video[0]
        seek_seconds = max(0.0, targets[0] - 2.0)
        if stream.time_base is not None:
            container.seek(
                int(seek_seconds / float(stream.time_base)),
                stream=stream,
                backward=True,
                any_frame=False,
            )
        fps = float(stream.average_rate) if stream.average_rate is not None else None
        for index, frame in enumerate(container.decode(stream)):
            timestamp = float(frame.time) if frame.time is not None else index / (fps or 30.0)
            if previous is None:
                previous, previous_time = frame, timestamp
                continue
            while target_index < len(targets) and targets[target_index] <= timestamp:
                target = targets[target_index]
                chosen = (
                    previous
                    if abs(previous_time - target) <= abs(timestamp - target)
                    else frame
                )
                selected.append(
                    ImageOps.pad(
                        chosen.to_image().convert("RGB"),
                        (config.frame_width, config.frame_height),
                        color=(0, 0, 0),
                    )
                )
                target_index += 1
            if target_index == len(targets):
                break
            previous, previous_time = frame, timestamp
    if previous is None:
        raise ValueError(f"No LongVideoBench frames: {window.window_id}")
    while target_index < len(targets):
        selected.append(
            ImageOps.pad(
                previous.to_image().convert("RGB"),
                (config.frame_width, config.frame_height),
                color=(0, 0, 0),
            )
        )
        target_index += 1
    return [
        selected[start : start + frames_per_block]
        for start in range(0, len(selected), frames_per_block)
    ]


def _valid(
    path: Path,
    window: VideoWindow,
    blocks: int,
    full_width: int,
    delta_width: int,
    config: CacheConfig,
    signature: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except (EOFError, OSError, RuntimeError):
        return False
    first, deltas, final_full = (
        value.get("first_full"),
        value.get("deltas"),
        value.get("final_full"),
    )
    return bool(
        value.get("schema") == CACHE_SCHEMA
        and value.get("cache_signature") == signature
        and value.get("window_id") == window.window_id
        and value.get("video_id") == window.video_id
        and value.get("blocks") == blocks
        and isinstance(first, torch.Tensor)
        and tuple(first.shape) == (config.expected_video_tokens, full_width)
        and isinstance(final_full, torch.Tensor)
        and tuple(final_full.shape) == (config.expected_video_tokens, full_width)
        and isinstance(deltas, torch.Tensor)
        and tuple(deltas.shape) == (blocks - 1, 1, delta_width)
        and all(
            tensor.dtype == torch.float16 and torch.isfinite(tensor).all()
            for tensor in (first, final_full, deltas)
        )
    )


@torch.no_grad()
def run(config_path: Path, provenance_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = config_path.resolve().parent.parent
    if not git_worktree_is_clean(root):
        raise RuntimeError("LongVideoBench cache requires a clean Git worktree")
    if sha256_file(config.deltatok_checkpoint) != config.deltatok_sha256:
        raise ValueError("LongVideoBench DeltaTok checksum mismatch")
    provenance = audit_provenance(provenance_path)
    media_policy_sha256 = require_media_policy(
        provenance, config.dataset_resource_name, config.media_policy
    )
    delta_config = load_deltatok_config(config.deltatok_config)
    signature = resolved_input_signature(
        config,
        {
            "annotations": config.annotations,
            "archive_index": config.archive_index,
            "media_policy": config.media_policy,
            "omni_config": config.omni_config,
            "deltatok_config": config.deltatok_config,
            "deltatok_checkpoint": config.deltatok_checkpoint,
        },
    )
    transformers_logging.set_verbosity_error()
    torch.set_num_threads(config.runtime.cpu_threads)
    with distributed_context(
        config.runtime.device,
        config.runtime.backend,
        nccl_compatibility_mode=config.runtime.nccl_compatibility_mode,
    ) as context:
        windows, questions = _select(config)
        local = windows[context.rank :: context.world_size]
        store = IndexedLongVideoBenchStore(config.archive_index)
        backend = QwenOmniThinkerEmbeddingBackend(
            load_omni_backbone_config(config.omni_config), context.device, provenance
        )
        delta_model = DeltaTok(delta_config.model).to(context.device).eval()
        checkpoint = torch.load(config.deltatok_checkpoint, map_location="cpu", weights_only=False)
        delta_model.load_state_dict(checkpoint["model"])
        delta_model.requires_grad_(False)
        started = time.perf_counter()
        for completed, window in enumerate(local, 1):
            frames_per_block = round(config.block_seconds * config.sample_fps)
            blocks = len(_targets(window, config)) // frames_per_block
            path = _cache_path(config, window)
            if not _valid(
                path,
                window,
                blocks,
                backend.output_dim,
                delta_config.model.model_dim,
                config,
                signature,
            ):
                features = _encode_batches(
                    backend.encode_video_chunks,
                    _video_blocks(store, window, config),
                    config.encoder_batch_size,
                    config.expected_video_tokens,
                    "video",
                )
                deltas = []
                with torch.autocast(device_type=context.device.type, dtype=torch.bfloat16):
                    for step in range(1, len(features)):
                        deltas.append(
                            delta_model.encode(features[step - 1][None], features[step][None])[0]
                            .cpu()
                            .to(torch.float16)
                        )
                _atomic_save(
                    path,
                    {
                        "schema": CACHE_SCHEMA,
                        "cache_signature": signature,
                        "window_id": window.window_id,
                        "video_id": window.video_id,
                        "video_path": window.video_path,
                        "window_index": window.index,
                        "start_seconds": window.start_seconds,
                        "end_seconds": window.end_seconds,
                        "blocks": len(features),
                        "first_full": features[0].cpu().to(torch.float16),
                        "deltas": torch.stack(deltas),
                        "final_full": features[-1].cpu().to(torch.float16),
                    },
                )
            if context.is_primary and (completed % 10 == 0 or completed == len(local)):
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * (len(local) - completed)
                print(
                    f"longvideobench_window_rank0={completed}/{len(local)} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
        if context.world_size > 1:
            torch.distributed.barrier()
        report = {}
        if context.is_primary:
            video_records: dict[str, list[dict[str, Any]]] = {
                video_id: [] for video_id in questions
            }
            total_bytes = 0
            total_blocks = 0
            for window in windows:
                frames_per_block = round(config.block_seconds * config.sample_fps)
                blocks = len(_targets(window, config)) // frames_per_block
                path = _cache_path(config, window)
                if not _valid(
                    path,
                    window,
                    blocks,
                    backend.output_dim,
                    delta_config.model.model_dim,
                    config,
                    signature,
                ):
                    raise ValueError(f"Invalid LongVideoBench window cache: {path}")
                size = path.stat().st_size
                total_bytes += size
                total_blocks += blocks
                video_records[window.video_id].append(
                    {
                        "window_id": window.window_id,
                        "cache_path": str(path),
                        "window_index": window.index,
                        "start_seconds": window.start_seconds,
                        "end_seconds": window.end_seconds,
                        "blocks": blocks,
                        "bytes": size,
                        "sha256": sha256_file(path),
                    }
                )
            manifest = {
                "schema": MANIFEST_SCHEMA,
                "code_revision": git_revision(root),
                "cache_signature": signature,
                "annotations": str(config.annotations),
                "annotations_sha256": sha256_file(config.annotations),
                "archive_index": str(config.archive_index),
                "archive_index_sha256": sha256_file(config.archive_index),
                "media_policy_sha256": media_policy_sha256,
                "omni_revision": backend.config.revision,
                "deltatok_sha256": config.deltatok_sha256,
                "window_seconds": config.window_seconds,
                "full_tokens": config.expected_video_tokens,
                "full_width": backend.output_dim,
                "delta_width": delta_config.model.model_dim,
                "videos": {
                    video_id: {
                        "windows": sorted(values, key=lambda value: value["window_index"]),
                        "questions": questions[video_id],
                    }
                    for video_id, values in video_records.items()
                },
            }
            _atomic_json(config.cache_root / "manifest.json", manifest)
            report = {
                "schema": "deltaomni.omni_longvideobench_summary.v1",
                "videos": len(video_records),
                "windows": len(windows),
                "blocks": total_blocks,
                "questions": sum(len(values) for values in questions.values()),
                "total_bytes": total_bytes,
                "world_size": context.world_size,
                "elapsed_seconds": time.perf_counter() - started,
                "completed_at_utc": datetime.now(UTC).isoformat(),
            }
            _atomic_json(config.report_path, report)
        if context.world_size > 1:
            torch.distributed.barrier()
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache LongVideoBench native FULL+delta windows")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/omni_longvideobench_cache_smoke.yaml")
    )
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    args = parser.parse_args()
    result = run(args.config, args.provenance)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
