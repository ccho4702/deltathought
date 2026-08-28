from __future__ import annotations

import argparse
import json
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

from deltaomni.data.canonicalize import read_canonical_dataset
from deltaomni.data.schema import CanonicalEpisode
from deltaomni.deltatok import DeltaTok
from deltaomni.deltatok_scale_train import load_config as load_deltatok_config
from deltaomni.distributed import distributed_context
from deltaomni.ego4d_dynamic_commits import (
    CommitWindow,
    DynamicCommitConfig,
    build_windows,
)
from deltaomni.ego4d_dynamic_commits import (
    load_config as load_dynamic_config,
)
from deltaomni.omni_backbones import QwenOmniThinkerEmbeddingBackend, load_omni_backbone_config
from deltaomni.omni_nextqa_joint_cache import _encode_batches
from deltaomni.provenance import audit as audit_provenance
from deltaomni.provenance import require_approved
from deltaomni.run_integrity import (
    git_revision,
    git_worktree_is_clean,
    resolved_input_signature,
    sha256_file,
    validate_license_record,
)
from deltaomni.train_sanity import _atomic_json

CACHE_SCHEMA = "deltaomni.omni_ego4d_goalstep_window.v1"
MANIFEST_SCHEMA = "deltaomni.omni_ego4d_goalstep_manifest.v1"


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
    canonical_manifest: Path
    license_record: Path
    dynamic_commit_config: Path
    omni_config: Path
    deltatok_config: Path
    deltatok_checkpoint: Path
    deltatok_sha256: str
    block_seconds: float
    sample_fps: float
    frame_width: int
    frame_height: int
    expected_video_tokens: int
    encoder_batch_size: int
    minimum_windows: dict[str, int]
    runtime: RuntimeConfig
    cache_root: Path
    report_path: Path


def load_config(path: Path) -> CacheConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    paths = {
        "canonical_manifest",
        "license_record",
        "dynamic_commit_config",
        "omni_config",
        "deltatok_config",
        "deltatok_checkpoint",
        "cache_root",
        "report_path",
    }
    values = {key: resolve(value) if key in paths else value for key, value in raw.items()}
    values["runtime"] = RuntimeConfig(**raw["runtime"])
    config = CacheConfig(**values)
    if set(config.minimum_windows) != {"train", "validation"}:
        raise ValueError("Ego4D GoalStep cache thresholds require train and validation")
    positive = (
        config.block_seconds,
        config.sample_fps,
        config.frame_width,
        config.frame_height,
        config.expected_video_tokens,
        config.encoder_batch_size,
        config.runtime.cpu_threads,
        *config.minimum_windows.values(),
    )
    if min(positive) <= 0 or len(config.deltatok_sha256) != 64:
        raise ValueError("Invalid Ego4D GoalStep cache controls")
    frames_per_block = config.block_seconds * config.sample_fps
    if not frames_per_block.is_integer():
        raise ValueError("Ego4D block duration and FPS must form an integer frame grid")
    return config


def _windows(
    config: CacheConfig,
    dynamic: DynamicCommitConfig,
) -> dict[str, list[tuple[CanonicalEpisode, CommitWindow]]]:
    canonical = read_canonical_dataset(config.canonical_manifest)
    if set(canonical) != {"train", "validation"}:
        raise ValueError("Ego4D GoalStep canonical splits changed")
    result = {}
    for split, episodes in canonical.items():
        values = [
            (episode, window)
            for episode in episodes
            for window in build_windows(episode, dynamic)
            if window.delta_updates > 0
        ]
        values.sort(key=lambda value: value[1].window_id)
        if len(values) < config.minimum_windows[split]:
            raise ValueError(
                f"Found {len(values)}/{config.minimum_windows[split]} Ego4D {split} windows"
            )
        result[split] = values
    return result


def _cache_path(config: CacheConfig, split: str, window_id: str) -> Path:
    digest = __import__("hashlib").sha256(window_id.encode()).hexdigest()
    return config.cache_root / split / digest[:2] / f"{digest}.pt"


def _atomic_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _window_targets(window: CommitWindow, config: CacheConfig) -> list[float]:
    frames_per_block = round(config.block_seconds * config.sample_fps)
    return [
        block * config.block_seconds + frame / config.sample_fps
        for block in range(window.anchor_block, window.final_block + 1)
        for frame in range(frames_per_block)
    ]


def _video_blocks(
    episode: CanonicalEpisode,
    window: CommitWindow,
    config: CacheConfig,
) -> list[list[Image.Image]]:
    assert episode.media.video is not None
    targets = _window_targets(window, config)
    frames_per_block = round(config.block_seconds * config.sample_fps)
    selected: list[Image.Image] = []
    target_index = 0
    previous = None
    previous_time = 0.0
    with av.open(str(episode.media.video.path)) as container:
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
                    previous if abs(previous_time - target) <= abs(timestamp - target) else frame
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
        raise ValueError(f"No Ego4D video frames: {episode.source_id}")
    while target_index < len(targets):
        selected.append(
            ImageOps.pad(
                previous.to_image().convert("RGB"),
                (config.frame_width, config.frame_height),
                color=(0, 0, 0),
            )
        )
        target_index += 1
    blocks = [
        selected[start : start + frames_per_block]
        for start in range(0, len(selected), frames_per_block)
    ]
    if len(blocks) != window.delta_updates + 1:
        raise ValueError(f"Ego4D window block count mismatch: {window.window_id}")
    return blocks


def _event_records(window: CommitWindow) -> list[dict[str, Any]]:
    return [
        {
            "caption_id": commit.caption_id,
            "text": commit.text,
            "start_seconds": commit.start_seconds,
            "end_seconds": commit.end_seconds,
            "commit_seconds": commit.commit_seconds,
            "delta_start": commit.previous_full_block - window.anchor_block,
            "delta_end": commit.current_full_block - window.anchor_block,
        }
        for commit in window.commits
    ]


def _valid(
    path: Path,
    *,
    episode: CanonicalEpisode,
    window: CommitWindow,
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
    first, deltas = value.get("first_full"), value.get("deltas")
    assert episode.media.video is not None
    return bool(
        value.get("schema") == CACHE_SCHEMA
        and value.get("cache_signature") == signature
        and value.get("window_id") == window.window_id
        and value.get("source_id") == episode.source_id
        and value.get("media_sha256") == episode.media.video.sha256
        and value.get("anchor_block") == window.anchor_block
        and value.get("final_block") == window.final_block
        and value.get("events") == _event_records(window)
        and isinstance(first, torch.Tensor)
        and tuple(first.shape) == (config.expected_video_tokens, full_width)
        and first.dtype == torch.float16
        and torch.isfinite(first).all()
        and isinstance(deltas, torch.Tensor)
        and tuple(deltas.shape) == (window.delta_updates, 1, delta_width)
        and deltas.dtype == torch.float16
        and torch.isfinite(deltas).all()
    )


@torch.no_grad()
def run(config_path: Path, provenance_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = config_path.resolve().parent.parent
    if not git_worktree_is_clean(root):
        raise RuntimeError("Ego4D GoalStep cache requires a clean Git worktree")
    license_record = validate_license_record(config.license_record)
    if license_record["dataset"] != "Ego4D":
        raise ValueError("Ego4D cache requires an Ego4D acceptance record")
    if sha256_file(config.deltatok_checkpoint) != config.deltatok_sha256:
        raise ValueError("Ego4D video DeltaTok checksum mismatch")
    provenance = audit_provenance(provenance_path)
    require_approved(provenance, [config.dataset_resource_name])
    dynamic = load_dynamic_config(config.dynamic_commit_config)
    if dynamic.block_seconds != config.block_seconds:
        raise ValueError("Ego4D dynamic commit/cache block duration mismatch")
    delta_config = load_deltatok_config(config.deltatok_config)
    signature = resolved_input_signature(
        config,
        {
            "canonical_manifest": config.canonical_manifest,
            "license_record": config.license_record,
            "dynamic_commit_config": config.dynamic_commit_config,
            "omni_config": config.omni_config,
            "deltatok_config": config.deltatok_config,
            "deltatok_checkpoint": config.deltatok_checkpoint,
            "provenance": provenance_path,
        },
    )
    transformers_logging.set_verbosity_error()
    torch.set_num_threads(config.runtime.cpu_threads)
    with distributed_context(
        config.runtime.device,
        config.runtime.backend,
        nccl_compatibility_mode=config.runtime.nccl_compatibility_mode,
    ) as context:
        windows = _windows(config, dynamic)
        flat = [
            (split, episode, window)
            for split, values in windows.items()
            for episode, window in values
        ]
        local = flat[context.rank :: context.world_size]
        backend = QwenOmniThinkerEmbeddingBackend(
            load_omni_backbone_config(config.omni_config), context.device, provenance
        )
        delta_model = DeltaTok(delta_config.model).to(context.device).eval()
        checkpoint = torch.load(config.deltatok_checkpoint, map_location="cpu", weights_only=False)
        delta_model.load_state_dict(checkpoint["model"])
        delta_model.requires_grad_(False)
        started = time.perf_counter()
        for completed, (split, episode, window) in enumerate(local, 1):
            path = _cache_path(config, split, window.window_id)
            if _valid(
                path,
                episode=episode,
                window=window,
                full_width=backend.output_dim,
                delta_width=delta_config.model.model_dim,
                config=config,
                signature=signature,
            ):
                continue
            features = _encode_batches(
                backend.encode_video_chunks,
                _video_blocks(episode, window, config),
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
            assert deltas
            assert episode.media.video is not None
            _atomic_save(
                path,
                {
                    "schema": CACHE_SCHEMA,
                    "cache_signature": signature,
                    "window_id": window.window_id,
                    "source_id": episode.source_id,
                    "source_group_id": episode.source_group_id,
                    "split": split,
                    "anchor_block": window.anchor_block,
                    "final_block": window.final_block,
                    "first_full": features[0].cpu().to(torch.float16),
                    "deltas": torch.stack(deltas),
                    "events": _event_records(window),
                    "truncated_precontext": window.truncated_precontext,
                    "media_sha256": episode.media.video.sha256,
                },
            )
            if context.is_primary and (completed % 10 == 0 or completed == len(local)):
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * (len(local) - completed)
                print(
                    f"ego4d_window_rank0={completed}/{len(local)} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
        if context.world_size > 1:
            torch.distributed.barrier()
        report = {}
        if context.is_primary:
            records = {split: [] for split in windows}
            total_bytes = 0
            total_blocks = 0
            total_commits = 0
            for split, values in windows.items():
                for episode, window in values:
                    path = _cache_path(config, split, window.window_id)
                    if not _valid(
                        path,
                        episode=episode,
                        window=window,
                        full_width=backend.output_dim,
                        delta_width=delta_config.model.model_dim,
                        config=config,
                        signature=signature,
                    ):
                        raise ValueError(f"Invalid Ego4D GoalStep cache: {path}")
                    size = path.stat().st_size
                    total_bytes += size
                    total_blocks += window.delta_updates + 1
                    total_commits += len(window.commits)
                    records[split].append(
                        {
                            "window_id": window.window_id,
                            "source_id": episode.source_id,
                            "source_group_id": episode.source_group_id,
                            "cache_path": str(path),
                            "blocks": window.delta_updates + 1,
                            "delta_updates": window.delta_updates,
                            "commits": len(window.commits),
                            "bytes": size,
                            "sha256": sha256_file(path),
                        }
                    )
            manifest = {
                "schema": MANIFEST_SCHEMA,
                "code_revision": git_revision(root),
                "cache_signature": signature,
                "canonical_manifest": str(config.canonical_manifest),
                "canonical_manifest_sha256": sha256_file(config.canonical_manifest),
                "license_record_sha256": sha256_file(config.license_record),
                "omni_revision": backend.config.revision,
                "deltatok_sha256": config.deltatok_sha256,
                "full_tokens": config.expected_video_tokens,
                "full_width": backend.output_dim,
                "delta_tokens_per_block": 1,
                "delta_width": delta_config.model.model_dim,
                "block_seconds": config.block_seconds,
                "splits": records,
            }
            _atomic_json(config.cache_root / "manifest.json", manifest)
            report = {
                "schema": "deltaomni.omni_ego4d_goalstep_summary.v1",
                "splits": {split: len(values) for split, values in windows.items()},
                "total_blocks": total_blocks,
                "total_commits": total_commits,
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
    parser = argparse.ArgumentParser(description="Cache Ego4D GoalStep FULL+delta windows")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/omni_ego4d_goalstep_cache.yaml")
    )
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    args = parser.parse_args()
    report = run(args.config, args.provenance)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
