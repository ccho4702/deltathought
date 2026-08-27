from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import yaml
from PIL import Image, ImageOps
from torch import Tensor
from transformers.utils import logging as transformers_logging

from deltaomni.data.canonicalize import read_canonical_dataset
from deltaomni.data.schema import CanonicalEpisode
from deltaomni.deltatok import DeltaTok
from deltaomni.deltatok_scale_train import load_config as load_deltatok_config
from deltaomni.distributed import distributed_context
from deltaomni.omni_audiocaps_prefix_cache import _atomic_torch_save
from deltaomni.omni_backbones import (
    QwenOmniThinkerEmbeddingBackend,
    load_omni_backbone_config,
)
from deltaomni.provenance import audit as audit_provenance
from deltaomni.run_integrity import (
    git_revision,
    git_worktree_is_clean,
    require_media_policy,
    resolved_input_signature,
    sha256_file,
)
from deltaomni.train_sanity import _atomic_json

CACHE_SCHEMA = "deltaomni.omni_nextqa_joint_prefix.v2"


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    backend: str
    nccl_compatibility_mode: bool
    cpu_threads: int


@dataclass(frozen=True)
class JointCacheConfig:
    seed: int
    dataset_resource_name: str
    media_policy: Path
    canonical_manifest: Path
    omni_config: Path
    video_deltatok_config: Path
    video_deltatok_checkpoint: Path
    video_deltatok_sha256: str
    audio_deltatok_config: Path
    audio_deltatok_checkpoint: Path
    audio_deltatok_sha256: str
    train_count: int
    validation_count: int
    test_count: int
    minimum_seconds: float
    maximum_seconds: float
    block_seconds: float
    sample_fps: float
    frame_width: int
    frame_height: int
    expected_video_tokens: int
    expected_audio_tokens: int
    video_batch_size: int
    audio_batch_size: int
    runtime: RuntimeConfig
    cache_root: Path
    report_path: Path


def load_config(path: Path) -> JointCacheConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    runtime = raw["runtime"]
    config = JointCacheConfig(
        **{
            **{
                key: raw[key]
                for key in raw
                if key
                not in {
                    "runtime",
                    "media_policy",
                    "canonical_manifest",
                    "omni_config",
                    "video_deltatok_config",
                    "video_deltatok_checkpoint",
                    "audio_deltatok_config",
                    "audio_deltatok_checkpoint",
                    "cache_root",
                    "report_path",
                }
            },
            "media_policy": resolve(raw["media_policy"]),
            "canonical_manifest": resolve(raw["canonical_manifest"]),
            "omni_config": resolve(raw["omni_config"]),
            "video_deltatok_config": resolve(raw["video_deltatok_config"]),
            "video_deltatok_checkpoint": resolve(raw["video_deltatok_checkpoint"]),
            "audio_deltatok_config": resolve(raw["audio_deltatok_config"]),
            "audio_deltatok_checkpoint": resolve(raw["audio_deltatok_checkpoint"]),
            "runtime": RuntimeConfig(**runtime),
            "cache_root": resolve(raw["cache_root"]),
            "report_path": resolve(raw["report_path"]),
        }
    )
    if (
        min(
            config.train_count,
            config.validation_count,
            config.test_count,
            config.minimum_seconds,
            config.maximum_seconds,
            config.block_seconds,
            config.sample_fps,
            config.frame_width,
            config.frame_height,
            config.expected_video_tokens,
            config.expected_audio_tokens,
            config.video_batch_size,
            config.audio_batch_size,
            config.runtime.cpu_threads,
        )
        <= 0
    ):
        raise ValueError("Invalid NExT-QA joint cache controls")
    frames_per_block = config.block_seconds * config.sample_fps
    if config.maximum_seconds < config.minimum_seconds or not frames_per_block.is_integer():
        raise ValueError("NExT-QA duration and frame-grid controls are inconsistent")
    if len(config.video_deltatok_sha256) != 64 or len(config.audio_deltatok_sha256) != 64:
        raise ValueError("NExT-QA DeltaTok checksums must be SHA-256 values")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def select_episodes(config: JointCacheConfig) -> dict[str, list[CanonicalEpisode]]:
    counts = {
        "train": config.train_count,
        "validation": config.validation_count,
        "test": config.test_count,
    }
    canonical = read_canonical_dataset(config.canonical_manifest)
    result = {}
    for split, count in counts.items():
        eligible = [
            episode
            for episode in canonical[split]
            if episode.media.audio is not None
            and episode.media.video is not None
            and episode.qa
            and episode.duration_seconds is not None
            and config.minimum_seconds <= episode.duration_seconds <= config.maximum_seconds
        ]
        eligible.sort(
            key=lambda episode: hashlib.sha256(
                f"{config.seed}:{episode.source_id}".encode()
            ).hexdigest()
        )
        if len(eligible) < count:
            raise ValueError(f"Found {len(eligible)}/{count} eligible NExT-QA {split} clips")
        result[split] = eligible[:count]
    return result


def _block_count(episode: CanonicalEpisode, config: JointCacheConfig) -> int:
    assert episode.duration_seconds is not None
    return max(2, math.floor((episode.duration_seconds + 1e-6) / config.block_seconds))


def _video_blocks(episode: CanonicalEpisode, config: JointCacheConfig) -> list[list[Image.Image]]:
    assert episode.media.video is not None
    frames_per_block = round(config.block_seconds * config.sample_fps)
    targets = [
        index / config.sample_fps
        for index in range(_block_count(episode, config) * frames_per_block)
    ]
    selected = []
    target_index = 0
    previous = None
    previous_time = 0.0
    with av.open(str(episode.media.video.path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate is not None else None
        for index, frame in enumerate(container.decode(stream)):
            timestamp = float(frame.time) if frame.time is not None else index / fps if fps else 0.0
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
        raise ValueError(f"No NExT-QA video frames: {episode.source_id}")
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


def _audio_blocks(
    episode: CanonicalEpisode, config: JointCacheConfig, sample_rate: int
) -> list[np.ndarray]:
    assert episode.media.audio is not None
    arrays = []
    with av.open(str(episode.media.audio.path)) as container:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        for frame in container.decode(audio=0):
            for converted in resampler.resample(frame):
                arrays.append(converted.to_ndarray().reshape(-1))
        for converted in resampler.resample(None):
            arrays.append(converted.to_ndarray().reshape(-1))
    waveform = np.concatenate(arrays).astype(np.float32, copy=False)
    samples = round(config.block_seconds * sample_rate)
    required = _block_count(episode, config) * samples
    if waveform.shape[0] < required:
        waveform = np.pad(waveform, (0, required - waveform.shape[0]))
    return [
        np.ascontiguousarray(waveform[start : start + samples])
        for start in range(0, required, samples)
    ]


def _encode_batches(backend, blocks, batch_size, expected_tokens, kind):
    features = []
    for start in range(0, len(blocks), batch_size):
        batch = blocks[start : start + batch_size]
        retained = len(batch)
        batch = batch + [batch[-1]] * (batch_size - retained)
        encoded, _ = backend(batch)
        for feature in encoded[:retained]:
            if feature.shape[0] != expected_tokens:
                raise ValueError(f"Unexpected NExT-QA {kind} shape: {tuple(feature.shape)}")
            features.append(feature)
    return features


def _deltas(model: DeltaTok, features: list[Tensor], device: torch.device) -> Tensor:
    values = []
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for step in range(1, len(features)):
            values.append(
                model.encode(features[step - 1][None], features[step][None])[0]
                .cpu()
                .to(torch.float16)
            )
    return torch.stack(values)


def _valid_cache(
    path: Path,
    *,
    episode: CanonicalEpisode,
    split: str,
    blocks: int,
    video_tokens: int,
    audio_tokens: int,
    full_width: int,
    delta_width: int,
    cache_signature: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (EOFError, OSError, RuntimeError):
        return False
    tensors = {
        "video_first": (video_tokens, full_width),
        "video_deltas": (blocks - 1, 1, delta_width),
        "audio_first": (audio_tokens, full_width),
        "audio_deltas": (blocks - 1, 1, delta_width),
    }
    assert episode.media.video is not None and episode.media.audio is not None
    return bool(
        payload.get("schema") == CACHE_SCHEMA
        and payload.get("cache_signature") == cache_signature
        and payload.get("source_id") == episode.source_id
        and payload.get("source_group_id") == episode.source_group_id
        and payload.get("split") == split
        and payload.get("blocks") == blocks
        and payload.get("video_media_sha256") == episode.media.video.sha256
        and payload.get("audio_media_sha256") == episode.media.audio.sha256
        and len(payload.get("qa", ())) == len(episode.qa or ())
        and all(
            isinstance(payload.get(name), torch.Tensor)
            and tuple(payload[name].shape) == shape
            and payload[name].dtype == torch.float16
            and torch.isfinite(payload[name]).all()
            for name, shape in tensors.items()
        )
    )


@torch.no_grad()
def run(config_path: Path, provenance_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    project_root = config_path.resolve().parent.parent
    if not git_worktree_is_clean(project_root):
        raise RuntimeError("NExT-QA cache generation requires a clean Git worktree")
    transformers_logging.set_verbosity_error()
    torch.set_num_threads(config.runtime.cpu_threads)
    provenance = audit_provenance(provenance_path)
    media_policy_sha256 = require_media_policy(
        provenance,
        config.dataset_resource_name,
        config.media_policy,
    )
    for path, expected in (
        (config.video_deltatok_checkpoint, config.video_deltatok_sha256),
        (config.audio_deltatok_checkpoint, config.audio_deltatok_sha256),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"DeltaTok checksum mismatch: {path}")
    omni_config = load_omni_backbone_config(config.omni_config)
    video_cfg, audio_cfg = (
        load_deltatok_config(config.video_deltatok_config),
        load_deltatok_config(config.audio_deltatok_config),
    )
    if video_cfg.model.model_dim != audio_cfg.model.model_dim:
        raise ValueError("Joint cache requires matching audio/video delta widths")
    cache_signature = resolved_input_signature(
        config,
        {
            "canonical_manifest": config.canonical_manifest,
            "omni_config": config.omni_config,
            "video_deltatok_config": config.video_deltatok_config,
            "video_deltatok_checkpoint": config.video_deltatok_checkpoint,
            "audio_deltatok_config": config.audio_deltatok_config,
            "audio_deltatok_checkpoint": config.audio_deltatok_checkpoint,
            "media_policy": config.media_policy,
            "provenance": provenance_path,
        },
    )
    with distributed_context(
        config.runtime.device,
        config.runtime.backend,
        nccl_compatibility_mode=config.runtime.nccl_compatibility_mode,
    ) as context:
        episodes = select_episodes(config)
        flat = [(split, episode) for split, values in episodes.items() for episode in values]
        local = flat[context.rank :: context.world_size]
        backend = QwenOmniThinkerEmbeddingBackend(
            omni_config,
            context.device,
            provenance,
        )
        video_model, audio_model = (
            DeltaTok(video_cfg.model).to(context.device).eval(),
            DeltaTok(audio_cfg.model).to(context.device).eval(),
        )
        video_model.load_state_dict(
            torch.load(config.video_deltatok_checkpoint, map_location="cpu", weights_only=False)[
                "model"
            ]
        )
        audio_model.load_state_dict(
            torch.load(config.audio_deltatok_checkpoint, map_location="cpu", weights_only=False)[
                "model"
            ]
        )
        started = time.perf_counter()
        for completed, (split, episode) in enumerate(local, 1):
            path = config.cache_root / split / f"{episode.source_id}.pt"
            blocks = _block_count(episode, config)
            if _valid_cache(
                path,
                episode=episode,
                split=split,
                blocks=blocks,
                video_tokens=config.expected_video_tokens,
                audio_tokens=config.expected_audio_tokens,
                full_width=backend.output_dim,
                delta_width=video_cfg.model.model_dim,
                cache_signature=cache_signature,
            ):
                continue
            video = _encode_batches(
                backend.encode_video_chunks,
                _video_blocks(episode, config),
                config.video_batch_size,
                config.expected_video_tokens,
                "video",
            )
            audio = _encode_batches(
                backend.encode_audio_chunks,
                _audio_blocks(episode, config, backend.config.sample_rate),
                config.audio_batch_size,
                config.expected_audio_tokens,
                "audio",
            )
            if len(video) != blocks or len(audio) != blocks:
                raise ValueError(
                    f"NExT-QA synchronized block mismatch for {episode.source_id}: "
                    f"video={len(video)} audio={len(audio)} expected={blocks}"
                )
            qa = [
                {
                    "question_id": item.question_id,
                    "question": item.question,
                    "answer": item.answer,
                    "choices": list(item.choices or ()),
                    "answer_index": item.answer_index,
                    "question_type": item.question_type,
                }
                for item in episode.qa or ()
            ]
            _atomic_torch_save(
                path,
                {
                    "schema": CACHE_SCHEMA,
                    "cache_signature": cache_signature,
                    "source_id": episode.source_id,
                    "source_group_id": episode.source_group_id,
                    "split": split,
                    "video_first": video[0].cpu().to(torch.float16),
                    "video_deltas": _deltas(video_model, video, context.device),
                    "audio_first": audio[0].cpu().to(torch.float16),
                    "audio_deltas": _deltas(audio_model, audio, context.device),
                    "qa": qa,
                    "blocks": len(video),
                    "video_media_sha256": episode.media.video.sha256,
                    "audio_media_sha256": episode.media.audio.sha256,
                },
            )
            if context.is_primary and (completed % 4 == 0 or completed == len(local)):
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * (len(local) - completed)
                print(
                    f"nextqa_joint_rank0={completed}/{len(local)} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
        if context.world_size > 1:
            torch.distributed.barrier()
        report = {}
        if context.is_primary:
            records = {split: [] for split in episodes}
            for split, values in episodes.items():
                for episode in values:
                    path = config.cache_root / split / f"{episode.source_id}.pt"
                    if not _valid_cache(
                        path,
                        episode=episode,
                        split=split,
                        blocks=_block_count(episode, config),
                        video_tokens=config.expected_video_tokens,
                        audio_tokens=config.expected_audio_tokens,
                        full_width=backend.output_dim,
                        delta_width=video_cfg.model.model_dim,
                        cache_signature=cache_signature,
                    ):
                        raise ValueError(f"Invalid or stale NExT-QA joint cache: {path}")
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    records[split].append(
                        {
                            "source_id": episode.source_id,
                            "source_group_id": episode.source_group_id,
                            "cache_path": str(path),
                            "blocks": payload["blocks"],
                            "qa": len(payload["qa"]),
                        }
                    )
            revision = git_revision(project_root)
            manifest = {
                "schema": "deltaomni.omni_nextqa_joint_manifest.v2",
                "code_revision": revision,
                "cache_signature": cache_signature,
                "canonical_manifest_sha256": sha256_file(config.canonical_manifest),
                "media_policy_sha256": media_policy_sha256,
                "omni_revision": omni_config.revision,
                "video_deltatok_sha256": config.video_deltatok_sha256,
                "audio_deltatok_sha256": config.audio_deltatok_sha256,
                "splits": records,
            }
            _atomic_json(config.cache_root / "manifest.json", manifest)
            report = {
                "schema": "deltaomni.omni_nextqa_joint_summary.v1",
                "splits": {s: len(v) for s, v in episodes.items()},
                "qa": {s: sum(r["qa"] for r in values) for s, values in records.items()},
                "elapsed_seconds": time.perf_counter() - started,
                "completed_at_utc": datetime.now(UTC).isoformat(),
            }
            _atomic_json(config.report_path, report)
        if context.world_size > 1:
            torch.distributed.barrier()
        return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/omni_nextqa_joint_poc.yaml"))
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    args = parser.parse_args()
    report = run(args.config, args.provenance)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
