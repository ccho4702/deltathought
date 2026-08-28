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

import torch
import yaml
from transformers.utils import logging as transformers_logging

from deltaomni.data.canonicalize import read_canonical_dataset
from deltaomni.data.schema import CanonicalEpisode
from deltaomni.deltatok import DeltaTok
from deltaomni.deltatok_scale_train import load_config as load_deltatok_config
from deltaomni.distributed import distributed_context
from deltaomni.omni_backbones import QwenOmniThinkerEmbeddingBackend, load_omni_backbone_config
from deltaomni.omni_nextqa_joint_cache import _block_count, _encode_batches, _video_blocks
from deltaomni.provenance import audit as audit_provenance
from deltaomni.run_integrity import (
    git_revision,
    git_worktree_is_clean,
    require_media_policy,
    resolved_input_signature,
    sha256_file,
)
from deltaomni.train_sanity import _atomic_json

CACHE_SCHEMA = "deltaomni.omni_msrvtt_video_prefix.v1"
MANIFEST_SCHEMA = "deltaomni.omni_msrvtt_video_manifest.v1"


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
    media_policy: Path
    canonical_manifest: Path
    omni_config: Path
    deltatok_config: Path
    deltatok_checkpoint: Path
    deltatok_sha256: str
    train_count: int
    validation_count: int
    test_count: int
    block_seconds: float
    sample_fps: float
    frame_width: int
    frame_height: int
    expected_video_tokens: int
    encoder_batch_size: int
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
        "media_policy",
        "canonical_manifest",
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
        config.train_count,
        config.validation_count,
        config.test_count,
        config.block_seconds,
        config.sample_fps,
        config.frame_width,
        config.frame_height,
        config.expected_video_tokens,
        config.encoder_batch_size,
        config.runtime.cpu_threads,
    )
    if min(positive) <= 0 or len(config.deltatok_sha256) != 64:
        raise ValueError("Invalid MSR-VTT video cache controls")
    frames_per_block = config.block_seconds * config.sample_fps
    if not frames_per_block.is_integer():
        raise ValueError("MSR-VTT block duration and FPS must form an integer frame grid")
    return config


def _select(config: CacheConfig) -> dict[str, list[CanonicalEpisode]]:
    canonical = read_canonical_dataset(config.canonical_manifest)
    counts = {
        "train": config.train_count,
        "validation": config.validation_count,
        "test": config.test_count,
    }
    result = {}
    for split, count in counts.items():
        episodes = sorted(
            canonical[split],
            key=lambda episode: __import__("hashlib").sha256(
                f"{config.seed}:{episode.source_id}".encode()
            ).hexdigest(),
        )
        if len(episodes) < count:
            raise ValueError(f"Found {len(episodes)}/{count} MSR-VTT {split} clips")
        result[split] = episodes[:count]
    return result


def _cache_path(config: CacheConfig, split: str, source_id: str) -> Path:
    shard = __import__("hashlib").sha256(source_id.encode()).hexdigest()[:2]
    return config.cache_root / split / shard / f"{source_id}.pt"


def _atomic_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _valid(
    path: Path,
    *,
    episode: CanonicalEpisode,
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
    first, deltas = value.get("first_full"), value.get("deltas")
    return bool(
        value.get("schema") == CACHE_SCHEMA
        and value.get("cache_signature") == signature
        and value.get("source_id") == episode.source_id
        and value.get("media_sha256") == episode.media.video.sha256
        and len(value.get("captions", ())) == len(episode.captions.video or ())
        and isinstance(first, torch.Tensor)
        and tuple(first.shape) == (config.expected_video_tokens, full_width)
        and first.dtype == torch.float16
        and torch.isfinite(first).all()
        and isinstance(deltas, torch.Tensor)
        and tuple(deltas.shape) == (blocks - 1, 1, delta_width)
        and deltas.dtype == torch.float16
        and torch.isfinite(deltas).all()
    )


@torch.no_grad()
def run(config_path: Path, provenance_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = config_path.resolve().parent.parent
    if not git_worktree_is_clean(root):
        raise RuntimeError("MSR-VTT cache generation requires a clean source worktree")
    if sha256_file(config.deltatok_checkpoint) != config.deltatok_sha256:
        raise ValueError("MSR-VTT video DeltaTok checksum mismatch")
    provenance = audit_provenance(provenance_path)
    policy_sha = require_media_policy(
        provenance, config.dataset_resource_name, config.media_policy
    )
    delta_config = load_deltatok_config(config.deltatok_config)
    signature = resolved_input_signature(
        config,
        {
            "canonical_manifest": config.canonical_manifest,
            "omni_config": config.omni_config,
            "deltatok_config": config.deltatok_config,
            "deltatok_checkpoint": config.deltatok_checkpoint,
            "media_policy": config.media_policy,
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
        episodes = _select(config)
        flat = [(split, episode) for split, values in episodes.items() for episode in values]
        local = flat[context.rank :: context.world_size]
        backend = QwenOmniThinkerEmbeddingBackend(
            load_omni_backbone_config(config.omni_config), context.device, provenance
        )
        delta_model = DeltaTok(delta_config.model).to(context.device).eval()
        checkpoint = torch.load(config.deltatok_checkpoint, map_location="cpu", weights_only=False)
        delta_model.load_state_dict(checkpoint["model"])
        delta_model.requires_grad_(False)
        started = time.perf_counter()
        for completed, (split, episode) in enumerate(local, 1):
            blocks = _block_count(episode, config)
            path = _cache_path(config, split, episode.source_id)
            if _valid(
                path,
                episode=episode,
                blocks=blocks,
                full_width=backend.output_dim,
                delta_width=delta_config.model.model_dim,
                config=config,
                signature=signature,
            ):
                continue
            features = _encode_batches(
                backend.encode_video_chunks,
                _video_blocks(episode, config),
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
            assert episode.media.video is not None
            _atomic_save(
                path,
                {
                    "schema": CACHE_SCHEMA,
                    "cache_signature": signature,
                    "source_id": episode.source_id,
                    "source_group_id": episode.source_group_id,
                    "split": split,
                    "first_full": features[0].cpu().to(torch.float16),
                    "deltas": torch.stack(deltas),
                    "captions": tuple(caption.text for caption in episode.captions.video or ()),
                    "blocks": len(features),
                    "media_sha256": episode.media.video.sha256,
                },
            )
            if context.is_primary and (completed % 25 == 0 or completed == len(local)):
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * (len(local) - completed)
                print(
                    f"msrvtt_video_rank0={completed}/{len(local)} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
        if context.world_size > 1:
            torch.distributed.barrier()
        report = {}
        if context.is_primary:
            records = {split: [] for split in episodes}
            total_bytes = 0
            for split, values in episodes.items():
                for episode in values:
                    path = _cache_path(config, split, episode.source_id)
                    blocks = _block_count(episode, config)
                    if not _valid(
                        path,
                        episode=episode,
                        blocks=blocks,
                        full_width=backend.output_dim,
                        delta_width=delta_config.model.model_dim,
                        config=config,
                        signature=signature,
                    ):
                        raise ValueError(f"Invalid MSR-VTT video cache: {path}")
                    size = path.stat().st_size
                    total_bytes += size
                    records[split].append(
                        {
                            "source_id": episode.source_id,
                            "source_group_id": episode.source_group_id,
                            "cache_path": str(path),
                            "blocks": blocks,
                            "captions": len(episode.captions.video or ()),
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
                "media_policy_sha256": policy_sha,
                "omni_revision": backend.config.revision,
                "deltatok_sha256": config.deltatok_sha256,
                "full_tokens": config.expected_video_tokens,
                "full_width": backend.output_dim,
                "delta_tokens_per_block": 1,
                "delta_width": delta_config.model.model_dim,
                "splits": records,
            }
            _atomic_json(config.cache_root / "manifest.json", manifest)
            report = {
                "schema": "deltaomni.omni_msrvtt_video_summary.v1",
                "splits": {split: len(values) for split, values in episodes.items()},
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
    parser = argparse.ArgumentParser(description="Cache native MSR-VTT video FULL+delta prefixes")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/omni_msrvtt_video_cache_smoke.yaml")
    )
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    args = parser.parse_args()
    report = run(args.config, args.provenance)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
