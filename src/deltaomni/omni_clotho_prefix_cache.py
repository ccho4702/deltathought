from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml
from transformers.utils import logging as transformers_logging

from deltaomni.data.schema import CanonicalEpisode, iter_jsonl
from deltaomni.deltatok_scale_train import load_config as load_deltatok_config
from deltaomni.deltatok_train import DeltaTok
from deltaomni.distributed import distributed_context
from deltaomni.omni_audiocaps_prefix_cache import CACHE_SCHEMA, _atomic_torch_save
from deltaomni.omni_backbones import (
    QwenOmniThinkerEmbeddingBackend,
    load_omni_backbone_config,
)
from deltaomni.provenance import audit as audit_provenance
from deltaomni.train_sanity import _atomic_json

MANIFEST_SCHEMA = "deltaomni.omni_clotho_prefix_manifest.v1"


@dataclass(frozen=True)
class CacheRuntime:
    device: str
    backend: str
    nccl_compatibility_mode: bool
    cpu_threads: int


@dataclass(frozen=True)
class ClothoPrefixConfig:
    seed: int
    canonical_manifest: Path
    omni_config: Path
    deltatok_config: Path
    deltatok_checkpoint: Path
    deltatok_checkpoint_sha256: str
    train_count: int
    validation_count: int
    test_count: int
    block_seconds: float
    minimum_blocks: int
    maximum_blocks: int
    expected_audio_tokens: int
    runtime: CacheRuntime
    cache_root: Path
    report_path: Path


def load_config(path: Path) -> ClothoPrefixConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    runtime = raw["runtime"]
    config = ClothoPrefixConfig(
        seed=int(raw["seed"]),
        canonical_manifest=resolve(raw["canonical_manifest"]),
        omni_config=resolve(raw["omni_config"]),
        deltatok_config=resolve(raw["deltatok_config"]),
        deltatok_checkpoint=resolve(raw["deltatok_checkpoint"]),
        deltatok_checkpoint_sha256=str(raw["deltatok_checkpoint_sha256"]),
        train_count=int(raw["train_count"]),
        validation_count=int(raw["validation_count"]),
        test_count=int(raw["test_count"]),
        block_seconds=float(raw["block_seconds"]),
        minimum_blocks=int(raw["minimum_blocks"]),
        maximum_blocks=int(raw["maximum_blocks"]),
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
    if (
        min(
            config.train_count,
            config.validation_count,
            config.test_count,
            config.block_seconds,
            config.minimum_blocks,
            config.maximum_blocks,
            config.expected_audio_tokens,
            config.runtime.cpu_threads,
        )
        <= 0
    ):
        raise ValueError("Clotho prefix controls must be positive")
    if config.minimum_blocks > config.maximum_blocks:
        raise ValueError("Clotho prefix block limits are reversed")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _split_path(manifest_path: Path, split: str) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "clotho":
        raise ValueError("Clotho prefix cache requires canonical Clotho")
    return manifest_path.parent / manifest["splits"][split]["path"]


def _rank(seed: int, episode: CanonicalEpisode) -> str:
    return hashlib.sha256(f"{seed}:{episode.source_id}".encode()).hexdigest()


def _block_count(episode: CanonicalEpisode, config: ClothoPrefixConfig) -> int:
    assert episode.duration_seconds is not None
    count = min(
        math.floor((episode.duration_seconds + 1e-6) / config.block_seconds),
        config.maximum_blocks,
    )
    if count < config.minimum_blocks:
        raise ValueError(f"Clotho clip is too short: {episode.episode_id}")
    return count


def select_episodes(config: ClothoPrefixConfig) -> dict[str, list[CanonicalEpisode]]:
    counts = {
        "train": config.train_count,
        "validation": config.validation_count,
        "test": config.test_count,
    }
    selected = {}
    for split, count in counts.items():
        episodes = list(iter_jsonl(_split_path(config.canonical_manifest, split)))
        episodes.sort(key=lambda episode: _rank(config.seed, episode))
        if len(episodes) < count:
            raise ValueError(f"Found {len(episodes)}/{count} Clotho {split} episodes")
        selected[split] = episodes[:count]
    return selected


def _audio_blocks(
    episode: CanonicalEpisode,
    config: ClothoPrefixConfig,
    target_rate: int,
) -> list[np.ndarray]:
    assert episode.media.audio is not None
    waveform, source_rate = sf.read(
        episode.media.audio.path,
        always_2d=True,
        dtype="float32",
    )
    mono = waveform.mean(axis=1)
    if source_rate != target_rate:
        mono = librosa.resample(mono, orig_sr=source_rate, target_sr=target_rate)
    samples_per_block = round(config.block_seconds * target_rate)
    count = _block_count(episode, config)
    required = count * samples_per_block
    if mono.shape[0] < required:
        raise ValueError(f"Clotho waveform shorter than canonical duration: {episode.source_id}")
    return [
        np.ascontiguousarray(mono[start : start + samples_per_block])
        for start in range(0, required, samples_per_block)
    ]


def _cache_path(config: ClothoPrefixConfig, split: str, source_id: str) -> Path:
    return config.cache_root / split / source_id[:2] / f"{source_id}.pt"


def _valid_cache(
    path: Path,
    episode: CanonicalEpisode,
    config: ClothoPrefixConfig,
    omni_revision: str,
    full_width: int,
    delta_width: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (EOFError, OSError, RuntimeError):
        return False
    blocks = _block_count(episode, config)
    full, deltas = payload.get("first_full"), payload.get("deltas")
    return bool(
        payload.get("schema") == CACHE_SCHEMA
        and payload.get("source_id") == episode.source_id
        and payload.get("omni_revision") == omni_revision
        and payload.get("deltatok_checkpoint_sha256") == config.deltatok_checkpoint_sha256
        and len(payload.get("captions", ())) == 5
        and isinstance(full, torch.Tensor)
        and tuple(full.shape) == (config.expected_audio_tokens, full_width)
        and full.dtype == torch.float16
        and torch.isfinite(full).all()
        and isinstance(deltas, torch.Tensor)
        and tuple(deltas.shape) == (blocks - 1, 1, delta_width)
        and deltas.dtype == torch.float16
        and torch.isfinite(deltas).all()
    )


@torch.no_grad()
def run(config_path: Path, provenance_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    transformers_logging.set_verbosity_error()
    torch.set_num_threads(config.runtime.cpu_threads)
    if _sha256(config.deltatok_checkpoint) != config.deltatok_checkpoint_sha256:
        raise ValueError("Audio DeltaTok checkpoint checksum mismatch")
    with distributed_context(
        config.runtime.device,
        config.runtime.backend,
        nccl_compatibility_mode=config.runtime.nccl_compatibility_mode,
    ) as context:
        episodes = select_episodes(config)
        flat = [(split, episode) for split, values in episodes.items() for episode in values]
        local = flat[context.rank :: context.world_size]
        backend = QwenOmniThinkerEmbeddingBackend(
            load_omni_backbone_config(config.omni_config),
            context.device,
            audit_provenance(provenance_path),
        )
        delta_config = load_deltatok_config(config.deltatok_config)
        delta_model = DeltaTok(delta_config.model).to(context.device).eval()
        checkpoint = torch.load(config.deltatok_checkpoint, map_location="cpu", weights_only=False)
        delta_model.load_state_dict(checkpoint["model"])
        delta_model.requires_grad_(False)
        torch.cuda.reset_peak_memory_stats(context.device)
        started = time.perf_counter()
        for completed, (split, episode) in enumerate(local, start=1):
            path = _cache_path(config, split, episode.source_id)
            if _valid_cache(
                path,
                episode,
                config,
                backend.config.revision,
                backend.output_dim,
                delta_config.model.model_dim,
            ):
                continue
            blocks = _audio_blocks(episode, config, backend.config.sample_rate)
            features = []
            metadata = []
            for block in blocks:
                encoded, block_metadata = backend.encode_audio_chunks([block])
                if encoded[0].shape != (config.expected_audio_tokens, backend.output_dim):
                    raise ValueError(f"Unexpected Omni audio shape: {tuple(encoded[0].shape)}")
                features.append(encoded[0])
                metadata.append(block_metadata[0])
            deltas = []
            with torch.autocast(device_type=context.device.type, dtype=torch.bfloat16):
                for step in range(1, len(features)):
                    delta = delta_model.encode(
                        features[step - 1].unsqueeze(0),
                        features[step].unsqueeze(0),
                    )
                    deltas.append(delta[0].cpu().to(torch.float16))
            assert episode.media.audio is not None
            _atomic_torch_save(
                path,
                {
                    "schema": CACHE_SCHEMA,
                    "source_id": episode.source_id,
                    "source_group_id": episode.source_group_id,
                    "split": split,
                    "first_full": features[0].cpu().to(torch.float16),
                    "deltas": torch.stack(deltas),
                    "captions": tuple(c.text for c in episode.captions.audio or ()),
                    "media_sha256": episode.media.audio.sha256,
                    "encoder_metadata": metadata,
                    "omni_revision": backend.config.revision,
                    "deltatok_checkpoint_sha256": config.deltatok_checkpoint_sha256,
                },
            )
            if context.is_primary and (completed % 25 == 0 or completed == len(local)):
                elapsed = time.perf_counter() - started
                eta = elapsed / completed * (len(local) - completed)
                print(
                    f"clotho_prefix_rank0={completed}/{len(local)} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
        if context.world_size > 1:
            torch.distributed.barrier()
        local_peak = torch.tensor(
            float(torch.cuda.max_memory_reserved(context.device)), device=context.device
        )
        if context.world_size > 1:
            torch.distributed.all_reduce(local_peak, op=torch.distributed.ReduceOp.MAX)
        report = {}
        if context.is_primary:
            code_revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=config_path.resolve().parent.parent,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            manifest_splits = {}
            total_bytes = 0
            horizons: Counter[int] = Counter()
            print("validating_clotho_prefix_cache", flush=True)
            for split, values in episodes.items():
                records = []
                for episode in values:
                    path = _cache_path(config, split, episode.source_id)
                    if not _valid_cache(
                        path,
                        episode,
                        config,
                        backend.config.revision,
                        backend.output_dim,
                        delta_config.model.model_dim,
                    ):
                        raise ValueError(f"Invalid Clotho prefix cache: {path}")
                    size = path.stat().st_size
                    updates = _block_count(episode, config) - 1
                    horizons[updates] += 1
                    total_bytes += size
                    records.append(
                        {
                            "source_id": episode.source_id,
                            "source_group_id": episode.source_group_id,
                            "cache_path": str(path),
                            "captions": 5,
                            "delta_updates": updates,
                            "bytes": size,
                            "sha256": _sha256(path),
                        }
                    )
                manifest_splits[split] = records
            manifest = {
                "schema": MANIFEST_SCHEMA,
                "code_revision": code_revision,
                "canonical_manifest": str(config.canonical_manifest),
                "canonical_manifest_sha256": _sha256(config.canonical_manifest),
                "omni_revision": backend.config.revision,
                "deltatok_checkpoint_sha256": config.deltatok_checkpoint_sha256,
                "full_tokens": config.expected_audio_tokens,
                "full_width": backend.output_dim,
                "delta_tokens_per_block": 1,
                "delta_width": delta_config.model.model_dim,
                "minimum_delta_updates": config.minimum_blocks - 1,
                "maximum_delta_updates": config.maximum_blocks - 1,
                "splits": manifest_splits,
            }
            _atomic_json(config.cache_root / "manifest.json", manifest)
            report = {
                "schema": "deltaomni.omni_clotho_prefix_summary.v1",
                "code_revision": code_revision,
                "splits": {split: len(values) for split, values in episodes.items()},
                "delta_update_histogram": dict(sorted(horizons.items())),
                "total_bytes": total_bytes,
                "world_size": context.world_size,
                "peak_reserved_bytes_per_rank_max": int(local_peak.item()),
                "elapsed_seconds": time.perf_counter() - started,
                "completed_at_utc": datetime.now(UTC).isoformat(),
            }
            _atomic_json(config.report_path, report)
        if context.world_size > 1:
            torch.distributed.barrier()
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache variable-horizon Clotho delta prefixes")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/omni_clotho_prefix_cache.yaml")
    )
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    args = parser.parse_args()
    report = run(args.config, args.provenance)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
