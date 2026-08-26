from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from deltaomni.backbones import DinoV2EmbeddingBackend, load_backbone_config
from deltaomni.model import ModalityDeltaCodec
from deltaomni.provenance import audit as audit_provenance
from deltaomni.ssv2_pilot import (
    _decode_uniform_frames,
    _evaluate_reconstruction,
    _retrieval_r1,
    load_pilot_config,
)
from deltaomni.train_sanity import _atomic_json


@dataclass(frozen=True)
class NextQAPilotConfig:
    seed: int
    device: str
    validation_clips: int
    frames_per_clip: int
    embedding_batch_size: int
    annotations: Path
    video_mapping: Path
    media_root: Path
    ssv2_config: Path
    cache_root: Path
    output_root: Path


def load_config(path: Path) -> NextQAPilotConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    return NextQAPilotConfig(
        seed=int(raw["seed"]),
        device=str(raw["device"]),
        validation_clips=int(raw["validation_clips"]),
        frames_per_clip=int(raw["frames_per_clip"]),
        embedding_batch_size=int(raw["embedding_batch_size"]),
        annotations=resolve(raw["annotations"]),
        video_mapping=resolve(raw["video_mapping"]),
        media_root=resolve(raw["media_root"]),
        ssv2_config=resolve(raw["ssv2_config"]),
        cache_root=resolve(raw["cache_root"]),
        output_root=resolve(raw["output_root"]),
    )


def _selected_videos(config: NextQAPilotConfig) -> list[dict[str, Any]]:
    with config.annotations.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    mapping = json.loads(config.video_mapping.read_text(encoding="utf-8"))
    by_video: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_video.setdefault(row["video"], []).append(row)
    candidates = sorted(
        by_video,
        key=lambda video: hashlib.sha256(f"{config.seed}:{video}".encode()).hexdigest(),
    )
    selected = []
    for video in candidates:
        relative = mapping.get(video)
        if relative is None:
            continue
        media_path = config.media_root / f"{relative}.mp4"
        if not media_path.is_file():
            continue
        selected.append(
            {
                "video": video,
                "relative_media": relative,
                "media_path": media_path,
                "qa_count": len(by_video[video]),
                "question_ids": [row["qid"] for row in by_video[video]],
            }
        )
        if len(selected) == config.validation_clips:
            break
    if len(selected) != config.validation_clips:
        raise ValueError(f"Found {len(selected)}/{config.validation_clips} NExT-QA videos")
    return selected


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def prepare(
    config: NextQAPilotConfig,
    backbone_config_path: Path,
    provenance_path: Path,
) -> dict[str, Any]:
    selected = _selected_videos(config)
    pending = []
    for record in selected:
        cache_path = config.cache_root / f"{record['video']}.pt"
        if cache_path.is_file():
            continue
        frames, media_info = _decode_uniform_frames(record["media_path"], config.frames_per_clip)
        pending.append((record, frames, media_info))
    if pending:
        backbone_config = load_backbone_config(backbone_config_path)
        backend = DinoV2EmbeddingBackend(
            backbone_config.video,
            backbone_config.cache_dir,
            torch.device(config.device),
            audit_provenance(provenance_path),
        )
        frames = [frame for _, record_frames, _ in pending for frame in record_frames]
        encoded_batches = []
        for start in range(0, len(frames), config.embedding_batch_size):
            batch = frames[start : start + config.embedding_batch_size]
            encoded_batches.append(backend.encode(batch).cpu())
            done = min(start + len(batch), len(frames))
            print(f"encode_nextqa_frames={done}/{len(frames)}", flush=True)
        encoded = torch.cat(encoded_batches)
        offset = 0
        for record, record_frames, media_info in pending:
            count = len(record_frames)
            embeddings = encoded[offset : offset + count]
            offset += count
            _atomic_torch_save(
                config.cache_root / f"{record['video']}.pt",
                {
                    "schema": "deltaomni.nextqa_embedding.v1",
                    "video": record["video"],
                    "media_path": str(record["media_path"]),
                    "media": media_info,
                    "embeddings": embeddings.to(torch.float16),
                },
            )
    manifest = {
        "schema": "deltaomni.nextqa_reconstruction_manifest.v1",
        "records": [
            {
                "video": record["video"],
                "cache_path": str(config.cache_root / f"{record['video']}.pt"),
                "qa_count": record["qa_count"],
                "question_ids": record["question_ids"],
            }
            for record in selected
        ],
    }
    _atomic_json(config.cache_root / "manifest.json", manifest)
    return manifest


def _load_embeddings(manifest: dict[str, Any], device: torch.device) -> torch.Tensor:
    embeddings = []
    for record in manifest["records"]:
        payload = torch.load(record["cache_path"], map_location="cpu", weights_only=False)
        embeddings.append(payload["embeddings"].float())
    return torch.stack(embeddings).to(device)


def _delta_checkpoint(ssv2_output_root: Path) -> tuple[str, Path]:
    summary = json.loads((ssv2_output_root / "latest_summary.json").read_text(encoding="utf-8"))
    run_dir = ssv2_output_root / summary["run_id"]
    checkpoints = sorted(run_dir.glob("checkpoints/step-*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No SSV2 delta checkpoint under {run_dir}")
    return summary["run_id"], checkpoints[-1]


def run(
    config_path: Path,
    backbone_config_path: Path,
    provenance_path: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    ssv2_config = load_pilot_config(config.ssv2_config)
    manifest = prepare(config, backbone_config_path, provenance_path)
    torch.cuda.empty_cache()
    device = torch.device(config.device)
    embeddings = _load_embeddings(manifest, device)
    source_run, checkpoint = _delta_checkpoint(ssv2_config.output_root)
    codec = ModalityDeltaCodec(ssv2_config.model).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    codec.load_state_dict(payload["model"])
    evaluation = _evaluate_reconstruction(codec, embeddings)
    anchor = embeddings[:, :1].expand_as(embeddings)
    metrics = {
        "validation_mse": evaluation["mse"],
        "anchor_mse": evaluation["anchor_mse"],
        "last_delta_mse": evaluation["last_delta_mse"],
        "shuffled_delta_mse": evaluation["shuffled_delta_mse"],
        "raw_pooled_delta_mse": evaluation["raw_pooled_delta_mse"],
        "retrieval_r1": _retrieval_r1(evaluation["reconstructed"], embeddings),
        "anchor_retrieval_r1": _retrieval_r1(anchor, embeddings),
        "videos": len(manifest["records"]),
        "questions": sum(record["qa_count"] for record in manifest["records"]),
    }
    checks = {
        "learned_beats_anchor": metrics["validation_mse"] < metrics["anchor_mse"],
        "learned_beats_last": metrics["validation_mse"] < metrics["last_delta_mse"],
        "learned_beats_shuffled": metrics["validation_mse"] < metrics["shuffled_delta_mse"],
        "retrieval_not_worse": metrics["retrieval_r1"] >= metrics["anchor_retrieval_r1"],
    }
    passed = all(checks.values())
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"nextqa-reconstruction-{timestamp}-{uuid.uuid4().hex[:8]}"
    report = {
        "run_id": run_id,
        "source_delta_run_id": source_run,
        "status": "signal" if passed else "inconclusive",
        "metrics": metrics,
        "checks": checks,
        "passed": passed,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    run_dir = config.output_root / run_id
    _atomic_json(run_dir / "summary.json", report)
    _atomic_json(config.output_root / "latest_summary.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Zero-shot NExT-QA delta reconstruction pilot")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/nextqa_reconstruction_pilot.yaml")
    )
    parser.add_argument("--backbones", type=Path, default=Path("configs/backbones.yaml"))
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    args = parser.parse_args()
    report = run(args.config, args.backbones, args.provenance)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
