from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import av
import yaml

from deltaomni.data.canonicalize import read_canonical_dataset, write_canonical_dataset
from deltaomni.data.nextqa import build_episode
from deltaomni.data.schema import MediaAsset, ProvenanceRecord
from deltaomni.provenance import audit as audit_provenance
from deltaomni.train_sanity import _atomic_json


@dataclass(frozen=True)
class NextQACanonicalConfig:
    dataset: str
    dataset_revision: str
    resource_name: str
    chunk_seconds: float
    cpu_workers: int
    annotations: dict[str, Path]
    video_mapping: Path
    media_root: Path
    license_record: Path
    cache_root: Path
    output_root: Path
    report_path: Path


def load_config(path: Path) -> NextQACanonicalConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    config = NextQACanonicalConfig(
        dataset=str(raw["dataset"]),
        dataset_revision=str(raw["dataset_revision"]),
        resource_name=str(raw["resource_name"]),
        chunk_seconds=float(raw["chunk_seconds"]),
        cpu_workers=int(raw["cpu_workers"]),
        annotations={name: resolve(value) for name, value in raw["annotations"].items()},
        video_mapping=resolve(raw["video_mapping"]),
        media_root=resolve(raw["media_root"]),
        license_record=resolve(raw["license_record"]),
        cache_root=resolve(raw["cache_root"]),
        output_root=resolve(raw["output_root"]),
        report_path=resolve(raw["report_path"]),
    )
    if config.dataset != "nextqa":
        raise ValueError("NExT-QA preprocessor requires dataset=nextqa")
    if set(config.annotations) != {"train", "validation", "test"}:
        raise ValueError("NExT-QA annotations require train/validation/test paths")
    if config.chunk_seconds <= 0 or config.cpu_workers <= 0:
        raise ValueError("NExT-QA preprocessing controls must be positive")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_revision(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _atomic_cache(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _inspect_media(source_id: str, path: Path, cache_root: Path) -> dict[str, Any]:
    stat = path.stat()
    cache_path = cache_root / "media" / f"{source_id}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached["bytes"] == stat.st_size and cached["mtime_ns"] == stat.st_mtime_ns:
            return cached
    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"NExT-QA media has no video stream: {path}")
        video = container.streams.video[0]
        duration = (
            float(container.duration / av.time_base)
            if container.duration is not None
            else float(video.duration * video.time_base)
        )
        video_codec = video.codec_context
        audio = container.streams.audio[0] if container.streams.audio else None
        audio_codec = audio.codec_context if audio is not None else None
        value = {
            "source_id": source_id,
            "path": str(path),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256(path),
            "duration_seconds": duration,
            "video": {
                "width": int(video_codec.width),
                "height": int(video_codec.height),
                "fps": (
                    float(video.average_rate) if video.average_rate is not None else None
                ),
            },
            "audio": (
                None
                if audio_codec is None
                else {
                    "sample_rate": int(audio_codec.sample_rate),
                    "channels": int(audio_codec.channels),
                }
            ),
        }
    if duration <= 0:
        raise ValueError(f"NExT-QA media has invalid duration: {path}")
    _atomic_cache(cache_path, value)
    return value


def _load_rows(config: NextQACanonicalConfig) -> dict[str, dict[str, list[dict[str, str]]]]:
    result = {}
    for split, path in config.annotations.items():
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(row["video"], []).append(row)
        result[split] = grouped
    return result


def _source_files(config: NextQACanonicalConfig) -> list[dict[str, Any]]:
    paths = [*config.annotations.values(), config.video_mapping]
    return [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in paths
    ]


def run(config_path: Path, provenance_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    project_root = config_path.resolve().parent.parent.parent
    code_revision = _code_revision(project_root)
    config_sha = _config_sha256(config_path)
    final_manifest = config.output_root / config.dataset / config.dataset_revision / "manifest.json"
    if final_manifest.is_file():
        loaded = read_canonical_dataset(final_manifest)
        report = {
            "schema": "deltaomni.canonical_preprocessing_summary.v1",
            "dataset": config.dataset,
            "dataset_revision": config.dataset_revision,
            "status": "already_complete",
            "manifest": str(final_manifest),
            "episodes": {split: len(values) for split, values in loaded.items()},
            "code_revision": code_revision,
        }
        _atomic_json(config.report_path, report)
        return report

    provenance = audit_provenance(provenance_path)
    if config.resource_name not in provenance["approved"]:
        raise ValueError("NExT-QA failed provenance gate")
    rows_by_split = _load_rows(config)
    mapping = json.loads(config.video_mapping.read_text(encoding="utf-8"))
    media_jobs = []
    for grouped in rows_by_split.values():
        for source_id in grouped:
            relative = mapping[source_id]
            path = config.media_root / f"{relative}.mp4"
            if not path.is_file():
                raise FileNotFoundError(path)
            media_jobs.append((source_id, path))
    media_jobs.sort()

    started = time.perf_counter()
    media_index = {}
    with ThreadPoolExecutor(max_workers=config.cpu_workers) as executor:
        futures = [
            executor.submit(_inspect_media, source_id, path, config.cache_root)
            for source_id, path in media_jobs
        ]
        for index, future in enumerate(futures, start=1):
            value = future.result()
            media_index[value["source_id"]] = value
            if index % 100 == 0 or index == len(futures):
                elapsed = time.perf_counter() - started
                eta = elapsed / index * (len(futures) - index)
                print(
                    f"nextqa_media={index}/{len(futures)} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )

    processed_at = datetime.now(UTC).isoformat()
    episodes_by_split = {}
    annotation_hashes = {
        split: _sha256(path) for split, path in config.annotations.items()
    }
    for split, grouped in rows_by_split.items():
        episodes = []
        for source_id, qa_rows in sorted(grouped.items()):
            media = media_index[source_id]
            video = MediaAsset(
                path=Path(media["path"]),
                sha256=media["sha256"],
                duration_seconds=media["duration_seconds"],
                mime_type="video/mp4",
                width=media["video"]["width"],
                height=media["video"]["height"],
                fps=media["video"]["fps"],
            )
            audio = (
                None
                if media["audio"] is None
                else MediaAsset(
                    path=Path(media["path"]),
                    sha256=media["sha256"],
                    duration_seconds=media["duration_seconds"],
                    mime_type="audio/mp4-container",
                    sample_rate=media["audio"]["sample_rate"],
                    channels=media["audio"]["channels"],
                )
            )
            episode = build_episode(
                source_id,
                qa_rows,
                video,
                split=split,
                dataset_revision=config.dataset_revision,
                chunk_seconds=config.chunk_seconds,
                provenance_report=provenance,
                audio_media=audio,
            )
            episode = replace(
                episode,
                source_group_id=f"yfcc100m:{source_id}",
                provenance=ProvenanceRecord(
                    resource_name=config.resource_name,
                    license_record=config.license_record,
                    annotation_path=config.annotations[split],
                    annotation_sha256=annotation_hashes[split],
                    preprocessing_config_sha256=config_sha,
                    code_revision=code_revision,
                    processed_at_utc=processed_at,
                ),
                metadata={
                    "mapped_media_id": mapping[source_id],
                    "annotation_frame_count": int(qa_rows[0]["frame_count"]),
                },
            )
            episode.validate_for_independent_qa()
            episodes.append(episode)
        episodes_by_split[split] = episodes

    manifest = write_canonical_dataset(
        config.output_root,
        config.dataset,
        config.dataset_revision,
        episodes_by_split,
        preprocessing_config_sha256=config_sha,
        code_revision=code_revision,
        source_files=_source_files(config),
    )
    loaded = read_canonical_dataset(manifest)
    report = {
        "schema": "deltaomni.canonical_preprocessing_summary.v1",
        "dataset": config.dataset,
        "dataset_revision": config.dataset_revision,
        "status": "complete",
        "manifest": str(manifest),
        "episodes": {split: len(values) for split, values in loaded.items()},
        "qa_items": {
            split: sum(len(episode.qa or ()) for episode in values)
            for split, values in loaded.items()
        },
        "media_with_audio": {
            split: sum(episode.media.audio is not None for episode in values)
            for split, values in loaded.items()
        },
        "code_revision": code_revision,
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(config.report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess NExT-QA into canonical episode v2")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/canonical/nextqa.yaml"),
    )
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    args = parser.parse_args()
    report = run(args.config, args.provenance)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
