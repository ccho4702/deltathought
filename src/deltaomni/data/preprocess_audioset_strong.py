from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from deltaomni.data.audioset_strong import build_episode, inspect_tsv
from deltaomni.data.canonicalize import read_canonical_dataset, write_canonical_dataset
from deltaomni.data.media import inspect_audio_media
from deltaomni.data.schema import MediaAsset, ProvenanceRecord
from deltaomni.provenance import audit as audit_provenance
from deltaomni.train_sanity import _atomic_json


@dataclass(frozen=True)
class AudioSetCanonicalConfig:
    dataset: str
    dataset_revision: str
    resource_name: str
    chunk_seconds: float
    nominal_duration_seconds: float
    cpu_workers: int
    annotations: dict[str, Path]
    labels: Path
    media_roots: dict[str, tuple[Path, ...]]
    license_record: Path | None
    cache_root: Path
    output_root: Path
    report_path: Path


def load_config(path: Path) -> AudioSetCanonicalConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    config = AudioSetCanonicalConfig(
        dataset=str(raw["dataset"]),
        dataset_revision=str(raw["dataset_revision"]),
        resource_name=str(raw["resource_name"]),
        chunk_seconds=float(raw["chunk_seconds"]),
        nominal_duration_seconds=float(raw["nominal_duration_seconds"]),
        cpu_workers=int(raw["cpu_workers"]),
        annotations={name: resolve(value) for name, value in raw["annotations"].items()},
        labels=resolve(raw["labels"]),
        media_roots={
            name: tuple(resolve(value) for value in values)
            for name, values in raw["media_roots"].items()
        },
        license_record=(
            None if raw.get("license_record") is None else resolve(raw["license_record"])
        ),
        cache_root=resolve(raw["cache_root"]),
        output_root=resolve(raw["output_root"]),
        report_path=resolve(raw["report_path"]),
    )
    if config.dataset != "audioset_strong":
        raise ValueError("AudioSet preprocessor requires dataset=audioset_strong")
    if set(config.annotations) != {"train", "validation"}:
        raise ValueError("AudioSet annotations require train/validation paths")
    if set(config.media_roots) != {"train", "validation"}:
        raise ValueError("AudioSet media roots require train/validation")
    if min(config.chunk_seconds, config.nominal_duration_seconds, config.cpu_workers) <= 0:
        raise ValueError("AudioSet preprocessing controls must be positive")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _audio_filename(clip_id: str) -> str:
    _, start_text = clip_id.rsplit("_", 1)
    return f"{clip_id}_{int(start_text) + 10_000}.flac"


def _cache_path(config: AudioSetCanonicalConfig, clip_id: str) -> Path:
    shard = hashlib.sha256(clip_id.encode()).hexdigest()[:2]
    return config.cache_root / "media" / shard / f"{clip_id}.json"


def _find_and_inspect(
    config: AudioSetCanonicalConfig,
    split: str,
    clip_id: str,
) -> dict[str, Any] | None:
    filename = _audio_filename(clip_id)
    for root in config.media_roots[split]:
        path = root / filename
        if path.is_file():
            return inspect_audio_media(clip_id, path, _cache_path(config, clip_id))
    return None


def _inspect_all_media(
    config: AudioSetCanonicalConfig,
    jobs: list[tuple[str, str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    started = time.perf_counter()
    result = {}
    missing = {"train": [], "validation": []}
    iterator = iter(jobs)
    pending: dict[Future[dict[str, Any] | None], tuple[str, str]] = {}

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            split, clip_id = next(iterator)
        except StopIteration:
            return False
        pending[executor.submit(_find_and_inspect, config, split, clip_id)] = (split, clip_id)
        return True

    completed = 0
    with ThreadPoolExecutor(max_workers=config.cpu_workers) as executor:
        for _ in range(config.cpu_workers * 2):
            if not submit_next(executor):
                break
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                split, clip_id = pending.pop(future)
                value = future.result()
                if value is None:
                    missing[split].append(clip_id)
                else:
                    result[clip_id] = value
                completed += 1
                submit_next(executor)
                if completed % 1000 == 0 or completed == len(jobs):
                    elapsed = time.perf_counter() - started
                    eta = elapsed / completed * (len(jobs) - completed)
                    print(
                        f"audioset_media={completed}/{len(jobs)} "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )
    return result, missing


def _exclusion_record(values: list[str]) -> dict[str, Any]:
    ordered = sorted(values)
    digest = hashlib.sha256("\n".join(ordered).encode()).hexdigest()
    return {
        "reason": "media_not_present_in_existing_read_only_roots",
        "count": len(ordered),
        "source_ids_sha256": digest,
        "examples": ordered[:20],
    }


def _invalid_event_record(values: tuple[Any, ...]) -> dict[str, Any]:
    serialized = [
        (
            f"{value.line_number}:{value.event.clip_id}:{value.event.start_seconds}:"
            f"{value.event.end_seconds}:{value.event.class_id}:{value.reason}"
        )
        for value in values
    ]
    return {
        "reason": "invalid_strong_annotation_quarantined",
        "count": len(serialized),
        "rows_sha256": hashlib.sha256("\n".join(serialized).encode()).hexdigest(),
        "examples": serialized[:20],
    }


def run(config_path: Path, provenance_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    project_root = config_path.resolve().parent.parent.parent
    code_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    final_manifest = config.output_root / config.dataset / config.dataset_revision / "manifest.json"
    if final_manifest.is_file():
        loaded = read_canonical_dataset(final_manifest)
        report = {
            "schema": "deltaomni.canonical_preprocessing_summary.v1",
            "dataset": config.dataset,
            "dataset_revision": config.dataset_revision,
            "status": "already_complete",
            "manifest": str(final_manifest),
            "episodes": {split: len(items) for split, items in loaded.items()},
            "code_revision": code_revision,
            "license_record_present": (
                config.license_record is not None and config.license_record.is_file()
            ),
        }
        _atomic_json(config.report_path, report)
        return report

    provenance = audit_provenance(provenance_path)
    if config.resource_name not in provenance["approved"]:
        raise ValueError("AudioSet Strong failed provenance gate")
    inspected = {split: inspect_tsv(path) for split, path in config.annotations.items()}
    labels = {
        fields[0]: fields[1]
        for line in config.labels.read_text(encoding="utf-8").splitlines()
        if len(fields := line.split("\t", 1)) == 2
    }
    jobs = sorted(
        (split, clip_id)
        for split, strong_file in inspected.items()
        for clip_id in strong_file.events
    )
    started = time.perf_counter()
    media_index, missing = _inspect_all_media(config, jobs)
    processed_at = datetime.now(UTC).isoformat()
    annotation_hashes = {
        split: _sha256(path) for split, path in config.annotations.items()
    }
    episodes_by_split = {}
    for split, strong_file in inspected.items():
        episodes = []
        for clip_id, events in sorted(strong_file.events.items()):
            media = media_index.get(clip_id)
            if media is None:
                continue
            audio = MediaAsset(
                path=Path(media["path"]),
                sha256=media["sha256"],
                duration_seconds=media["duration_seconds"],
                mime_type="audio/flac",
                sample_rate=media["audio"]["sample_rate"],
                channels=media["audio"]["channels"],
            )
            timeline_duration = max(
                config.nominal_duration_seconds,
                audio.duration_seconds or 0.0,
                max(event.end_seconds for event in events),
            )
            episode = build_episode(
                clip_id,
                events,
                labels,
                audio,
                split=split,
                dataset_revision=config.dataset_revision,
                chunk_seconds=config.chunk_seconds,
                provenance_report=provenance,
                timeline_duration_seconds=timeline_duration,
            )
            episode = replace(
                episode,
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
                    "media_format": media["audio"]["format"],
                    "media_subtype": media["audio"]["subtype"],
                    "nominal_duration_seconds": config.nominal_duration_seconds,
                },
            )
            episode.validate_for_caption_training()
            episodes.append(episode)
        episodes_by_split[split] = episodes
    episodes_by_split["test"] = []

    source_paths = [*config.annotations.values(), config.labels]
    exclusions = {
        "missing_media": {
            split: _exclusion_record(values) for split, values in missing.items()
        },
        "invalid_events": {
            split: _invalid_event_record(strong_file.invalid_events)
            for split, strong_file in inspected.items()
        },
    }
    manifest = write_canonical_dataset(
        config.output_root,
        config.dataset,
        config.dataset_revision,
        episodes_by_split,
        preprocessing_config_sha256=config_sha,
        code_revision=code_revision,
        source_files=[
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in source_paths
        ],
        exclusions=exclusions,
    )
    loaded = read_canonical_dataset(manifest)
    report = {
        "schema": "deltaomni.canonical_preprocessing_summary.v1",
        "dataset": config.dataset,
        "dataset_revision": config.dataset_revision,
        "status": "complete",
        "manifest": str(manifest),
        "episodes": {split: len(items) for split, items in loaded.items()},
        "caption_items": {
            split: sum(len(episode.captions.audio or ()) for episode in items)
            for split, items in loaded.items()
        },
        "event_items": {
            split: sum(len(episode.events or ()) for episode in items)
            for split, items in loaded.items()
        },
        "missing_media": {split: len(values) for split, values in missing.items()},
        "invalid_events": {
            split: len(strong_file.invalid_events)
            for split, strong_file in inspected.items()
        },
        "code_revision": code_revision,
        "license_record_present": (
            config.license_record is not None and config.license_record.is_file()
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(config.report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess AudioSet Strong into canonical episode v2"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/canonical/audioset_strong.yaml"),
    )
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    args = parser.parse_args()
    report = run(args.config, args.provenance)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
