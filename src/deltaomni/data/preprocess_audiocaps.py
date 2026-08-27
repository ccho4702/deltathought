from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from deltaomni.data.canonicalize import read_canonical_dataset, write_canonical_dataset
from deltaomni.data.media import inspect_audio_media
from deltaomni.data.schema import (
    CanonicalEpisode,
    CaptionAnnotation,
    CaptionBundle,
    MediaAsset,
    MediaBundle,
    ProvenanceRecord,
    TextBundle,
    temporal_grid,
)
from deltaomni.provenance import audit as audit_provenance
from deltaomni.train_sanity import _atomic_json


@dataclass(frozen=True)
class AudioCapsConfig:
    dataset: str
    dataset_revision: str
    resource_name: str
    chunk_seconds: float
    nominal_duration_seconds: float
    cpu_workers: int
    annotations: dict[str, Path]
    media_roots: tuple[Path, ...]
    license_record: Path
    cache_root: Path
    output_root: Path
    report_path: Path


@dataclass(frozen=True)
class AudioCapsRow:
    caption_id: str
    youtube_id: str
    start_time: int
    caption: str

    @property
    def source_id(self) -> str:
        start_ms = self.start_time * 1000
        return f"{self.youtube_id}_{start_ms}_{start_ms + 10_000}"


def load_config(path: Path) -> AudioCapsConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    config = AudioCapsConfig(
        dataset=str(raw["dataset"]),
        dataset_revision=str(raw["dataset_revision"]),
        resource_name=str(raw["resource_name"]),
        chunk_seconds=float(raw["chunk_seconds"]),
        nominal_duration_seconds=float(raw["nominal_duration_seconds"]),
        cpu_workers=int(raw["cpu_workers"]),
        annotations={split: resolve(value) for split, value in raw["annotations"].items()},
        media_roots=tuple(resolve(value) for value in raw["media_roots"]),
        license_record=resolve(raw["license_record"]),
        cache_root=resolve(raw["cache_root"]),
        output_root=resolve(raw["output_root"]),
        report_path=resolve(raw["report_path"]),
    )
    if config.dataset != "audiocaps":
        raise ValueError("AudioCaps preprocessor requires dataset=audiocaps")
    if set(config.annotations) != {"train", "validation", "test"}:
        raise ValueError("AudioCaps requires official train/validation/test annotations")
    if (
        min(
            config.chunk_seconds,
            config.nominal_duration_seconds,
            config.cpu_workers,
            len(config.media_roots),
        )
        <= 0
    ):
        raise ValueError("AudioCaps preprocessing controls must be positive")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[AudioCapsRow]:
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected = {"audiocap_id", "youtube_id", "start_time", "caption"}
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(f"Unexpected AudioCaps columns: {path}")
        for line_number, raw in enumerate(reader, start=2):
            caption = raw["caption"].strip()
            youtube_id = raw["youtube_id"].strip()
            try:
                start_time = int(raw["start_time"])
            except ValueError as error:
                raise ValueError(f"Invalid start_time at {path}:{line_number}") from error
            if not raw["audiocap_id"].strip() or not youtube_id or not caption or start_time < 0:
                raise ValueError(f"Invalid AudioCaps row at {path}:{line_number}")
            rows.append(
                AudioCapsRow(
                    caption_id=raw["audiocap_id"].strip(),
                    youtube_id=youtube_id,
                    start_time=start_time,
                    caption=caption,
                )
            )
    if len({row.caption_id for row in rows}) != len(rows):
        raise ValueError(f"Duplicate AudioCaps caption ID: {path}")
    return rows


def _group_rows(rows: list[AudioCapsRow]) -> dict[str, list[AudioCapsRow]]:
    result: dict[str, list[AudioCapsRow]] = {}
    for row in rows:
        result.setdefault(row.source_id, []).append(row)
    for values in result.values():
        first = values[0]
        if any(
            value.youtube_id != first.youtube_id or value.start_time != first.start_time
            for value in values
        ):
            raise ValueError(f"AudioCaps source collision: {first.source_id}")
    return result


def _find_media(config: AudioCapsConfig, source_id: str) -> Path | None:
    filename = f"{source_id}.flac"
    for root in config.media_roots:
        candidate = root / filename
        if candidate.is_file():
            return candidate
    return None


def _cache_path(config: AudioCapsConfig, source_id: str) -> Path:
    shard = hashlib.sha256(source_id.encode()).hexdigest()[:2]
    return config.cache_root / "media" / shard / f"{source_id}.json"


def _inspect(
    config: AudioCapsConfig,
    job: tuple[str, str],
) -> tuple[str, str, dict[str, Any] | None]:
    split, source_id = job
    path = _find_media(config, source_id)
    if path is None:
        return split, source_id, None
    return split, source_id, inspect_audio_media(source_id, path, _cache_path(config, source_id))


def _attrition(values: list[str]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "reason": "official_caption_clip_missing_from_existing_read_only_audioset_media",
        "count": len(ordered),
        "source_ids_sha256": hashlib.sha256("\n".join(ordered).encode()).hexdigest(),
        "examples": ordered[:20],
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
        raise ValueError("AudioCaps failed provenance gate")
    if not config.license_record.is_file():
        raise FileNotFoundError(f"AudioCaps license record is missing: {config.license_record}")
    rows = {split: _read_rows(path) for split, path in config.annotations.items()}
    grouped = {split: _group_rows(values) for split, values in rows.items()}
    jobs = [
        (split, source_id) for split, sources in grouped.items() for source_id in sorted(sources)
    ]
    started = time.perf_counter()
    inspected: dict[tuple[str, str], dict[str, Any]] = {}
    missing = {"train": [], "validation": [], "test": []}
    with ThreadPoolExecutor(max_workers=config.cpu_workers) as executor:
        iterator = executor.map(lambda job: _inspect(config, job), jobs)
        for index, (split, source_id, media) in enumerate(iterator, start=1):
            if media is None:
                missing[split].append(source_id)
            else:
                inspected[(split, source_id)] = media
            if index % 1000 == 0 or index == len(jobs):
                elapsed = time.perf_counter() - started
                eta = elapsed / index * (len(jobs) - index)
                print(
                    f"audiocaps_media={index}/{len(jobs)} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )

    config_sha256 = _sha256(config_path)
    annotation_hashes = {split: _sha256(path) for split, path in config.annotations.items()}
    processed_at = datetime.now(UTC).isoformat()
    episodes: dict[str, list[CanonicalEpisode]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for split, sources in grouped.items():
        for source_id, captions in sorted(sources.items()):
            media = inspected.get((split, source_id))
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
            assert audio.duration_seconds is not None
            caption_end = min(config.nominal_duration_seconds, audio.duration_seconds)
            episode = CanonicalEpisode(
                episode_id=f"audiocaps:{split}:{source_id}",
                dataset=config.dataset,
                dataset_revision=config.dataset_revision,
                split=split,
                source_id=source_id,
                source_group_id=f"youtube:{captions[0].youtube_id}",
                media=MediaBundle(image=None, video=None, audio=audio),
                duration_seconds=audio.duration_seconds,
                temporal_blocks=temporal_grid(audio.duration_seconds, config.chunk_seconds),
                captions=CaptionBundle(
                    image=None,
                    video=None,
                    audio=tuple(
                        CaptionAnnotation(
                            caption_id=f"audiocaps:{caption.caption_id}",
                            scope="audio",
                            text=caption.caption,
                            start_seconds=0.0,
                            end_seconds=caption_end,
                            commit_seconds=caption_end,
                            language="en",
                            annotation_origin="official_human_crowdsourced",
                            timing_origin="clip_level_10_seconds",
                            independent_from_qa=True,
                        )
                        for caption in captions
                    ),
                    joint=None,
                ),
                text=TextBundle(transcript=None, subtitle=None, ocr=None),
                events=None,
                qa=None,
                provenance=ProvenanceRecord(
                    resource_name=config.resource_name,
                    source_url="https://github.com/cdjkim/audiocaps",
                    license_record=config.license_record,
                    annotation_path=config.annotations[split],
                    annotation_sha256=annotation_hashes[split],
                    preprocessing_config_sha256=config_sha256,
                    code_revision=code_revision,
                    processed_at_utc=processed_at,
                ),
                metadata={
                    "youtube_id": captions[0].youtube_id,
                    "start_time_seconds": captions[0].start_time,
                    "reference_captions": len(captions),
                    "media_format": media["audio"]["format"],
                    "media_subtype": media["audio"]["subtype"],
                },
            )
            episode.validate_for_caption_training()
            episodes[split].append(episode)

    source_paths = [*config.annotations.values(), config.license_record]
    manifest = write_canonical_dataset(
        config.output_root,
        config.dataset,
        config.dataset_revision,
        episodes,
        preprocessing_config_sha256=config_sha256,
        code_revision=code_revision,
        source_files=[
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in source_paths
        ],
        exclusions={
            "missing_media": {split: _attrition(values) for split, values in missing.items()}
        },
    )
    loaded = read_canonical_dataset(manifest)
    report = {
        "schema": "deltaomni.canonical_preprocessing_summary.v1",
        "dataset": config.dataset,
        "dataset_revision": config.dataset_revision,
        "status": "complete",
        "manifest": str(manifest),
        "episodes": {split: len(values) for split, values in loaded.items()},
        "caption_items": {
            split: sum(len(episode.captions.audio or ()) for episode in values)
            for split, values in loaded.items()
        },
        "missing_media": {split: len(values) for split, values in missing.items()},
        "code_revision": code_revision,
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(config.report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess AudioCaps into canonical episode v2")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/canonical/audiocaps.yaml"),
    )
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.provenance), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
