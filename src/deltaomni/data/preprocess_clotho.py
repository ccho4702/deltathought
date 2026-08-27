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
from soundfile import LibsndfileError

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
class ClothoConfig:
    dataset: str
    dataset_revision: str
    resource_name: str
    chunk_seconds: float
    cpu_workers: int
    annotations: dict[str, Path]
    metadata: dict[str, Path]
    media_roots: dict[str, Path]
    license_record: Path
    cache_root: Path
    output_root: Path
    report_path: Path


@dataclass(frozen=True)
class ClothoRecord:
    file_name: str
    captions: tuple[str, ...]
    sound_id: str
    sound_link: str
    keywords: tuple[str, ...]
    manufacturer: str
    media_license: str
    start_end_samples: str | None


def load_config(path: Path) -> ClothoConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    config = ClothoConfig(
        dataset=str(raw["dataset"]),
        dataset_revision=str(raw["dataset_revision"]),
        resource_name=str(raw["resource_name"]),
        chunk_seconds=float(raw["chunk_seconds"]),
        cpu_workers=int(raw["cpu_workers"]),
        annotations={split: resolve(value) for split, value in raw["annotations"].items()},
        metadata={split: resolve(value) for split, value in raw["metadata"].items()},
        media_roots={split: resolve(value) for split, value in raw["media_roots"].items()},
        license_record=resolve(raw["license_record"]),
        cache_root=resolve(raw["cache_root"]),
        output_root=resolve(raw["output_root"]),
        report_path=resolve(raw["report_path"]),
    )
    splits = {"train", "validation", "test"}
    if config.dataset != "clotho" or not all(
        set(mapping) == splits
        for mapping in (config.annotations, config.metadata, config.media_roots)
    ):
        raise ValueError("Clotho requires official train/validation/test resources")
    if min(config.chunk_seconds, config.cpu_workers) <= 0:
        raise ValueError("Clotho preprocessing controls must be positive")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_records(captions_path: Path, metadata_path: Path) -> list[ClothoRecord]:
    # Official v2.1 development/validation metadata contain Latin-1 manufacturer names.
    with metadata_path.open(newline="", encoding="latin-1") as stream:
        metadata_rows = {row["file_name"]: row for row in csv.DictReader(stream)}
    records = []
    with captions_path.open(newline="", encoding="utf-8-sig") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), start=2):
            file_name = row["file_name"]
            metadata = metadata_rows.get(file_name)
            captions = tuple(row[f"caption_{index}"].strip() for index in range(1, 6))
            if metadata is None or any(not caption for caption in captions):
                raise ValueError(f"Invalid Clotho record at {captions_path}:{line_number}")
            records.append(
                ClothoRecord(
                    file_name=file_name,
                    captions=captions,
                    sound_id=metadata["sound_id"],
                    sound_link=metadata["sound_link"],
                    keywords=tuple(value for value in metadata["keywords"].split(";") if value),
                    manufacturer=metadata["manufacturer"],
                    media_license=metadata["license"],
                    start_end_samples=metadata["start_end_samples"] or None,
                )
            )
    if len({record.file_name for record in records}) != len(records):
        raise ValueError(f"Duplicate Clotho filename: {captions_path}")
    if set(metadata_rows) != {record.file_name for record in records}:
        raise ValueError(f"Clotho caption/metadata filename mismatch: {captions_path}")
    return records


def _source_id(file_name: str) -> str:
    digest = hashlib.sha256(file_name.encode()).hexdigest()[:20]
    return f"clotho-{digest}"


def _deduplicate_sources(
    records: dict[str, list[ClothoRecord]],
) -> tuple[dict[str, list[ClothoRecord]], dict[str, dict[str, str]]]:
    kept = {"test": list(records["test"]), "validation": [], "train": []}
    excluded: dict[str, dict[str, str]] = {
        "train": {},
        "validation": {},
        "test": {},
    }
    test_sources = {record.sound_id for record in kept["test"]}
    for record in records["validation"]:
        if record.sound_id in test_sources:
            excluded["validation"][record.file_name] = "sound_id_overlap_with_test"
        else:
            kept["validation"].append(record)
    held_out_sources = test_sources | {record.sound_id for record in kept["validation"]}
    for record in records["train"]:
        if record.sound_id in held_out_sources:
            excluded["train"][record.file_name] = "sound_id_overlap_with_validation_or_test"
        else:
            kept["train"].append(record)
    return {split: kept[split] for split in ("train", "validation", "test")}, excluded


def _cache_path(config: ClothoConfig, split: str, source_id: str) -> Path:
    return config.cache_root / "media" / split / source_id[:2] / f"{source_id}.json"


def _inspect(
    config: ClothoConfig,
    job: tuple[str, ClothoRecord],
) -> tuple[str, ClothoRecord, dict[str, Any] | None, str | None]:
    split, record = job
    path = config.media_roots[split] / record.file_name
    if not path.is_file():
        return split, record, None, "missing"
    if path.stat().st_size == 0:
        return split, record, None, "empty_file"
    source_id = _source_id(record.file_name)
    try:
        media = inspect_audio_media(source_id, path, _cache_path(config, split, source_id))
    except (json.JSONDecodeError, LibsndfileError, OSError, ValueError) as error:
        return split, record, None, f"{type(error).__name__}:{str(error)[:500]}"
    return split, record, media, None


def _exclusion(values: dict[str, str], reason: str) -> dict[str, Any]:
    serialized = [f"{name}:{detail}" for name, detail in sorted(values.items())]
    return {
        "reason": reason,
        "count": len(serialized),
        "records_sha256": hashlib.sha256("\n".join(serialized).encode()).hexdigest(),
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
        raise ValueError("Clotho failed provenance gate")
    if not config.license_record.is_file():
        raise FileNotFoundError(f"Clotho license record missing: {config.license_record}")
    raw_records = {
        split: _read_records(config.annotations[split], config.metadata[split])
        for split in config.annotations
    }
    records, source_overlap = _deduplicate_sources(raw_records)
    jobs = [(split, record) for split, values in records.items() for record in values]
    inspected: dict[tuple[str, str], dict[str, Any]] = {}
    missing: dict[str, dict[str, str]] = {split: {} for split in records}
    invalid: dict[str, dict[str, str]] = {split: {} for split in records}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=config.cpu_workers) as executor:
        iterator = executor.map(lambda job: _inspect(config, job), jobs)
        for index, (split, record, media, reason) in enumerate(iterator, start=1):
            if media is not None:
                inspected[(split, record.file_name)] = media
            elif reason == "missing":
                missing[split][record.file_name] = reason
            else:
                invalid[split][record.file_name] = reason or "unknown_inspection_failure"
            if index % 500 == 0 or index == len(jobs):
                elapsed = time.perf_counter() - started
                eta = elapsed / index * (len(jobs) - index)
                print(
                    f"clotho_media={index}/{len(jobs)} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
    config_sha256 = _sha256(config_path)
    annotation_hashes = {split: _sha256(path) for split, path in config.annotations.items()}
    processed_at = datetime.now(UTC).isoformat()
    episodes: dict[str, list[CanonicalEpisode]] = {split: [] for split in records}
    for split, values in records.items():
        for record in values:
            media = inspected.get((split, record.file_name))
            if media is None:
                continue
            source_id = _source_id(record.file_name)
            audio = MediaAsset(
                path=Path(media["path"]),
                sha256=media["sha256"],
                duration_seconds=media["duration_seconds"],
                mime_type="audio/wav",
                sample_rate=media["audio"]["sample_rate"],
                channels=media["audio"]["channels"],
            )
            assert audio.duration_seconds is not None
            episode = CanonicalEpisode(
                episode_id=f"clotho:{split}:{source_id}",
                dataset=config.dataset,
                dataset_revision=config.dataset_revision,
                split=split,
                source_id=source_id,
                source_group_id=f"freesound:{record.sound_id}",
                media=MediaBundle(image=None, video=None, audio=audio),
                duration_seconds=audio.duration_seconds,
                temporal_blocks=temporal_grid(audio.duration_seconds, config.chunk_seconds),
                captions=CaptionBundle(
                    image=None,
                    video=None,
                    audio=tuple(
                        CaptionAnnotation(
                            caption_id=f"{source_id}:caption:{index}",
                            scope="audio",
                            text=caption,
                            start_seconds=0.0,
                            end_seconds=audio.duration_seconds,
                            commit_seconds=audio.duration_seconds,
                            language="en",
                            annotation_origin="official_human_crowdsourced",
                            timing_origin="full_audio_clip",
                            independent_from_qa=True,
                        )
                        for index, caption in enumerate(record.captions)
                    ),
                    joint=None,
                ),
                text=TextBundle(transcript=None, subtitle=None, ocr=None),
                events=None,
                qa=None,
                provenance=ProvenanceRecord(
                    resource_name=config.resource_name,
                    source_url="https://zenodo.org/records/4783391",
                    license_record=config.license_record,
                    annotation_path=config.annotations[split],
                    annotation_sha256=annotation_hashes[split],
                    preprocessing_config_sha256=config_sha256,
                    code_revision=code_revision,
                    processed_at_utc=processed_at,
                ),
                metadata={
                    "file_name": record.file_name,
                    "keywords": list(record.keywords),
                    "sound_id": record.sound_id,
                    "sound_link": record.sound_link,
                    "manufacturer": record.manufacturer,
                    "media_license": record.media_license,
                    "start_end_samples": record.start_end_samples,
                    "reference_captions": 5,
                },
            )
            episode.validate_for_caption_training()
            episodes[split].append(episode)
    source_paths = [
        *config.annotations.values(),
        *config.metadata.values(),
        config.license_record,
    ]
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
            "cross_split_source_overlap": {
                split: _exclusion(values, "freesound_sound_id_overlap_quarantined")
                for split, values in source_overlap.items()
            },
            "missing_media": {
                split: _exclusion(values, "official_archive_media_missing")
                for split, values in missing.items()
            },
            "invalid_media": {
                split: _exclusion(values, "media_failed_integrity_or_decoder_inspection")
                for split, values in invalid.items()
            },
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
        "invalid_media": {split: len(values) for split, values in invalid.items()},
        "cross_split_source_overlap": {
            split: len(values) for split, values in source_overlap.items()
        },
        "code_revision": code_revision,
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(config.report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess Clotho v2.1 to canonical episode v2")
    parser.add_argument("--config", type=Path, default=Path("configs/canonical/clotho.yaml"))
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.provenance), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
