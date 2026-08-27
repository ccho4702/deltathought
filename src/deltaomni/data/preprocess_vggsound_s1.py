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
from deltaomni.data.media import inspect_audio_media, inspect_av_media
from deltaomni.data.schema import (
    CanonicalEpisode,
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
class VGGSoundS1Config:
    dataset: str
    dataset_revision: str
    resource_name: str
    seed: int
    excluded_source_ids: frozenset[str]
    train_count: int
    validation_count: int
    test_count: int
    chunk_seconds: float
    cpu_workers: int
    train_metadata: Path
    test_metadata: Path
    video_root: Path
    audio_root: Path
    cache_root: Path
    output_root: Path
    report_path: Path


def load_config(path: Path) -> VGGSoundS1Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    config = VGGSoundS1Config(
        dataset=str(raw["dataset"]),
        dataset_revision=str(raw["dataset_revision"]),
        resource_name=str(raw["resource_name"]),
        seed=int(raw["seed"]),
        excluded_source_ids=frozenset(str(value) for value in raw.get("excluded_source_ids", [])),
        train_count=int(raw["train_count"]),
        validation_count=int(raw["validation_count"]),
        test_count=int(raw["test_count"]),
        chunk_seconds=float(raw["chunk_seconds"]),
        cpu_workers=int(raw["cpu_workers"]),
        train_metadata=resolve(raw["train_metadata"]),
        test_metadata=resolve(raw["test_metadata"]),
        video_root=resolve(raw["video_root"]),
        audio_root=resolve(raw["audio_root"]),
        cache_root=resolve(raw["cache_root"]),
        output_root=resolve(raw["output_root"]),
        report_path=resolve(raw["report_path"]),
    )
    if config.dataset != "vggsound":
        raise ValueError("VGGSound preprocessor requires dataset=vggsound")
    positive = (
        config.train_count,
        config.validation_count,
        config.test_count,
        config.chunk_seconds,
        config.cpu_workers,
    )
    if min(positive) <= 0:
        raise ValueError("VGGSound preprocessing controls must be positive")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_names(path: Path) -> list[str]:
    names = []
    with path.open(newline="", encoding="utf-8") as stream:
        for line_number, row in enumerate(csv.reader(stream), start=1):
            if len(row) < 2 or not row[0].endswith(".mp4"):
                raise ValueError(f"Malformed VGGSound metadata row {path}:{line_number}")
            names.append(row[0])
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate VGGSound filename in {path}")
    return names


def _source_group(name: str) -> str:
    return f"youtube:{Path(name).stem.rsplit('_', 1)[0]}"


def _rank(names: list[str], seed: int) -> list[str]:
    return sorted(
        names,
        key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(),
    )


def _select_available(
    names: list[str],
    *,
    count: int,
    seed: int,
    video_root: Path,
    audio_root: Path,
    excluded_groups: set[str] | None = None,
    excluded_source_ids: frozenset[str] = frozenset(),
) -> tuple[list[str], set[str], int]:
    selected = []
    groups = set(excluded_groups or ())
    unavailable = 0
    for name in _rank(names, seed):
        if Path(name).stem in excluded_source_ids:
            continue
        group = _source_group(name)
        if group in groups:
            continue
        if (
            not (video_root / name).is_file()
            or not (audio_root / Path(name).with_suffix(".wav")).is_file()
        ):
            unavailable += 1
            continue
        selected.append(name)
        groups.add(group)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)}/{count} requested VGGSound pairs are available")
    return selected, groups, unavailable


def _select_splits(config: VGGSoundS1Config) -> tuple[dict[str, list[str]], dict[str, int]]:
    train_names = _read_names(config.train_metadata)
    test_names = _read_names(config.test_metadata)
    train, used_groups, unavailable_train = _select_available(
        train_names,
        count=config.train_count,
        seed=config.seed,
        video_root=config.video_root,
        audio_root=config.audio_root,
        excluded_source_ids=config.excluded_source_ids,
    )
    validation, used_groups, unavailable_validation = _select_available(
        test_names,
        count=config.validation_count,
        seed=config.seed + 1,
        video_root=config.video_root,
        audio_root=config.audio_root,
        excluded_groups=used_groups,
        excluded_source_ids=config.excluded_source_ids,
    )
    test, _, unavailable_test = _select_available(
        test_names,
        count=config.test_count,
        seed=config.seed + 2,
        video_root=config.video_root,
        audio_root=config.audio_root,
        excluded_groups=used_groups,
        excluded_source_ids=config.excluded_source_ids,
    )
    return (
        {"train": train, "validation": validation, "test": test},
        {
            "train": unavailable_train,
            "validation": unavailable_validation,
            "test": unavailable_test,
        },
    )


def _inspect_pair(
    config: VGGSoundS1Config,
    job: tuple[str, str],
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    split, name = job
    source_id = Path(name).stem
    video = inspect_av_media(
        source_id,
        config.video_root / name,
        config.cache_root / "video" / f"{source_id}.json",
    )
    audio = inspect_audio_media(
        source_id,
        config.audio_root / Path(name).with_suffix(".wav"),
        config.cache_root / "audio" / f"{source_id}.json",
    )
    return split, name, video, audio


def _build_episode(
    config: VGGSoundS1Config,
    split: str,
    name: str,
    video_info: dict[str, Any],
    audio_info: dict[str, Any],
    *,
    config_sha256: str,
    code_revision: str,
    processed_at_utc: str,
    annotation_sha256: str,
) -> CanonicalEpisode:
    video = MediaAsset(
        path=Path(video_info["path"]),
        sha256=video_info["sha256"],
        duration_seconds=video_info["duration_seconds"],
        mime_type="video/mp4",
        width=video_info["video"]["width"],
        height=video_info["video"]["height"],
        fps=video_info["video"]["fps"],
    )
    audio = MediaAsset(
        path=Path(audio_info["path"]),
        sha256=audio_info["sha256"],
        duration_seconds=audio_info["duration_seconds"],
        mime_type="audio/wav",
        sample_rate=audio_info["audio"]["sample_rate"],
        channels=audio_info["audio"]["channels"],
    )
    assert video.duration_seconds is not None and audio.duration_seconds is not None
    episode_duration = max(video.duration_seconds, audio.duration_seconds)
    annotation_path = config.train_metadata if split == "train" else config.test_metadata
    source_id = Path(name).stem
    episode = CanonicalEpisode(
        episode_id=f"vggsound:{split}:{source_id}",
        dataset=config.dataset,
        dataset_revision=config.dataset_revision,
        split=split,
        source_id=source_id,
        source_group_id=_source_group(name),
        media=MediaBundle(image=None, video=video, audio=audio),
        duration_seconds=episode_duration,
        temporal_blocks=temporal_grid(episode_duration, config.chunk_seconds),
        captions=CaptionBundle(image=None, video=None, audio=None, joint=None),
        text=TextBundle(transcript=None, subtitle=None, ocr=None),
        events=None,
        qa=None,
        provenance=ProvenanceRecord(
            resource_name=config.resource_name,
            source_url="https://github.com/hche11/VGGSound",
            annotation_path=annotation_path,
            annotation_sha256=annotation_sha256,
            preprocessing_config_sha256=config_sha256,
            code_revision=code_revision,
            processed_at_utc=processed_at_utc,
        ),
        metadata={
            "aligned_duration_seconds": min(video.duration_seconds, audio.duration_seconds),
            "weak_class_label_ignored": True,
        },
    )
    episode.validate()
    return episode


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
            "episodes": {split: len(items) for split, items in loaded.items()},
            "code_revision": code_revision,
        }
        _atomic_json(config.report_path, report)
        return report

    provenance = audit_provenance(provenance_path)
    if config.resource_name not in provenance["approved"]:
        raise ValueError("VGGSound failed provenance gate")
    selected, unavailable = _select_splits(config)
    jobs = [(split, name) for split, names in selected.items() for name in names]
    started = time.perf_counter()
    inspected = []
    with ThreadPoolExecutor(max_workers=config.cpu_workers) as executor:
        iterator = executor.map(lambda job: _inspect_pair(config, job), jobs)
        for index, value in enumerate(iterator, start=1):
            inspected.append(value)
            if index % 100 == 0 or index == len(jobs):
                elapsed = time.perf_counter() - started
                eta = elapsed / index * (len(jobs) - index)
                print(
                    f"vggsound_media={index}/{len(jobs)} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )

    config_sha256 = _sha256(config_path)
    annotation_hashes = {
        "train": _sha256(config.train_metadata),
        "test": _sha256(config.test_metadata),
    }
    processed_at_utc = datetime.now(UTC).isoformat()
    episodes: dict[str, list[CanonicalEpisode]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for split, name, video_info, audio_info in inspected:
        annotation_key = "train" if split == "train" else "test"
        episodes[split].append(
            _build_episode(
                config,
                split,
                name,
                video_info,
                audio_info,
                config_sha256=config_sha256,
                code_revision=code_revision,
                processed_at_utc=processed_at_utc,
                annotation_sha256=annotation_hashes[annotation_key],
            )
        )
    source_paths = (config.train_metadata, config.test_metadata)
    manifest = write_canonical_dataset(
        config.output_root,
        config.dataset,
        config.dataset_revision,
        episodes,
        preprocessing_config_sha256=config_sha256,
        code_revision=code_revision,
        source_files=[
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in source_paths
        ],
        exclusions={
            "unavailable_pairs_encountered_during_selection": unavailable,
            "decode_audit_excluded_source_ids": sorted(config.excluded_source_ids),
            "weak_class_labels": "intentionally_ignored_for_self_supervised_s1",
        },
    )
    loaded = read_canonical_dataset(manifest)
    report = {
        "schema": "deltaomni.canonical_preprocessing_summary.v1",
        "dataset": config.dataset,
        "dataset_revision": config.dataset_revision,
        "status": "complete",
        "manifest": str(manifest),
        "episodes": {split: len(items) for split, items in loaded.items()},
        "unavailable_pairs_encountered": unavailable,
        "code_revision": code_revision,
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(config.report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a deterministic paired VGGSound subset for S1 DeltaTok"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/canonical/vggsound_s1.yaml"),
    )
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.provenance), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
