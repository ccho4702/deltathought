from __future__ import annotations

import argparse
import csv
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

from deltaomni.data.canonicalize import read_canonical_dataset, write_canonical_dataset
from deltaomni.data.media import inspect_av_media
from deltaomni.data.schema import MediaAsset, ProvenanceRecord
from deltaomni.data.something_something import build_episode
from deltaomni.provenance import audit as audit_provenance
from deltaomni.train_sanity import _atomic_json


@dataclass(frozen=True)
class SSV2CanonicalConfig:
    dataset: str
    dataset_revision: str
    resource_name: str
    chunk_seconds: float
    cpu_workers: int
    annotations: dict[str, Path]
    test_answers: Path
    media_root: Path
    license_record: Path | None
    cache_root: Path
    output_root: Path
    report_path: Path


def load_config(path: Path) -> SSV2CanonicalConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    config = SSV2CanonicalConfig(
        dataset=str(raw["dataset"]),
        dataset_revision=str(raw["dataset_revision"]),
        resource_name=str(raw["resource_name"]),
        chunk_seconds=float(raw["chunk_seconds"]),
        cpu_workers=int(raw["cpu_workers"]),
        annotations={name: resolve(value) for name, value in raw["annotations"].items()},
        test_answers=resolve(raw["test_answers"]),
        media_root=resolve(raw["media_root"]),
        license_record=(
            None if raw.get("license_record") is None else resolve(raw["license_record"])
        ),
        cache_root=resolve(raw["cache_root"]),
        output_root=resolve(raw["output_root"]),
        report_path=resolve(raw["report_path"]),
    )
    if config.dataset != "something_something_v2":
        raise ValueError("SSV2 preprocessor requires dataset=something_something_v2")
    if set(config.annotations) != {"train", "validation", "test"}:
        raise ValueError("SSV2 annotations require train/validation/test paths")
    if config.chunk_seconds <= 0 or config.cpu_workers <= 0:
        raise ValueError("SSV2 preprocessing controls must be positive")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _code_revision(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_annotations(config: SSV2CanonicalConfig) -> dict[str, list[dict[str, Any]]]:
    values = {
        split: json.loads(path.read_text(encoding="utf-8"))
        for split, path in config.annotations.items()
    }
    with config.test_answers.open(encoding="utf-8", newline="") as stream:
        answers = {row[0]: row[1] for row in csv.reader(stream, delimiter=";")}
    test_ids = {str(record["id"]) for record in values["test"]}
    if test_ids != set(answers):
        raise ValueError("SSV2 test answers do not exactly cover test annotations")
    values["test"] = [
        {**record, "label": answers[str(record["id"])], "template": answers[str(record["id"])]}
        for record in values["test"]
    ]
    return values


def _media_cache_path(config: SSV2CanonicalConfig, source_id: str) -> Path:
    shard = source_id[-2:].zfill(2)
    return config.cache_root / "media" / shard / f"{source_id}.json"


def _inspect_all_media(
    config: SSV2CanonicalConfig,
    jobs: list[tuple[str, Path]],
) -> dict[str, dict[str, Any]]:
    started = time.perf_counter()
    result = {}
    iterator = iter(jobs)
    pending: dict[Future[dict[str, Any]], str] = {}

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            source_id, path = next(iterator)
        except StopIteration:
            return False
        future = executor.submit(
            inspect_av_media,
            source_id,
            path,
            _media_cache_path(config, source_id),
        )
        pending[future] = source_id
        return True

    completed = 0
    with ThreadPoolExecutor(max_workers=config.cpu_workers) as executor:
        for _ in range(config.cpu_workers * 2):
            if not submit_next(executor):
                break
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                source_id = pending.pop(future)
                result[source_id] = future.result()
                completed += 1
                submit_next(executor)
                if completed % 1000 == 0 or completed == len(jobs):
                    elapsed = time.perf_counter() - started
                    eta = elapsed / completed * (len(jobs) - completed)
                    print(
                        f"ssv2_media={completed}/{len(jobs)} "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )
    return result


def run(config_path: Path, provenance_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    project_root = config_path.resolve().parent.parent.parent
    code_revision = _code_revision(project_root)
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
        raise ValueError("SSV2 failed provenance gate")
    annotations = _load_annotations(config)
    split_ids = {
        split: {str(record["id"]) for record in records}
        for split, records in annotations.items()
    }
    if (
        split_ids["train"] & split_ids["validation"]
        or split_ids["train"] & split_ids["test"]
        or split_ids["validation"] & split_ids["test"]
    ):
        raise ValueError("SSV2 official splits overlap")
    jobs = []
    for source_id in sorted(set().union(*split_ids.values())):
        path = config.media_root / f"{source_id}.webm"
        if not path.is_file():
            raise FileNotFoundError(path)
        jobs.append((source_id, path))

    started = time.perf_counter()
    media_index = _inspect_all_media(config, jobs)
    processed_at = datetime.now(UTC).isoformat()
    annotation_hashes = {
        split: _sha256(path) for split, path in config.annotations.items()
    }
    episodes_by_split = {}
    for split, records in annotations.items():
        episodes = []
        for annotation in records:
            source_id = str(annotation["id"])
            media = media_index[source_id]
            video = MediaAsset(
                path=Path(media["path"]),
                sha256=media["sha256"],
                duration_seconds=media["duration_seconds"],
                mime_type="video/webm",
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
                    mime_type="audio/webm-container",
                    sample_rate=media["audio"]["sample_rate"],
                    channels=media["audio"]["channels"],
                )
            )
            episode = build_episode(
                annotation,
                video,
                split=split,
                dataset_revision=config.dataset_revision,
                chunk_seconds=config.chunk_seconds,
                provenance_report=provenance,
                audio_media=audio,
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
                    **episode.metadata,
                    "placeholders": annotation.get("placeholders"),
                },
            )
            episode.validate_for_caption_training()
            episodes.append(episode)
        episodes_by_split[split] = episodes

    source_paths = [*config.annotations.values(), config.test_answers]
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
            split: sum(len(episode.captions.video or ()) for episode in items)
            for split, items in loaded.items()
        },
        "media_with_audio": {
            split: sum(episode.media.audio is not None for episode in items)
            for split, items in loaded.items()
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
    parser = argparse.ArgumentParser(description="Preprocess SSV2 into canonical episode v2")
    parser.add_argument("--config", type=Path, default=Path("configs/canonical/ssv2.yaml"))
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    args = parser.parse_args()
    report = run(args.config, args.provenance)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
