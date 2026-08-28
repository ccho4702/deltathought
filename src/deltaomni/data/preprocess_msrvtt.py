from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from deltaomni.data.canonicalize import read_canonical_dataset, write_canonical_dataset
from deltaomni.data.media import inspect_av_media
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
from deltaomni.run_integrity import require_media_policy
from deltaomni.train_sanity import _atomic_json


@dataclass(frozen=True)
class MSRVTTConfig:
    dataset: str
    dataset_revision: str
    resource_name: str
    chunk_seconds: float
    cpu_workers: int
    annotations: dict[str, Path]
    media_root: Path
    media_policy: Path
    cache_root: Path
    output_root: Path
    report_path: Path


def load_config(path: Path) -> MSRVTTConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    config = MSRVTTConfig(
        dataset=str(raw["dataset"]),
        dataset_revision=str(raw["dataset_revision"]),
        resource_name=str(raw["resource_name"]),
        chunk_seconds=float(raw["chunk_seconds"]),
        cpu_workers=int(raw["cpu_workers"]),
        annotations={split: resolve(value) for split, value in raw["annotations"].items()},
        media_root=resolve(raw["media_root"]),
        media_policy=resolve(raw["media_policy"]),
        cache_root=resolve(raw["cache_root"]),
        output_root=resolve(raw["output_root"]),
        report_path=resolve(raw["report_path"]),
    )
    if config.dataset != "msrvtt" or set(config.annotations) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("MSR-VTT requires the official train/validation/test splits")
    if min(config.chunk_seconds, config.cpu_workers) <= 0:
        raise ValueError("MSR-VTT preprocessing controls must be positive")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(path: Path) -> dict[str, tuple[str, ...]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"MSR-VTT metadata must be a mapping: {path}")
    result = {}
    for source_id, captions in raw.items():
        if not isinstance(captions, dict) or len(captions) != 20:
            raise ValueError(f"MSR-VTT requires 20 captions: {source_id}")
        values = tuple(str(value).strip() for _, value in sorted(captions.items()))
        if not all(values):
            raise ValueError(f"MSR-VTT contains an empty caption: {source_id}")
        result[str(source_id)] = values
    return result


def run(config_path: Path, provenance_path: Path) -> dict:
    config = load_config(config_path)
    root = config_path.resolve().parent.parent.parent
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    final = config.output_root / config.dataset / config.dataset_revision / "manifest.json"
    if final.is_file():
        loaded = read_canonical_dataset(final)
        return {"status": "already_complete", "episodes": {k: len(v) for k, v in loaded.items()}}
    provenance = audit_provenance(provenance_path)
    policy_sha256 = require_media_policy(
        provenance, config.resource_name, config.media_policy
    )
    metadata = {split: _metadata(path) for split, path in config.annotations.items()}
    groups = {split: set(values) for split, values in metadata.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if groups[left] & groups[right]:
            raise ValueError(f"MSR-VTT split overlap: {left}/{right}")
    jobs = [
        (split, source_id)
        for split, values in metadata.items()
        for source_id in sorted(values)
    ]
    started = time.perf_counter()

    def inspect(job: tuple[str, str]):
        split, source_id = job
        media = config.media_root / f"{source_id}.mp4"
        cache = config.cache_root / "media" / f"{source_id}.json"
        return split, source_id, inspect_av_media(source_id, media, cache)

    inspected = {}
    with ThreadPoolExecutor(max_workers=config.cpu_workers) as executor:
        for index, (split, source_id, media) in enumerate(executor.map(inspect, jobs), 1):
            inspected[(split, source_id)] = media
            if index % 500 == 0 or index == len(jobs):
                elapsed = time.perf_counter() - started
                eta = elapsed / index * (len(jobs) - index)
                print(
                    f"msrvtt_media={index}/{len(jobs)} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
    config_sha = _sha256(config_path)
    annotation_sha = {split: _sha256(path) for split, path in config.annotations.items()}
    processed_at = datetime.now(UTC).isoformat()
    episodes = {split: [] for split in metadata}
    for split, values in metadata.items():
        for source_id, captions in sorted(values.items()):
            media = inspected[(split, source_id)]
            video = MediaAsset(
                path=Path(media["path"]),
                sha256=media["sha256"],
                duration_seconds=media["duration_seconds"],
                mime_type="video/mp4",
                width=media["video"]["width"],
                height=media["video"]["height"],
                fps=media["video"]["fps"],
            )
            duration = float(media["duration_seconds"])
            episode = CanonicalEpisode(
                episode_id=f"msrvtt:{split}:{source_id}",
                dataset=config.dataset,
                dataset_revision=config.dataset_revision,
                split=split,
                source_id=source_id,
                source_group_id=f"msrvtt:{source_id}",
                media=MediaBundle(image=None, video=video, audio=None),
                duration_seconds=duration,
                temporal_blocks=temporal_grid(duration, config.chunk_seconds),
                captions=CaptionBundle(
                    image=None,
                    video=tuple(
                        CaptionAnnotation(
                            caption_id=f"msrvtt:{source_id}:{index:02d}",
                            scope="video",
                            text=text,
                            start_seconds=0.0,
                            end_seconds=duration,
                            commit_seconds=duration,
                            language="en",
                            annotation_origin="official_human_crowdsourced",
                            timing_origin="clip_level",
                            independent_from_qa=True,
                        )
                        for index, text in enumerate(captions)
                    ),
                    audio=None,
                    joint=None,
                ),
                text=TextBundle(transcript=None, subtitle=None, ocr=None),
                events=None,
                qa=None,
                provenance=ProvenanceRecord(
                    resource_name=config.resource_name,
                    source_url="https://www.microsoft.com/en-us/research/?p=301679",
                    annotation_path=config.annotations[split],
                    annotation_sha256=annotation_sha[split],
                    preprocessing_config_sha256=config_sha,
                    code_revision=revision,
                    processed_at_utc=processed_at,
                ),
                metadata={"reference_captions": 20, "media_policy_sha256": policy_sha256},
            )
            episode.validate_for_caption_training()
            episodes[split].append(episode)
    manifest = write_canonical_dataset(
        config.output_root,
        config.dataset,
        config.dataset_revision,
        episodes,
        preprocessing_config_sha256=config_sha,
        code_revision=revision,
        source_files=[
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (*config.annotations.values(), config.media_policy)
        ],
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
            split: sum(len(episode.captions.video or ()) for episode in values)
            for split, values in loaded.items()
        },
        "elapsed_seconds": time.perf_counter() - started,
        "code_revision": revision,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(config.report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Canonicalize official MSR-VTT")
    parser.add_argument("--config", type=Path, default=Path("configs/canonical/msrvtt.yaml"))
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.provenance), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
