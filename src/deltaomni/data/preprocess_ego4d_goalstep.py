from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
class GoalStepConfig:
    dataset: str
    dataset_revision: str
    resource_name: str
    chunk_seconds: float
    minimum_segment_seconds: float
    minimum_commits_per_video: int
    cpu_workers: int
    minimum_available_videos: dict[str, int]
    maximum_videos: dict[str, int] | None
    annotations: dict[str, Path]
    metadata: Path
    media_root: Path
    media_policy: Path
    cache_root: Path
    output_root: Path
    report_path: Path


def load_config(path: Path) -> GoalStepConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    config = GoalStepConfig(
        dataset=str(raw["dataset"]),
        dataset_revision=str(raw["dataset_revision"]),
        resource_name=str(raw["resource_name"]),
        chunk_seconds=float(raw["chunk_seconds"]),
        minimum_segment_seconds=float(raw["minimum_segment_seconds"]),
        minimum_commits_per_video=int(raw["minimum_commits_per_video"]),
        cpu_workers=int(raw["cpu_workers"]),
        minimum_available_videos={
            split: int(value) for split, value in raw["minimum_available_videos"].items()
        },
        maximum_videos=(
            None
            if raw.get("maximum_videos") is None
            else {split: int(value) for split, value in raw["maximum_videos"].items()}
        ),
        annotations={split: resolve(value) for split, value in raw["annotations"].items()},
        metadata=resolve(raw["metadata"]),
        media_root=resolve(raw["media_root"]),
        media_policy=resolve(raw["media_policy"]),
        cache_root=resolve(raw["cache_root"]),
        output_root=resolve(raw["output_root"]),
        report_path=resolve(raw["report_path"]),
    )
    if config.dataset != "ego4d_goalstep" or set(config.annotations) != {
        "train",
        "validation",
    }:
        raise ValueError("Ego4D GoalStep requires official train and validation splits")
    if set(config.minimum_available_videos) != set(config.annotations):
        raise ValueError("Ego4D GoalStep availability thresholds must cover both splits")
    if config.maximum_videos is not None and set(config.maximum_videos) != set(config.annotations):
        raise ValueError("Ego4D GoalStep maximum counts must cover both splits")
    if min(
        config.chunk_seconds,
        config.minimum_segment_seconds,
        config.minimum_commits_per_video,
        config.cpu_workers,
        *config.minimum_available_videos.values(),
        *(config.maximum_videos or {}).values(),
    ) <= 0:
        raise ValueError("Ego4D GoalStep preprocessing controls must be positive")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _goalstep(path: Path, expected_split: str) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    official_split = "val" if expected_split == "validation" else expected_split
    if not isinstance(value, dict) or value.get("split") != official_split:
        raise ValueError(f"Unexpected Ego4D GoalStep split: {path}")
    videos = value.get("videos")
    if not isinstance(videos, list):
        raise ValueError(f"Malformed Ego4D GoalStep videos: {path}")
    return videos


def _leaf_segments(
    video: dict[str, Any], minimum_seconds: float
) -> list[dict[str, Any]]:
    leaves: list[dict[str, Any]] = []

    def visit(segment: dict[str, Any], depth: int) -> None:
        children = segment.get("segments") or []
        if children:
            for child in children:
                visit(child, depth + 1)
            return
        try:
            start = float(segment["start_time"])
            end = float(segment["end_time"])
        except (KeyError, TypeError, ValueError):
            return
        text = str(segment.get("step_description", "")).strip()
        relevance = str(segment.get("is_relevant", "")).strip().lower()
        if (
            not text
            or start < 0
            or end - start < minimum_seconds
            or relevance == "irrelevant"
        ):
            return
        leaves.append(
            {
                "start_seconds": start,
                "end_seconds": end,
                "commit_seconds": end,
                "text": text,
                "depth": depth,
                "relevance": relevance or None,
                "step_category": segment.get("step_category"),
            }
        )

    for segment in video.get("segments") or []:
        visit(segment, 0)
    unique = {
        (
            round(segment["start_seconds"], 6),
            round(segment["end_seconds"], 6),
            segment["text"].casefold(),
        ): segment
        for segment in leaves
    }
    return sorted(
        unique.values(),
        key=lambda segment: (
            segment["commit_seconds"],
            segment["start_seconds"],
            segment["text"],
        ),
    )


def _video_metadata(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    videos = value.get("videos") if isinstance(value, dict) else None
    if not isinstance(videos, list):
        raise ValueError(f"Malformed Ego4D metadata: {path}")
    return {str(video["video_uid"]): video for video in videos}


def run(config_path: Path, provenance_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = config_path.resolve().parent.parent.parent
    final = config.output_root / config.dataset / config.dataset_revision / "manifest.json"
    if final.is_file():
        loaded = read_canonical_dataset(final)
        return {"status": "already_complete", "episodes": {k: len(v) for k, v in loaded.items()}}

    provenance = audit_provenance(provenance_path)
    media_policy_sha256 = require_media_policy(
        provenance, config.resource_name, config.media_policy
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    metadata = _video_metadata(config.metadata)
    raw_splits = {
        split: _goalstep(path, split) for split, path in config.annotations.items()
    }
    split_ids = {
        split: {str(video["video_uid"]) for video in videos}
        for split, videos in raw_splits.items()
    }
    if split_ids["train"] & split_ids["validation"]:
        raise ValueError("Ego4D GoalStep train/validation source overlap")

    selected: dict[str, list[tuple[dict[str, Any], list[dict[str, Any]]]]] = {}
    missing_media: dict[str, list[str]] = {}
    for split, videos in raw_splits.items():
        selected[split] = []
        missing_media[split] = []
        for video in videos:
            segments = _leaf_segments(video, config.minimum_segment_seconds)
            if len(segments) < config.minimum_commits_per_video:
                continue
            uid = str(video["video_uid"])
            if uid not in metadata:
                raise ValueError(f"Missing Ego4D metadata: {uid}")
            media = config.media_root / f"{uid}.mp4"
            if not media.is_file():
                missing_media[split].append(uid)
                continue
            selected[split].append((video, segments))
        selected[split].sort(key=lambda value: str(value[0]["video_uid"]))
        if config.maximum_videos is not None:
            selected[split] = selected[split][: config.maximum_videos[split]]
        if len(selected[split]) < config.minimum_available_videos[split]:
            raise ValueError(
                f"Ego4D local media coverage fell below the pinned {split} minimum: "
                f"{len(selected[split])}/{config.minimum_available_videos[split]}"
            )

    jobs = [
        str(video["video_uid"])
        for values in selected.values()
        for video, _ in values
    ]
    started = time.perf_counter()

    def inspect(uid: str) -> tuple[str, dict[str, Any]]:
        return uid, inspect_av_media(
            uid,
            config.media_root / f"{uid}.mp4",
            config.cache_root / "media" / f"{uid}.json",
        )

    inspected = {}
    with ThreadPoolExecutor(max_workers=config.cpu_workers) as executor:
        for index, (uid, media) in enumerate(executor.map(inspect, jobs), 1):
            inspected[uid] = media
            if index % 20 == 0 or index == len(jobs):
                elapsed = time.perf_counter() - started
                eta = elapsed / index * (len(jobs) - index)
                print(
                    f"ego4d_goalstep_media={index}/{len(jobs)} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )

    config_sha = _sha256(config_path)
    annotation_sha = {split: _sha256(path) for split, path in config.annotations.items()}
    metadata_sha = _sha256(config.metadata)
    processed_at = datetime.now(UTC).isoformat()
    episodes: dict[str, list[CanonicalEpisode]] = {split: [] for split in selected}
    intervals = []
    for split, values in selected.items():
        for video, segments in values:
            uid = str(video["video_uid"])
            media = inspected[uid]
            duration = float(media["duration_seconds"])
            valid_segments = [
                segment for segment in segments if segment["commit_seconds"] <= duration + 0.5
            ]
            if len(valid_segments) < config.minimum_commits_per_video:
                continue
            commits = [segment["commit_seconds"] for segment in valid_segments]
            intervals.extend(
                right - left for left, right in zip(commits, commits[1:], strict=False)
            )
            video_metadata = metadata[uid]["video_metadata"]
            captions = tuple(
                CaptionAnnotation(
                    caption_id=f"ego4d-goalstep:{uid}:{index:04d}",
                    scope="video",
                    text=segment["text"],
                    start_seconds=segment["start_seconds"],
                    end_seconds=segment["end_seconds"],
                    commit_seconds=segment["commit_seconds"],
                    language="en",
                    annotation_origin="official_human_goalstep",
                    timing_origin="official_goalstep_segment_end",
                    independent_from_qa=True,
                )
                for index, segment in enumerate(valid_segments)
            )
            episode = CanonicalEpisode(
                episode_id=f"ego4d-goalstep:{split}:{uid}",
                dataset=config.dataset,
                dataset_revision=config.dataset_revision,
                split=split,
                source_id=uid,
                source_group_id=f"ego4d:{uid}",
                media=MediaBundle(
                    image=None,
                    video=MediaAsset(
                        path=Path(media["path"]),
                        sha256=media["sha256"],
                        duration_seconds=duration,
                        mime_type="video/mp4",
                        width=media["video"]["width"],
                        height=media["video"]["height"],
                        fps=media["video"]["fps"],
                    ),
                    audio=None,
                ),
                duration_seconds=duration,
                temporal_blocks=temporal_grid(duration, config.chunk_seconds),
                captions=CaptionBundle(image=None, video=captions, audio=None, joint=None),
                text=TextBundle(transcript=None, subtitle=None, ocr=None),
                events=None,
                qa=None,
                provenance=ProvenanceRecord(
                    resource_name=config.resource_name,
                    source_url="https://ego4d-data.org/docs/data/annotation-guidelines/",
                    annotation_path=config.annotations[split],
                    annotation_sha256=annotation_sha[split],
                    preprocessing_config_sha256=config_sha,
                    code_revision=revision,
                    processed_at_utc=processed_at,
                ),
                metadata={
                    "goal_category": video.get("goal_category"),
                    "goal_description": video.get("goal_description"),
                    "metadata_sha256": metadata_sha,
                    "media_policy_sha256": media_policy_sha256,
                    "dynamic_commit_boundaries": True,
                    "hierarchy_policy": "deepest_non_irrelevant_segments",
                    "source_video_codec": video_metadata.get("video_codec"),
                },
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
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (*config.annotations.values(), config.metadata, config.media_policy)
        ],
    )
    loaded = read_canonical_dataset(manifest)
    report = {
        "schema": "deltaomni.ego4d_goalstep_preprocessing.v1",
        "status": "complete",
        "dataset_revision": config.dataset_revision,
        "manifest": str(manifest),
        "episodes": {split: len(values) for split, values in loaded.items()},
        "commits": {
            split: sum(len(episode.captions.video or ()) for episode in values)
            for split, values in loaded.items()
        },
        "missing_eligible_media": {
            split: len(values) for split, values in missing_media.items()
        },
        "commit_interval_seconds": {
            "minimum": min(intervals),
            "median": statistics.median(intervals),
            "maximum": max(intervals),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "code_revision": revision,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(config.report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Canonicalize official Ego4D GoalStep")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/canonical/ego4d_goalstep.yaml")
    )
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.provenance), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
