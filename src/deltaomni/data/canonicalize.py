from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deltaomni.data.schema import SCHEMA_VERSION, CanonicalEpisode, read_jsonl, write_jsonl

MANIFEST_SCHEMA = "deltaomni.dataset_manifest.v2"
SPLITS = ("train", "validation", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _coverage(episodes: Sequence[CanonicalEpisode]) -> dict[str, Any]:
    return {
        "episodes": len(episodes),
        "media": {
            scope: sum(getattr(episode.media, scope) is not None for episode in episodes)
            for scope in ("image", "video", "audio")
        },
        "caption_annotation_available": {
            scope: sum(getattr(episode.captions, scope) is not None for episode in episodes)
            for scope in ("image", "video", "audio", "joint")
        },
        "caption_items": {
            scope: sum(len(getattr(episode.captions, scope) or ()) for episode in episodes)
            for scope in ("image", "video", "audio", "joint")
        },
        "text_annotation_available": {
            kind: sum(getattr(episode.text, kind) is not None for episode in episodes)
            for kind in ("transcript", "subtitle", "ocr")
        },
        "text_items": {
            kind: sum(len(getattr(episode.text, kind) or ()) for episode in episodes)
            for kind in ("transcript", "subtitle", "ocr")
        },
        "event_annotation_available": sum(episode.events is not None for episode in episodes),
        "event_items": sum(len(episode.events or ()) for episode in episodes),
        "qa_annotation_available": sum(episode.qa is not None for episode in episodes),
        "qa_items": sum(len(episode.qa or ()) for episode in episodes),
    }


def _validate_dataset(
    dataset: str,
    revision: str,
    episodes_by_split: Mapping[str, Sequence[CanonicalEpisode]],
) -> None:
    unknown = set(episodes_by_split) - set(SPLITS)
    if unknown:
        raise ValueError(f"Unsupported canonical splits: {sorted(unknown)}")
    episode_ids = set()
    split_sources: dict[str, set[str]] = {}
    for split, episodes in episodes_by_split.items():
        split_sources[split] = set()
        for episode in episodes:
            episode.validate()
            if episode.dataset != dataset or episode.dataset_revision != revision:
                raise ValueError(f"Canonical dataset identity mismatch: {episode.episode_id}")
            if episode.split != split:
                raise ValueError(f"Canonical split mismatch: {episode.episode_id}")
            if episode.episode_id in episode_ids:
                raise ValueError(f"Duplicate canonical episode_id: {episode.episode_id}")
            episode_ids.add(episode.episode_id)
            split_sources[split].add(episode.source_group_id)
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = split_sources.get(left, set()) & split_sources.get(right, set())
            if overlap:
                example = sorted(overlap)[0]
                raise ValueError(f"Cross-split source_group_id overlap: {example}")


def write_canonical_dataset(
    output_root: Path,
    dataset: str,
    revision: str,
    episodes_by_split: Mapping[str, Sequence[CanonicalEpisode]],
    *,
    preprocessing_config_sha256: str,
    code_revision: str,
    source_files: Sequence[Mapping[str, Any]],
) -> Path:
    _validate_dataset(dataset, revision, episodes_by_split)
    target = output_root / dataset / revision
    if target.exists():
        raise FileExistsError(f"Canonical dataset revision already exists: {target}")
    temporary = target.parent / f".{revision}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        split_manifest = {}
        for split in SPLITS:
            episodes = list(episodes_by_split.get(split, ()))
            path = temporary / f"{split}.jsonl"
            write_jsonl(path, episodes)
            split_manifest[split] = {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "coverage": _coverage(episodes),
            }
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "episode_schema": SCHEMA_VERSION,
            "dataset": dataset,
            "dataset_revision": revision,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "preprocessing_config_sha256": preprocessing_config_sha256,
            "code_revision": code_revision,
            "source_files": [dict(value) for value in source_files],
            "splits": split_manifest,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        with manifest_path.open("rb") as stream:
            os.fsync(stream.fileno())
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, target)
    except BaseException:
        if temporary.is_dir():
            for path in temporary.iterdir():
                path.unlink()
            temporary.rmdir()
        raise
    return target / "manifest.json"


def read_canonical_dataset(manifest_path: Path) -> dict[str, list[CanonicalEpisode]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("Unsupported canonical dataset manifest")
    if manifest.get("episode_schema") != SCHEMA_VERSION:
        raise ValueError("Canonical manifest episode schema mismatch")
    result = {}
    for split in SPLITS:
        split_record = manifest["splits"][split]
        path = manifest_path.parent / split_record["path"]
        if _sha256(path) != split_record["sha256"]:
            raise ValueError(f"Canonical split checksum mismatch: {split}")
        episodes = read_jsonl(path)
        if len(episodes) != split_record["coverage"]["episodes"]:
            raise ValueError(f"Canonical split count mismatch: {split}")
        result[split] = episodes
    _validate_dataset(
        str(manifest["dataset"]),
        str(manifest["dataset_revision"]),
        result,
    )
    return result
