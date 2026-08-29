from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from deltaomni.run_integrity import git_revision, git_worktree_is_clean, sha256_file
from deltaomni.train_sanity import _atomic_json


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    source_manifest: Path
    train_fraction: float
    output_manifest: Path
    report_path: Path


def load_config(path: Path) -> SplitConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    config = SplitConfig(
        seed=int(raw["seed"]),
        source_manifest=resolve(raw["source_manifest"]),
        train_fraction=float(raw["train_fraction"]),
        output_manifest=resolve(raw["output_manifest"]),
        report_path=resolve(raw["report_path"]),
    )
    if not 0 < config.train_fraction < 1:
        raise ValueError("LongVideoBench diagnostic train fraction must be between zero and one")
    return config


def split_videos(
    videos: dict[str, dict[str, Any]], seed: int, train_fraction: float
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, list[str]] = {}
    for video_id, value in videos.items():
        questions = value.get("questions") or []
        groups = {str(question["duration_group"]) for question in questions}
        if len(groups) != 1:
            raise ValueError(f"LongVideoBench video has inconsistent duration groups: {video_id}")
        grouped.setdefault(groups.pop(), []).append(video_id)
    train_ids = set()
    for duration_group, video_ids in grouped.items():
        ordered = sorted(
            video_ids,
            key=lambda video_id: hashlib.sha256(
                f"{seed}:{duration_group}:{video_id}".encode()
            ).hexdigest(),
        )
        count = min(len(ordered) - 1, max(1, round(len(ordered) * train_fraction)))
        train_ids.update(ordered[:count])
    return {
        "train": {key: videos[key] for key in sorted(train_ids)},
        "validation": {key: videos[key] for key in sorted(videos.keys() - train_ids)},
    }


def _summary(videos: dict[str, dict[str, Any]]) -> dict[str, Any]:
    questions = [question for value in videos.values() for question in value["questions"]]
    return {
        "videos": len(videos),
        "questions": len(questions),
        "windows": sum(len(value["windows"]) for value in videos.values()),
        "duration_groups": dict(
            sorted(Counter(str(q["duration_group"]) for q in questions).items())
        ),
        "question_categories": dict(
            sorted(Counter(str(q["question_category"]) for q in questions).items())
        ),
    }


def run(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = config_path.resolve().parent.parent
    if not git_worktree_is_clean(root):
        raise RuntimeError("LongVideoBench diagnostic split requires a clean Git worktree")
    source = json.loads(config.source_manifest.read_text(encoding="utf-8"))
    if source.get("schema") != "deltaomni.omni_longvideobench_manifest.v1":
        raise ValueError("Unexpected LongVideoBench source manifest")
    splits = split_videos(source["videos"], config.seed, config.train_fraction)
    overlap = splits["train"].keys() & splits["validation"].keys()
    if overlap or set(splits["train"]) | set(splits["validation"]) != set(source["videos"]):
        raise ValueError("LongVideoBench diagnostic split coverage failed")
    manifest = {
        "schema": "deltaomni.longvideobench_diagnostic_split.v1",
        "benchmark_status": "validation_labels_consumed_for_diagnostic_training",
        "source_manifest": str(config.source_manifest),
        "source_manifest_sha256": sha256_file(config.source_manifest),
        "seed": config.seed,
        "train_fraction": config.train_fraction,
        "code_revision": git_revision(root),
        "splits": splits,
    }
    _atomic_json(config.output_manifest, manifest)
    report = {
        "schema": "deltaomni.longvideobench_diagnostic_split_summary.v1",
        "benchmark_status": manifest["benchmark_status"],
        "source_overlap": 0,
        "coverage_videos": sum(len(values) for values in splits.values()),
        "coverage_questions": sum(
            len(value["questions"]) for values in splits.values() for value in values.values()
        ),
        "splits": {name: _summary(values) for name, values in splits.items()},
        "code_revision": manifest["code_revision"],
    }
    _atomic_json(config.report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a non-benchmark LongVideoBench train/val")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/longvideobench_diagnostic_split.yaml")
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
