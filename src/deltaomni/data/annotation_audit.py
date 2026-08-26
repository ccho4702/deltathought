from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from deltaomni.data.audioset_strong import inspect_tsv
from deltaomni.train_sanity import _atomic_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _label_map(path: Path) -> dict[str, str]:
    labels = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split("\t")
        if len(fields) != 2 or not all(field.strip() for field in fields):
            raise ValueError(f"Malformed label map row {line_number}")
        if fields[0] in labels:
            raise ValueError(f"Duplicate AudioSet class ID: {fields[0]}")
        labels[fields[0]] = fields[1]
    return labels


def audit_audioset(annotation_dir: Path) -> dict[str, Any]:
    train_path = annotation_dir / "audioset_train_strong.tsv"
    eval_path = annotation_dir / "audioset_eval_strong.tsv"
    label_path = annotation_dir / "mid_to_display_name.tsv"
    train_file = inspect_tsv(train_path)
    eval_file = inspect_tsv(eval_path)
    train = train_file.events
    evaluation = eval_file.events
    labels = _label_map(label_path)
    train_events = [event for events in train.values() for event in events]
    eval_events = [event for events in evaluation.values() for event in events]
    class_ids = {event.class_id for event in train_events + eval_events}
    invalid_bounds = [
        event
        for event in train_events + eval_events
        if event.start_seconds < 0 or event.end_seconds > 10
    ]
    known_zero_duration = {
        ("vqj-pMnJ9Zg_18000", 9.260, 9.260, "/m/07rbp7_"),
    }
    observed_invalid = {
        (
            invalid.event.clip_id,
            invalid.event.start_seconds,
            invalid.event.end_seconds,
            invalid.event.class_id,
        )
        for invalid in train_file.invalid_events + eval_file.invalid_events
    }
    checks = {
        "train_clips": len(train) == 103_463,
        "train_events": train_file.data_rows == 934_821,
        "eval_clips": len(evaluation) == 16_996,
        "eval_events": eval_file.data_rows == 139_538,
        "label_rows": len(labels) == 456,
        "all_event_classes_named": not (class_ids - labels.keys()),
        "all_bounds_within_ten_seconds": not invalid_bounds,
        "known_invalid_rows_only": observed_invalid == known_zero_duration,
    }
    return {
        "train_clips": len(train),
        "train_events": train_file.data_rows,
        "train_usable_events": len(train_events),
        "eval_clips": len(evaluation),
        "eval_events": eval_file.data_rows,
        "eval_usable_events": len(eval_events),
        "event_classes": len(class_ids),
        "label_rows": len(labels),
        "quarantined_events": [
            {
                "line_number": invalid.line_number,
                "clip_id": invalid.event.clip_id,
                "start_seconds": invalid.event.start_seconds,
                "end_seconds": invalid.event.end_seconds,
                "class_id": invalid.event.class_id,
                "reason": invalid.reason,
            }
            for invalid in train_file.invalid_events + eval_file.invalid_events
        ],
        "checks": checks,
        "passed": all(checks.values()),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (train_path, eval_path, label_path)
        },
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def audit_nextqa(annotation_dir: Path) -> dict[str, Any]:
    paths = {split: annotation_dir / f"{split}.csv" for split in ("train", "val", "test")}
    rows = {split: _read_csv(path) for split, path in paths.items()}
    mapping_path = annotation_dir / "map_vid_vidorID.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    expected_rows = {"train": 34_132, "val": 4_996, "test": 8_564}
    expected_videos = {"train": 3_870, "val": 570, "test": 1_000}
    video_sets = {split: {row["video"] for row in values} for split, values in rows.items()}
    all_rows = [row for values in rows.values() for row in values]
    required_columns = {"video", "question", "answer", "qid", "a0", "a1", "a2", "a3", "a4"}
    checks = {
        "row_counts": all(
            len(rows[split]) == expected for split, expected in expected_rows.items()
        ),
        "video_counts": all(
            len(video_sets[split]) == expected for split, expected in expected_videos.items()
        ),
        "split_video_disjoint": not (
            video_sets["train"] & video_sets["val"]
            or video_sets["train"] & video_sets["test"]
            or video_sets["val"] & video_sets["test"]
        ),
        "required_columns": all(required_columns <= row.keys() for row in all_rows),
        "answer_indices_valid": all(row["answer"] in {"0", "1", "2", "3", "4"} for row in all_rows),
        "questions_nonempty": all(row["question"].strip() for row in all_rows),
        "mapping_covers_videos": set().union(*video_sets.values()) <= set(mapping),
    }
    return {
        "rows": {split: len(values) for split, values in rows.items()},
        "videos": {split: len(values) for split, values in video_sets.items()},
        "mapping_rows": len(mapping),
        "checks": checks,
        "passed": all(checks.values()),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (*paths.values(), mapping_path)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit downloaded official annotation files")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/annotation_audit.json"),
    )
    args = parser.parse_args()
    data_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    configured_root = Path(data_config["raw_root"])
    project_root = args.config.resolve().parent.parent
    raw_root = args.raw_root or (
        configured_root if configured_root.is_absolute() else project_root / configured_root
    )
    report = {
        "raw_root": str(raw_root),
        "audioset_strong": audit_audioset(raw_root / "audioset_strong/annotations"),
        "nextqa": audit_nextqa(raw_root / "nextqa/annotations"),
    }
    report["passed"] = report["audioset_strong"]["passed"] and report["nextqa"]["passed"]
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
