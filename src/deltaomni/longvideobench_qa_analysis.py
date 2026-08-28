from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from deltaomni.train_sanity import _atomic_json


@dataclass(frozen=True)
class AnalysisConfig:
    predictions: Path
    frozen_validation_manifest: Path
    required_arms: tuple[str, ...]
    comparisons: tuple[tuple[str, str], ...]
    bootstrap_samples: int
    seed: int
    report_path: Path


def load_config(path: Path) -> AnalysisConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    config = AnalysisConfig(
        predictions=resolve(raw["predictions"]),
        frozen_validation_manifest=resolve(raw["frozen_validation_manifest"]),
        required_arms=tuple(str(value) for value in raw["required_arms"]),
        comparisons=tuple(
            (str(value["left"]), str(value["right"])) for value in raw["comparisons"]
        ),
        bootstrap_samples=int(raw["bootstrap_samples"]),
        seed=int(raw["seed"]),
        report_path=resolve(raw["report_path"]),
    )
    if not config.required_arms or config.bootstrap_samples <= 0:
        raise ValueError("LongVideoBench analysis requires arms and bootstrap samples")
    if len(set(config.required_arms)) != len(config.required_arms):
        raise ValueError("LongVideoBench analysis contains duplicate arms")
    unknown = {
        arm for pair in config.comparisons for arm in pair if arm not in config.required_arms
    }
    if unknown:
        raise ValueError(f"LongVideoBench comparisons contain unknown arms: {sorted(unknown)}")
    return config


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            required = {
                "arm",
                "id",
                "video_id",
                "prediction",
                "parsed_choice",
                "correct_choice",
                "duration_group",
                "question_category",
            }
            if not isinstance(value, dict) or not required <= value.keys():
                raise ValueError(f"Malformed LongVideoBench prediction line {line_number}")
            rows.append(value)
    return rows


def _paired_exact_p(left: list[bool], right: list[bool]) -> float:
    left_only = sum(a and not b for a, b in zip(left, right, strict=True))
    right_only = sum(b and not a for a, b in zip(left, right, strict=True))
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _cluster_bootstrap(
    left: dict[str, list[float]],
    right: dict[str, list[float]],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    clusters = sorted(left)
    if clusters != sorted(right) or not clusters:
        raise ValueError("LongVideoBench paired clusters do not match")
    generator = random.Random(seed)
    differences = []
    for _ in range(samples):
        selected = [clusters[generator.randrange(len(clusters))] for _ in clusters]
        left_values = [value for cluster in selected for value in left[cluster]]
        right_values = [value for cluster in selected for value in right[cluster]]
        differences.append(
            sum(left_values) / len(left_values) - sum(right_values) / len(right_values)
        )
    differences.sort()
    return [
        differences[int(0.025 * (samples - 1))],
        differences[int(0.975 * (samples - 1))],
    ]


def analyze(
    rows: list[dict[str, Any]],
    frozen: dict[str, Any],
    config: AnalysisConfig,
) -> dict[str, Any]:
    if frozen.get("schema") != "deltaomni.longvideobench_frozen_validation.v1":
        raise ValueError("Unsupported frozen LongVideoBench validation manifest")
    frozen_questions = list(frozen["questions"])
    ids = [str(value["id"]) for value in frozen_questions]
    expected = set(ids)
    by_arm: dict[str, dict[str, dict[str, Any]]] = {
        arm: {} for arm in config.required_arms
    }
    for row in rows:
        arm = str(row["arm"])
        question_id = str(row["id"])
        if arm not in by_arm:
            raise ValueError(f"Unexpected LongVideoBench prediction arm: {arm}")
        if question_id not in expected:
            raise ValueError(
                f"Prediction is outside frozen LongVideoBench validation: {question_id}"
            )
        if question_id in by_arm[arm]:
            raise ValueError(f"Duplicate LongVideoBench prediction: {arm}/{question_id}")
        by_arm[arm][question_id] = row
    for arm, values in by_arm.items():
        missing = expected - values.keys()
        if missing:
            raise ValueError(f"LongVideoBench arm {arm} is missing {len(missing)} questions")

    def correct(row: dict[str, Any]) -> bool:
        parsed = row["parsed_choice"]
        return parsed is not None and int(parsed) == int(row["correct_choice"])

    arms = {}
    for arm, values in by_arm.items():
        ordered = [values[question_id] for question_id in ids]
        overall = [correct(row) for row in ordered]
        parsed = [row["parsed_choice"] is not None for row in ordered]
        duration: dict[str, list[bool]] = defaultdict(list)
        category: dict[str, list[bool]] = defaultdict(list)
        for row, is_correct in zip(ordered, overall, strict=True):
            duration[str(row["duration_group"])].append(is_correct)
            category[str(row["question_category"])].append(is_correct)
        arms[arm] = {
            "accuracy": sum(overall) / len(overall),
            "parse_rate": sum(parsed) / len(parsed),
            "questions": len(overall),
            "videos": len({str(row["video_id"]) for row in ordered}),
            "by_duration_group": {
                key: {"accuracy": sum(group) / len(group), "questions": len(group)}
                for key, group in sorted(duration.items())
            },
            "by_question_category": {
                key: {"accuracy": sum(group) / len(group), "questions": len(group)}
                for key, group in sorted(category.items())
            },
        }

    comparisons = {}
    for comparison_index, (left_name, right_name) in enumerate(config.comparisons):
        left_rows = [by_arm[left_name][question_id] for question_id in ids]
        right_rows = [by_arm[right_name][question_id] for question_id in ids]
        left_correct = [correct(row) for row in left_rows]
        right_correct = [correct(row) for row in right_rows]
        left_clusters: dict[str, list[float]] = defaultdict(list)
        right_clusters: dict[str, list[float]] = defaultdict(list)
        for left_row, right_row, left_value, right_value in zip(
            left_rows,
            right_rows,
            left_correct,
            right_correct,
            strict=True,
        ):
            if left_row["video_id"] != right_row["video_id"]:
                raise ValueError("LongVideoBench paired rows have different source videos")
            source = str(left_row["video_id"])
            left_clusters[source].append(float(left_value))
            right_clusters[source].append(float(right_value))
        difference = sum(left_correct) / len(ids) - sum(right_correct) / len(ids)
        key = f"{left_name}_minus_{right_name}"
        comparisons[key] = {
            "accuracy_difference": difference,
            "cluster_bootstrap_95ci": _cluster_bootstrap(
                left_clusters,
                right_clusters,
                samples=config.bootstrap_samples,
                seed=config.seed + comparison_index,
            ),
            "paired_exact_p": _paired_exact_p(left_correct, right_correct),
            "left_only_correct": sum(
                left and not right
                for left, right in zip(left_correct, right_correct, strict=True)
            ),
            "right_only_correct": sum(
                right and not left
                for left, right in zip(left_correct, right_correct, strict=True)
            ),
        }
    return {
        "schema": "deltaomni.longvideobench_qa_analysis.v1",
        "dataset_revision": frozen["dataset_revision"],
        "annotation_sha256": frozen["annotation_sha256"],
        "arms": arms,
        "comparisons": comparisons,
        "all_parse_rates_complete": all(value["parse_rate"] == 1.0 for value in arms.values()),
        "bootstrap_samples": config.bootstrap_samples,
        "source_clusters": len({str(row["video_id"]) for row in rows}),
    }


def run(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    frozen = json.loads(config.frozen_validation_manifest.read_text(encoding="utf-8"))
    report = analyze(_rows(config.predictions), frozen, config)
    _atomic_json(config.report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze frozen LongVideoBench QA controls")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/longvideobench_qa_analysis.yaml")
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
