from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from deltaomni.run_integrity import git_revision, sha256_file
from deltaomni.train_sanity import _atomic_json

CONTROLS = ("multimodal", "text_only", "video_only", "audio_only")
PAIRS = (
    ("video_only", "multimodal"),
    ("video_only", "text_only"),
    ("multimodal", "text_only"),
    ("audio_only", "text_only"),
)


def _sign_test(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    smaller = min(left_only, right_only)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1))
    return min(1.0, 2 * tail / (2**discordant))


def analyze(rows: list[dict[str, Any]], *, seed: int, bootstrap_samples: int) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    values = {
        (str(row["source_id"]), str(row["question_id"]), str(row["control"])): int(
            row["correct"]
        )
        for row in rows
        if row.get("task") == "nextqa_multiple_choice"
    }
    qa_keys = sorted({(source, question) for source, question, _ in values})
    expected = {(source, question, control) for source, question in qa_keys for control in CONTROLS}
    missing = sorted(expected - set(values))
    if missing:
        raise ValueError(f"Missing matched control predictions: {missing[0]}")
    by_source: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in qa_keys:
        by_source[key[0]].append(key)
    sources = sorted(by_source)
    rng = np.random.default_rng(seed)
    accuracy_draws = {control: [] for control in CONTROLS}
    difference_draws = {pair: [] for pair in PAIRS}
    for _ in range(bootstrap_samples):
        sampled_sources = rng.choice(sources, size=len(sources), replace=True)
        sampled = [key for source in sampled_sources for key in by_source[source]]
        accuracy = {
            control: sum(values[key + (control,)] for key in sampled) / len(sampled)
            for control in CONTROLS
        }
        for control in CONTROLS:
            accuracy_draws[control].append(accuracy[control])
        for pair in PAIRS:
            difference_draws[pair].append(accuracy[pair[0]] - accuracy[pair[1]])
    accuracy = {}
    for control in CONTROLS:
        correct = sum(values[key + (control,)] for key in qa_keys)
        low, high = np.quantile(accuracy_draws[control], (0.025, 0.975))
        accuracy[control] = {
            "correct": correct,
            "examples": len(qa_keys),
            "accuracy": correct / len(qa_keys),
            "cluster_bootstrap_95ci": [float(low), float(high)],
        }
    comparisons = {}
    for left, right in PAIRS:
        left_only = sum(
            values[key + (left,)] == 1 and values[key + (right,)] == 0 for key in qa_keys
        )
        right_only = sum(
            values[key + (left,)] == 0 and values[key + (right,)] == 1 for key in qa_keys
        )
        low, high = np.quantile(difference_draws[(left, right)], (0.025, 0.975))
        comparisons[f"{left}_minus_{right}"] = {
            "accuracy_difference": accuracy[left]["accuracy"] - accuracy[right]["accuracy"],
            "cluster_bootstrap_95ci": [float(low), float(high)],
            "left_only_correct": left_only,
            "right_only_correct": right_only,
            "paired_exact_sign_p": _sign_test(left_only, right_only),
        }
    return {
        "controls": accuracy,
        "comparisons": comparisons,
        "qa_examples": len(qa_keys),
        "source_clusters": len(sources),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def run(prediction_root: Path, output_path: Path, *, seed: int, bootstrap_samples: int) -> dict:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(prediction_root.glob("*.json"))
    ]
    result = {
        "schema": "deltaomni.vanilla_control_analysis.v1",
        "code_revision": git_revision(Path(__file__).resolve().parents[2]),
        "prediction_root": str(prediction_root),
        "analysis": analyze(rows, seed=seed, bootstrap_samples=bootstrap_samples),
        "prediction_files": len(rows),
    }
    signature_path = prediction_root.parent / "run_signature.json"
    if signature_path.is_file():
        result["run_signature_sha256"] = sha256_file(signature_path)
    _atomic_json(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze matched vanilla Omni modality controls")
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path(
            "outputs/baselines/qwen2_5_omni_vanilla_controls_v2/"
            "qwen2-5-omni-vanilla-controls-v2/predictions"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/qwen2_5_omni_vanilla_controls_v2_analysis.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    args = parser.parse_args()
    result = run(
        args.prediction_root,
        args.output,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
