from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from deltaomni.train_sanity import _atomic_json


@dataclass(frozen=True)
class TrialSpec:
    name: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class SearchConfig:
    base_config: Path
    seeds: tuple[int, ...]
    gpu_count: int
    required_pass_fraction: float
    trials: tuple[TrialSpec, ...]
    output_root: Path
    temp_root: Path


def load_config(path: Path) -> SearchConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    config = SearchConfig(
        base_config=resolve(raw["base_config"]),
        seeds=tuple(int(seed) for seed in raw["seeds"]),
        gpu_count=int(raw["gpu_count"]),
        required_pass_fraction=float(raw["required_pass_fraction"]),
        trials=tuple(
            TrialSpec(str(trial["name"]), dict(trial.get("overrides", {})))
            for trial in raw["trials"]
        ),
        output_root=resolve(raw["output_root"]),
        temp_root=resolve(raw["temp_root"]),
    )
    if len(config.seeds) < 3 or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("search requires at least three unique seeds")
    if config.gpu_count <= 0 or not 0 < config.required_pass_fraction <= 1:
        raise ValueError("invalid GPU count or pass fraction")
    if len({trial.name for trial in config.trials}) != len(config.trials):
        raise ValueError("trial names must be unique")
    return config


def _set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    current = mapping
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            raise ValueError(f"Cannot override {dotted_key}: {key} is not a mapping")
        current = child
    if keys[-1] not in current:
        raise ValueError(f"Cannot override unknown key {dotted_key}")
    current[keys[-1]] = value


def _absolute_paths(raw: dict[str, Any], base_path: Path) -> None:
    root = base_path.resolve().parent.parent
    for key in ("ssv2_config", "semantic_config", "output_root", "log_root"):
        candidate = Path(raw[key])
        raw[key] = str(candidate if candidate.is_absolute() else root / candidate)


def _atomic_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def _trial_config(
    config: SearchConfig,
    trial: TrialSpec,
    seed: int,
    evaluation_split: str,
) -> dict[str, Any]:
    raw = yaml.safe_load(config.base_config.read_text(encoding="utf-8"))
    _absolute_paths(raw, config.base_config)
    raw["seed"] = seed
    raw["evaluation_split"] = evaluation_split
    for key, value in trial.overrides.items():
        _set_nested(raw, key, value)
    return raw


def _run_trial(
    *,
    generated_config: Path,
    run_id: str,
    output_root: Path,
    gpu_count: int,
    log_path: Path,
) -> dict[str, Any]:
    summary_path = output_root / run_id / "summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={gpu_count}",
        "-m",
        "deltaomni.ssv2_semantic_token_pilot",
        "--config",
        str(generated_config),
        "--run-id",
        run_id,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("trial process has no output stream")
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if not summary_path.is_file():
        raise RuntimeError(f"Trial {run_id} exited {return_code} without a summary")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _causal_gap(summary: dict[str, Any]) -> float:
    metrics = summary["metrics"]
    normal = float(metrics["hard_normal"]["accuracy"])
    controls = (
        float(metrics["hard_zero"]["accuracy"]),
        float(metrics["hard_last"]["accuracy"]),
        float(metrics["hard_shuffled"]["accuracy_max"]),
    )
    return normal - max(controls)


def run(config_path: Path, search_id: str | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    search_id = search_id or (
        f"delta-search-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )
    search_dir = config.output_root / search_id
    generated_root = config.temp_root / search_id / "configs"
    search_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    validation_results: dict[str, list[dict[str, Any]]] = {}
    completed = 0
    total_validation = len(config.trials) * len(config.seeds)
    required = math.ceil(config.required_pass_fraction * len(config.seeds))
    for trial in config.trials:
        validation_results[trial.name] = []
        for seed_index, seed in enumerate(config.seeds):
            raw = _trial_config(config, trial, seed, "validation")
            generated = generated_root / f"{trial.name}-seed{seed}-validation.yaml"
            _atomic_yaml(generated, raw)
            run_id = f"{search_id}-{trial.name}-seed{seed}-validation"
            summary = _run_trial(
                generated_config=generated,
                run_id=run_id,
                output_root=Path(raw["output_root"]),
                gpu_count=config.gpu_count,
                log_path=search_dir / "logs" / f"{run_id}.log",
            )
            validation_results[trial.name].append(summary)
            completed += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / completed * (total_validation - completed)
            print(
                f"search_validation={completed}/{total_validation} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
            _atomic_json(
                search_dir / "validation_runs.json",
                {name: values for name, values in validation_results.items()},
            )
            passed_so_far = sum(
                bool(value["passed"]) for value in validation_results[trial.name]
            )
            remaining = len(config.seeds) - seed_index - 1
            if passed_so_far + remaining < required:
                print(
                    f"prune_trial={trial.name} passed={passed_so_far} "
                    f"remaining={remaining} required={required}",
                    flush=True,
                )
                break

    aggregates = {}
    for trial in config.trials:
        summaries = validation_results[trial.name]
        gaps = [_causal_gap(summary) for summary in summaries]
        passed = sum(bool(summary["passed"]) for summary in summaries)
        aggregates[trial.name] = {
            "passed_seeds": passed,
            "required_passed_seeds": required,
            "eligible": passed >= required,
            "mean_causal_gap": sum(gaps) / len(gaps),
            "worst_seed_causal_gap": min(gaps),
            "run_ids": [summary["run_id"] for summary in summaries],
        }
    eligible = [trial for trial in config.trials if aggregates[trial.name]["eligible"]]
    if not eligible:
        report = {
            "search_id": search_id,
            "status": "no_delta_setting_passed",
            "aggregates": aggregates,
            "selected_trial": None,
            "test_runs": [],
            "passed": False,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_json(search_dir / "summary.json", report)
        _atomic_json(config.output_root / "latest_summary.json", report)
        return report

    selected = max(
        eligible,
        key=lambda trial: (
            aggregates[trial.name]["worst_seed_causal_gap"],
            aggregates[trial.name]["mean_causal_gap"],
        ),
    )
    test_results = []
    for seed in config.seeds:
        raw = _trial_config(config, selected, seed, "test")
        generated = generated_root / f"{selected.name}-seed{seed}-test.yaml"
        _atomic_yaml(generated, raw)
        run_id = f"{search_id}-{selected.name}-seed{seed}-test"
        test_results.append(
            _run_trial(
                generated_config=generated,
                run_id=run_id,
                output_root=Path(raw["output_root"]),
                gpu_count=config.gpu_count,
                log_path=search_dir / "logs" / f"{run_id}.log",
            )
        )
    test_passed = all(bool(summary["passed"]) for summary in test_results)
    report = {
        "search_id": search_id,
        "status": "delta_gate_passed" if test_passed else "test_gate_failed",
        "aggregates": aggregates,
        "selected_trial": selected.name,
        "selection_rule": "eligible pass fraction, then worst-seed and mean causal gap",
        "test_runs": [summary["run_id"] for summary in test_results],
        "test_causal_gaps": [_causal_gap(summary) for summary in test_results],
        "passed": test_passed,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(search_dir / "summary.json", report)
    _atomic_json(config.output_root / "latest_summary.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-seed A6000 delta setting search")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ssv2_delta_search_a6000.yaml"),
    )
    parser.add_argument("--search-id")
    args = parser.parse_args()
    report = run(args.config, args.search_id)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
