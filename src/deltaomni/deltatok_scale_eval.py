from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from deltaomni.deltatok import DeltaTok
from deltaomni.deltatok_scale_train import (
    PairDataset,
    _evaluate_pairs,
    _evaluate_rollout,
    _evaluation_checks,
    load_config,
    run_signature,
)
from deltaomni.train_sanity import _atomic_json, _set_seed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    output_path: Path,
) -> dict:
    if split not in {"validation", "test"}:
        raise ValueError("DeltaTok evaluation split must be validation or test")
    if output_path.exists():
        raise FileExistsError(f"Evaluation output already exists: {output_path}")
    config = load_config(config_path)
    _set_seed(config.seed)
    torch.set_num_threads(config.runtime.cpu_threads)
    manifest = json.loads(config.cache_manifest.read_text(encoding="utf-8"))
    data = PairDataset(manifest, split, config.runtime.cache_entries)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    signature = run_signature(config)
    legacy_signature = json.dumps(asdict(config), sort_keys=True, default=str)
    if payload.get("config_signature") not in {signature, legacy_signature}:
        raise ValueError("Evaluation checkpoint/configuration mismatch")
    signature_version = (
        "content-sha256-v2"
        if payload.get("config_signature") == signature
        else "legacy-path-only-v1"
    )
    device = torch.device(config.runtime.device)
    model = DeltaTok(config.model).to(device).eval()
    model.load_state_dict(payload["model"])
    teacher = _evaluate_pairs(model, data, config, device)
    rollout = _evaluate_rollout(model, data, device)
    checks = _evaluation_checks(teacher, rollout, config)
    code_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=config_path.resolve().parent.parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema": "deltaomni.deltatok_scale_evaluation.v1",
        "modality": config.modality,
        "split": split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "training_code_revision": payload.get("code_revision"),
        "evaluation_code_revision": code_revision,
        "training_signature_version": signature_version,
        "cache_manifest_sha256": _sha256(config.cache_manifest),
        "teacher_forced": teacher,
        "autoregressive_rollout": rollout,
        "checks": checks,
        "passed": all(checks.values()),
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a fixed DeltaTok checkpoint")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.checkpoint, args.split, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
