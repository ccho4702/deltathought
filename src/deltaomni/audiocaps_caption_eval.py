from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import torch
from peft import set_peft_model_state_dict

from deltaomni.audiocaps_caption_lora import (
    PrefixDataset,
    _load_model,
    evaluate,
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


def _checks(metrics: dict, minimum_gap: float) -> dict[str, bool]:
    return {
        "normal_nll_beats_zero": metrics["nll"]["normal"] < metrics["nll"]["zero"],
        "normal_nll_beats_shuffled": (metrics["nll"]["normal"] < metrics["nll"]["shuffled"]),
        "word_f1_beats_zero": (
            metrics["generation"]["normal"]["word_f1"]
            >= metrics["generation"]["zero"]["word_f1"] + minimum_gap
        ),
        "word_f1_beats_shuffled": (
            metrics["generation"]["normal"]["word_f1"]
            >= metrics["generation"]["shuffled"]["word_f1"] + minimum_gap
        ),
    }


def run(
    config_path: Path,
    checkpoint_path: Path,
    output_path: Path,
) -> dict:
    if output_path.exists():
        raise FileExistsError(f"Caption evaluation output already exists: {output_path}")
    config = load_config(config_path)
    signature = run_signature(config)
    legacy_signature = json.dumps(asdict(config), sort_keys=True, default=str)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("signature") not in {signature, legacy_signature}:
        raise ValueError("Caption evaluation checkpoint/configuration mismatch")
    signature_version = (
        "content-sha256-v2"
        if payload.get("signature") == signature
        else "legacy-path-only-v1"
    )
    _set_seed(config.seed)
    torch.set_num_threads(config.runtime.cpu_threads)
    device = torch.device(config.runtime.device)
    model, _ = _load_model(config, device)
    set_peft_model_state_dict(model.thinker, payload["lora"])
    model.adapter.load_state_dict(payload["adapter"])
    manifest = json.loads(config.prefix_manifest.read_text(encoding="utf-8"))
    test = PrefixDataset(manifest, "test", config.runtime.cache_entries)
    evaluation = replace(config.evaluation, nll_examples=len(test))
    evaluation_config = replace(config, evaluation=evaluation)
    metrics = evaluate(model, test, evaluation_config, device)
    checks = _checks(metrics, config.evaluation.minimum_control_gap)
    code_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=config_path.resolve().parent.parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema": "deltaomni.audiocaps_caption_evaluation.v1",
        "split": "test",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "training_code_revision": payload.get("code_revision"),
        "evaluation_code_revision": code_revision,
        "training_signature_version": signature_version,
        "prefix_manifest_sha256": _sha256(config.prefix_manifest),
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate fixed AudioCaps Caption LoRA")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.checkpoint, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
