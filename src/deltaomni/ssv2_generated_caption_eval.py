from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from deltaomni.backbones import load_backbone_config
from deltaomni.language import FrozenCausalCaptionBackend, SemanticTokenLanguageAdapter
from deltaomni.provenance import audit as audit_provenance
from deltaomni.ssv2_semantic_caption_pilot import _load_delta_model, _tokens, load_config
from deltaomni.ssv2_semantic_token_pilot import (
    CachedEmbeddingSplit,
    _assert_evaluation_checkpoint_compatible,
    _latest_checkpoint,
)
from deltaomni.train_sanity import _atomic_json


def _atomic_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@torch.no_grad()
def run(
    config_path: Path,
    checkpoint_run_id: str,
    *,
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, object]:
    if batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError("batch_size and max_new_tokens must be positive")
    config = load_config(config_path)
    if not config.runtime.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("generated caption evaluation requires an available CUDA device")
    device = torch.device("cuda:0")
    delta_model, ssv2 = _load_delta_model(config, device)
    manifest = json.loads((ssv2.cache_root / "manifest.json").read_text(encoding="utf-8"))
    split = CachedEmbeddingSplit(manifest, config.evaluation_split)
    backbone = load_backbone_config(Path("configs/backbones.yaml"))
    backend = FrozenCausalCaptionBackend(
        backbone.language_large,
        backbone.cache_dir,
        device,
        audit_provenance(Path("configs/provenance.yaml")),
        dtype=torch.bfloat16,
    )
    adapter = SemanticTokenLanguageAdapter(
        delta_model.bottleneck.codebook.shape[1],
        backend.hidden_size,
    ).to(device)
    checkpoint = _latest_checkpoint(config.output_root / checkpoint_run_id)
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint for caption run {checkpoint_run_id}")
    checkpoint_path, payload = checkpoint
    signature = json.dumps(asdict(config), sort_keys=True, default=str)
    _assert_evaluation_checkpoint_compatible(payload["config_signature"], signature)
    adapter.load_state_dict(payload["model"])
    adapter.eval()

    started = time.perf_counter()
    records: list[dict[str, object]] = []
    predictions: Counter[str] = Counter()
    for start in range(0, len(split), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(split)))
        full, labels = split.load_batch(indices, None)
        slots, _, _ = delta_model.condition(full.to(device))
        prefix = adapter(_tokens(delta_model, slots, hard=config.hard_tokens), 1)
        generated = backend.generate_captions(
            prefix,
            ssv2.caption.prompt,
            max_new_tokens=max_new_tokens,
        )
        for index, label, caption in zip(
            indices.tolist(), labels.tolist(), generated, strict=True
        ):
            normalized = caption.strip().lower()
            target = ssv2.caption.targets[label]
            predictions[normalized] += 1
            records.append(
                {
                    "source_id": split.records[index]["source_id"],
                    "class_index": label,
                    "target": target,
                    "generated": normalized,
                    "exact": normalized == target,
                }
            )
        completed = len(records)
        elapsed = time.perf_counter() - started
        eta = elapsed / completed * (len(split) - completed)
        print(
            f"generated={completed}/{len(split)} elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )

    correct = sum(bool(record["exact"]) for record in records)
    valid_targets = set(ssv2.caption.targets)
    out_of_set = sum(
        count for caption, count in predictions.items() if caption not in valid_targets
    )
    output_dir = (
        config.output_root
        / checkpoint_run_id
        / "evaluations"
        / f"{config.evaluation_split}_generated"
    )
    summary: dict[str, object] = {
        "schema": "deltaomni.ssv2_generated_caption.v1",
        "checkpoint_run_id": checkpoint_run_id,
        "checkpoint_path": str(checkpoint_path),
        "evaluation_split": config.evaluation_split,
        "count": len(records),
        "correct": correct,
        "exact_accuracy": correct / len(records),
        "out_of_target_set": out_of_set,
        "predictions": dict(sorted(predictions.items())),
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_jsonl(output_dir / "captions.jsonl", records)
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate greedy SSV2 captions")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-run-id", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    summary = run(
        args.config,
        args.checkpoint_run_id,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
