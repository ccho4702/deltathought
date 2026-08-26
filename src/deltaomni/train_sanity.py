from __future__ import annotations

import argparse
import json
import os
import random
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from deltaomni.config import SanityConfig, load_config
from deltaomni.interleaving import StreamingDeltaEngine, render_interleaving
from deltaomni.model import DeltaCodecModel, PairDeltaEncoder
from deltaomni.synthetic import SyntheticInterleavedDataset, collate_examples, token_text


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(run_dir.glob("checkpoints/step-*.pt"))
    return checkpoints[-1] if checkpoints else None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _loader(
    dataset: SyntheticInterleavedDataset,
    config: SanityConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed + (1 if shuffle else 2))
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.training.num_workers,
        collate_fn=collate_examples,
        generator=generator,
    )


def _training_batch(
    dataset: SyntheticInterleavedDataset,
    config: SanityConfig,
    step: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(config.seed * 1_000_003 + step)
    indices = torch.randint(
        0,
        len(dataset),
        (config.training.batch_size,),
        generator=generator,
    )
    return collate_examples([dataset[int(index)] for index in indices])


@torch.no_grad()
def evaluate(
    model: DeltaCodecModel,
    loader: DataLoader,
    config: SanityConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, list[float]] = {}
    for batch in loader:
        moved = {key: value.to(device) for key, value in batch.items()}
        losses = model.forward_sequence(**moved, weights=config.loss).detached()
        for key, value in losses.items():
            totals.setdefault(key, []).append(value)
    return {key: sum(values) / len(values) for key, values in totals.items()}


def _render_teacher_interleaving(
    model: DeltaCodecModel,
    example: Any,
    config: SanityConfig,
) -> str:
    engine = StreamingDeltaEngine(model, config.stream)
    initial = {
        modality: example.full_embeddings[0, index].unsqueeze(0)
        for index, modality in enumerate(config.modalities)
    }
    events = engine.initialize(0.0, initial)
    for time_index in range(1, example.full_embeddings.shape[0]):
        embeddings = {
            modality: example.full_embeddings[time_index, index].unsqueeze(0)
            for index, modality in enumerate(config.modalities)
        }
        forced = {
            modality
            for index, modality in enumerate(config.modalities)
            if bool(example.commit_targets[time_index, index])
        }
        events.extend(
            engine.step(
                float(time_index),
                embeddings,
                force_commits=forced,
                token_text=token_text(),
            )
        )
    return render_interleaving(events)


def train(
    config: SanityConfig,
    *,
    run_id: str | None = None,
    stop_after_step: int | None = None,
) -> dict[str, Any]:
    if config.training.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"Configured device {config.training.device!r} is unavailable")
    torch.set_num_threads(config.training.cpu_threads)
    _set_seed(config.seed)
    device = torch.device(config.training.device)
    run_id = run_id or f"delta-sanity-{_timestamp()}-{uuid.uuid4().hex[:8]}"
    run_dir = config.training.run_root / run_id
    log_path = config.training.log_root / run_id / "metrics.jsonl"
    if run_dir.exists() and config.training.resume != "auto":
        raise FileExistsError(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved = asdict(config)
    resolved["modalities"] = [modality.value for modality in config.modalities]
    config_signature = json.dumps(resolved, sort_keys=True, default=str)
    _atomic_json(run_dir / "resolved_config.json", resolved)
    _atomic_json(
        run_dir / "status.json",
        {"run_id": run_id, "status": "running", "started_at_utc": datetime.now(UTC).isoformat()},
    )

    train_dataset = SyntheticInterleavedDataset(config, config.training.examples, split_seed=10_000)
    validation_dataset = SyntheticInterleavedDataset(
        config,
        config.training.validation_examples,
        split_seed=20_000,
    )
    validation_loader = _loader(validation_dataset, config, shuffle=False)
    model = DeltaCodecModel(config.model, config.modalities).to(device)
    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate)
    start_step = 0
    initial: dict[str, float] | None = None

    checkpoint = _latest_checkpoint(run_dir) if config.training.resume == "auto" else None
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload["config_signature"] != config_signature:
            raise ValueError("Checkpoint configuration is incompatible with the current run")
        if payload.get("delta_algorithm") != PairDeltaEncoder.ALGORITHM_VERSION:
            raise ValueError(
                "Checkpoint delta algorithm is incompatible with the current code: "
                f"{payload.get('delta_algorithm', 'unversioned')} != "
                f"{PairDeltaEncoder.ALGORITHM_VERSION}"
            )
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
        initial = payload.get("initial_validation_losses")
        random.setstate(payload["python_rng_state"])
        torch.random.set_rng_state(payload["torch_rng_state"])
        _append_jsonl(
            log_path,
            {"event": "resume", "step": start_step, "checkpoint": str(checkpoint)},
        )
        if start_step >= config.training.max_steps:
            summary_path = run_dir / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError("Completed checkpoint has no retained summary")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            _atomic_json(
                run_dir / "status.json",
                {
                    "run_id": run_id,
                    "status": summary["status"],
                    "completed_at_utc": summary["completed_at_utc"],
                    "resume_noop_at_utc": datetime.now(UTC).isoformat(),
                },
            )
            return summary

    if initial is None:
        initial = evaluate(model, validation_loader, config, device)
    final_step = min(config.training.max_steps, stop_after_step or config.training.max_steps)
    if final_step <= start_step:
        raise ValueError("stop_after_step must be greater than the resumed checkpoint step")
    started = time.perf_counter()
    for step in range(start_step + 1, final_step + 1):
        batch = _training_batch(train_dataset, config, step)
        moved = {key: value.to(device) for key, value in batch.items()}
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_output = model.forward_sequence(**moved, weights=config.loss)
        loss_output.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        metrics = {"event": "train", "step": step, **loss_output.detached()}
        _append_jsonl(log_path, metrics)

        if step % 10 == 0 or step == final_step:
            elapsed = time.perf_counter() - started
            completed = step - start_step
            eta = elapsed / completed * (final_step - step)
            print(
                f"step={step}/{final_step} loss={metrics['total']:.4f} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
        if step % config.training.checkpoint_interval_steps == 0 or step == final_step:
            _atomic_checkpoint(
                run_dir / "checkpoints" / f"step-{step:06d}.pt",
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "delta_algorithm": PairDeltaEncoder.ALGORITHM_VERSION,
                    "config_signature": config_signature,
                    "python_rng_state": random.getstate(),
                    "torch_rng_state": torch.random.get_rng_state(),
                    "initial_validation_losses": initial,
                },
            )

    if final_step < config.training.max_steps:
        current = evaluate(model, validation_loader, config, device)
        interrupted = {
            "run_id": run_id,
            "status": "interrupted",
            "step": final_step,
            "initial_validation_losses": initial,
            "current_validation_losses": current,
            "passed": False,
            "interrupted_at_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_json(run_dir / "interruption.json", interrupted)
        _atomic_json(
            run_dir / "status.json",
            {
                "run_id": run_id,
                "status": "interrupted",
                "step": final_step,
                "interrupted_at_utc": interrupted["interrupted_at_utc"],
            },
        )
        return interrupted

    final = evaluate(model, validation_loader, config, device)
    interleaving = _render_teacher_interleaving(model, validation_dataset[0], config)
    improved = {key: final[key] < initial[key] for key in initial}
    required = ("total", "reconstruction", "trigger", "caption", "length")
    passed = all(improved[key] for key in required)
    summary = {
        "run_id": run_id,
        "status": "complete" if passed else "failed_sanity",
        "initial_validation_losses": initial,
        "final_validation_losses": final,
        "loss_decreased": improved,
        "required_losses": list(required),
        "passed": passed,
        "interleaving": interleaving,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(run_dir / "summary.json", summary)
    _atomic_json(
        run_dir / "status.json",
        {
            "run_id": run_id,
            "status": summary["status"],
            "completed_at_utc": summary["completed_at_utc"],
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the deterministic DeltaOmni sanity model")
    parser.add_argument("--config", type=Path, default=Path("configs/sanity.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    summary = train(
        load_config(args.config),
        run_id=args.run_id,
        stop_after_step=args.stop_after_step,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] or summary["status"] == "interrupted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
