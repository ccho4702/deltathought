from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from deltaomni.backbones import load_backbone_config
from deltaomni.evaluation import cross_label_permutations
from deltaomni.language import DeltaLanguageProjector, FrozenCausalCaptionBackend
from deltaomni.model import ModalityDeltaCodec
from deltaomni.provenance import audit as audit_provenance
from deltaomni.ssv2_pilot import PilotConfig, _load_embeddings, load_pilot_config
from deltaomni.train_sanity import _atomic_json, _set_seed


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _latest_delta_checkpoint(config: PilotConfig) -> tuple[dict[str, Any], Path]:
    summary_path = config.output_root / "latest_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Run the SSV2 delta pilot before caption alignment")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir = config.output_root / summary["run_id"]
    checkpoints = sorted(run_dir.glob("checkpoints/step-*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No delta checkpoint under {run_dir}")
    return summary, checkpoints[-1]


@torch.no_grad()
def _conditioning(codec: ModalityDeltaCodec, full: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    codec.eval()
    anchor = full[:, 0]
    previous = anchor
    slots = torch.zeros(
        full.shape[0],
        codec.delta_encoder.queries.shape[0],
        full.shape[-1],
        device=full.device,
    )
    last_delta = None
    for time_index in range(1, full.shape[1]):
        current = full[:, time_index]
        last_delta = codec.delta_encoder(previous, current)
        slots = codec.accumulator(slots, last_delta)
        previous = current
    if last_delta is None:
        raise ValueError("Caption pilot requires at least two frames")
    return anchor, slots, last_delta


def _conditioning_train(
    codec: ModalityDeltaCodec,
    full: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    anchor = full[:, 0]
    previous = anchor
    slots = torch.zeros(
        full.shape[0],
        codec.delta_encoder.queries.shape[0],
        full.shape[-1],
        device=full.device,
    )
    losses = []
    last_delta = None
    for time_index in range(1, full.shape[1]):
        current = full[:, time_index]
        last_delta = codec.delta_encoder(previous, current)
        slots = codec.accumulator(slots, last_delta)
        reconstructed = codec.reconstructor(anchor, slots)
        losses.append(F.smooth_l1_loss(reconstructed, current))
        previous = current
    if last_delta is None:
        raise ValueError("Caption pilot requires at least two frames")
    return anchor, slots, last_delta, torch.stack(losses).mean()


@torch.no_grad()
def _evaluate_condition(
    backend: FrozenCausalCaptionBackend,
    projector: DeltaLanguageProjector,
    anchors: Tensor,
    delta_states: Tensor,
    labels: Tensor,
    config: PilotConfig,
) -> dict[str, float]:
    projector.eval()
    prefix = projector(anchors, delta_states, 1)
    candidate_losses = backend.candidate_caption_losses(
        prefix,
        config.caption.prompt,
        config.caption.targets,
    )
    predicted = candidate_losses.argmin(dim=-1)
    target_losses = candidate_losses.gather(1, labels[:, None]).squeeze(1)
    return {
        "accuracy": float(predicted.eq(labels).float().mean()),
        "target_nll": float(target_losses.mean()),
    }


def _mean_condition_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        raise ValueError("condition metrics cannot be empty")
    result = {
        key: sum(value[key] for value in values) / len(values)
        for key in values[0]
    }
    result["accuracy_min"] = min(value["accuracy"] for value in values)
    result["accuracy_max"] = max(value["accuracy"] for value in values)
    return result


def run(
    config_path: Path,
    backbone_config_path: Path,
    provenance_path: Path,
) -> dict[str, Any]:
    config = load_pilot_config(config_path)
    backbone_config = load_backbone_config(backbone_config_path)
    provenance = audit_provenance(provenance_path)
    device = torch.device(config.device)
    manifest = json.loads((config.cache_root / "manifest.json").read_text(encoding="utf-8"))
    delta_summary, delta_checkpoint = _latest_delta_checkpoint(config)
    train_full, train_labels = _load_embeddings(manifest, "train")
    validation_full, validation_labels = _load_embeddings(manifest, "validation")
    train_full = train_full.to(device)
    validation_full = validation_full.to(device)
    train_labels = train_labels.to(device)
    validation_labels = validation_labels.to(device)

    codec = ModalityDeltaCodec(config.model).to(device)
    payload = torch.load(delta_checkpoint, map_location=device, weights_only=False)
    codec.load_state_dict(payload["model"])
    train_anchor, train_delta, train_last = _conditioning(codec, train_full)
    validation_anchor, validation_delta, validation_last = _conditioning(codec, validation_full)
    if not config.caption.train_delta_encoder:
        del codec, train_full, validation_full
        torch.cuda.empty_cache()

    backend = FrozenCausalCaptionBackend(
        backbone_config.language,
        backbone_config.cache_dir,
        device,
        provenance,
    )
    _set_seed(config.seed)
    projector = DeltaLanguageProjector(config.model.embedding_dim, backend.hidden_size).to(device)
    projector_parameters = list(projector.parameters())
    trainable_parameters = list(projector_parameters)
    parameter_groups: list[dict[str, Any]] = [
        {"params": projector_parameters, "lr": config.caption.learning_rate}
    ]
    if config.caption.train_delta_encoder:
        codec_parameters = [
            *codec.delta_encoder.parameters(),
            *codec.accumulator.parameters(),
            *codec.reconstructor.parameters(),
        ]
        trainable_parameters.extend(codec_parameters)
        parameter_groups.append(
            {"params": codec_parameters, "lr": config.caption.delta_learning_rate}
        )
    optimizer = torch.optim.AdamW(parameter_groups)
    initial = _evaluate_condition(
        backend,
        projector,
        validation_anchor,
        validation_delta,
        validation_labels,
        config,
    )
    run_id = f"ssv2-caption-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = config.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    for step in range(1, config.caption.max_steps + 1):
        generator = torch.Generator().manual_seed(config.seed * 1_000_003 + step)
        indices = torch.randint(
            0,
            train_anchor.shape[0],
            (config.caption.batch_size,),
            generator=generator,
        ).to(device)
        projector.train()
        reconstruction_loss = torch.zeros((), device=device)
        if config.caption.train_delta_encoder:
            codec.train()
            anchor, delta_state, _, reconstruction_loss = _conditioning_train(
                codec,
                train_full[indices],
            )
        else:
            anchor = train_anchor[indices]
            delta_state = train_delta[indices]
        optimizer.zero_grad(set_to_none=True)
        prefix = projector(anchor, delta_state, 1)
        candidate_losses = backend.candidate_caption_losses(
            prefix,
            config.caption.prompt,
            config.caption.targets,
        )
        label = train_labels[indices]
        caption_loss = candidate_losses.gather(1, label[:, None]).mean()
        ranking_loss = F.cross_entropy(-candidate_losses, label)
        loss = (
            caption_loss
            + config.caption.ranking_weight * ranking_loss
            + config.caption.reconstruction_weight * reconstruction_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()
        if step % 10 == 0 or step == config.caption.max_steps:
            elapsed = time.perf_counter() - started
            eta = elapsed / step * (config.caption.max_steps - step)
            print(
                f"caption_step={step}/{config.caption.max_steps} loss={float(loss):.5f} "
                f"rank={float(ranking_loss):.5f} "
                f"recon={float(reconstruction_loss):.5f} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
        if step % config.caption.checkpoint_interval_steps == 0:
            _atomic_torch_save(
                run_dir / "checkpoints" / f"step-{step:06d}.pt",
                {
                    "step": step,
                    "projector": projector.state_dict(),
                    "codec": codec.state_dict() if config.caption.train_delta_encoder else None,
                    "optimizer": optimizer.state_dict(),
                    "delta_checkpoint": str(delta_checkpoint),
                },
            )

    if config.caption.train_delta_encoder:
        train_anchor, train_delta, train_last = _conditioning(codec, train_full)
        validation_anchor, validation_delta, validation_last = _conditioning(
            codec,
            validation_full,
        )

    normal = _evaluate_condition(
        backend,
        projector,
        validation_anchor,
        validation_delta,
        validation_labels,
        config,
    )
    zero = _evaluate_condition(
        backend,
        projector,
        validation_anchor,
        torch.zeros_like(validation_delta),
        validation_labels,
        config,
    )
    last = _evaluate_condition(
        backend,
        projector,
        validation_anchor,
        validation_last,
        validation_labels,
        config,
    )
    shuffle_indices = cross_label_permutations(
        validation_labels,
        repeats=4,
        seed=config.seed + 20_000,
    )
    shuffled = _mean_condition_metrics(
        [
            _evaluate_condition(
                backend,
                projector,
                validation_anchor,
                validation_delta[indices],
                validation_labels,
                config,
            )
            for indices in shuffle_indices
        ]
    )
    metrics = {
        "initial_validation_accuracy": initial["accuracy"],
        "initial_validation_target_nll": initial["target_nll"],
        "validation_accuracy": normal["accuracy"],
        "validation_target_nll": normal["target_nll"],
        "zero_delta_accuracy": zero["accuracy"],
        "zero_delta_target_nll": zero["target_nll"],
        "last_delta_accuracy": last["accuracy"],
        "last_delta_target_nll": last["target_nll"],
        "shuffled_delta_accuracy": shuffled["accuracy"],
        "shuffled_delta_target_nll": shuffled["target_nll"],
        "chance_accuracy": 1 / len(config.caption.targets),
    }
    checks = {
        "validation_nll_decreased": (
            metrics["validation_target_nll"] < metrics["initial_validation_target_nll"]
        ),
        "accuracy_above_chance": metrics["validation_accuracy"] > metrics["chance_accuracy"],
        "normal_nll_beats_zero": (
            metrics["validation_target_nll"] < metrics["zero_delta_target_nll"]
        ),
        "normal_nll_beats_shuffled": (
            metrics["validation_target_nll"] < metrics["shuffled_delta_target_nll"]
        ),
        "accumulation_not_worse_than_last": (
            metrics["validation_accuracy"] >= metrics["last_delta_accuracy"]
        ),
    }
    passed = all(checks.values())
    report = {
        "run_id": run_id,
        "delta_run_id": delta_summary["run_id"],
        "train_delta_encoder": config.caption.train_delta_encoder,
        "status": "signal" if passed else "inconclusive",
        "metrics": metrics,
        "checks": checks,
        "passed": passed,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(run_dir / "summary.json", report)
    _atomic_json(config.output_root / "latest_caption_summary.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SSV2 delta-to-frozen-Qwen caption pilot")
    parser.add_argument("--config", type=Path, default=Path("configs/ssv2_pilot.yaml"))
    parser.add_argument("--backbones", type=Path, default=Path("configs/backbones.yaml"))
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    args = parser.parse_args()
    report = run(args.config, args.backbones, args.provenance)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
