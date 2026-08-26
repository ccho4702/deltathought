from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from torch.nn import functional as F

from deltaomni.backbones import load_backbone_config
from deltaomni.language import ChangeAwareResampler, FrozenCausalCaptionBackend
from deltaomni.model import ModalityDeltaCodec
from deltaomni.provenance import audit as audit_provenance
from deltaomni.ssv2_caption_pilot import _conditioning
from deltaomni.ssv2_pilot import _load_embeddings, load_pilot_config
from deltaomni.ssv2_semantic_pilot import load_config as load_semantic_config
from deltaomni.train_sanity import _atomic_json, _set_seed


@dataclass(frozen=True)
class ResamplerPilotConfig:
    seed: int
    device: str
    ssv2_config: Path
    semantic_config: Path
    query_tokens: int
    num_heads: int
    temperature: float
    alignment_steps: int
    alignment_learning_rate: float
    caption_steps: int
    caption_learning_rate: float
    caption_ranking_weight: float
    alignment_guard_weight: float
    output_root: Path


def load_config(path: Path) -> ResamplerPilotConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    return ResamplerPilotConfig(
        seed=int(raw["seed"]),
        device=str(raw["device"]),
        ssv2_config=resolve(raw["ssv2_config"]),
        semantic_config=resolve(raw["semantic_config"]),
        query_tokens=int(raw["query_tokens"]),
        num_heads=int(raw["num_heads"]),
        temperature=float(raw["temperature"]),
        alignment_steps=int(raw["alignment_steps"]),
        alignment_learning_rate=float(raw["alignment_learning_rate"]),
        caption_steps=int(raw["caption_steps"]),
        caption_learning_rate=float(raw["caption_learning_rate"]),
        caption_ranking_weight=float(raw["caption_ranking_weight"]),
        alignment_guard_weight=float(raw["alignment_guard_weight"]),
        output_root=resolve(raw["output_root"]),
    )


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _alignment_logits(prefix: Tensor, text_embeddings: Tensor, temperature: float) -> Tensor:
    prefix_embedding = F.normalize(prefix.mean(dim=1), dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    return prefix_embedding @ text_embeddings.T / temperature


@torch.no_grad()
def _alignment_metrics(
    resampler: ChangeAwareResampler,
    anchors: Tensor,
    deltas: Tensor,
    labels: Tensor,
    text_embeddings: Tensor,
    temperature: float,
) -> dict[str, float]:
    resampler.eval()
    logits = _alignment_logits(resampler(anchors, deltas, 1), text_embeddings, temperature)
    return {
        "accuracy": float(logits.argmax(dim=-1).eq(labels).float().mean()),
        "loss": float(F.cross_entropy(logits, labels)),
    }


@torch.no_grad()
def _caption_metrics(
    backend: FrozenCausalCaptionBackend,
    resampler: ChangeAwareResampler,
    anchors: Tensor,
    deltas: Tensor,
    labels: Tensor,
    prompt: str,
    targets: tuple[str, ...],
) -> dict[str, float]:
    resampler.eval()
    predictions = []
    target_losses = []
    for index in range(anchors.shape[0]):
        prefix = resampler(anchors[index : index + 1], deltas[index : index + 1], 1)
        losses = [float(backend.caption_loss(prefix, prompt, target)) for target in targets]
        predictions.append(min(range(len(losses)), key=losses.__getitem__))
        target_losses.append(losses[int(labels[index])])
    predictions_tensor = torch.tensor(predictions, device=labels.device)
    return {
        "accuracy": float(predictions_tensor.eq(labels).float().mean()),
        "target_nll": sum(target_losses) / len(target_losses),
    }


def run(
    config_path: Path,
    backbone_config_path: Path,
    provenance_path: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    ssv2 = load_pilot_config(config.ssv2_config)
    manifest = json.loads((ssv2.cache_root / "manifest.json").read_text(encoding="utf-8"))
    train_full, train_labels = _load_embeddings(manifest, "train")
    validation_full, validation_labels = _load_embeddings(manifest, "validation")
    device = torch.device(config.device)
    train_full, train_labels = train_full.to(device), train_labels.to(device)
    validation_full = validation_full.to(device)
    validation_labels = validation_labels.to(device)
    semantic_config = load_semantic_config(config.semantic_config)
    semantic_summary = json.loads(
        (semantic_config.output_root / "latest_summary.json").read_text(encoding="utf-8")
    )
    semantic_run_dir = semantic_config.output_root / semantic_summary["run_id"]
    semantic_checkpoints = sorted(semantic_run_dir.glob("checkpoints/step-*.pt"))
    if not semantic_checkpoints:
        raise FileNotFoundError(f"No semantic checkpoint under {semantic_run_dir}")
    delta_checkpoint = semantic_checkpoints[-1]
    codec = ModalityDeltaCodec(ssv2.model).to(device)
    semantic_payload = torch.load(delta_checkpoint, map_location=device, weights_only=False)
    codec.load_state_dict(semantic_payload["codec"])
    train_anchor, train_delta, train_last = _conditioning(codec, train_full)
    validation_anchor, validation_delta, validation_last = _conditioning(codec, validation_full)
    del codec, train_full, validation_full
    torch.cuda.empty_cache()

    backbone_config = load_backbone_config(backbone_config_path)
    backend = FrozenCausalCaptionBackend(
        backbone_config.language,
        backbone_config.cache_dir,
        device,
        audit_provenance(provenance_path),
    )
    text_embeddings = backend.encode_text(list(ssv2.caption.targets)).detach()
    _set_seed(config.seed)
    resampler = ChangeAwareResampler(
        input_dim=ssv2.model.embedding_dim,
        language_dim=backend.hidden_size,
        query_tokens=config.query_tokens,
        num_heads=config.num_heads,
    ).to(device)
    initial_alignment = _alignment_metrics(
        resampler,
        validation_anchor,
        validation_delta,
        validation_labels,
        text_embeddings,
        config.temperature,
    )
    optimizer = torch.optim.AdamW(
        resampler.parameters(),
        lr=config.alignment_learning_rate,
    )
    started = time.perf_counter()
    for step in range(1, config.alignment_steps + 1):
        generator = torch.Generator().manual_seed(config.seed * 1_000_003 + step)
        indices = torch.randint(0, train_anchor.shape[0], (8,), generator=generator).to(device)
        resampler.train()
        optimizer.zero_grad(set_to_none=True)
        prefix = resampler(train_anchor[indices], train_delta[indices], 1)
        logits = _alignment_logits(prefix, text_embeddings, config.temperature)
        loss = F.cross_entropy(logits, train_labels[indices])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(resampler.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0:
            elapsed = time.perf_counter() - started
            eta = elapsed / step * (config.alignment_steps - step)
            print(
                f"align_step={step}/{config.alignment_steps} loss={float(loss):.5f} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )

    aligned = _alignment_metrics(
        resampler,
        validation_anchor,
        validation_delta,
        validation_labels,
        text_embeddings,
        config.temperature,
    )
    aligned_zero = _alignment_metrics(
        resampler,
        validation_anchor,
        torch.zeros_like(validation_delta),
        validation_labels,
        text_embeddings,
        config.temperature,
    )
    aligned_last = _alignment_metrics(
        resampler,
        validation_anchor,
        validation_last,
        validation_labels,
        text_embeddings,
        config.temperature,
    )
    aligned_shuffled = _alignment_metrics(
        resampler,
        validation_anchor,
        validation_delta.roll(1, dims=0),
        validation_labels,
        text_embeddings,
        config.temperature,
    )

    optimizer = torch.optim.AdamW(resampler.parameters(), lr=config.caption_learning_rate)
    for step in range(1, config.caption_steps + 1):
        index = (config.seed * 97 + step * 17) % train_anchor.shape[0]
        label = train_labels[index].view(1)
        resampler.train()
        optimizer.zero_grad(set_to_none=True)
        prefix = resampler(train_anchor[index : index + 1], train_delta[index : index + 1], 1)
        candidate_losses = torch.stack(
            [
                backend.caption_loss(prefix, ssv2.caption.prompt, target)
                for target in ssv2.caption.targets
            ]
        )
        caption_loss = candidate_losses[label].squeeze(0)
        ranking_loss = F.cross_entropy((-candidate_losses).unsqueeze(0), label)
        alignment_loss = F.cross_entropy(
            _alignment_logits(prefix, text_embeddings, config.temperature),
            label,
        )
        loss = (
            caption_loss
            + config.caption_ranking_weight * ranking_loss
            + config.alignment_guard_weight * alignment_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(resampler.parameters(), 1.0)
        optimizer.step()
        if step % 20 == 0:
            print(
                f"resampler_caption_step={step}/{config.caption_steps} loss={float(loss):.5f}",
                flush=True,
            )

    caption = _caption_metrics(
        backend,
        resampler,
        validation_anchor,
        validation_delta,
        validation_labels,
        ssv2.caption.prompt,
        ssv2.caption.targets,
    )
    caption_zero = _caption_metrics(
        backend,
        resampler,
        validation_anchor,
        torch.zeros_like(validation_delta),
        validation_labels,
        ssv2.caption.prompt,
        ssv2.caption.targets,
    )
    caption_last = _caption_metrics(
        backend,
        resampler,
        validation_anchor,
        validation_last,
        validation_labels,
        ssv2.caption.prompt,
        ssv2.caption.targets,
    )
    caption_shuffled = _caption_metrics(
        backend,
        resampler,
        validation_anchor,
        validation_delta.roll(1, dims=0),
        validation_labels,
        ssv2.caption.prompt,
        ssv2.caption.targets,
    )
    metrics = {
        "initial_alignment": initial_alignment,
        "aligned": aligned,
        "aligned_zero": aligned_zero,
        "aligned_last": aligned_last,
        "aligned_shuffled": aligned_shuffled,
        "caption": caption,
        "caption_zero": caption_zero,
        "caption_last": caption_last,
        "caption_shuffled": caption_shuffled,
        "chance_accuracy": 1 / len(ssv2.caption.targets),
    }
    checks = {
        "alignment_above_chance": aligned["accuracy"] > metrics["chance_accuracy"],
        "alignment_beats_zero": aligned["accuracy"] > aligned_zero["accuracy"],
        "alignment_beats_shuffled": aligned["accuracy"] > aligned_shuffled["accuracy"],
        "caption_above_chance": caption["accuracy"] > metrics["chance_accuracy"],
        "caption_beats_zero": caption["accuracy"] > caption_zero["accuracy"],
        "caption_beats_shuffled": caption["accuracy"] > caption_shuffled["accuracy"],
    }
    passed = all(checks.values())
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"ssv2-resampler-{timestamp}-{uuid.uuid4().hex[:8]}"
    report = {
        "run_id": run_id,
        "source_semantic_run_id": semantic_summary["run_id"],
        "status": "signal" if passed else "inconclusive",
        "metrics": metrics,
        "checks": checks,
        "passed": passed,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    run_dir = config.output_root / run_id
    _atomic_json(run_dir / "summary.json", report)
    _atomic_torch_save(
        run_dir / "resampler.pt",
        {"resampler": resampler.state_dict(), "delta_checkpoint": str(delta_checkpoint)},
    )
    _atomic_json(config.output_root / "latest_summary.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run change-aware SSV2 resampler alignment pilot")
    parser.add_argument("--config", type=Path, default=Path("configs/ssv2_resampler_pilot.yaml"))
    parser.add_argument("--backbones", type=Path, default=Path("configs/backbones.yaml"))
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    args = parser.parse_args()
    report = run(args.config, args.backbones, args.provenance)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
