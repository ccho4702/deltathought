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
from torch import Tensor, nn
from torch.nn import functional as F

from deltaomni.evaluation import cross_label_permutations
from deltaomni.model import ModalityDeltaCodec
from deltaomni.ssv2_caption_pilot import _latest_delta_checkpoint
from deltaomni.ssv2_pilot import _evaluate_reconstruction, _load_embeddings, load_pilot_config
from deltaomni.train_sanity import _atomic_json, _set_seed


@dataclass(frozen=True)
class SemanticPilotConfig:
    seed: int
    device: str
    ssv2_config: Path
    batch_size: int
    learning_rate: float
    max_steps: int
    checkpoint_interval_steps: int
    reconstruction_weight: float
    semantic_weight: float
    shuffle_repeats: int
    output_root: Path


def load_config(path: Path) -> SemanticPilotConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    return SemanticPilotConfig(
        seed=int(raw["seed"]),
        device=str(raw["device"]),
        ssv2_config=resolve(raw["ssv2_config"]),
        batch_size=int(raw["batch_size"]),
        learning_rate=float(raw["learning_rate"]),
        max_steps=int(raw["max_steps"]),
        checkpoint_interval_steps=int(raw["checkpoint_interval_steps"]),
        reconstruction_weight=float(raw["reconstruction_weight"]),
        semantic_weight=float(raw["semantic_weight"]),
        shuffle_repeats=int(raw.get("shuffle_repeats", 8)),
        output_root=resolve(raw["output_root"]),
    )


class SemanticHead(nn.Module):
    def __init__(self, dimension: int, classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension // 2),
            nn.GELU(),
            nn.Linear(dimension // 2, classes),
        )

    def forward(self, slots: Tensor) -> Tensor:
        return self.network(slots.mean(dim=1))


def _forward(
    codec: ModalityDeltaCodec,
    head: SemanticHead,
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
        raise ValueError("Semantic pilot requires multiple frames")
    return head(slots), slots, last_delta, torch.stack(losses).mean()


@torch.no_grad()
def _semantic_metrics(
    codec: ModalityDeltaCodec,
    head: SemanticHead,
    full: Tensor,
    labels: Tensor,
    *,
    shuffle_repeats: int,
    seed: int,
) -> dict[str, float]:
    codec.eval()
    head.eval()
    logits, slots, last_delta, _ = _forward(codec, head, full)
    normal = float(logits.argmax(dim=-1).eq(labels).float().mean())
    zero = float(head(torch.zeros_like(slots)).argmax(dim=-1).eq(labels).float().mean())
    last = float(head(last_delta).argmax(dim=-1).eq(labels).float().mean())
    shuffled_accuracies = []
    for indices in cross_label_permutations(labels, repeats=shuffle_repeats, seed=seed):
        shuffled_accuracies.append(
            head(slots[indices]).argmax(dim=-1).eq(labels).float().mean()
        )
    shuffled = torch.stack(shuffled_accuracies)
    return {
        "normal": normal,
        "zero": zero,
        "last": last,
        "shuffled": float(shuffled.mean()),
        "shuffled_std": float(shuffled.std(unbiased=False)),
        "shuffled_min": float(shuffled.min()),
        "shuffled_max": float(shuffled.max()),
    }


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def run(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    ssv2 = load_pilot_config(config.ssv2_config)
    manifest = json.loads((ssv2.cache_root / "manifest.json").read_text(encoding="utf-8"))
    train_full, train_labels = _load_embeddings(manifest, "train")
    validation_full, validation_labels = _load_embeddings(manifest, "validation")
    device = torch.device(config.device)
    train_full, train_labels = train_full.to(device), train_labels.to(device)
    validation_full = validation_full.to(device)
    validation_labels = validation_labels.to(device)
    delta_summary, checkpoint = _latest_delta_checkpoint(ssv2)
    _set_seed(config.seed)
    codec = ModalityDeltaCodec(ssv2.model).to(device)
    codec.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    head = SemanticHead(ssv2.model.embedding_dim, len(ssv2.classes)).to(device)
    parameters = [
        *codec.delta_encoder.parameters(),
        *codec.accumulator.parameters(),
        *codec.reconstructor.parameters(),
        *head.parameters(),
    ]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    initial_reconstruction = _evaluate_reconstruction(codec, validation_full)
    initial_semantic = _semantic_metrics(
        codec,
        head,
        validation_full,
        validation_labels,
        shuffle_repeats=config.shuffle_repeats,
        seed=config.seed + 10_000,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"ssv2-semantic-{timestamp}-{uuid.uuid4().hex[:8]}"
    run_dir = config.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    for step in range(1, config.max_steps + 1):
        generator = torch.Generator().manual_seed(config.seed * 1_000_003 + step)
        indices = torch.randint(0, train_full.shape[0], (config.batch_size,), generator=generator)
        indices = indices.to(device)
        codec.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _, _, reconstruction_loss = _forward(codec, head, train_full[indices])
        semantic_loss = F.cross_entropy(logits, train_labels[indices])
        loss = (
            config.reconstruction_weight * reconstruction_loss
            + config.semantic_weight * semantic_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step % 25 == 0:
            elapsed = time.perf_counter() - started
            eta = elapsed / step * (config.max_steps - step)
            print(
                f"semantic_step={step}/{config.max_steps} loss={float(loss):.5f} "
                f"recon={float(reconstruction_loss):.5f} cls={float(semantic_loss):.5f} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
        if step % config.checkpoint_interval_steps == 0:
            _atomic_torch_save(
                run_dir / "checkpoints" / f"step-{step:06d}.pt",
                {"step": step, "codec": codec.state_dict(), "head": head.state_dict()},
            )

    reconstruction = _evaluate_reconstruction(codec, validation_full)
    semantic = _semantic_metrics(
        codec,
        head,
        validation_full,
        validation_labels,
        shuffle_repeats=config.shuffle_repeats,
        seed=config.seed + 20_000,
    )
    metrics = {
        "initial_reconstruction_mse": initial_reconstruction["mse"],
        "reconstruction_mse": reconstruction["mse"],
        "anchor_mse": reconstruction["anchor_mse"],
        "initial_semantic": initial_semantic,
        "semantic": semantic,
        "chance_accuracy": 1 / len(ssv2.classes),
    }
    checks = {
        "semantic_above_chance": semantic["normal"] > metrics["chance_accuracy"],
        "semantic_beats_zero": semantic["normal"] > semantic["zero"],
        "semantic_beats_last": semantic["normal"] > semantic["last"],
        "semantic_beats_shuffled": semantic["normal"] > semantic["shuffled"],
        "reconstruction_preserved": reconstruction["mse"] < reconstruction["anchor_mse"],
    }
    passed = all(checks.values())
    report = {
        "run_id": run_id,
        "source_delta_run_id": delta_summary["run_id"],
        "status": "signal" if passed else "inconclusive",
        "metrics": metrics,
        "checks": checks,
        "passed": passed,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(run_dir / "summary.json", report)
    _atomic_json(config.output_root / "latest_summary.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train semantic auxiliary delta objective")
    parser.add_argument("--config", type=Path, default=Path("configs/ssv2_semantic_pilot.yaml"))
    args = parser.parse_args()
    report = run(args.config)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
