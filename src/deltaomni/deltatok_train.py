from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F

from deltaomni.train_sanity import _atomic_json, _set_seed


@dataclass(frozen=True)
class DeltaTokConfig:
    seed: int
    cache_manifest: Path
    device: str
    precision: str
    cpu_threads: int
    num_workers: int
    batch_size: int
    evaluation_batch_size: int
    input_dim: int
    model_dim: int
    tokens_per_frame: int
    delta_tokens: int
    depth: int
    num_heads: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    checkpoint_interval_steps: int
    resume: str
    output_root: Path
    log_root: Path


def load_config(path: Path) -> DeltaTokConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    config = DeltaTokConfig(
        seed=int(raw["seed"]),
        cache_manifest=resolve(raw["cache_manifest"]),
        device=str(raw["device"]),
        precision=str(raw["precision"]),
        cpu_threads=int(raw["cpu_threads"]),
        num_workers=int(raw["num_workers"]),
        batch_size=int(raw["batch_size"]),
        evaluation_batch_size=int(raw["evaluation_batch_size"]),
        input_dim=int(raw["input_dim"]),
        model_dim=int(raw["model_dim"]),
        tokens_per_frame=int(raw["tokens_per_frame"]),
        delta_tokens=int(raw["delta_tokens"]),
        depth=int(raw["depth"]),
        num_heads=int(raw["num_heads"]),
        learning_rate=float(raw["learning_rate"]),
        weight_decay=float(raw["weight_decay"]),
        warmup_steps=int(raw["warmup_steps"]),
        max_steps=int(raw["max_steps"]),
        checkpoint_interval_steps=int(raw["checkpoint_interval_steps"]),
        resume=str(raw["resume"]),
        output_root=resolve(raw["output_root"]),
        log_root=resolve(raw["log_root"]),
    )
    if config.precision != "bfloat16" or config.resume not in {"auto", "never"}:
        raise ValueError("DeltaTok requires bfloat16 and valid resume mode")
    if config.model_dim % config.num_heads or config.delta_tokens <= 0:
        raise ValueError("Invalid DeltaTok attention dimensions")
    return config


class PairDataset:
    def __init__(self, manifest: dict[str, Any], split: str) -> None:
        self.pairs = []
        for record in manifest["splits"][split]:
            payload = torch.load(record["cache_path"], map_location="cpu", weights_only=False)
            for step in range(1, payload["embeddings"].shape[0]):
                self.pairs.append((record["cache_path"], step))

    def __len__(self) -> int:
        return len(self.pairs)

    def load_batch(
        self, indices: Tensor, executor: ThreadPoolExecutor | None
    ) -> tuple[Tensor, Tensor]:
        selected = indices.cpu().tolist()

        def load(index: int) -> tuple[Tensor, Tensor]:
            path, step = self.pairs[index]
            values = torch.load(path, map_location="cpu", weights_only=False)["embeddings"].float()
            return values[step - 1], values[step]

        values = list(executor.map(load, selected)) if executor else [load(i) for i in selected]
        return torch.stack([v[0] for v in values]), torch.stack([v[1] for v in values])


class DeltaTok(nn.Module):
    def __init__(self, config: DeltaTokConfig) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(config.input_dim)
        self.input_projection = nn.Linear(config.input_dim, config.model_dim)
        self.output_projection = nn.Linear(config.model_dim, config.input_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.positions = nn.Parameter(torch.randn(config.tokens_per_frame, config.model_dim) * 0.02)
        self.types = nn.Parameter(torch.randn(3, config.model_dim) * 0.02)
        self.delta_queries = nn.Parameter(torch.randn(config.delta_tokens, config.model_dim) * 0.02)
        def layer() -> nn.TransformerEncoderLayer:
            return nn.TransformerEncoderLayer(
                config.model_dim,
                config.num_heads,
                4 * config.model_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        self.encoder = nn.TransformerEncoder(layer(), config.depth)
        self.decoder = nn.TransformerEncoder(layer(), config.depth)

    def encode(self, previous: Tensor, current: Tensor) -> Tensor:
        previous = self.input_projection(self.input_norm(previous)) + self.positions + self.types[0]
        current = self.input_projection(self.input_norm(current)) + self.positions + self.types[1]
        queries = self.delta_queries.unsqueeze(0).expand(previous.shape[0], -1, -1)
        encoded = self.encoder(torch.cat((queries, previous, current), dim=1))
        return encoded[:, : self.delta_queries.shape[0]]

    def decode(self, previous: Tensor, delta: Tensor) -> Tensor:
        projected = (
            self.input_projection(self.input_norm(previous)) + self.positions + self.types[0]
        )
        decoded = self.decoder(torch.cat((projected, delta + self.types[2]), dim=1))
        residual = self.output_projection(decoded[:, : previous.shape[1]])
        return previous + residual

    def forward(self, previous: Tensor, current: Tensor) -> tuple[Tensor, Tensor]:
        delta = self.encode(previous, current)
        return self.decode(previous, delta), delta


def _atomic_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def evaluate(
    model: DeltaTok, data: PairDataset, config: DeltaTokConfig, device: torch.device, executor
):
    model.eval()
    squared = cosine = copy_squared = 0.0
    reconstructed = []
    targets = []
    count = 0
    for start in range(0, len(data), config.evaluation_batch_size):
        idx = torch.arange(start, min(start + config.evaluation_batch_size, len(data)))
        previous, current = data.load_batch(idx, executor)
        previous, current = previous.to(device), current.to(device)
        predicted, _ = model(previous, current)
        squared += float((predicted.float() - current.float()).square().mean()) * len(idx)
        copy_squared += float((previous.float() - current.float()).square().mean()) * len(idx)
        cosine += float(
            F.cosine_similarity(predicted.float().flatten(1), current.float().flatten(1)).mean()
        ) * len(idx)
        reconstructed.append(predicted.mean(1).float().cpu())
        targets.append(current.mean(1).float().cpu())
        count += len(idx)
    rec, tgt = torch.cat(reconstructed), torch.cat(targets)
    similarity = F.normalize(rec) @ F.normalize(tgt).T
    retrieval = float(similarity.argmax(1).eq(torch.arange(count)).float().mean())
    return {
        "mse": squared / count,
        "copy_previous_mse": copy_squared / count,
        "cosine": cosine / count,
        "retrieval_r1": retrieval,
        "pairs": count,
    }


def run(config_path: Path, run_id: str | None) -> dict[str, Any]:
    config = load_config(config_path)
    torch.set_num_threads(config.cpu_threads)
    _set_seed(config.seed)
    device = torch.device(config.device)
    manifest = json.loads(config.cache_manifest.read_text())
    train, validation = PairDataset(manifest, "train"), PairDataset(manifest, "validation")
    model = DeltaTok(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    selected = run_id or f"deltatok-video-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = config.output_root / selected
    signature = json.dumps(asdict(config), sort_keys=True, default=str)
    start_step = 1
    checkpoints = (
        sorted((run_dir / "checkpoints").glob("step-*.pt")) if config.resume == "auto" else []
    )
    if checkpoints:
        payload = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
        if payload["signature"] != signature:
            raise ValueError("DeltaTok checkpoint mismatch")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = payload["next_step"]
    executor = ThreadPoolExecutor(config.num_workers) if config.num_workers else None
    started = time.perf_counter()
    try:
        for step in range(start_step, config.max_steps + 1):
            generator = torch.Generator().manual_seed(config.seed * 1_000_003 + step)
            idx = torch.randint(0, len(train), (config.batch_size,), generator=generator)
            previous, current = train.load_batch(idx, executor)
            previous, current = previous.to(device), current.to(device)
            optimizer.zero_grad(set_to_none=True)
            warmup = min(1.0, step / max(config.warmup_steps, 1))
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate * warmup
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted, _ = model(previous, current)
                loss = F.mse_loss(predicted.float(), current.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-2)
            optimizer.step()
            if step % 10 == 0:
                elapsed = time.perf_counter() - started
                eta = elapsed / (step - start_step + 1) * (config.max_steps - step)
                print(
                    f"deltatok_step={step}/{config.max_steps} "
                    f"mse={float(loss):.6f} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
            if step % config.checkpoint_interval_steps == 0 or step == config.max_steps:
                _atomic_save(
                    run_dir / "checkpoints" / f"step-{step:06d}.pt",
                    {
                        "next_step": step + 1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "signature": signature,
                    },
                )
        metrics = evaluate(model, validation, config, device, executor)
    finally:
        if executor:
            executor.shutdown()
    report = {
        "schema": "deltaomni.deltatok_training.v1",
        "run_id": selected,
        "training_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "passed": metrics["mse"] < metrics["copy_previous_mse"],
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(run_dir / "summary.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/deltatok_video_integration.yaml")
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.run_id), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
