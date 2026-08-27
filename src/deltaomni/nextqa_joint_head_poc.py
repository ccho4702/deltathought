from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import uuid
from collections import OrderedDict
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
class JointHeadConfig:
    seed: int
    joint_manifest: Path
    device: str
    cpu_threads: int
    text_buckets: int
    hidden_width: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_steps: int
    checkpoint_interval_steps: int
    keep_last_checkpoints: int
    resume: str
    output_root: Path
    log_root: Path
    report_path: Path


def load_config(path: Path) -> JointHeadConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    config = JointHeadConfig(
        **{
            **{
                key: raw[key]
                for key in raw
                if key not in {"joint_manifest", "output_root", "log_root", "report_path"}
            },
            "joint_manifest": resolve(raw["joint_manifest"]),
            "output_root": resolve(raw["output_root"]),
            "log_root": resolve(raw["log_root"]),
            "report_path": resolve(raw["report_path"]),
        }
    )
    if config.resume not in {"auto", "never"}:
        raise ValueError("Invalid joint head resume mode")
    return config


def _token_ids(text: str, buckets: int) -> list[int]:
    tokens = re.findall(r"[a-z0-9]+", text.lower()) or ["<empty>"]
    return [int(hashlib.sha256(token.encode()).hexdigest()[:16], 16) % buckets for token in tokens]


class JointQADataset:
    def __init__(self, manifest: dict[str, Any], split: str, buckets: int) -> None:
        self.records = list(manifest["splits"][split])
        self.examples = []
        self.buckets = buckets
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for record_index, record in enumerate(self.records):
            payload = torch.load(record["cache_path"], map_location="cpu", weights_only=False)
            for qa_index, qa in enumerate(payload["qa"]):
                if len(qa["choices"]) >= 2 and qa["answer_index"] is not None:
                    self.examples.append((record_index, qa_index))

    def __len__(self) -> int:
        return len(self.examples)

    def _payload(self, record_index: int) -> dict[str, Any]:
        path = self.records[record_index]["cache_path"]
        payload = self.cache.get(path)
        if payload is not None:
            self.cache.move_to_end(path)
            return payload
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.cache[path] = payload
        while len(self.cache) > 128:
            self.cache.popitem(last=False)
        return payload

    def batch(self, indices: Tensor) -> dict[str, Any]:
        payloads = []
        qa_values = []
        for example_index in indices.tolist():
            record_index, qa_index = self.examples[example_index]
            payload = self._payload(record_index)
            payloads.append(payload)
            qa_values.append(payload["qa"][qa_index])
        return {
            "video_full": torch.stack([p["video_first"].float().mean(0) for p in payloads]),
            "video_delta": torch.stack([p["video_deltas"].float().mean((0, 1)) for p in payloads]),
            "audio_full": torch.stack([p["audio_first"].float().mean(0) for p in payloads]),
            "audio_delta": torch.stack([p["audio_deltas"].float().mean((0, 1)) for p in payloads]),
            "questions": [_token_ids(qa["question"], self.buckets) for qa in qa_values],
            "choices": [
                [_token_ids(choice, self.buckets) for choice in qa["choices"]] for qa in qa_values
            ],
            "labels": torch.tensor([qa["answer_index"] for qa in qa_values]),
            "answers": [qa["answer"] for qa in qa_values],
            "source_ids": [p["source_id"] for p in payloads],
        }


class JointQAHead(nn.Module):
    def __init__(self, buckets: int, hidden: int) -> None:
        super().__init__()
        self.text = nn.Embedding(buckets, hidden)
        self.video_full = nn.Linear(3584, hidden)
        self.audio_full = nn.Linear(3584, hidden)
        self.video_delta = nn.Linear(768, hidden)
        self.audio_delta = nn.Linear(768, hidden)
        self.context = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())

    def _text(self, values: list[list[int]], device: torch.device) -> Tensor:
        return torch.stack(
            [self.text(torch.tensor(value, device=device)).mean(0) for value in values]
        )

    def forward(self, batch: dict[str, Any], control: str = "normal") -> Tensor:
        device = self.text.weight.device
        vf, vd = batch["video_full"].to(device), batch["video_delta"].to(device)
        af, ad = batch["audio_full"].to(device), batch["audio_delta"].to(device)
        if control == "video_zero":
            vf, vd = torch.zeros_like(vf), torch.zeros_like(vd)
        elif control == "audio_zero":
            af, ad = torch.zeros_like(af), torch.zeros_like(ad)
        elif control == "delta_zero":
            vd, ad = torch.zeros_like(vd), torch.zeros_like(ad)
        elif control == "delta_shuffled":
            vd, ad = vd.roll(1, 0), ad.roll(1, 0)
        elif control != "normal":
            raise ValueError(f"Unknown joint QA control: {control}")
        question = self._text(batch["questions"], device)
        context = self.context(
            question
            + self.video_full(vf)
            + self.video_delta(vd)
            + self.audio_full(af)
            + self.audio_delta(ad)
        )
        rows = []
        for index, choices in enumerate(batch["choices"]):
            encoded = self._text(choices, device)
            rows.append(encoded @ context[index] / (context.shape[-1] ** 0.5))
        return torch.stack(rows)


@torch.no_grad()
def evaluate(model, data, config, control="normal"):
    model.eval()
    correct = total = 0
    examples = []
    for start in range(0, len(data), config.batch_size):
        indices = torch.arange(start, min(start + config.batch_size, len(data)))
        batch = data.batch(indices)
        predicted = model(batch, control).argmax(1).cpu()
        labels = batch["labels"]
        correct += int(predicted.eq(labels).sum())
        total += len(indices)
        if len(examples) < 10:
            for offset, choice_index in enumerate(predicted.tolist()):
                if len(examples) == 10:
                    break
                examples.append(
                    {
                        "source_id": batch["source_ids"][offset],
                        "question": " ".join(map(str, batch["questions"][offset])),
                        "prediction": choice_index,
                        "target": int(labels[offset]),
                        "answer": batch["answers"][offset],
                    }
                )
    return {
        "accuracy": correct / total,
        "correct": correct,
        "examples": total,
        "predictions": examples,
    }


def _atomic_save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _latest_checkpoint(run_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    for path in sorted((run_dir / "checkpoints").glob("step-*.pt"), reverse=True):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except (EOFError, OSError, RuntimeError):
            continue
        if {"model", "optimizer", "next_step", "signature", "rng"} <= payload.keys():
            return path, payload
    return None


def run(config_path: Path, run_id: str | None, stop_after_step: int | None):
    config = load_config(config_path)
    _set_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    device = torch.device(config.device)
    manifest = json.loads(config.joint_manifest.read_text())
    train, validation, test = (
        JointQADataset(manifest, split, config.text_buckets)
        for split in ("train", "validation", "test")
    )
    model = JointQAHead(config.text_buckets, config.hidden_width).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    selected = run_id or f"nextqa-joint-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = config.output_root / selected
    run_dir.mkdir(parents=True, exist_ok=True)
    signature = json.dumps(asdict(config), sort_keys=True, default=str)
    start_step = 1
    initial = None
    resumed = _latest_checkpoint(run_dir) if config.resume == "auto" else None
    if resumed is not None:
        checkpoint_path, payload = resumed
        if payload["signature"] != signature:
            raise ValueError("Joint QA checkpoint configuration mismatch")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        torch.random.set_rng_state(payload["rng"]["torch"])
        torch.cuda.set_rng_state(payload["rng"]["cuda"], device)
        start_step = int(payload["next_step"])
        initial = payload.get("initial")
        print(f"resume={checkpoint_path} next_step={start_step}", flush=True)
    if initial is None:
        initial = {
            split: evaluate(model, data, config)
            for split, data in (("train", train), ("validation", validation), ("test", test))
        }
    final_step = min(config.max_steps, stop_after_step or config.max_steps)
    started = time.perf_counter()
    for step in range(start_step, final_step + 1):
        generator = torch.Generator().manual_seed(config.seed * 1_000_003 + step)
        indices = torch.randint(0, len(train), (config.batch_size,), generator=generator)
        batch = train.batch(indices)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == final_step:
            elapsed = time.perf_counter() - started
            eta = elapsed / (step - start_step + 1) * (final_step - step)
            print(
                f"joint_qa_step={step}/{final_step} loss={float(loss):.4f} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
        if step % config.checkpoint_interval_steps == 0 or step == final_step:
            _atomic_save(
                run_dir / "checkpoints" / f"step-{step:06d}.pt",
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "next_step": step + 1,
                    "signature": signature,
                    "initial": initial,
                    "rng": {
                        "torch": torch.random.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state(device),
                    },
                },
            )
    if final_step < config.max_steps:
        return {"run_id": selected, "status": "interrupted", "step": final_step}
    normal = {
        split: evaluate(model, data, config)
        for split, data in (("train", train), ("validation", validation), ("test", test))
    }
    controls = {
        name: evaluate(model, validation, config, name)
        for name in ("video_zero", "audio_zero", "delta_zero", "delta_shuffled")
    }
    report = {
        "schema": "deltaomni.nextqa_joint_head_poc.v1",
        "run_id": selected,
        "initial": initial,
        "final": normal,
        "validation_controls": controls,
        "passed": normal["train"]["accuracy"] >= 0.9
        and normal["validation"]["accuracy"] > initial["validation"]["accuracy"],
        "training_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(run_dir / "summary.json", report)
    _atomic_json(config.report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/nextqa_joint_head_poc.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.run_id, args.stop_after_step), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
