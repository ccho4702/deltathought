from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from deltaomni.types import Modality


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int
    hidden_dim: int
    embedding_tokens: int
    delta_tokens: int
    num_heads: int
    caption_vocab_size: int
    max_caption_length: int


@dataclass(frozen=True)
class StreamConfig:
    trigger_threshold: float
    load_threshold: float
    max_section_steps: int


@dataclass(frozen=True)
class TrainingConfig:
    device: str
    cpu_threads: int
    num_workers: int
    examples: int
    validation_examples: int
    sequence_steps: int
    batch_size: int
    learning_rate: float
    max_steps: int
    checkpoint_interval_steps: int
    run_root: Path
    log_root: Path
    resume: str


@dataclass(frozen=True)
class LossConfig:
    reconstruction: float
    identity: float
    trigger: float
    caption: float
    length: float


@dataclass(frozen=True)
class SanityConfig:
    seed: int
    modalities: tuple[Modality, ...]
    model: ModelConfig
    stream: StreamConfig
    training: TrainingConfig
    loss: LossConfig


def _require(mapping: dict[str, Any], key: str, expected_type: type) -> Any:
    value = mapping.get(key)
    if not isinstance(value, expected_type):
        raise ValueError(f"Expected {key!r} to be {expected_type.__name__}")
    return value


def load_config(path: Path) -> SanityConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    model = _require(raw, "model", dict)
    stream = _require(raw, "stream", dict)
    training = _require(raw, "training", dict)
    loss = _require(raw, "loss", dict)
    modalities = tuple(Modality(item) for item in _require(raw, "modalities", list))
    if len(modalities) != len(set(modalities)) or not modalities:
        raise ValueError("modalities must be a non-empty unique list")
    root = path.resolve().parent.parent

    return SanityConfig(
        seed=int(raw["seed"]),
        modalities=modalities,
        model=ModelConfig(
            embedding_dim=int(model["embedding_dim"]),
            hidden_dim=int(model["hidden_dim"]),
            embedding_tokens=int(model.get("embedding_tokens", 3)),
            delta_tokens=int(model["delta_tokens"]),
            num_heads=int(model["num_heads"]),
            caption_vocab_size=int(model["caption_vocab_size"]),
            max_caption_length=int(model["max_caption_length"]),
        ),
        stream=StreamConfig(
            trigger_threshold=float(stream["trigger_threshold"]),
            load_threshold=float(stream["load_threshold"]),
            max_section_steps=int(stream["max_section_steps"]),
        ),
        training=TrainingConfig(
            device=str(training["device"]),
            cpu_threads=int(training["cpu_threads"]),
            num_workers=int(training["num_workers"]),
            examples=int(training["examples"]),
            validation_examples=int(training["validation_examples"]),
            sequence_steps=int(training["sequence_steps"]),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            max_steps=int(training["max_steps"]),
            checkpoint_interval_steps=int(training["checkpoint_interval_steps"]),
            run_root=root / training["run_root"],
            log_root=root / training["log_root"],
            resume=str(training["resume"]),
        ),
        loss=LossConfig(
            reconstruction=float(loss["reconstruction"]),
            identity=float(loss["identity"]),
            trigger=float(loss["trigger"]),
            caption=float(loss["caption"]),
            length=float(loss["length"]),
        ),
    )
