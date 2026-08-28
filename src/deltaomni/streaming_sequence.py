from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SectionRef:
    source_id: str
    source_group_id: str
    cache_path: Path
    delta_updates: int
    captions: int


@dataclass(frozen=True)
class StreamingSequence:
    sequence_id: str
    split: str
    sections: tuple[SectionRef, ...]

    def validate(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid streaming split: {self.split}")
        if len(self.sections) < 2:
            raise ValueError("Streaming sequence requires multiple caption/full cycles")
        if len({section.source_group_id for section in self.sections}) != len(self.sections):
            raise ValueError("Synthetic streaming sequence repeats a source group")
        if any(section.delta_updates <= 0 or section.captions <= 0 for section in self.sections):
            raise ValueError("Streaming section requires deltas and captions")

    def timeline(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        self.validate()
        commit = []
        refresh = []
        elapsed = []
        section_index = []
        for index, section in enumerate(self.sections):
            for step in range(section.delta_updates):
                commit.append(step == section.delta_updates - 1)
                refresh.append(step == 0)
                elapsed.append(step + 1)
                section_index.append(index)
        return (
            torch.tensor(commit, dtype=torch.float32),
            torch.tensor(refresh, dtype=torch.bool),
            torch.tensor(elapsed, dtype=torch.float32),
            torch.tensor(section_index, dtype=torch.long),
        )


def build_sequences(
    prefix_manifest: Path,
    *,
    sections_per_sequence: int,
    seed: int,
) -> tuple[dict[str, list[StreamingSequence]], dict[str, int]]:
    if sections_per_sequence < 2:
        raise ValueError("Multi-commit sequences need at least two sections")
    manifest = json.loads(prefix_manifest.read_text(encoding="utf-8"))
    result = {}
    discarded = {}

    def delta_updates(record: dict[str, Any]) -> int:
        if "delta_updates" in record:
            return int(record["delta_updates"])
        if "blocks" in record:
            return int(record["blocks"]) - 1
        raise ValueError(f"Prefix record has no delta length: {record.get('source_id')}")

    for split, raw_records in manifest["splits"].items():
        records = sorted(
            raw_records,
            key=lambda record: hashlib.sha256(f"{seed}:{record['source_id']}".encode()).hexdigest(),
        )
        usable = len(records) // sections_per_sequence * sections_per_sequence
        discarded[split] = len(records) - usable
        sequences = []
        for start in range(0, usable, sections_per_sequence):
            sections = tuple(
                SectionRef(
                    source_id=str(record["source_id"]),
                    source_group_id=str(record["source_group_id"]),
                    cache_path=Path(record["cache_path"]),
                    delta_updates=delta_updates(record),
                    captions=int(record["captions"]),
                )
                for record in records[start : start + sections_per_sequence]
            )
            sequence = StreamingSequence(
                sequence_id=f"{split}:{start // sections_per_sequence:08d}",
                split=split,
                sections=sections,
            )
            sequence.validate()
            sequences.append(sequence)
        result[split] = sequences
    return result, discarded


class CommitHead(nn.Module):
    def __init__(self, delta_width: int, hidden_width: int) -> None:
        super().__init__()
        self.delta_norm = nn.LayerNorm(delta_width)
        self.delta_projection = nn.Linear(delta_width, hidden_width)
        self.elapsed_projection = nn.Sequential(
            nn.Linear(1, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
        )
        self.cell = nn.GRUCell(hidden_width, hidden_width)
        self.output = nn.Linear(hidden_width, 1)

    def forward(
        self,
        deltas: Tensor,
        elapsed: Tensor,
        refresh: Tensor,
        valid: Tensor,
    ) -> Tensor:
        if deltas.ndim != 3 or elapsed.shape != deltas.shape[:2]:
            raise ValueError("CommitHead delta/elapsed shape mismatch")
        if refresh.shape != elapsed.shape or valid.shape != elapsed.shape:
            raise ValueError("CommitHead mask shape mismatch")
        hidden = torch.zeros(
            deltas.shape[0], self.output.in_features, device=deltas.device, dtype=deltas.dtype
        )
        logits = []
        for step in range(deltas.shape[1]):
            hidden = torch.where(refresh[:, step : step + 1], torch.zeros_like(hidden), hidden)
            features = self.delta_projection(self.delta_norm(deltas[:, step]))
            features = features + self.elapsed_projection(elapsed[:, step : step + 1])
            updated = self.cell(features, hidden)
            hidden = torch.where(valid[:, step : step + 1], updated, hidden)
            logits.append(self.output(hidden).squeeze(-1))
        return torch.stack(logits, dim=1)


def commit_loss(
    logits: Tensor,
    targets: Tensor,
    valid: Tensor,
    *,
    positive_weight: float,
) -> Tensor:
    if logits.shape != targets.shape or valid.shape != targets.shape:
        raise ValueError("Commit loss shape mismatch")
    losses = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=torch.tensor(positive_weight, device=logits.device),
        reduction="none",
    )
    return losses.masked_select(valid).mean()


def sequence_to_dict(sequence: StreamingSequence) -> dict[str, Any]:
    return {
        "sequence_id": sequence.sequence_id,
        "split": sequence.split,
        "sections": [
            {
                "source_id": section.source_id,
                "source_group_id": section.source_group_id,
                "cache_path": str(section.cache_path),
                "delta_updates": section.delta_updates,
                "captions": section.captions,
            }
            for section in sequence.sections
        ],
    }
