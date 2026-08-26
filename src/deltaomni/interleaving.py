from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from deltaomni.config import StreamConfig
from deltaomni.model import DeltaCodecModel
from deltaomni.types import EventKind, Modality, StreamEvent


@dataclass
class RuntimeState:
    anchor: Tensor
    previous: Tensor
    delta_slots: Tensor
    load: Tensor
    section_start_seconds: float
    section_steps: int = 0


class StreamingDeltaEngine:
    """Causal A/V engine with independent clocks and reset scopes."""

    def __init__(self, model: DeltaCodecModel, stream_config: StreamConfig) -> None:
        self.model = model
        self.stream_config = stream_config
        self.states: dict[Modality, RuntimeState] = {}

    def initialize(
        self,
        timestamp_seconds: float,
        full_embeddings: dict[Modality, Tensor],
    ) -> list[StreamEvent]:
        if self.states:
            raise RuntimeError("StreamingDeltaEngine is already initialized")
        events = []
        for modality in self.model.modalities:
            embedding = full_embeddings[modality]
            if embedding.ndim != 3 or embedding.shape[0] != 1:
                raise ValueError("runtime full embeddings must be [1, tokens, dim]")
            self.states[modality] = RuntimeState(
                anchor=embedding,
                previous=embedding,
                delta_slots=torch.zeros(
                    1,
                    self.model.config.delta_tokens,
                    self.model.config.embedding_dim,
                    dtype=embedding.dtype,
                    device=embedding.device,
                ),
                load=torch.zeros(1, dtype=embedding.dtype, device=embedding.device),
                section_start_seconds=timestamp_seconds,
            )
            events.append(
                StreamEvent(EventKind.FULL, modality, timestamp_seconds, timestamp_seconds)
            )
        return events

    @torch.no_grad()
    def step(
        self,
        timestamp_seconds: float,
        full_embeddings: dict[Modality, Tensor],
        *,
        force_commits: set[Modality] | None = None,
        token_text: dict[int, str] | None = None,
    ) -> list[StreamEvent]:
        if not self.states:
            raise RuntimeError("initialize must be called before step")
        force_commits = force_commits or set()
        token_text = token_text or {}
        events: list[StreamEvent] = []
        decisions: dict[Modality, tuple[bool, Tensor]] = {}

        for modality in self.model.modalities:
            codec = self.model.codecs[modality.value]
            state = self.states[modality]
            current = full_embeddings[modality]
            delta = codec.delta_encoder(state.previous, current)
            state.delta_slots = codec.accumulator(state.delta_slots, delta)
            state.load = state.load + codec.policy.novelty_score(delta)
            state.section_steps += 1
            trigger_logit, _ = codec.policy(state.delta_slots, state.load)
            trigger_probability = float(torch.sigmoid(trigger_logit).item())
            should_commit = (
                modality in force_commits
                or trigger_probability >= self.stream_config.trigger_threshold
                or float(state.load.item()) >= self.stream_config.load_threshold
                or state.section_steps >= self.stream_config.max_section_steps
            )
            decisions[modality] = (should_commit, current)
            state.previous = current
            events.append(
                StreamEvent(
                    EventKind.DELTA,
                    modality,
                    timestamp_seconds,
                    state.section_start_seconds,
                    metadata={
                        "load": float(state.load.item()),
                        "trigger_probability": trigger_probability,
                    },
                )
            )

        for modality in self.model.modalities:
            should_commit, current = decisions[modality]
            if not should_commit:
                continue
            codec = self.model.codecs[modality.value]
            state = self.states[modality]
            token_ids = codec.caption_decoder.generate(
                state.anchor,
                state.delta_slots,
                bos_token_id=1,
                eos_token_id=2,
            )
            text = " ".join(token_text.get(token, str(token)) for token in token_ids[1:-1])
            events.append(
                StreamEvent(
                    EventKind.CAPTION,
                    modality,
                    timestamp_seconds,
                    state.section_start_seconds,
                    text=text,
                    metadata={"token_ids": token_ids},
                )
            )
            state.anchor = current
            state.delta_slots = torch.zeros_like(state.delta_slots)
            state.load = torch.zeros_like(state.load)
            state.section_start_seconds = timestamp_seconds
            state.section_steps = 0
            events.append(
                StreamEvent(EventKind.FULL, modality, timestamp_seconds, timestamp_seconds)
            )
        return events


def render_interleaving(events: list[StreamEvent]) -> str:
    return " ".join(event.render() for event in events)
