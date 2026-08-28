from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ContinuousKVState:
    past_key_values: Any
    attention_mask: Tensor
    tokens: int


class ContinuousKVRunner:
    """Append multimodal/text chunks without resetting autoregressive state."""

    def __init__(self, model: nn.Module, *, position_axes: int = 3) -> None:
        if position_axes not in {1, 3}:
            raise ValueError("Continuous KV position axes must be one or three")
        self.model = model
        self.position_axes = position_axes

    def _position_ids(self, start: int, width: int, batch: int, device: torch.device) -> Tensor:
        values = torch.arange(start, start + width, device=device).unsqueeze(0).expand(batch, -1)
        return values if self.position_axes == 1 else values.unsqueeze(0).expand(3, -1, -1)

    def append(
        self,
        *,
        state: ContinuousKVState | None,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> tuple[Tensor, ContinuousKVState]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Continuous KV append requires exactly one input representation")
        values = input_ids if input_ids is not None else inputs_embeds
        assert values is not None
        if values.ndim not in {2, 3}:
            raise ValueError(f"Unexpected continuous KV input shape: {tuple(values.shape)}")
        batch, width = values.shape[:2]
        device = values.device
        if state is None:
            attention_mask = torch.ones(batch, width, dtype=torch.bool, device=device)
            start = 0
            past = None
        else:
            if state.attention_mask.shape[0] != batch:
                raise ValueError("Continuous KV batch size cannot change mid-stream")
            attention_mask = torch.cat(
                (
                    state.attention_mask,
                    torch.ones(batch, width, dtype=torch.bool, device=device),
                ),
                dim=1,
            )
            start = state.tokens
            past = state.past_key_values
        kwargs = {
            "attention_mask": attention_mask,
            "position_ids": self._position_ids(start, width, batch, device),
            "past_key_values": past,
            "use_cache": True,
            "cache_position": torch.arange(start, start + width, device=device),
        }
        if input_ids is not None:
            kwargs["input_ids"] = input_ids
        else:
            kwargs["inputs_embeds"] = inputs_embeds
        output = self.model(**kwargs)
        next_state = ContinuousKVState(
            past_key_values=output.past_key_values,
            attention_mask=attention_mask,
            tokens=start + width,
        )
        return output.logits, next_state

    @torch.no_grad()
    def greedy_append(
        self,
        logits: Tensor,
        state: ContinuousKVState,
        *,
        end_token_id: int,
        max_new_tokens: int,
    ) -> tuple[Tensor, ContinuousKVState]:
        if max_new_tokens <= 0:
            raise ValueError("Continuous KV generation length must be positive")
        generated = []
        finished = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
        for _ in range(max_new_tokens):
            next_ids = logits[:, -1].argmax(dim=-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, end_token_id), next_ids)
            generated.append(next_ids)
            finished |= next_ids.eq(end_token_id)
            logits, state = self.append(state=state, input_ids=next_ids.unsqueeze(1))
            if bool(finished.all()):
                break
        return torch.stack(generated, dim=1), state
