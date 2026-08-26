from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from deltaomni.config import LossConfig, ModelConfig
from deltaomni.types import Modality


class PairDeltaEncoder(nn.Module):
    """Compress an anchor/current embedding pair into ordered delta slots."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dimension = config.embedding_dim
        self.gate = nn.Linear(2 * dimension, 2 * dimension)
        self.pair_projection = nn.Sequential(
            nn.Linear(4 * dimension, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, dimension),
            nn.LayerNorm(dimension),
        )
        self.direct_projection = nn.Linear(dimension, dimension, bias=False)
        nn.init.eye_(self.direct_projection.weight)
        self.queries = nn.Parameter(torch.randn(config.delta_tokens, dimension) * 0.02)
        self.attention = nn.MultiheadAttention(
            dimension,
            config.num_heads,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(dimension)

    def forward(self, anchor: Tensor, current: Tensor) -> Tensor:
        if anchor.shape != current.shape or anchor.ndim != 3:
            raise ValueError("anchor and current must have matching [batch, tokens, dim] shapes")
        gates = torch.sigmoid(self.gate(torch.cat((anchor, current), dim=-1)))
        anchor_gate, current_gate = gates.chunk(2, dim=-1)
        gated_anchor = anchor_gate * anchor
        gated_current = current_gate * current
        pair = torch.cat(
            (gated_anchor, gated_current, gated_current - gated_anchor, anchor * current),
            dim=-1,
        )
        pair_tokens = self.pair_projection(pair)
        queries = self.queries.unsqueeze(0).expand(anchor.shape[0], -1, -1)
        attended, _ = self.attention(queries, pair_tokens, pair_tokens, need_weights=False)
        magnitude = (current - anchor).square().mean(dim=(1, 2), keepdim=True).sqrt()
        direct = self.direct_projection((current - anchor).mean(dim=1)).unsqueeze(1)
        learned_residual = self.output_norm(queries + attended) * magnitude
        return direct + 0.1 * learned_residual


class DeltaAccumulator(nn.Module):
    """Keep a bounded, ordered recurrent memory rather than summing deltas."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.cell = nn.GRUCell(config.embedding_dim, config.embedding_dim)
        self.residual_gate = nn.Linear(2 * config.embedding_dim, config.embedding_dim)

    def forward(self, state: Tensor, delta: Tensor) -> Tensor:
        if state.shape != delta.shape:
            raise ValueError("state and delta slots must have identical shapes")
        batch, slots, dimension = state.shape
        flattened_state = state.reshape(batch * slots, dimension)
        flattened_delta = delta.reshape(batch * slots, dimension)
        recurrent = self.cell(flattened_delta, flattened_state)
        gate = torch.sigmoid(
            self.residual_gate(torch.cat((flattened_state, flattened_delta), dim=-1))
        )
        ordered_residual = gate * (recurrent - flattened_state)
        updated = flattened_state + flattened_delta + 0.1 * ordered_residual
        return updated.reshape(batch, slots, dimension)


class FullEmbeddingReconstructor(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.embedding_dim,
            config.num_heads,
            batch_first=True,
        )
        self.token_positions = nn.Parameter(
            torch.randn(config.embedding_tokens, config.embedding_dim) * 0.02
        )
        self.update = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

    def forward(self, anchor: Tensor, accumulated_delta: Tensor) -> Tensor:
        attended, _ = self.attention(
            anchor,
            accumulated_delta,
            accumulated_delta,
            need_weights=False,
        )
        global_delta = accumulated_delta.mean(dim=1, keepdim=True)
        update_input = attended + global_delta + self.token_positions.unsqueeze(0)
        return anchor + self.update(update_input)


class CommitAndLengthHead(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.novelty = nn.Linear(config.embedding_dim, 1)
        self.trigger = nn.Linear(config.embedding_dim + 1, 1)
        self.length = nn.Linear(config.embedding_dim + 1, config.max_caption_length + 1)

    def novelty_score(self, delta: Tensor) -> Tensor:
        return F.softplus(self.novelty(delta.mean(dim=1))).squeeze(-1)

    def forward(self, accumulated_delta: Tensor, load: Tensor) -> tuple[Tensor, Tensor]:
        features = torch.cat((accumulated_delta.mean(dim=1), load.unsqueeze(-1)), dim=-1)
        return self.trigger(features).squeeze(-1), self.length(features)


class DeltaCaptionDecoder(nn.Module):
    """Small causal decoder used to validate the future LLM-prefix contract."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.max_length = config.max_caption_length
        self.embedding = nn.Embedding(config.caption_vocab_size, config.hidden_dim)
        self.context = nn.Sequential(
            nn.Linear(2 * config.embedding_dim, config.hidden_dim),
            nn.Tanh(),
        )
        self.decoder = nn.GRU(config.hidden_dim, config.hidden_dim, batch_first=True)
        self.output = nn.Linear(config.hidden_dim, config.caption_vocab_size)

    def forward(self, anchor: Tensor, delta_state: Tensor, input_ids: Tensor) -> Tensor:
        context = self.context(torch.cat((anchor.mean(dim=1), delta_state.mean(dim=1)), dim=-1))
        decoded, _ = self.decoder(self.embedding(input_ids), context.unsqueeze(0))
        return self.output(decoded)

    @torch.no_grad()
    def generate(
        self,
        anchor: Tensor,
        delta_state: Tensor,
        *,
        bos_token_id: int,
        eos_token_id: int,
    ) -> list[int]:
        if anchor.shape[0] != 1:
            raise ValueError("Greedy generation currently accepts one stream at a time")
        generated = [bos_token_id]
        for _ in range(self.max_length - 1):
            inputs = torch.tensor([generated], dtype=torch.long, device=anchor.device)
            next_token = int(self(anchor, delta_state, inputs)[0, -1].argmax().item())
            generated.append(next_token)
            if next_token == eos_token_id:
                break
        return generated


class ModalityDeltaCodec(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.delta_encoder = PairDeltaEncoder(config)
        self.accumulator = DeltaAccumulator(config)
        self.reconstructor = FullEmbeddingReconstructor(config)
        self.policy = CommitAndLengthHead(config)
        self.caption_decoder = DeltaCaptionDecoder(config)


@dataclass(frozen=True)
class LossOutput:
    total: Tensor
    reconstruction: Tensor
    step_reconstruction: Tensor
    section_reconstruction: Tensor
    identity: Tensor
    trigger: Tensor
    caption: Tensor
    length: Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "reconstruction": float(self.reconstruction.detach()),
            "step_reconstruction": float(self.step_reconstruction.detach()),
            "section_reconstruction": float(self.section_reconstruction.detach()),
            "identity": float(self.identity.detach()),
            "trigger": float(self.trigger.detach()),
            "caption": float(self.caption.detach()),
            "length": float(self.length.detach()),
        }


class DeltaCodecModel(nn.Module):
    def __init__(self, config: ModelConfig, modalities: tuple[Modality, ...]) -> None:
        super().__init__()
        self.config = config
        self.modalities = modalities
        self.codecs = nn.ModuleDict(
            {modality.value: ModalityDeltaCodec(config) for modality in modalities}
        )

    def forward_sequence(
        self,
        full_embeddings: Tensor,
        commit_targets: Tensor,
        caption_targets: Tensor,
        caption_lengths: Tensor,
        weights: LossConfig,
    ) -> LossOutput:
        if full_embeddings.ndim != 5:
            raise ValueError("full_embeddings must be [batch, time, modality, tokens, dim]")
        _, time_steps, modality_count, _, _ = full_embeddings.shape
        if modality_count != len(self.modalities):
            raise ValueError("full_embeddings modality axis does not match configured modalities")

        reconstruction_losses: list[Tensor] = []
        step_reconstruction_losses: list[Tensor] = []
        section_reconstruction_losses: list[Tensor] = []
        identity_losses: list[Tensor] = []
        trigger_losses: list[Tensor] = []
        caption_losses: list[Tensor] = []
        length_losses: list[Tensor] = []

        for modality_index, modality in enumerate(self.modalities):
            codec = self.codecs[modality.value]
            anchor = full_embeddings[:, 0, modality_index]
            previous = anchor
            slots = torch.zeros(
                anchor.shape[0],
                self.config.delta_tokens,
                self.config.embedding_dim,
                device=anchor.device,
                dtype=anchor.dtype,
            )
            load = torch.zeros(anchor.shape[0], device=anchor.device, dtype=anchor.dtype)

            for time_index in range(1, time_steps):
                current = full_embeddings[:, time_index, modality_index]
                delta = codec.delta_encoder(previous, current)
                slots = codec.accumulator(slots, delta)
                load = load + codec.policy.novelty_score(delta)
                trigger_logits, length_logits = codec.policy(slots, load)
                step_reconstructed = codec.reconstructor(previous, delta)
                section_reconstructed = codec.reconstructor(anchor, slots)

                step_loss = F.smooth_l1_loss(step_reconstructed, current)
                section_loss = F.smooth_l1_loss(section_reconstructed, current)
                step_reconstruction_losses.append(step_loss)
                section_reconstruction_losses.append(section_loss)
                reconstruction_losses.append(0.5 * (step_loss + section_loss))
                identity_losses.append(codec.delta_encoder(current, current).square().mean())
                target_commit = commit_targets[:, time_index, modality_index].float()
                trigger_losses.append(
                    F.binary_cross_entropy_with_logits(trigger_logits, target_commit)
                )

                selected = target_commit.bool()
                if selected.any():
                    selected_targets = caption_targets[selected, time_index, modality_index]
                    caption_logits = codec.caption_decoder(
                        anchor[selected],
                        slots[selected],
                        selected_targets[:, :-1],
                    )
                    caption_losses.append(
                        F.cross_entropy(
                            caption_logits.reshape(-1, caption_logits.shape[-1]),
                            selected_targets[:, 1:].reshape(-1),
                            ignore_index=0,
                        )
                    )
                    length_losses.append(
                        F.cross_entropy(
                            length_logits[selected],
                            caption_lengths[selected, time_index, modality_index],
                        )
                    )

                reset = selected[:, None, None]
                anchor = torch.where(reset, current, anchor)
                slots = torch.where(reset, torch.zeros_like(slots), slots)
                load = torch.where(selected, torch.zeros_like(load), load)
                previous = current

        def mean_or_zero(values: list[Tensor]) -> Tensor:
            return torch.stack(values).mean() if values else full_embeddings.new_zeros(())

        reconstruction = mean_or_zero(reconstruction_losses)
        step_reconstruction = mean_or_zero(step_reconstruction_losses)
        section_reconstruction = mean_or_zero(section_reconstruction_losses)
        identity = mean_or_zero(identity_losses)
        trigger = mean_or_zero(trigger_losses)
        caption = mean_or_zero(caption_losses)
        length = mean_or_zero(length_losses)
        total = (
            weights.reconstruction * reconstruction
            + weights.identity * identity
            + weights.trigger * trigger
            + weights.caption * caption
            + weights.length * length
        )
        return LossOutput(
            total,
            reconstruction,
            step_reconstruction,
            section_reconstruction,
            identity,
            trigger,
            caption,
            length,
        )
