from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SemanticTokenOutput:
    class_logits: Tensor
    assignment_logits: Tensor
    assignment_probabilities: Tensor
    code_ids: Tensor
    tokens: Tensor


class SemanticTokenBottleneck(nn.Module):
    """Compress accumulated modality deltas into typed soft or discrete semantic tokens."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        token_count: int,
        codebook_size: int,
        classes: int,
        num_heads: int,
        modalities: int = 3,
    ) -> None:
        super().__init__()
        if token_count <= 0 or codebook_size < classes:
            raise ValueError("token_count must be positive and codebook_size must cover classes")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )
        self.queries = nn.Parameter(torch.randn(token_count, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.assignment = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, codebook_size),
        )
        self.codebook = nn.Parameter(torch.randn(codebook_size, hidden_dim) * 0.02)
        self.type_embedding = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.modality_embeddings = nn.Embedding(modalities, hidden_dim)
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, classes),
        )

    def forward(
        self,
        delta_state: Tensor,
        *,
        modality_index: int,
        temperature: float,
        hard: bool,
    ) -> SemanticTokenOutput:
        if delta_state.ndim != 3:
            raise ValueError("delta_state must be [batch, slots, input_dim]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        evidence = self.input_projection(delta_state)
        queries = self.queries.unsqueeze(0).expand(delta_state.shape[0], -1, -1)
        features, _ = self.attention(queries, evidence, evidence, need_weights=False)
        assignment_logits = self.assignment(queries + features)
        probabilities = F.softmax(assignment_logits / temperature, dim=-1)
        code_ids = probabilities.argmax(dim=-1)
        if hard:
            one_hot = F.one_hot(
                code_ids,
                num_classes=self.codebook.shape[0],
            ).to(probabilities.dtype)
            weights = one_hot + probabilities - probabilities.detach() if self.training else one_hot
        else:
            weights = probabilities
        modality = self.modality_embeddings.weight[modality_index]
        tokens = weights @ self.codebook + self.type_embedding + modality
        class_logits = self.decoder(tokens.mean(dim=1))
        return SemanticTokenOutput(
            class_logits=class_logits,
            assignment_logits=assignment_logits,
            assignment_probabilities=probabilities,
            code_ids=code_ids,
            tokens=tokens,
        )


def assignment_statistics(probabilities: Tensor) -> dict[str, Tensor]:
    if probabilities.ndim != 3:
        raise ValueError("probabilities must be [batch, tokens, codes]")
    clamped = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    sample_entropy = -(clamped * clamped.log()).sum(dim=-1).mean()
    mean_usage = probabilities.mean(dim=(0, 1))
    usage = mean_usage.clamp_min(torch.finfo(mean_usage.dtype).tiny)
    usage_entropy = -(usage * usage.log()).sum()
    effective_codes = usage_entropy.exp()
    return {
        "sample_entropy": sample_entropy,
        "usage_entropy": usage_entropy,
        "effective_codes": effective_codes,
    }
