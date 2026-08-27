from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor, nn


class DeltaTokModelConfig(Protocol):
    input_dim: int
    model_dim: int
    tokens_per_frame: int
    delta_tokens: int
    depth: int
    num_heads: int


class DeltaTok(nn.Module):
    def __init__(self, config: DeltaTokModelConfig) -> None:
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
