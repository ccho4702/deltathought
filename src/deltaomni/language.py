from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltaomni.backbones import BackboneSpec
from deltaomni.provenance import require_approved


class DeltaLanguageProjector(nn.Module):
    def __init__(self, input_dim: int, language_dim: int, modalities: int = 3) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, language_dim),
        )
        self.type_embeddings = nn.Parameter(torch.randn(2, language_dim) * 0.02)
        self.modality_embeddings = nn.Embedding(modalities, language_dim)

    def forward(self, anchor: Tensor, delta: Tensor, modality_index: int) -> Tensor:
        if anchor.ndim != 3 or delta.ndim != 3 or anchor.shape[-1] != delta.shape[-1]:
            raise ValueError("anchor and delta must be [batch, tokens, shared_dim]")
        modality = self.modality_embeddings.weight[modality_index]
        full_prefix = self.projection(anchor) + self.type_embeddings[0] + modality
        delta_prefix = self.projection(delta) + self.type_embeddings[1] + modality
        return torch.cat((full_prefix, delta_prefix), dim=1)


class ChangeAwareResampler(nn.Module):
    def __init__(
        self,
        input_dim: int,
        language_dim: int,
        query_tokens: int,
        num_heads: int,
        modalities: int = 3,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, language_dim),
        )
        self.type_embeddings = nn.Parameter(torch.randn(2, language_dim) * 0.02)
        self.modality_embeddings = nn.Embedding(modalities, language_dim)
        self.queries = nn.Parameter(torch.randn(query_tokens, language_dim) * 0.02)
        self.attention = nn.MultiheadAttention(language_dim, num_heads, batch_first=True)
        self.output = nn.Sequential(
            nn.LayerNorm(language_dim),
            nn.Linear(language_dim, 4 * language_dim),
            nn.GELU(),
            nn.Linear(4 * language_dim, language_dim),
        )
        self.output_norm = nn.LayerNorm(language_dim)

    def forward(self, anchor: Tensor, delta: Tensor, modality_index: int) -> Tensor:
        if anchor.ndim != 3 or delta.ndim != 3 or anchor.shape[-1] != delta.shape[-1]:
            raise ValueError("anchor and delta must be [batch, tokens, shared_dim]")
        modality = self.modality_embeddings.weight[modality_index]
        full_tokens = self.projection(anchor) + self.type_embeddings[0] + modality
        delta_tokens = self.projection(delta) + self.type_embeddings[1] + modality
        evidence = torch.cat((full_tokens, delta_tokens), dim=1)
        queries = self.queries.unsqueeze(0).expand(anchor.shape[0], -1, -1)
        attended, _ = self.attention(queries, evidence, evidence, need_weights=False)
        return self.output_norm(queries + attended + self.output(attended))


class FrozenCausalCaptionBackend(nn.Module):
    def __init__(
        self,
        spec: BackboneSpec,
        cache_dir: Path,
        device: torch.device,
        provenance_report: dict[str, Any],
    ) -> None:
        super().__init__()
        require_approved(provenance_report, [spec.resource_name])
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            cache_dir=cache_dir,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            cache_dir=cache_dir,
        ).eval()
        self.model.requires_grad_(False)
        self.model.to(device)
        self.device = device
        self.hidden_size = int(self.model.config.hidden_size)

    def caption_loss(
        self,
        prefix: Tensor,
        prompt: str,
        caption: str,
    ) -> Tensor:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        caption_ids = self.tokenizer.encode(
            caption + self.tokenizer.eos_token,
            add_special_tokens=False,
        )
        text_ids = torch.tensor([prompt_ids + caption_ids], device=self.device)
        text_embeddings = self.model.get_input_embeddings()(text_ids)
        inputs_embeds = torch.cat((prefix, text_embeddings), dim=1)
        ignored = torch.full(
            (1, prefix.shape[1] + len(prompt_ids)),
            -100,
            dtype=torch.long,
            device=self.device,
        )
        labels = torch.cat(
            (ignored, torch.tensor([caption_ids], dtype=torch.long, device=self.device)),
            dim=1,
        )
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        ).loss

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> Tensor:
        tokens = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        outputs = self.model(**tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        mask = tokens["attention_mask"].unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
