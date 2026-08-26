from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
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


class SemanticTokenLanguageAdapter(nn.Module):
    """Map already-bottlenecked semantic tokens into a frozen language model."""

    def __init__(self, input_dim: int, language_dim: int, modalities: int = 3) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, language_dim),
            nn.GELU(),
            nn.Linear(language_dim, language_dim),
        )
        self.type_embedding = nn.Parameter(torch.randn(language_dim) * 0.02)
        self.modality_embeddings = nn.Embedding(modalities, language_dim)

    def forward(self, tokens: Tensor, modality_index: int) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError("semantic tokens must be [batch, tokens, input_dim]")
        return (
            self.projection(tokens)
            + self.type_embedding
            + self.modality_embeddings.weight[modality_index]
        )


class FrozenCausalCaptionBackend(nn.Module):
    def __init__(
        self,
        spec: BackboneSpec,
        cache_dir: Path,
        device: torch.device,
        provenance_report: dict[str, Any],
        dtype: torch.dtype | None = None,
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
            torch_dtype=dtype,
        ).eval()
        self.model.requires_grad_(False)
        self.model.to(device)
        self.model.config.use_cache = False
        self.device = device
        self.hidden_size = int(self.model.config.hidden_size)

    def caption_loss(
        self,
        prefix: Tensor,
        prompt: str,
        caption: str,
    ) -> Tensor:
        return self.caption_losses(prefix, prompt, [caption]).mean()

    def caption_losses(
        self,
        prefix: Tensor,
        prompt: str,
        captions: list[str],
    ) -> Tensor:
        if prefix.ndim != 3 or prefix.shape[0] != len(captions):
            raise ValueError("prefix batch must match captions")
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        caption_ids = [
            self.tokenizer.encode(
                caption + self.tokenizer.eos_token,
                add_special_tokens=False,
            )
            for caption in captions
        ]
        sequences = [prompt_ids + ids for ids in caption_ids]
        maximum = max(map(len, sequences))
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("caption tokenizer requires a pad token")
        text_ids = torch.full(
            (len(sequences), maximum),
            pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        text_mask = torch.zeros_like(text_ids)
        labels = torch.full(
            (len(sequences), prefix.shape[1] + maximum),
            -100,
            dtype=torch.long,
            device=self.device,
        )
        for index, (sequence, target) in enumerate(zip(sequences, caption_ids, strict=True)):
            text_ids[index, : len(sequence)] = torch.tensor(sequence, device=self.device)
            text_mask[index, : len(sequence)] = 1
            target_start = prefix.shape[1] + len(prompt_ids)
            labels[index, target_start : target_start + len(target)] = torch.tensor(
                target,
                device=self.device,
            )
        text_embeddings = self.model.get_input_embeddings()(text_ids)
        inputs_embeds = torch.cat((prefix.to(text_embeddings.dtype), text_embeddings), dim=1)
        prefix_mask = torch.ones(
            prefix.shape[:2],
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.cat((prefix_mask, text_mask), dim=1)
        logits = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        ).logits
        shifted_logits = logits[:, :-1].float()
        shifted_labels = labels[:, 1:]
        token_losses = F.cross_entropy(
            shifted_logits.transpose(1, 2),
            shifted_labels,
            ignore_index=-100,
            reduction="none",
        )
        valid = shifted_labels.ne(-100)
        return (token_losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    def candidate_caption_losses(
        self,
        prefix: Tensor,
        prompt: str,
        targets: tuple[str, ...],
    ) -> Tensor:
        batch, prefix_tokens, hidden = prefix.shape
        expanded = (
            prefix[:, None]
            .expand(batch, len(targets), prefix_tokens, hidden)
            .reshape(batch * len(targets), prefix_tokens, hidden)
        )
        losses = self.caption_losses(expanded, prompt, list(targets) * batch)
        return losses.reshape(batch, len(targets))

    @torch.no_grad()
    def generate_captions(
        self,
        prefix: Tensor,
        prompt: str,
        *,
        max_new_tokens: int,
    ) -> list[str]:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        text_ids = torch.tensor(
            [prompt_ids] * prefix.shape[0],
            dtype=torch.long,
            device=self.device,
        )
        text_embeddings = self.model.get_input_embeddings()(text_ids)
        inputs_embeds = torch.cat((prefix.to(text_embeddings.dtype), text_embeddings), dim=1)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=self.device,
        )
        generated = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

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
