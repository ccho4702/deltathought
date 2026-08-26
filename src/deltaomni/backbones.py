from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor, nn
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, ClapModel

from deltaomni.provenance import require_approved


@dataclass(frozen=True)
class BackboneSpec:
    resource_name: str
    model_id: str
    revision: str
    delta_tokens: int | None
    sample_rate: int | None = None


@dataclass(frozen=True)
class BackboneConfig:
    cache_dir: Path
    device: str
    video: BackboneSpec
    audio: BackboneSpec
    language: BackboneSpec
    language_smoke_anchor_tokens: int
    language_smoke_steps: int
    language_smoke_learning_rate: float


def load_backbone_config(path: Path) -> BackboneConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def spec(name: str) -> BackboneSpec:
        value = raw["models"][name]
        return BackboneSpec(
            resource_name=str(value["resource_name"]),
            model_id=str(value["model_id"]),
            revision=str(value["revision"]),
            delta_tokens=int(value["delta_tokens"]) if "delta_tokens" in value else None,
            sample_rate=int(value["sample_rate"]) if "sample_rate" in value else None,
        )

    cache_dir = Path(raw["cache_dir"])
    return BackboneConfig(
        cache_dir=cache_dir if cache_dir.is_absolute() else root / cache_dir,
        device=str(raw["device"]),
        video=spec("video"),
        audio=spec("audio"),
        language=spec("language"),
        language_smoke_anchor_tokens=int(raw["models"]["language"]["smoke_anchor_tokens"]),
        language_smoke_steps=int(raw["models"]["language"]["smoke_steps"]),
        language_smoke_learning_rate=float(
            raw["models"]["language"]["smoke_learning_rate"]
        ),
    )


class DinoV2EmbeddingBackend(nn.Module):
    def __init__(
        self,
        spec: BackboneSpec,
        cache_dir: Path,
        device: torch.device,
        provenance_report: dict[str, Any],
    ) -> None:
        super().__init__()
        require_approved(provenance_report, [spec.resource_name])
        self.processor = AutoImageProcessor.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            cache_dir=cache_dir,
            use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            cache_dir=cache_dir,
        ).eval()
        self.model.requires_grad_(False)
        self.model.to(device)
        self.device = device

    @torch.no_grad()
    def encode(self, images: list[Image.Image]) -> Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        return self.model(**inputs).last_hidden_state


class ClapEmbeddingBackend(nn.Module):
    def __init__(
        self,
        spec: BackboneSpec,
        cache_dir: Path,
        device: torch.device,
        provenance_report: dict[str, Any],
    ) -> None:
        super().__init__()
        require_approved(provenance_report, [spec.resource_name])
        if spec.sample_rate is None:
            raise ValueError("CLAP requires a configured sample_rate")
        self.processor = AutoProcessor.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            cache_dir=cache_dir,
        )
        self.model = ClapModel.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            cache_dir=cache_dir,
        ).eval()
        self.model.requires_grad_(False)
        self.model.to(device)
        self.device = device
        self.sample_rate = spec.sample_rate

    @torch.no_grad()
    def encode(self, waveforms: list[np.ndarray]) -> Tensor:
        inputs = self.processor(
            audios=waveforms,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        return self.model.get_audio_features(**inputs).unsqueeze(1)
