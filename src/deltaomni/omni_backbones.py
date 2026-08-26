from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor, nn
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

from deltaomni.provenance import require_approved


@dataclass(frozen=True)
class OmniVideoConfig:
    fps: float
    min_pixels: int
    max_pixels: int


@dataclass(frozen=True)
class OmniBackboneConfig:
    resource_name: str
    model_id: str
    revision: str
    cache_dir: Path
    component: str
    precision: str
    attention_implementation: str
    sample_rate: int
    seconds_per_chunk: int
    position_id_per_seconds: int
    video: OmniVideoConfig


def load_omni_backbone_config(path: Path) -> OmniBackboneConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent
    cache_dir = Path(raw["cache_dir"])
    video = raw["video"]
    config = OmniBackboneConfig(
        resource_name=str(raw["resource_name"]),
        model_id=str(raw["model_id"]),
        revision=str(raw["revision"]),
        cache_dir=cache_dir if cache_dir.is_absolute() else root / cache_dir,
        component=str(raw["component"]),
        precision=str(raw["precision"]),
        attention_implementation=str(raw["attention_implementation"]),
        sample_rate=int(raw["sample_rate"]),
        seconds_per_chunk=int(raw["seconds_per_chunk"]),
        position_id_per_seconds=int(raw["position_id_per_seconds"]),
        video=OmniVideoConfig(
            fps=float(video["fps"]),
            min_pixels=int(video["min_pixels"]),
            max_pixels=int(video["max_pixels"]),
        ),
    )
    if config.component != "thinker":
        raise ValueError("DeltaThought uses the Qwen2.5-Omni Thinker component only")
    if config.precision != "bfloat16":
        raise ValueError("Qwen2.5-Omni experiments require bfloat16")
    if config.sample_rate != 16_000:
        raise ValueError("Qwen2.5-Omni audio requires 16 kHz waveforms")
    if config.attention_implementation not in {"eager", "sdpa", "flash_attention_2"}:
        raise ValueError("Unsupported Qwen2.5-Omni attention implementation")
    if config.video.min_pixels <= 0 or config.video.max_pixels < config.video.min_pixels:
        raise ValueError("Invalid Qwen2.5-Omni video pixel bounds")
    if len(config.revision) != 40:
        raise ValueError("Qwen2.5-Omni revision must be a full commit SHA")
    return config


class QwenOmniThinkerEmbeddingBackend(nn.Module):
    def __init__(
        self,
        config: OmniBackboneConfig,
        device: torch.device,
        provenance_report: dict[str, Any],
        *,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        require_approved(provenance_report, [config.resource_name])
        self.config = config
        self.device = device
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            config.model_id,
            revision=config.revision,
            cache_dir=config.cache_dir,
            local_files_only=local_files_only,
        )
        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            config.model_id,
            revision=config.revision,
            cache_dir=config.cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation=config.attention_implementation,
            local_files_only=local_files_only,
        ).eval()
        self.thinker.requires_grad_(False)
        self.thinker.to(device)

    @property
    def output_dim(self) -> int:
        return int(self.thinker.config.text_config.hidden_size)

    def _video_text(self) -> str:
        return (
            f"{self.processor.vision_bos_token}"
            f"{self.processor.video_token}"
            f"{self.processor.vision_eos_token}"
        )

    def _audio_text(self) -> str:
        return (
            f"{self.processor.audio_bos_token}"
            f"{self.processor.audio_token}"
            f"{self.processor.audio_eos_token}"
        )

    @torch.no_grad()
    def encode_video_chunks(
        self,
        videos: list[list[Image.Image]],
    ) -> tuple[list[Tensor], list[dict[str, Any]]]:
        if not videos or any(not frames for frames in videos):
            raise ValueError("videos must contain at least one non-empty frame sequence")
        inputs = self.processor(
            text=[self._video_text()] * len(videos),
            videos=videos,
            return_tensors="pt",
            padding=True,
            videos_kwargs={
                "fps": self.config.video.fps,
                "min_pixels": self.config.video.min_pixels,
                "max_pixels": self.config.video.max_pixels,
                "seconds_per_chunk": self.config.seconds_per_chunk,
                "position_id_per_seconds": self.config.position_id_per_seconds,
                "use_audio_in_video": False,
            },
        )
        pixel_values = inputs["pixel_values_videos"].to(self.device)
        grid_thw = inputs["video_grid_thw"].to(self.device)
        features = self.thinker.get_video_features(pixel_values, grid_thw)
        merge = int(self.thinker.config.vision_config.spatial_merge_size)
        counts = (grid_thw.prod(dim=1) // (merge**2)).tolist()
        chunks = list(features.split(counts, dim=0))
        metadata = [
            {
                "grid_thw": [int(value) for value in grid.tolist()],
                "tokens": int(count),
                "width": int(chunk.shape[-1]),
            }
            for grid, count, chunk in zip(grid_thw.cpu(), counts, chunks, strict=True)
        ]
        return chunks, metadata

    @torch.no_grad()
    def encode_audio_chunks(
        self,
        waveforms: list[np.ndarray],
    ) -> tuple[list[Tensor], list[dict[str, Any]]]:
        if not waveforms or any(waveform.ndim != 1 for waveform in waveforms):
            raise ValueError("waveforms must contain non-empty mono arrays")
        inputs = self.processor(
            text=[self._audio_text()] * len(waveforms),
            audio=waveforms,
            return_tensors="pt",
            padding=True,
            audio_kwargs={"sampling_rate": self.config.sample_rate},
        )
        input_features = inputs["input_features"].to(self.device)
        feature_mask = inputs["feature_attention_mask"].to(self.device)
        feature_lengths = feature_mask.sum(dim=1)
        _, output_lengths = self.thinker.audio_tower._get_feat_extract_output_lengths(
            feature_lengths
        )
        features = self.thinker.get_audio_features(
            input_features,
            feature_attention_mask=feature_mask,
        )
        counts = [int(value) for value in output_lengths.tolist()]
        chunks = list(features.split(counts, dim=0))
        metadata = [
            {
                "samples": int(waveform.shape[0]),
                "seconds": float(waveform.shape[0] / self.config.sample_rate),
                "tokens": int(count),
                "width": int(chunk.shape[-1]),
            }
            for waveform, count, chunk in zip(waveforms, counts, chunks, strict=True)
        ]
        return chunks, metadata
