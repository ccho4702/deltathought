from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from deltaomni.backbones import (
    ClapEmbeddingBackend,
    DinoV2EmbeddingBackend,
    load_backbone_config,
)
from deltaomni.config import load_config
from deltaomni.model import PairDeltaEncoder
from deltaomni.provenance import audit
from deltaomni.train_sanity import _atomic_json


def run(
    backbone_config_path: Path,
    provenance_path: Path,
    sanity_config_path: Path,
) -> dict:
    backbone_config = load_backbone_config(backbone_config_path)
    provenance = audit(provenance_path)
    device = torch.device(backbone_config.device)
    gradient = np.tile(np.arange(224, dtype=np.uint8)[None, :], (224, 1))
    image = Image.fromarray(gradient[:, :, None].repeat(3, axis=2))
    changed_image = Image.fromarray(np.roll(np.asarray(image), 32, axis=1))
    time_axis = np.arange(backbone_config.audio.sample_rate)
    waveform = np.sin(2 * np.pi * 440 * time_axis / backbone_config.audio.sample_rate).astype(
        np.float32
    )
    changed_waveform = np.sin(
        2 * np.pi * 880 * time_axis / backbone_config.audio.sample_rate
    ).astype(np.float32)

    video_backend = DinoV2EmbeddingBackend(
        backbone_config.video,
        backbone_config.cache_dir,
        device,
        provenance,
    )
    video_first = video_backend.encode([image])
    video_second = video_backend.encode([image])
    video_changed = video_backend.encode([changed_image])
    audio_backend = ClapEmbeddingBackend(
        backbone_config.audio,
        backbone_config.cache_dir,
        device,
        provenance,
    )
    audio_first = audio_backend.encode([waveform])
    audio_second = audio_backend.encode([waveform])
    audio_changed = audio_backend.encode([changed_waveform])

    sanity = load_config(sanity_config_path)
    video_delta_config = replace(
        sanity.model,
        embedding_dim=video_first.shape[-1],
        embedding_tokens=video_first.shape[1],
        delta_tokens=backbone_config.video.delta_tokens,
    )
    audio_delta_config = replace(
        sanity.model,
        embedding_dim=audio_first.shape[-1],
        embedding_tokens=audio_first.shape[1],
        delta_tokens=backbone_config.audio.delta_tokens,
    )
    video_delta = PairDeltaEncoder(video_delta_config)(video_first, video_second)
    audio_delta = PairDeltaEncoder(audio_delta_config)(audio_first, audio_second)
    video_change_delta = PairDeltaEncoder(video_delta_config)(video_first, video_changed)
    audio_change_delta = PairDeltaEncoder(audio_delta_config)(audio_first, audio_changed)
    checks = {
        "video_rank_three": video_first.ndim == 3,
        "audio_rank_three": audio_first.ndim == 3,
        "video_deterministic": torch.equal(video_first, video_second),
        "audio_deterministic": torch.equal(audio_first, audio_second),
        "video_same_input_zero_delta": torch.count_nonzero(video_delta).item() == 0,
        "audio_same_input_zero_delta": torch.count_nonzero(audio_delta).item() == 0,
        "video_change_is_nonzero": float(video_change_delta.norm()) > 0,
        "audio_change_is_nonzero": float(audio_change_delta.norm()) > 0,
        "video_delta_is_compressed": video_delta.shape[1] < video_first.shape[1],
        "audio_delta_does_not_expand": audio_delta.shape[1] <= audio_first.shape[1],
        "models_frozen": (
            not any(parameter.requires_grad for parameter in video_backend.parameters())
            and not any(parameter.requires_grad for parameter in audio_backend.parameters())
        ),
    }
    return {
        "video": {
            "model_id": backbone_config.video.model_id,
            "revision": backbone_config.video.revision,
            "embedding_shape": list(video_first.shape),
            "delta_shape": list(video_delta.shape),
            "changed_delta_norm": float(video_change_delta.norm()),
        },
        "audio": {
            "model_id": backbone_config.audio.model_id,
            "revision": backbone_config.audio.revision,
            "embedding_shape": list(audio_first.shape),
            "delta_shape": list(audio_delta.shape),
            "changed_delta_norm": float(audio_change_delta.norm()),
            "sample_rate": backbone_config.audio.sample_rate,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test pinned real embedding backbones")
    parser.add_argument("--config", type=Path, default=Path("configs/backbones.yaml"))
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    parser.add_argument("--sanity-config", type=Path, default=Path("configs/sanity.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/backbone_smoke.json"))
    args = parser.parse_args()
    report = run(args.config, args.provenance, args.sanity_config)
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
