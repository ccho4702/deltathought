from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import av
import librosa
import numpy as np
import soundfile as sf
import torch
from PIL import Image

from deltaomni.omni_backbones import (
    QwenOmniThinkerEmbeddingBackend,
    load_omni_backbone_config,
)
from deltaomni.provenance import audit as audit_provenance
from deltaomni.train_sanity import _atomic_json


def _video_chunks(path: Path, frames_per_chunk: int = 4) -> list[list[Image.Image]]:
    with av.open(str(path), mode="r") as container:
        decoded = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
    required = 2 * frames_per_chunk
    if len(decoded) < required:
        raise ValueError(f"Video has {len(decoded)} frames, requires at least {required}: {path}")
    indices = torch.linspace(0, len(decoded) - 1, required).round().long().tolist()
    selected = [decoded[index] for index in indices]
    return [selected[:frames_per_chunk], selected[frames_per_chunk:]]


def _audio_chunks(path: Path, sample_rate: int) -> list[np.ndarray]:
    waveform, source_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = waveform.mean(axis=1)
    if source_rate != sample_rate:
        mono = librosa.resample(mono, orig_sr=source_rate, target_sr=sample_rate)
    required = 4 * sample_rate
    if mono.shape[0] < required:
        mono = np.pad(mono, (0, required - mono.shape[0]))
    return [mono[: 2 * sample_rate], mono[2 * sample_rate : required]]


def run(
    config_path: Path,
    provenance_path: Path,
    video_path: Path,
    audio_path: Path,
    output_path: Path,
) -> dict[str, object]:
    config = load_omni_backbone_config(config_path)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen2.5-Omni smoke requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    print("loading_qwen2_5_omni_thinker", flush=True)
    backend = QwenOmniThinkerEmbeddingBackend(
        config,
        device,
        audit_provenance(provenance_path),
    )
    loaded_seconds = time.perf_counter() - started
    print(f"thinker_loaded elapsed={loaded_seconds:.1f}s", flush=True)

    video_chunks = _video_chunks(video_path)
    first_video, first_video_metadata = backend.encode_video_chunks([video_chunks[0]])
    second_video, second_video_metadata = backend.encode_video_chunks([video_chunks[1]])
    video_features = [first_video[0], second_video[0]]
    video_metadata = [first_video_metadata[0], second_video_metadata[0]]
    repeated_video, _ = backend.encode_video_chunks([video_chunks[0]])
    audio_chunks = _audio_chunks(audio_path, config.sample_rate)
    first_audio, first_audio_metadata = backend.encode_audio_chunks([audio_chunks[0]])
    second_audio, second_audio_metadata = backend.encode_audio_chunks([audio_chunks[1]])
    audio_features = [first_audio[0], second_audio[0]]
    audio_metadata = [first_audio_metadata[0], second_audio_metadata[0]]
    repeated_audio, _ = backend.encode_audio_chunks([audio_chunks[0]])
    video_repeat_error = (video_features[0] - repeated_video[0]).abs().float()
    audio_repeat_error = (audio_features[0] - repeated_audio[0]).abs().float()

    checks = {
        "video_width_matches_thinker": all(
            feature.shape[-1] == backend.output_dim for feature in video_features
        ),
        "audio_width_matches_thinker": all(
            feature.shape[-1] == backend.output_dim for feature in audio_features
        ),
        "video_repeat_deterministic": torch.equal(video_features[0], repeated_video[0]),
        "audio_repeat_deterministic": torch.equal(audio_features[0], repeated_audio[0]),
        "video_change_nonzero": not torch.equal(video_features[0], video_features[1]),
        "audio_change_nonzero": not torch.equal(audio_features[0], audio_features[1]),
    }
    report: dict[str, object] = {
        "schema": "deltaomni.qwen2_5_omni_backbone_smoke.v1",
        "model_id": config.model_id,
        "revision": config.revision,
        "component": config.component,
        "output_dim": backend.output_dim,
        "video": {
            "source": str(video_path),
            "chunks": video_metadata,
            "change_mean_absolute": float(
                (video_features[1] - video_features[0]).abs().float().mean()
            ),
            "repeat_error_max_absolute": float(video_repeat_error.max()),
            "repeat_error_mean_absolute": float(video_repeat_error.mean()),
        },
        "audio": {
            "source": str(audio_path),
            "chunks": audio_metadata,
            "change_mean_absolute": float(
                (audio_features[1] - audio_features[0]).abs().float().mean()
            ),
            "repeat_error_max_absolute": float(audio_repeat_error.max()),
            "repeat_error_mean_absolute": float(audio_repeat_error.mean()),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "load_seconds": loaded_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test native Qwen2.5-Omni encoders")
    parser.add_argument("--config", type=Path, default=Path("configs/qwen2_5_omni.yaml"))
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("/mnt/nfs_shared_data/dataset/ssv2/20bn-something-something-v2/57188.webm"),
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path(
            "/mnt/nfs_shared_data/dataset/omniembed/audioset/eval_segments/audio/"
            "--4gqARaEJE_0_10000.flac"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/qwen2_5_omni_backbone_smoke.json"),
    )
    args = parser.parse_args()
    report = run(args.config, args.provenance, args.video, args.audio, args.output)
    print(json.dumps(report, indent=2))
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
