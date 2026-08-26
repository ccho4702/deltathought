from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import av
import soundfile as sf


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_av_media(source_id: str, path: Path, cache_path: Path) -> dict[str, Any]:
    stat = path.stat()
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached["bytes"] == stat.st_size and cached["mtime_ns"] == stat.st_mtime_ns:
            return cached
    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"Media has no video stream: {path}")
        video = container.streams.video[0]
        duration = (
            float(container.duration / av.time_base)
            if container.duration is not None
            else float(video.duration * video.time_base)
        )
        video_codec = video.codec_context
        audio = container.streams.audio[0] if container.streams.audio else None
        audio_codec = audio.codec_context if audio is not None else None
        value = {
            "source_id": source_id,
            "path": str(path),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256(path),
            "duration_seconds": duration,
            "video": {
                "width": int(video_codec.width),
                "height": int(video_codec.height),
                "fps": (
                    float(video.average_rate) if video.average_rate is not None else None
                ),
            },
            "audio": (
                None
                if audio_codec is None
                else {
                    "sample_rate": int(audio_codec.sample_rate),
                    "channels": int(audio_codec.channels),
                }
            ),
        }
    if duration <= 0:
        raise ValueError(f"Media has invalid duration: {path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.replace(temporary, cache_path)
    return value


def inspect_audio_media(source_id: str, path: Path, cache_path: Path) -> dict[str, Any]:
    stat = path.stat()
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached["bytes"] == stat.st_size and cached["mtime_ns"] == stat.st_mtime_ns:
            return cached
    info = sf.info(path)
    duration = float(info.frames / info.samplerate)
    if duration <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise ValueError(f"Media has invalid audio metadata: {path}")
    value = {
        "source_id": source_id,
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(path),
        "duration_seconds": duration,
        "audio": {
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "format": str(info.format),
            "subtype": str(info.subtype),
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.replace(temporary, cache_path)
    return value
