from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import transformers
import yaml
from PIL import Image, ImageOps
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
from transformers.utils import logging as transformers_logging

from deltaomni.audiocaps_caption_lora import _rouge_l, _tokens, _word_f1
from deltaomni.data.schema import CanonicalEpisode, iter_jsonl
from deltaomni.distributed import distributed_context
from deltaomni.omni_backbones import load_omni_backbone_config

REPORT_SCHEMA = "deltaomni.qwen2_5_omni_vanilla_baseline.v1"


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    backend: str
    nccl_compatibility_mode: bool
    cpu_threads: int


@dataclass(frozen=True)
class BaselineConfig:
    seed: int
    omni_config: Path
    nextqa_manifest: Path
    nextqa_selection_manifest: Path
    msrvtt_metadata: Path
    msrvtt_video_root: Path
    msrvtt_count: int
    minimum_seconds: float
    maximum_seconds: float
    sample_fps: float
    frame_width: int
    frame_height: int
    freeform_max_new_tokens: int
    multiple_choice_max_new_tokens: int
    caption_max_new_tokens: int
    runtime: RuntimeConfig
    run_id: str
    output_root: Path
    log_root: Path
    report_path: Path
    comparison_report: Path


def load_config(path: Path) -> BaselineConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    path_fields = {
        "omni_config",
        "nextqa_manifest",
        "nextqa_selection_manifest",
        "msrvtt_metadata",
        "msrvtt_video_root",
        "output_root",
        "log_root",
        "report_path",
        "comparison_report",
    }
    values = {key: resolve(value) if key in path_fields else value for key, value in raw.items()}
    values["runtime"] = RuntimeConfig(**raw["runtime"])
    config = BaselineConfig(**values)
    positive = (
        config.msrvtt_count,
        config.minimum_seconds,
        config.maximum_seconds,
        config.sample_fps,
        config.frame_width,
        config.frame_height,
        config.freeform_max_new_tokens,
        config.multiple_choice_max_new_tokens,
        config.caption_max_new_tokens,
        config.runtime.cpu_threads,
    )
    if min(positive) <= 0 or config.maximum_seconds < config.minimum_seconds:
        raise ValueError("Invalid vanilla baseline controls")
    if not config.run_id or "/" in config.run_id:
        raise ValueError("run_id must be a non-empty path component")
    return config


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _split_path(manifest_path: Path, split: str) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest_path.parent / manifest["splits"][split]["path"]


def _nextqa_episodes(config: BaselineConfig) -> list[CanonicalEpisode]:
    selection = json.loads(config.nextqa_selection_manifest.read_text(encoding="utf-8"))
    selected = [record["source_id"] for record in selection["splits"]["test"]]
    by_id = {
        episode.source_id: episode
        for episode in iter_jsonl(_split_path(config.nextqa_manifest, "test"))
        if episode.source_id in set(selected)
    }
    missing = [source_id for source_id in selected if source_id not in by_id]
    if missing:
        raise ValueError(f"Missing selected NExT-QA episodes: {missing}")
    return [by_id[source_id] for source_id in selected]


def _duration_seconds(path: Path) -> float:
    with av.open(str(path)) as container:
        if container.duration is not None:
            return float(container.duration / av.time_base)
        streams = [*container.streams.video, *container.streams.audio]
        durations = [
            float(stream.duration * stream.time_base) for stream in streams if stream.duration
        ]
    if not durations:
        raise ValueError(f"Media has no duration: {path}")
    return max(durations)


def _msrvtt_items(config: BaselineConfig) -> list[dict[str, Any]]:
    metadata = json.loads(config.msrvtt_metadata.read_text(encoding="utf-8"))
    ordered = sorted(
        metadata,
        key=lambda source_id: hashlib.sha256(
            f"{config.seed}:{source_id}".encode()
        ).hexdigest(),
    )
    result = []
    for source_id in ordered:
        path = config.msrvtt_video_root / f"{source_id}.mp4"
        if not path.is_file():
            continue
        duration = _duration_seconds(path)
        if config.minimum_seconds <= duration <= config.maximum_seconds:
            result.append(
                {
                    "source_id": source_id,
                    "path": path,
                    "duration_seconds": duration,
                    "references": tuple(metadata[source_id].values()),
                }
            )
        if len(result) == config.msrvtt_count:
            break
    if len(result) != config.msrvtt_count:
        raise ValueError(f"Found {len(result)}/{config.msrvtt_count} eligible MSR-VTT clips")
    return result


def _sample_video(path: Path, fps: float, size: tuple[int, int]) -> list[Image.Image]:
    duration = _duration_seconds(path)
    targets = np.arange(0.0, duration, 1.0 / fps).tolist()
    selected: list[Image.Image] = []
    target_index = 0
    previous = None
    previous_time = 0.0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream_fps = float(stream.average_rate) if stream.average_rate is not None else None
        for index, frame in enumerate(container.decode(stream)):
            timestamp = float(frame.time) if frame.time is not None else index / (stream_fps or fps)
            if previous is None:
                previous, previous_time = frame, timestamp
                continue
            while target_index < len(targets) and targets[target_index] <= timestamp:
                target = targets[target_index]
                chosen = (
                    previous
                    if abs(previous_time - target) <= abs(timestamp - target)
                    else frame
                )
                selected.append(
                    ImageOps.pad(chosen.to_image().convert("RGB"), size, color=(0, 0, 0))
                )
                target_index += 1
            if target_index == len(targets):
                break
            previous, previous_time = frame, timestamp
    if previous is None:
        raise ValueError(f"No video frames: {path}")
    while target_index < len(targets):
        selected.append(ImageOps.pad(previous.to_image().convert("RGB"), size, color=(0, 0, 0)))
        target_index += 1
    return selected


def _decode_audio(path: Path, sample_rate: int) -> np.ndarray | None:
    arrays = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        for frame in container.decode(audio=0):
            for converted in resampler.resample(frame):
                arrays.append(converted.to_ndarray().reshape(-1))
        for converted in resampler.resample(None):
            arrays.append(converted.to_ndarray().reshape(-1))
    return np.concatenate(arrays).astype(np.float32, copy=False) if arrays else None


def _clean_prediction(text: str) -> str:
    first = text.strip().splitlines()[0] if text.strip() else ""
    return re.sub(r"^(assistant|answer)\s*:\s*", "", first, flags=re.IGNORECASE).strip()


def _parse_choice(text: str, count: int) -> int | None:
    cleaned = _clean_prediction(text).upper()
    match = re.search(r"(?:^|[^A-Z])([A-Z])(?:[^A-Z]|$)", cleaned)
    if match is None:
        return None
    index = ord(match.group(1)) - ord("A")
    return index if 0 <= index < count else None


def _text_metrics(prediction: str, references: tuple[str, ...]) -> dict[str, float]:
    normalized = " ".join(_tokens(prediction))
    return {
        "exact_match": float(any(normalized == " ".join(_tokens(ref)) for ref in references)),
        "word_f1": max(_word_f1(prediction, ref) for ref in references),
        "rouge_l": max(_rouge_l(prediction, ref) for ref in references),
    }


def _lexical_choice(prediction: str, choices: tuple[str, ...]) -> int:
    scores = [(_word_f1(prediction, choice), _rouge_l(prediction, choice)) for choice in choices]
    return max(range(len(scores)), key=scores.__getitem__)


class VanillaGenerator:
    def __init__(self, config: BaselineConfig, device: torch.device) -> None:
        omni = load_omni_backbone_config(config.omni_config)
        self.config = config
        self.omni = omni
        self.device = device
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            omni.model_id,
            revision=omni.revision,
            cache_dir=omni.cache_dir,
            local_files_only=True,
        )
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            omni.model_id,
            revision=omni.revision,
            cache_dir=omni.cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation=omni.attention_implementation,
            local_files_only=True,
        ).eval().to(device)

    @torch.inference_mode()
    def generate(
        self,
        frames: list[Image.Image],
        audio: np.ndarray | None,
        prompt: str,
        max_new_tokens: int,
    ) -> tuple[str, float]:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "local-media"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        use_audio = audio is not None
        inputs = self.processor(
            text=text,
            videos=[frames],
            audio=[audio] if use_audio else None,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio,
            videos_kwargs={
                "fps": self.config.sample_fps,
                "min_pixels": self.omni.video.min_pixels,
                "max_pixels": self.omni.video.max_pixels,
                "seconds_per_chunk": self.omni.seconds_per_chunk,
                "position_id_per_seconds": self.omni.position_id_per_seconds,
            },
            audio_kwargs={"sampling_rate": self.omni.sample_rate},
        )
        inputs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        started = time.perf_counter()
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_audio_in_video=use_audio,
        )
        latency = time.perf_counter() - started
        generated = output[:, inputs["input_ids"].shape[1] :]
        decoded = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return _clean_prediction(decoded), latency


def _prediction_path(run_dir: Path, key: str) -> Path:
    digest = hashlib.sha256(key.encode()).hexdigest()
    return run_dir / "predictions" / f"{digest}.json"


def _save_prediction(run_dir: Path, key: str, payload: dict[str, Any]) -> None:
    path = _prediction_path(run_dir, key)
    if not path.exists():
        _atomic_json(path, {"key": key, **payload})


def _pending(run_dir: Path, keys: list[str]) -> bool:
    return any(not _prediction_path(run_dir, key).is_file() for key in keys)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def _by_question_type(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    types = sorted({str(row["question_type"]) for row in rows})
    return {
        question_type: _mean(
            [row for row in rows if row["question_type"] == question_type], metric
        )
        for question_type in types
    }


def _consolidate(
    config: BaselineConfig, run_dir: Path, hardware: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "predictions").glob("*.json"))
    ]
    freeform = [row for row in rows if row["task"] == "nextqa_freeform"]
    multiple_choice = [row for row in rows if row["task"] == "nextqa_multiple_choice"]
    captions = [row for row in rows if row["task"] == "msrvtt_caption"]
    comparison = None
    if config.comparison_report.is_file():
        current = json.loads(config.comparison_report.read_text(encoding="utf-8"))
        comparison = {
            "report": str(config.comparison_report),
            "delta_joint_head_test_accuracy": current["final"]["test"]["accuracy"],
            "vanilla_raw_omni_test_accuracy": _mean(multiple_choice, "correct"),
            "warning": "The delta result is a trained lightweight classifier, not Qwen LoRA.",
        }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=config.omni_config.parent.parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "schema": REPORT_SCHEMA,
        "run_id": config.run_id,
        "code_revision": revision,
        "resolved_config": json.loads(json.dumps(asdict(config), default=str)),
        "inputs": {
            "config_sha256": _sha256(config.omni_config),
            "nextqa_manifest_sha256": _sha256(config.nextqa_manifest),
            "nextqa_selection_manifest_sha256": _sha256(config.nextqa_selection_manifest),
            "msrvtt_metadata_sha256": _sha256(config.msrvtt_metadata),
        },
        "nextqa_freeform": {
            "examples": len(freeform),
            "exact_match": _mean(freeform, "exact_match"),
            "word_f1": _mean(freeform, "word_f1"),
            "rouge_l": _mean(freeform, "rouge_l"),
            "lexical_choice_accuracy": _mean(freeform, "lexical_choice_correct"),
            "word_f1_by_question_type": _by_question_type(freeform, "word_f1"),
        },
        "nextqa_multiple_choice": {
            "examples": len(multiple_choice),
            "accuracy": _mean(multiple_choice, "correct"),
            "parse_rate": _mean(multiple_choice, "parsed"),
            "accuracy_by_question_type": _by_question_type(multiple_choice, "correct"),
        },
        "msrvtt_caption": {
            "examples": len(captions),
            "exact_match": _mean(captions, "exact_match"),
            "word_f1": _mean(captions, "word_f1"),
            "rouge_l": _mean(captions, "rouge_l"),
        },
        "comparison": comparison,
        "mean_generation_latency_seconds": _mean(rows, "generation_latency_seconds"),
        "hardware": hardware,
        "software": {"torch": torch.__version__, "transformers": transformers.__version__},
        "examples": rows[:12],
        "limitations": [
            "NExT-QA references are multiple-choice answer phrases; free-form metrics are a "
            "derived evaluation.",
            "Lexical-choice accuracy maps generated text to the most overlapping choice and is "
            "diagnostic, not a standard open-QA metric.",
            "MSR-VTT scores use maximum word-F1/ROUGE-L over 20 references, not the standard "
            "COCO caption evaluation suite.",
            "This PoC uses 16 source-disjoint NExT-QA test videos and 16 short MSR-VTT test "
            "videos.",
        ],
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }


def run(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    if config.report_path.exists():
        existing = json.loads(config.report_path.read_text(encoding="utf-8"))
        if existing.get("run_id") == config.run_id and existing.get("schema") == REPORT_SCHEMA:
            return existing
        raise FileExistsError(f"Refusing to overwrite baseline report: {config.report_path}")
    transformers_logging.set_verbosity_error()
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.runtime.cpu_threads)
    with distributed_context(
        config.runtime.device,
        config.runtime.backend,
        nccl_compatibility_mode=config.runtime.nccl_compatibility_mode,
    ) as context:
        run_dir = config.output_root / config.run_id
        log_path = config.log_root / config.run_id / f"rank-{context.rank:04d}.jsonl"
        episodes = _nextqa_episodes(config)
        captions = _msrvtt_items(config)
        tasks: list[tuple[str, Any]] = [
            *(("nextqa", episode) for episode in episodes),
            *(("msrvtt", item) for item in captions),
        ]
        local = tasks[context.rank :: context.world_size]
        expected = sum(
            2 * len(value.qa or ()) if kind == "nextqa" else 1 for kind, value in local
        )
        _append_log(
            log_path,
            {
                "event": "start_or_resume",
                "rank": context.rank,
                "world_size": context.world_size,
                "tasks": len(local),
                "expected_generations": expected,
                "timestamp_utc": datetime.now(UTC).isoformat(),
            },
        )
        generator = VanillaGenerator(config, context.device)
        started = time.perf_counter()
        local_completed = 0
        for kind, value in local:
            if kind == "nextqa":
                episode: CanonicalEpisode = value
                keys = [
                    f"nextqa:{episode.source_id}:{qa.question_id}:{mode}"
                    for qa in episode.qa or ()
                    for mode in ("freeform", "multiple_choice")
                ]
                if not _pending(run_dir, keys):
                    local_completed += len(keys)
                    continue
                assert episode.media.video is not None
                media_path = episode.media.video.path
                frames = _sample_video(
                    media_path, config.sample_fps, (config.frame_width, config.frame_height)
                )
                audio = _decode_audio(media_path, generator.omni.sample_rate)
                for qa in episode.qa or ():
                    assert qa.choices is not None and qa.answer_index is not None
                    free_key = f"nextqa:{episode.source_id}:{qa.question_id}:freeform"
                    if not _prediction_path(run_dir, free_key).is_file():
                        prediction, latency = generator.generate(
                            frames,
                            audio,
                            "Answer the question about the video in a short phrase. "
                            "Do not explain. "
                            f"Question: {qa.question}",
                            config.freeform_max_new_tokens,
                        )
                        metrics = _text_metrics(prediction, (qa.answer,))
                        lexical = _lexical_choice(prediction, qa.choices)
                        _save_prediction(
                            run_dir,
                            free_key,
                            {
                                "task": "nextqa_freeform",
                                "source_id": episode.source_id,
                                "question_id": qa.question_id,
                                "question_type": qa.question_type,
                                "question": qa.question,
                                "prediction": prediction,
                                "reference": qa.answer,
                                "choices": qa.choices,
                                **metrics,
                                "lexical_choice_index": lexical,
                                "lexical_choice_correct": float(lexical == qa.answer_index),
                                "generation_latency_seconds": latency,
                            },
                        )
                        local_completed += 1
                    mc_key = f"nextqa:{episode.source_id}:{qa.question_id}:multiple_choice"
                    if not _prediction_path(run_dir, mc_key).is_file():
                        choices = "\n".join(
                            f"{chr(65 + index)}. {choice}"
                            for index, choice in enumerate(qa.choices)
                        )
                        prediction, latency = generator.generate(
                            frames,
                            audio,
                            "Choose the best answer to the video question. "
                            "Reply with only one letter.\n"
                            f"Question: {qa.question}\n{choices}",
                            config.multiple_choice_max_new_tokens,
                        )
                        parsed = _parse_choice(prediction, len(qa.choices))
                        _save_prediction(
                            run_dir,
                            mc_key,
                            {
                                "task": "nextqa_multiple_choice",
                                "source_id": episode.source_id,
                                "question_id": qa.question_id,
                                "question_type": qa.question_type,
                                "question": qa.question,
                                "prediction": prediction,
                                "reference": qa.answer,
                                "answer_index": qa.answer_index,
                                "parsed_index": parsed,
                                "parsed": float(parsed is not None),
                                "correct": float(parsed == qa.answer_index),
                                "generation_latency_seconds": latency,
                            },
                        )
                        local_completed += 1
                    elapsed = time.perf_counter() - started
                    eta = elapsed / max(local_completed, 1) * max(expected - local_completed, 0)
                    print(
                        f"vanilla_rank={context.rank} progress={local_completed}/{expected} "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )
            else:
                item = value
                key = f"msrvtt:{item['source_id']}:caption"
                if _prediction_path(run_dir, key).is_file():
                    local_completed += 1
                    continue
                frames = _sample_video(
                    item["path"], config.sample_fps, (config.frame_width, config.frame_height)
                )
                audio = _decode_audio(item["path"], generator.omni.sample_rate)
                prediction, latency = generator.generate(
                    frames,
                    audio,
                    "Summarize the video in one concise sentence.",
                    config.caption_max_new_tokens,
                )
                metrics = _text_metrics(prediction, item["references"])
                _save_prediction(
                    run_dir,
                    key,
                    {
                        "task": "msrvtt_caption",
                        "source_id": item["source_id"],
                        "duration_seconds": item["duration_seconds"],
                        "prediction": prediction,
                        "references": item["references"],
                        **metrics,
                        "generation_latency_seconds": latency,
                    },
                )
                local_completed += 1
                elapsed = time.perf_counter() - started
                eta = elapsed / max(local_completed, 1) * max(expected - local_completed, 0)
                print(
                    f"vanilla_rank={context.rank} progress={local_completed}/{expected} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
        hardware_path = run_dir / "hardware" / f"rank-{context.rank:04d}.json"
        properties = torch.cuda.get_device_properties(context.device)
        _atomic_json(
            hardware_path,
            {
                "rank": context.rank,
                "device": str(context.device),
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(context.device),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        _append_log(
            log_path,
            {
                "event": "rank_complete",
                "rank": context.rank,
                "completed_generations": local_completed,
                "timestamp_utc": datetime.now(UTC).isoformat(),
            },
        )
        if context.world_size > 1:
            torch.distributed.barrier()
        report = {}
        if context.is_primary:
            hardware = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted((run_dir / "hardware").glob("rank-*.json"))
            ]
            report = _consolidate(config, run_dir, hardware)
            expected_total = sum(2 * len(episode.qa or ()) for episode in episodes) + len(captions)
            actual_total = len(list((run_dir / "predictions").glob("*.json")))
            if actual_total != expected_total:
                raise RuntimeError(f"Incomplete prediction set: {actual_total}/{expected_total}")
            _atomic_json(run_dir / "summary.json", report)
            _atomic_json(config.report_path, report)
        if context.world_size > 1:
            torch.distributed.barrier()
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate vanilla Qwen2.5-Omni generation")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/qwen2_5_omni_vanilla_baseline_poc.yaml")
    )
    args = parser.parse_args()
    report = run(args.config)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
