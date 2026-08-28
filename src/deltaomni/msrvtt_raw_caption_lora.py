from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
from transformers.utils import logging as transformers_logging

from deltaomni.audiocaps_caption_lora import (
    _append_jsonl,
    _atomic_torch_save,
    _broadcast_string,
    _caption_metrics,
    _gather_rng_states,
    _prune_checkpoints,
    _restore_rng,
)
from deltaomni.data.canonicalize import read_canonical_dataset
from deltaomni.data.schema import CanonicalEpisode
from deltaomni.distributed import distributed_context, reduce_sums
from deltaomni.omni_backbones import load_omni_backbone_config
from deltaomni.omni_vanilla_baseline import _sample_video
from deltaomni.run_integrity import (
    git_revision,
    git_worktree_is_clean,
    resolved_input_signature,
    sha256_file,
)
from deltaomni.train_sanity import _atomic_json, _set_seed


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    backend: str
    nccl_compatibility_mode: bool
    precision: str
    cpu_threads: int
    gradient_accumulation_steps: int
    frame_cache_entries: int


@dataclass(frozen=True)
class TrainingConfig:
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules_regex: str
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    checkpoint_interval_steps: int
    keep_last_checkpoints: int
    gradient_clip_norm: float
    resume: str
    caption_sampling: str = "random"


@dataclass(frozen=True)
class EvaluationConfig:
    examples: int
    max_new_tokens: int
    split: str = "validation"


@dataclass(frozen=True)
class RawCaptionConfig:
    seed: int
    canonical_manifest: Path
    omni_config: Path
    train_count: int
    validation_count: int
    sample_fps: float
    frame_width: int
    frame_height: int
    max_target_tokens: int
    system_prompt: str
    user_prompt: str
    runtime: RuntimeConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    output_root: Path
    log_root: Path
    report_path: Path


def load_config(path: Path) -> RawCaptionConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    config = RawCaptionConfig(
        seed=int(raw["seed"]),
        canonical_manifest=resolve(raw["canonical_manifest"]),
        omni_config=resolve(raw["omni_config"]),
        train_count=int(raw["train_count"]),
        validation_count=int(raw["validation_count"]),
        sample_fps=float(raw["sample_fps"]),
        frame_width=int(raw["frame_width"]),
        frame_height=int(raw["frame_height"]),
        max_target_tokens=int(raw["max_target_tokens"]),
        system_prompt=str(raw["system_prompt"]),
        user_prompt=str(raw["user_prompt"]),
        runtime=RuntimeConfig(**raw["runtime"]),
        training=TrainingConfig(**raw["training"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        output_root=resolve(raw["output_root"]),
        log_root=resolve(raw["log_root"]),
        report_path=resolve(raw["report_path"]),
    )
    if config.runtime.precision != "bfloat16" or config.training.resume not in {"auto", "never"}:
        raise ValueError("Raw caption LoRA requires bfloat16 and a valid resume mode")
    if config.training.caption_sampling not in {"random", "first"}:
        raise ValueError("Raw caption sampling must be random or first")
    if config.evaluation.split not in {"train", "validation"}:
        raise ValueError("Raw caption evaluation split must be train or validation")
    positive = (
        config.train_count,
        config.validation_count,
        config.sample_fps,
        config.frame_width,
        config.frame_height,
        config.max_target_tokens,
        config.runtime.cpu_threads,
        config.runtime.gradient_accumulation_steps,
        config.runtime.frame_cache_entries,
        config.training.lora_rank,
        config.training.lora_alpha,
        config.training.learning_rate,
        config.training.warmup_steps,
        config.training.max_steps,
        config.training.checkpoint_interval_steps,
        config.training.keep_last_checkpoints,
        config.training.gradient_clip_norm,
        config.evaluation.examples,
        config.evaluation.max_new_tokens,
    )
    if min(positive) <= 0 or not 0 <= config.training.lora_dropout < 1:
        raise ValueError("Raw caption LoRA controls must be positive")
    return config


class RawVideoDataset:
    def __init__(
        self,
        episodes: list[CanonicalEpisode],
        config: RawCaptionConfig,
    ) -> None:
        self.episodes = episodes
        self.config = config
        self.cache: OrderedDict[str, list] = OrderedDict()

    def __len__(self) -> int:
        return len(self.episodes)

    def item(self, index: int, *, reference_index: int = 0) -> dict[str, Any]:
        episode = self.episodes[index]
        assert episode.media.video is not None and episode.captions.video
        references = tuple(caption.text for caption in episode.captions.video)
        return {
            "source_id": episode.source_id,
            "path": episode.media.video.path,
            "caption": references[reference_index % len(references)],
            "references": references,
        }

    def frames(self, index: int) -> list:
        item = self.item(index)
        key = str(item["path"])
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.move_to_end(key)
            return cached
        frames = _sample_video(
            item["path"],
            self.config.sample_fps,
            (self.config.frame_width, self.config.frame_height),
        )
        self.cache[key] = frames
        self.cache.move_to_end(key)
        while len(self.cache) > self.config.runtime.frame_cache_entries:
            self.cache.popitem(last=False)
        return frames


def _select(config: RawCaptionConfig) -> tuple[RawVideoDataset, RawVideoDataset]:
    data = read_canonical_dataset(config.canonical_manifest)

    def choose(split: str, count: int) -> list[CanonicalEpisode]:
        values = sorted(
            data[split],
            key=lambda episode: hashlib.sha256(
                f"{config.seed}:{episode.source_id}".encode()
            ).hexdigest(),
        )
        if len(values) < count:
            raise ValueError(f"Found {len(values)}/{count} raw caption {split} videos")
        return values[:count]

    return (
        RawVideoDataset(choose("train", config.train_count), config),
        RawVideoDataset(choose("validation", config.validation_count), config),
    )


class RawCaptionModel(nn.Module):
    def __init__(self, thinker, processor, config: RawCaptionConfig) -> None:
        super().__init__()
        self.thinker = thinker
        self.processor = processor
        self.config = config
        self.end_token_id = int(processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        self.prompt = self._prompt()

    def _prompt(self) -> str:
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.config.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "local-media"},
                    {"type": "text", "text": self.config.user_prompt},
                ],
            },
        ]
        return self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

    def _inputs(self, frames: list, device: torch.device) -> dict[str, Tensor]:
        omni = load_omni_backbone_config(self.config.omni_config)
        values = self.processor(
            text=self.prompt,
            videos=[frames],
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
            videos_kwargs={
                "fps": self.config.sample_fps,
                "min_pixels": omni.video.min_pixels,
                "max_pixels": omni.video.max_pixels,
                "seconds_per_chunk": omni.seconds_per_chunk,
                "position_id_per_seconds": omni.position_id_per_seconds,
            },
        )
        return {key: value.to(device) for key, value in values.items() if torch.is_tensor(value)}

    def forward(self, frames: list, caption: str) -> tuple[Tensor, Tensor]:
        device = next(self.thinker.parameters()).device
        inputs = self._inputs(frames, device)
        target_ids = self.processor.tokenizer(caption, add_special_tokens=False)["input_ids"]
        target_ids = target_ids[: self.config.max_target_tokens - 1] + [self.end_token_id]
        target = torch.tensor(target_ids, device=device).unsqueeze(0)
        prompt_width = inputs["input_ids"].shape[1]
        inputs["input_ids"] = torch.cat((inputs["input_ids"], target), dim=1)
        inputs["attention_mask"] = torch.cat(
            (inputs["attention_mask"], torch.ones_like(target)), dim=1
        )
        labels = torch.full_like(inputs["input_ids"], -100)
        labels[:, prompt_width:] = target
        output = self.thinker(
            **inputs,
            labels=labels,
            use_cache=False,
            use_audio_in_video=False,
        )
        return output.loss, labels.ne(-100).sum()

    @torch.no_grad()
    def generate(self, frames: list, max_new_tokens: int) -> str:
        device = next(self.thinker.parameters()).device
        inputs = self._inputs(frames, device)
        output = self.thinker.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_audio_in_video=False,
        )
        generated = output[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


def _load_model(config: RawCaptionConfig, device: torch.device) -> RawCaptionModel:
    omni = load_omni_backbone_config(config.omni_config)
    processor = Qwen2_5OmniProcessor.from_pretrained(
        omni.model_id,
        revision=omni.revision,
        cache_dir=omni.cache_dir,
        local_files_only=True,
    )
    base = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        omni.model_id,
        revision=omni.revision,
        cache_dir=omni.cache_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation=omni.attention_implementation,
        local_files_only=True,
    )
    base.requires_grad_(False)
    thinker = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.training.lora_rank,
            lora_alpha=config.training.lora_alpha,
            lora_dropout=config.training.lora_dropout,
            target_modules=config.training.target_modules_regex,
            bias="none",
        ),
    ).to(device)
    return RawCaptionModel(thinker, processor, config).to(device)


@torch.no_grad()
def evaluate(
    model: RawCaptionModel,
    data: RawVideoDataset,
    config: RawCaptionConfig,
) -> dict[str, Any]:
    model.eval()
    count = min(config.evaluation.examples, len(data))
    predictions = {"full_video": [], "first_block": []}
    references = []
    examples = []
    started = time.perf_counter()
    for index in range(count):
        item = data.item(index)
        frames = data.frames(index)
        predictions["full_video"].append(
            model.generate(frames, config.evaluation.max_new_tokens)
        )
        predictions["first_block"].append(
            model.generate(frames[:2], config.evaluation.max_new_tokens)
        )
        references.append(item["references"])
        if len(examples) < 10:
            examples.append(
                {
                    "source_id": item["source_id"],
                    "references": list(item["references"]),
                    "full_video": predictions["full_video"][-1],
                    "first_block": predictions["first_block"][-1],
                }
            )
        completed = index + 1
        elapsed = time.perf_counter() - started
        eta = elapsed / completed * (count - completed)
        if completed % 10 == 0 or completed == count:
            print(
                f"raw_caption_eval={completed}/{count} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    return {
        "metrics": {
            name: _caption_metrics(values, references)
            for name, values in predictions.items()
        },
        "examples": examples,
    }


def _checkpoint(path: Path) -> dict[str, Any] | None:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except (EOFError, OSError, RuntimeError):
        return None
    required = {
        "next_step",
        "lora",
        "optimizer",
        "signature",
        "rng_states",
        "world_size",
        "code_revision",
    }
    return value if required <= value.keys() else None


def _latest_checkpoint(run_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    for path in sorted((run_dir / "checkpoints").glob("step-*.pt"), reverse=True):
        value = _checkpoint(path)
        if value is not None:
            return path, value
    return None


def run(
    config_path: Path,
    run_id_override: str | None,
    stop_after_step: int | None,
) -> dict[str, Any]:
    config = load_config(config_path)
    root = config_path.resolve().parent.parent
    if not git_worktree_is_clean(root):
        raise RuntimeError("Raw caption LoRA requires a clean source worktree")
    _set_seed(config.seed)
    torch.set_num_threads(config.runtime.cpu_threads)
    train, validation = _select(config)
    evaluation_data = train if config.evaluation.split == "train" else validation
    signature = resolved_input_signature(
        config,
        {
            "canonical_manifest": config.canonical_manifest,
            "omni_config": config.omni_config,
        },
    )
    code_revision = git_revision(root)
    transformers_logging.set_verbosity_error()
    with distributed_context(
        config.runtime.device,
        config.runtime.backend,
        nccl_compatibility_mode=config.runtime.nccl_compatibility_mode,
    ) as context:
        core = _load_model(config, context.device)
        model: nn.Module = core
        if context.world_size > 1:
            model = DistributedDataParallel(model, device_ids=[context.local_rank])
        optimizer = torch.optim.AdamW(
            [parameter for parameter in core.thinker.parameters() if parameter.requires_grad],
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        selected = run_id_override or (
            f"msrvtt-raw-caption-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        run_id = _broadcast_string(selected if context.is_primary else None, context)
        run_dir = config.output_root / run_id
        log_path = config.log_root / run_id / "metrics.jsonl"
        events_path = config.log_root / run_id / "events.jsonl"
        if context.is_primary:
            run_dir.mkdir(parents=True, exist_ok=True)
            new_run = not (run_dir / "resolved_config.json").is_file()
            if new_run:
                _atomic_json(run_dir / "resolved_config.json", asdict(config))
                _atomic_json(
                    run_dir / "metadata.json",
                    {
                        "code_revision": code_revision,
                        "world_size": context.world_size,
                        "gpu": torch.cuda.get_device_name(context.device),
                        "torch_version": torch.__version__,
                        "cuda_version": torch.version.cuda,
                        "started_at_utc": datetime.now(UTC).isoformat(),
                    },
                )
                _append_jsonl(
                    events_path,
                    {"event": "start", "at_utc": datetime.now(UTC).isoformat()},
                )
                _atomic_json(
                    run_dir / "status.json",
                    {"status": "running", "started_at_utc": datetime.now(UTC).isoformat()},
                )
        if context.world_size > 1:
            torch.distributed.barrier()
        start_step = 1
        resumed = _latest_checkpoint(run_dir) if config.training.resume == "auto" else None
        if resumed is not None:
            _, value = resumed
            if (
                value["signature"] != signature
                or value["world_size"] != context.world_size
                or value["code_revision"] != code_revision
            ):
                raise ValueError("Raw caption checkpoint is incompatible")
            set_peft_model_state_dict(core.thinker, value["lora"])
            optimizer.load_state_dict(value["optimizer"])
            _restore_rng(value["rng_states"][context.rank], context.device)
            start_step = int(value["next_step"])
            if context.is_primary:
                _append_jsonl(
                    events_path,
                    {
                        "event": "resume",
                        "next_step": start_step,
                        "at_utc": datetime.now(UTC).isoformat(),
                    },
                )
                _atomic_json(
                    run_dir / "status.json",
                    {
                        "status": "running",
                        "resumed_at_utc": datetime.now(UTC).isoformat(),
                        "next_step": start_step,
                    },
                )
        final_step = min(config.training.max_steps, stop_after_step or config.training.max_steps)
        started = time.perf_counter()
        for step in range(start_step, final_step + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated = torch.zeros(2, device=context.device)
            for accumulation in range(config.runtime.gradient_accumulation_steps):
                generator = torch.Generator().manual_seed(
                    config.seed * 1_000_003 + step * 101 + accumulation
                )
                global_indices = torch.randint(
                    0, len(train), (context.world_size,), generator=generator
                )
                index = int(global_indices[context.rank])
                references = train.item(index)["references"]
                reference_index = (
                    int(torch.randint(0, len(references), (), generator=generator))
                    if config.training.caption_sampling == "random"
                    else 0
                )
                item = train.item(index, reference_index=reference_index)
                frames = train.frames(index)
                sync = (
                    nullcontext()
                    if accumulation == config.runtime.gradient_accumulation_steps - 1
                    or not isinstance(model, DistributedDataParallel)
                    else model.no_sync()
                )
                with sync:
                    with torch.autocast(device_type=context.device.type, dtype=torch.bfloat16):
                        loss, tokens = model(frames, item["caption"])
                        scaled = loss / config.runtime.gradient_accumulation_steps
                    scaled.backward()
                    accumulated += torch.stack((scaled.detach(), tokens.detach().float()))
                    del loss, scaled, tokens
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            warmup = min(1.0, step / config.training.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = config.training.learning_rate * warmup
            optimizer.step()
            reduced = reduce_sums({"metrics": accumulated})["metrics"] / context.world_size
            if context.is_primary:
                elapsed = time.perf_counter() - started
                eta = elapsed / (step - start_step + 1) * (final_step - step)
                record = {"step": step, "loss": float(reduced[0]), "tokens": float(reduced[1])}
                _append_jsonl(log_path, record)
                if step % 5 == 0 or step == final_step:
                    print(
                        f"raw_caption_step={step}/{final_step} loss={record['loss']:.5f} "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )
            if step % config.training.checkpoint_interval_steps == 0 or step == final_step:
                torch.cuda.empty_cache()
                states = _gather_rng_states(context)
                if context.is_primary:
                    _atomic_torch_save(
                        run_dir / "checkpoints" / f"step-{step:06d}.pt",
                        {
                            "next_step": step + 1,
                            "lora": get_peft_model_state_dict(core.thinker),
                            "optimizer": optimizer.state_dict(),
                            "signature": signature,
                            "rng_states": states,
                            "world_size": context.world_size,
                            "code_revision": code_revision,
                        },
                    )
                    _prune_checkpoints(run_dir, config.training.keep_last_checkpoints)
                    _append_jsonl(
                        events_path,
                        {
                            "event": "checkpoint",
                            "step": step,
                            "at_utc": datetime.now(UTC).isoformat(),
                        },
                    )
        if context.world_size > 1:
            torch.distributed.barrier()
        result = {}
        if final_step < config.training.max_steps:
            result = {"run_id": run_id, "status": "interrupted", "step": final_step}
            if context.is_primary:
                _atomic_json(run_dir / "status.json", result)
        elif context.is_primary:
            metrics = evaluate(core, evaluation_data, config)
            full = metrics["metrics"]["full_video"]["word_f1"]
            first = metrics["metrics"]["first_block"]["word_f1"]
            result = {
                "schema": "deltaomni.msrvtt_raw_caption_lora.v2",
                "run_id": run_id,
                "status": "complete",
                "evaluation_split": config.evaluation.split,
                "evaluation": metrics,
                "checks": {"full_video_beats_first_block": full > first},
                "passed": full > first,
                "training_seconds": time.perf_counter() - started,
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "canonical_manifest_sha256": sha256_file(config.canonical_manifest),
            }
            _atomic_json(run_dir / "summary.json", result)
            _atomic_json(
                run_dir / "status.json",
                {
                    "status": "complete",
                    "step": final_step,
                    "completed_at_utc": result["completed_at_utc"],
                },
            )
            _atomic_json(config.report_path, result)
        if context.world_size > 1:
            torch.distributed.barrier()
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune raw-video Qwen on MSR-VTT captions")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/msrvtt_raw_caption_smoke.yaml")
    )
    parser.add_argument("--run-id")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    result = run(args.config, args.run_id, args.stop_after_step)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
