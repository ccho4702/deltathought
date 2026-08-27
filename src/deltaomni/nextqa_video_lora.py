from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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

from deltaomni.audiocaps_caption_lora import (
    _append_jsonl,
    _atomic_torch_save,
    _broadcast_string,
    _gather_rng_states,
    _prune_checkpoints,
    _restore_rng,
)
from deltaomni.distributed import distributed_context, reduce_sums
from deltaomni.omni_backbones import load_omni_backbone_config
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
    per_device_batch_size: int
    gradient_accumulation_steps: int
    cache_entries: int


@dataclass(frozen=True)
class InterfaceConfig:
    delta_width: int
    hidden_width: int
    max_delta_updates: int
    max_prompt_tokens: int
    max_new_tokens: int
    system_prompt: str


@dataclass(frozen=True)
class TrainingConfig:
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules_regex: str
    lora_learning_rate: float
    interface_learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    checkpoint_interval_steps: int
    keep_last_checkpoints: int
    gradient_clip_norm: float
    control_loss_weight: float
    control_margin: float
    resume: str


@dataclass(frozen=True)
class EvaluationConfig:
    validation_examples: int
    train_examples: int
    controls: tuple[str, ...]


@dataclass(frozen=True)
class VideoQAConfig:
    seed: int
    joint_manifest: Path
    omni_config: Path
    runtime: RuntimeConfig
    interface: InterfaceConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    output_root: Path
    log_root: Path
    report_path: Path


def load_config(path: Path) -> VideoQAConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    evaluation = dict(raw["evaluation"])
    evaluation["controls"] = tuple(evaluation["controls"])
    config = VideoQAConfig(
        seed=int(raw["seed"]),
        joint_manifest=resolve(raw["joint_manifest"]),
        omni_config=resolve(raw["omni_config"]),
        runtime=RuntimeConfig(**raw["runtime"]),
        interface=InterfaceConfig(**raw["interface"]),
        training=TrainingConfig(**raw["training"]),
        evaluation=EvaluationConfig(**evaluation),
        output_root=resolve(raw["output_root"]),
        log_root=resolve(raw["log_root"]),
        report_path=resolve(raw["report_path"]),
    )
    if config.runtime.precision != "bfloat16" or config.training.resume not in {"auto", "never"}:
        raise ValueError("Video QA LoRA requires bfloat16 and a valid resume mode")
    allowed = {"normal", "delta_zero", "last_only", "reversed", "cross_source"}
    if set(config.evaluation.controls) - allowed or "normal" not in config.evaluation.controls:
        raise ValueError("Invalid video QA controls")
    positive = (
        config.runtime.cpu_threads,
        config.runtime.per_device_batch_size,
        config.runtime.gradient_accumulation_steps,
        config.runtime.cache_entries,
        config.interface.delta_width,
        config.interface.hidden_width,
        config.interface.max_delta_updates,
        config.interface.max_prompt_tokens,
        config.interface.max_new_tokens,
        config.training.lora_rank,
        config.training.lora_alpha,
        config.training.lora_learning_rate,
        config.training.interface_learning_rate,
        config.training.warmup_steps,
        config.training.max_steps,
        config.training.checkpoint_interval_steps,
        config.training.keep_last_checkpoints,
        config.training.gradient_clip_norm,
        config.training.control_loss_weight,
        config.training.control_margin,
        config.evaluation.validation_examples,
        config.evaluation.train_examples,
    )
    if min(positive) <= 0 or not 0 <= config.training.lora_dropout < 1:
        raise ValueError("Video QA LoRA controls must be positive")
    return config


class VideoQADataset:
    def __init__(self, manifest: dict[str, Any], split: str, cache_entries: int) -> None:
        self.records = list(manifest["splits"][split])
        self.examples: list[tuple[int, int]] = []
        self.cache_entries = cache_entries
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for record_index, record in enumerate(self.records):
            payload = self._load_path(str(record["cache_path"]))
            for qa_index, qa in enumerate(payload["qa"]):
                if len(qa["choices"]) == 5 and qa["answer_index"] is not None:
                    self.examples.append((record_index, qa_index))

    def __len__(self) -> int:
        return len(self.examples)

    def _load_path(self, path: str) -> dict[str, Any]:
        cached = self.cache.get(path)
        if cached is not None:
            self.cache.move_to_end(path)
            return cached
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.cache[path] = payload
        self.cache.move_to_end(path)
        while len(self.cache) > self.cache_entries:
            self.cache.popitem(last=False)
        return payload

    def item(self, index: int, *, delta_index: int | None = None) -> dict[str, Any]:
        record_index, qa_index = self.examples[index]
        delta_record_index, _ = self.examples[index if delta_index is None else delta_index]
        payload = self._load_path(str(self.records[record_index]["cache_path"]))
        delta_payload = self._load_path(str(self.records[delta_record_index]["cache_path"]))
        qa = payload["qa"][qa_index]
        video_deltas = delta_payload["video_deltas"].float()
        target_updates = int(payload["video_deltas"].shape[0])
        if video_deltas.shape[0] != target_updates:
            positions = torch.linspace(0, video_deltas.shape[0] - 1, target_updates).round().long()
            video_deltas = video_deltas[positions]
        return {
            "source_id": str(payload["source_id"]),
            "question_id": str(qa["question_id"]),
            "question_type": str(qa["question_type"]),
            "question": str(qa["question"]),
            "choices": tuple(str(value) for value in qa["choices"]),
            "answer_index": int(qa["answer_index"]),
            "video_first": payload["video_first"].float(),
            "video_deltas": video_deltas,
        }

    def batch(self, indices: Tensor) -> list[dict[str, Any]]:
        return [self.item(index) for index in indices.tolist()]

    def cross_source_donors(self, count: int) -> Tensor:
        selected = list(range(min(count, len(self))))
        donors = []
        for index in selected:
            item = self.item(index)
            distinct = [
                candidate
                for candidate in range(len(self))
                if self.item(candidate)["source_id"] != item["source_id"]
            ]
            if not distinct:
                raise ValueError("Cross-source control requires another source")
            rank = int(
                hashlib.sha256(
                    f"{item['source_id']}:{self.examples[index][1]}".encode()
                ).hexdigest()[:16],
                16,
            )
            donors.append(distinct[rank % len(distinct)])
        return torch.tensor(donors, dtype=torch.long)


class VideoDeltaAdapter(nn.Module):
    def __init__(self, config: InterfaceConfig) -> None:
        super().__init__()
        self.delta_norm = nn.LayerNorm(config.delta_width)
        self.delta_projection = nn.Linear(config.delta_width, config.hidden_width, bias=False)
        self.anchor_type = nn.Parameter(torch.zeros(config.hidden_width))
        self.delta_type = nn.Parameter(torch.zeros(config.hidden_width))
        self.delta_positions = nn.Parameter(
            torch.randn(config.max_delta_updates, config.hidden_width) * 0.02
        )

    def forward(self, first: Tensor, deltas: Tensor) -> tuple[Tensor, Tensor]:
        if first.ndim != 2 or deltas.ndim != 3 or deltas.shape[1] != 1:
            raise ValueError("Unexpected video QA prefix shape")
        if not 0 < deltas.shape[0] <= self.delta_positions.shape[0]:
            raise ValueError("Video delta horizon exceeds configured maximum")
        projected = self.delta_projection(self.delta_norm(deltas[:, 0]))
        projected = projected + self.delta_type + self.delta_positions[: deltas.shape[0]]
        return first + self.anchor_type, projected


class VideoQAModel(nn.Module):
    def __init__(self, thinker, processor, config: InterfaceConfig) -> None:
        super().__init__()
        self.thinker = thinker
        self.tokenizer = processor.tokenizer
        self.adapter = VideoDeltaAdapter(config)
        self.config = config
        self.end_token_id = int(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        self.video_start_id = int(thinker.config.vision_start_token_id)
        self.video_end_id = int(thinker.config.vision_end_token_id)

    def _prompt_ids(self, item: dict[str, Any], device: torch.device) -> Tensor:
        choices = "\n".join(f"{chr(65 + i)}. {value}" for i, value in enumerate(item["choices"]))
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.config.system_prompt},
                {
                    "role": "user",
                    "content": "Choose the best answer. Reply with one letter only.\n"
                    f"Question: {item['question']}\n{choices}",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        return torch.tensor(ids[-self.config.max_prompt_tokens :], device=device)

    def _controlled(self, deltas: Tensor, control: str) -> Tensor:
        if control == "normal" or control == "cross_source":
            return deltas
        if control == "delta_zero":
            return torch.zeros_like(deltas)
        if control == "last_only":
            value = torch.zeros_like(deltas)
            value[-1] = deltas[-1]
            return value
        if control == "reversed":
            return deltas.flip(0)
        raise ValueError(f"Unknown video QA control: {control}")

    def _prefix(self, item: dict[str, Any], control: str, device: torch.device) -> Tensor:
        first = item["video_first"].to(device)
        deltas = self._controlled(item["video_deltas"].to(device), control)
        anchors, projected = self.adapter(first, deltas)
        embeddings = self.thinker.get_input_embeddings()
        special = torch.tensor([self.video_start_id, self.video_end_id], device=device)
        return torch.cat(
            (
                embeddings(special[:1]),
                anchors.to(embeddings.weight.dtype),
                projected.to(embeddings.weight.dtype),
                embeddings(special[1:]),
            ),
            dim=0,
        )

    @staticmethod
    def _position_ids(mask: Tensor, width: int | None = None) -> Tensor:
        positions = mask.long().cumsum(-1) - 1
        positions.masked_fill_(~mask, 0)
        if width is not None:
            positions = positions[:, -width:]
        return positions.unsqueeze(0).expand(3, -1, -1)

    def forward(
        self,
        items: list[dict[str, Any]],
        control: str = "normal",
    ) -> tuple[Tensor, Tensor]:
        device = self.adapter.anchor_type.device
        embeddings = self.thinker.get_input_embeddings()
        sequences, labels = [], []
        for item in items:
            prefix = self._prefix(item, control, device)
            prompt = self._prompt_ids(item, device)
            target = torch.tensor(
                self.tokenizer(chr(65 + item["answer_index"]), add_special_tokens=False)[
                    "input_ids"
                ]
                + [self.end_token_id],
                device=device,
            )
            text = torch.cat((prompt, target))
            sequences.append(torch.cat((prefix, embeddings(text)), dim=0))
            labels.append(
                torch.cat(
                    (
                        torch.full(
                            (prefix.shape[0] + prompt.shape[0],),
                            -100,
                            device=device,
                            dtype=torch.long,
                        ),
                        target,
                    )
                )
            )
        width = max(value.shape[0] for value in sequences)
        hidden = sequences[0].shape[-1]
        inputs = torch.zeros(len(items), width, hidden, device=device, dtype=sequences[0].dtype)
        target_labels = torch.full((len(items), width), -100, device=device, dtype=torch.long)
        mask = torch.zeros(len(items), width, device=device, dtype=torch.bool)
        for index, (sequence, target) in enumerate(zip(sequences, labels, strict=True)):
            inputs[index, : sequence.shape[0]] = sequence
            target_labels[index, : target.shape[0]] = target
            mask[index, : sequence.shape[0]] = True
        output = self.thinker(
            inputs_embeds=inputs,
            attention_mask=mask,
            position_ids=self._position_ids(mask),
            labels=target_labels,
            use_cache=False,
        )
        return output.loss, target_labels.ne(-100).sum()

    @torch.no_grad()
    def generate(self, item: dict[str, Any], control: str) -> str:
        device = self.adapter.anchor_type.device
        prefix = self._prefix(item, control, device)
        prompt = self._prompt_ids(item, device)
        inputs = torch.cat((prefix, self.thinker.get_input_embeddings()(prompt)), dim=0)[None]
        mask = torch.ones(1, inputs.shape[1], dtype=torch.bool, device=device)
        output = self.thinker(
            inputs_embeds=inputs,
            attention_mask=mask,
            position_ids=self._position_ids(mask),
            use_cache=True,
        )
        next_id = output.logits[:, -1].argmax(-1)
        generated, past = [], output.past_key_values
        for step in range(self.config.max_new_tokens):
            generated.append(next_id)
            if int(next_id.item()) == self.end_token_id:
                break
            mask = torch.cat((mask, torch.ones(1, 1, dtype=torch.bool, device=device)), 1)
            output = self.thinker(
                input_ids=next_id[:, None],
                attention_mask=mask,
                position_ids=self._position_ids(mask, 1),
                past_key_values=past,
                use_cache=True,
                cache_position=torch.tensor([inputs.shape[1] + step], device=device),
            )
            next_id, past = output.logits[:, -1].argmax(-1), output.past_key_values
        return self.tokenizer.decode(torch.cat(generated), skip_special_tokens=True).strip()


def _load_model(config: VideoQAConfig, device: torch.device) -> VideoQAModel:
    omni = load_omni_backbone_config(config.omni_config)
    processor = Qwen2_5OmniProcessor.from_pretrained(
        omni.model_id, revision=omni.revision, cache_dir=omni.cache_dir, local_files_only=True
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
    return VideoQAModel(thinker, processor, config.interface).to(device)


@torch.no_grad()
def evaluate(
    model: VideoQAModel,
    data: VideoQADataset,
    config: VideoQAConfig,
    *,
    count: int,
    controls: tuple[str, ...],
) -> dict[str, Any]:
    model.eval()
    count = min(count, len(data))
    donors = data.cross_source_donors(count) if "cross_source" in controls else None
    results = {control: 0 for control in controls}
    examples = []
    started = time.perf_counter()
    for index in range(count):
        base = data.item(index)
        for control in controls:
            item = (
                data.item(index, delta_index=int(donors[index]))
                if control == "cross_source"
                else base
            )
            prediction = model.generate(item, control)
            parsed = re.search(r"(?:^|[^A-Z])([A-E])(?:[^A-Z]|$)", prediction.upper())
            predicted = None if parsed is None else ord(parsed.group(1)) - 65
            results[control] += int(predicted == base["answer_index"])
        if len(examples) < 10:
            examples.append(
                {
                    "source_id": base["source_id"],
                    "question_id": base["question_id"],
                    "target": base["answer_index"],
                }
            )
        completed = index + 1
        elapsed = time.perf_counter() - started
        eta = elapsed / completed * (count - completed)
        if completed % 10 == 0 or completed == count:
            print(
                f"video_qa_eval={completed}/{count} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    return {
        "accuracy": {name: value / count for name, value in results.items()},
        "correct": results,
        "examples": count,
        "samples": examples,
    }


def _checkpoint(path: Path) -> dict[str, Any] | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (EOFError, OSError, RuntimeError):
        return None
    required = {
        "next_step",
        "lora",
        "adapter",
        "optimizer",
        "signature",
        "rng_states",
        "world_size",
        "code_revision",
    }
    return payload if required <= payload.keys() else None


def _latest_checkpoint(run_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    for path in sorted((run_dir / "checkpoints").glob("step-*.pt"), reverse=True):
        payload = _checkpoint(path)
        if payload is not None:
            return path, payload
    return None


def run(
    config_path: Path, run_id_override: str | None, stop_after_step: int | None
) -> dict[str, Any]:
    config = load_config(config_path)
    root = config_path.resolve().parent.parent
    if not git_worktree_is_clean(root):
        raise RuntimeError("Video QA LoRA requires a clean source worktree")
    _set_seed(config.seed)
    torch.set_num_threads(config.runtime.cpu_threads)
    manifest = json.loads(config.joint_manifest.read_text())
    if manifest.get("schema") != "deltaomni.omni_nextqa_joint_manifest.v2":
        raise ValueError("Video QA LoRA requires a signed v2 joint manifest")
    signature = resolved_input_signature(
        config, {"joint_manifest": config.joint_manifest, "omni_config": config.omni_config}
    )
    code_revision = git_revision(root)
    with distributed_context(
        config.runtime.device,
        config.runtime.backend,
        nccl_compatibility_mode=config.runtime.nccl_compatibility_mode,
    ) as context:
        train = VideoQADataset(manifest, "train", config.runtime.cache_entries)
        validation = VideoQADataset(manifest, "validation", config.runtime.cache_entries)
        core = _load_model(config, context.device)
        model: nn.Module = core
        if context.world_size > 1:
            model = DistributedDataParallel(model, device_ids=[context.local_rank])
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [p for p in core.thinker.parameters() if p.requires_grad],
                    "lr": config.training.lora_learning_rate,
                },
                {
                    "params": core.adapter.parameters(),
                    "lr": config.training.interface_learning_rate,
                },
            ],
            weight_decay=config.training.weight_decay,
        )
        selected = (
            run_id_override
            or "nextqa-video-lora-"
            f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        )
        run_id = _broadcast_string(selected if context.is_primary else None, context)
        run_dir = config.output_root / run_id
        log_path = config.log_root / run_id / "metrics.jsonl"
        if context.is_primary:
            run_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(run_dir / "resolved_config.json", asdict(config))
        if context.world_size > 1:
            torch.distributed.barrier()
        start_step = 1
        resumed = _latest_checkpoint(run_dir) if config.training.resume == "auto" else None
        if resumed:
            _, payload = resumed
            if (
                payload["signature"] != signature
                or payload["world_size"] != context.world_size
                or payload["code_revision"] != code_revision
            ):
                raise ValueError("Video QA checkpoint is incompatible")
            set_peft_model_state_dict(core.thinker, payload["lora"])
            core.adapter.load_state_dict(payload["adapter"])
            optimizer.load_state_dict(payload["optimizer"])
            _restore_rng(payload["rng_states"][context.rank], context.device)
            start_step = int(payload["next_step"])
        final_step = min(config.training.max_steps, stop_after_step or config.training.max_steps)
        started = time.perf_counter()
        for step in range(start_step, final_step + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated = torch.zeros(4, device=context.device)
            for accumulation in range(config.runtime.gradient_accumulation_steps):
                generator = torch.Generator().manual_seed(
                    config.seed * 1_000_003 + step * 101 + accumulation
                )
                global_batch = config.runtime.per_device_batch_size * context.world_size
                indices = torch.randint(0, len(train), (global_batch,), generator=generator)
                local = indices.reshape(context.world_size, -1)[context.rank]
                items = train.batch(local)
                sync = (
                    nullcontext()
                    if accumulation == config.runtime.gradient_accumulation_steps - 1
                    or not isinstance(model, DistributedDataParallel)
                    else model.no_sync()
                )
                with sync:
                    with torch.autocast(device_type=context.device.type, dtype=torch.bfloat16):
                        normal_loss, tokens = model(items, "normal")
                        zero_loss, _ = model(items, "delta_zero")
                        ranking = torch.relu(
                            config.training.control_margin + normal_loss - zero_loss
                        )
                        loss = normal_loss + config.training.control_loss_weight * ranking
                        scaled = loss / config.runtime.gradient_accumulation_steps
                    scaled.backward()
                    accumulated += torch.stack(
                        (
                            scaled.detach(),
                            tokens.detach().float(),
                            normal_loss.detach() / config.runtime.gradient_accumulation_steps,
                            zero_loss.detach() / config.runtime.gradient_accumulation_steps,
                        )
                    )
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            warmup = min(1.0, step / config.training.warmup_steps)
            optimizer.param_groups[0]["lr"] = config.training.lora_learning_rate * warmup
            optimizer.param_groups[1]["lr"] = config.training.interface_learning_rate * warmup
            optimizer.step()
            reduced = reduce_sums({"v": accumulated})["v"] / context.world_size
            if context.is_primary:
                elapsed = time.perf_counter() - started
                eta = elapsed / (step - start_step + 1) * (final_step - step)
                record = {
                    "step": step,
                    "loss": float(reduced[0]),
                    "tokens": float(reduced[1]),
                    "normal_loss": float(reduced[2]),
                    "zero_loss": float(reduced[3]),
                }
                _append_jsonl(log_path, record)
                if step % 5 == 0 or step == final_step:
                    print(
                        f"video_qa_step={step}/{final_step} loss={record['loss']:.5f} "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )
            if step % config.training.checkpoint_interval_steps == 0 or step == final_step:
                states = _gather_rng_states(context)
                if context.is_primary:
                    _atomic_torch_save(
                        run_dir / "checkpoints" / f"step-{step:06d}.pt",
                        {
                            "next_step": step + 1,
                            "lora": get_peft_model_state_dict(core.thinker),
                            "adapter": core.adapter.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "signature": signature,
                            "rng_states": states,
                            "world_size": context.world_size,
                            "code_revision": code_revision,
                        },
                    )
                    _prune_checkpoints(run_dir, config.training.keep_last_checkpoints)
        if context.world_size > 1:
            torch.distributed.barrier()
        result = {}
        if final_step < config.training.max_steps:
            result = {"run_id": run_id, "status": "interrupted", "step": final_step}
        elif context.is_primary:
            train_metrics = evaluate(
                core, train, config, count=config.evaluation.train_examples, controls=("normal",)
            )
            validation_metrics = evaluate(
                core,
                validation,
                config,
                count=config.evaluation.validation_examples,
                controls=config.evaluation.controls,
            )
            accuracy = validation_metrics["accuracy"]
            checks = {
                "train_overfit": train_metrics["accuracy"]["normal"] >= 0.7,
                "normal_beats_zero": accuracy["normal"] > accuracy.get("delta_zero", -1),
                "normal_beats_cross_source": accuracy["normal"] > accuracy.get("cross_source", -1),
            }
            result = {
                "schema": "deltaomni.nextqa_video_lora.v1",
                "run_id": run_id,
                "status": "complete",
                "train": train_metrics,
                "validation": validation_metrics,
                "checks": checks,
                "passed": all(checks.values()),
                "training_seconds": time.perf_counter() - started,
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "joint_manifest_sha256": sha256_file(config.joint_manifest),
            }
            _atomic_json(run_dir / "summary.json", result)
            _atomic_json(config.report_path, result)
        if context.world_size > 1:
            torch.distributed.barrier()
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Qwen Thinker LoRA on variable video deltas")
    parser.add_argument("--config", type=Path, default=Path("configs/nextqa_video_lora_smoke.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    result = run(args.config, args.run_id, args.stop_after_step)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
