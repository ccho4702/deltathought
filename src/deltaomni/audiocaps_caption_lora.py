from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
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
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from transformers.utils import logging as transformers_logging

from deltaomni.distributed import distributed_context, reduce_sums, unwrap
from deltaomni.omni_backbones import load_omni_backbone_config
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
class LoRAConfig:
    rank: int
    alpha: int
    dropout: float
    target_modules_regex: str


@dataclass(frozen=True)
class InterfaceConfig:
    delta_width: int
    hidden_width: int
    delta_updates: int
    max_target_tokens: int
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True)
class TrainingConfig:
    lora_learning_rate: float
    interface_learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    checkpoint_interval_steps: int
    keep_last_checkpoints: int
    gradient_clip_norm: float
    resume: str


@dataclass(frozen=True)
class EvaluationConfig:
    nll_batch_size: int
    nll_examples: int
    generation_examples: int
    max_new_tokens: int
    minimum_control_gap: float


@dataclass(frozen=True)
class CaptionConfig:
    seed: int
    prefix_manifest: Path
    omni_config: Path
    runtime: RuntimeConfig
    lora: LoRAConfig
    interface: InterfaceConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    output_root: Path
    log_root: Path


def load_config(path: Path) -> CaptionConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    config = CaptionConfig(
        seed=int(raw["seed"]),
        prefix_manifest=resolve(raw["prefix_manifest"]),
        omni_config=resolve(raw["omni_config"]),
        runtime=RuntimeConfig(**raw["runtime"]),
        lora=LoRAConfig(**raw["lora"]),
        interface=InterfaceConfig(**raw["interface"]),
        training=TrainingConfig(**raw["training"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        output_root=resolve(raw["output_root"]),
        log_root=resolve(raw["log_root"]),
    )
    if config.runtime.precision != "bfloat16":
        raise ValueError("AudioCaps Caption LoRA requires bfloat16")
    if config.training.resume not in {"auto", "never"}:
        raise ValueError("Invalid Caption LoRA resume mode")
    if re.fullmatch(config.lora.target_modules_regex, "model.layers.0.self_attn.q_proj") is None:
        raise ValueError("Caption LoRA regex does not target Thinker text attention")
    if re.fullmatch(config.lora.target_modules_regex, "audio_tower.layers.0.self_attn.q_proj"):
        raise ValueError("Caption LoRA must not target the audio tower")
    positive = (
        config.runtime.cpu_threads,
        config.runtime.per_device_batch_size,
        config.runtime.gradient_accumulation_steps,
        config.runtime.cache_entries,
        config.lora.rank,
        config.lora.alpha,
        config.interface.delta_width,
        config.interface.hidden_width,
        config.interface.delta_updates,
        config.interface.max_target_tokens,
        config.training.lora_learning_rate,
        config.training.interface_learning_rate,
        config.training.warmup_steps,
        config.training.max_steps,
        config.training.checkpoint_interval_steps,
        config.training.keep_last_checkpoints,
        config.training.gradient_clip_norm,
        config.evaluation.nll_batch_size,
        config.evaluation.nll_examples,
        config.evaluation.generation_examples,
        config.evaluation.max_new_tokens,
        config.evaluation.minimum_control_gap,
    )
    if min(positive) <= 0 or not 0 <= config.lora.dropout < 1:
        raise ValueError("Caption LoRA controls must be positive")
    return config


class PrefixDataset:
    def __init__(self, manifest: dict[str, Any], split: str, cache_entries: int) -> None:
        self.records = list(manifest["splits"][split])
        self.cache_entries = cache_entries
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def load(self, index: int) -> dict[str, Any]:
        path = self.records[index]["cache_path"]
        cached = self.cache.get(path)
        if cached is not None:
            self.cache.move_to_end(path)
            return cached
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("source_id") != self.records[index]["source_id"]:
            raise ValueError(f"Prefix cache source mismatch: {path}")
        self.cache[path] = payload
        self.cache.move_to_end(path)
        while len(self.cache) > self.cache_entries:
            self.cache.popitem(last=False)
        return payload

    def batch(
        self,
        indices: Tensor,
        *,
        delta_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, list[str], list[tuple[str, ...]], list[str]]:
        selected = indices.cpu().tolist()
        delta_selected = selected if delta_indices is None else delta_indices.cpu().tolist()
        payloads = [self.load(index) for index in selected]
        delta_payloads = [self.load(index) for index in delta_selected]
        return (
            torch.stack([payload["first_full"].float() for payload in payloads]),
            torch.stack([payload["deltas"].float() for payload in delta_payloads]),
            [payload["captions"][0] for payload in payloads],
            [tuple(payload["captions"]) for payload in payloads],
            [str(payload["source_id"]) for payload in payloads],
        )


class DeltaPrefixAdapter(nn.Module):
    def __init__(self, config: InterfaceConfig) -> None:
        super().__init__()
        self.delta_norm = nn.LayerNorm(config.delta_width)
        self.delta_projection = nn.Linear(config.delta_width, config.hidden_width, bias=False)
        self.anchor_type = nn.Parameter(torch.zeros(config.hidden_width))
        self.delta_type = nn.Parameter(torch.zeros(config.hidden_width))
        self.delta_positions = nn.Parameter(
            torch.randn(config.delta_updates, config.hidden_width) * 0.02
        )

    def forward(self, first_full: Tensor, deltas: Tensor) -> tuple[Tensor, Tensor]:
        if deltas.shape[1] != self.delta_positions.shape[0] or deltas.shape[2] != 1:
            raise ValueError(f"Unexpected delta prefix shape: {tuple(deltas.shape)}")
        anchors = first_full + self.anchor_type
        squeezed = deltas[:, :, 0]
        projected = self.delta_projection(self.delta_norm(squeezed))
        projected = projected + self.delta_type + self.delta_positions
        return anchors, projected


class CaptionInterface:
    def __init__(self, processor: Qwen2_5OmniProcessor, config: InterfaceConfig) -> None:
        self.tokenizer = processor.tokenizer
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": config.user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        self.prompt_ids = tuple(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        self.max_target_tokens = config.max_target_tokens
        self.end_token_id = int(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        self.pad_token_id = int(self.tokenizer.pad_token_id)

    def targets(self, captions: list[str], device: torch.device) -> tuple[Tensor, Tensor]:
        values = []
        for caption in captions:
            ids = self.tokenizer(caption, add_special_tokens=False)["input_ids"]
            values.append(ids[: self.max_target_tokens - 1] + [self.end_token_id])
        width = max(len(ids) for ids in values)
        target_ids = torch.full(
            (len(values), width),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        target_mask = torch.zeros((len(values), width), dtype=torch.bool, device=device)
        for index, ids in enumerate(values):
            target_ids[index, : len(ids)] = torch.tensor(ids, device=device)
            target_mask[index, : len(ids)] = True
        return target_ids, target_mask

    def prompt(self, batch: int, device: torch.device) -> Tensor:
        return torch.tensor(self.prompt_ids, device=device).unsqueeze(0).expand(batch, -1)

    def decode(self, token_ids: Tensor) -> list[str]:
        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


class AudioCaptionModel(nn.Module):
    def __init__(
        self,
        thinker: nn.Module,
        adapter: DeltaPrefixAdapter,
        interface: CaptionInterface,
        audio_start_token_id: int,
        audio_end_token_id: int,
    ) -> None:
        super().__init__()
        self.thinker = thinker
        self.adapter = adapter
        self.interface = interface
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id

    def _prefix(
        self,
        first_full: Tensor,
        deltas: Tensor,
    ) -> Tensor:
        anchors, projected = self.adapter(first_full, deltas)
        embeddings = self.thinker.get_input_embeddings()
        batch = first_full.shape[0]
        start_ids = torch.full(
            (batch, 1), self.audio_start_token_id, dtype=torch.long, device=first_full.device
        )
        end_ids = torch.full(
            (batch, 1), self.audio_end_token_id, dtype=torch.long, device=first_full.device
        )
        dtype = embeddings.weight.dtype
        return torch.cat(
            (
                embeddings(start_ids),
                anchors.to(dtype),
                projected.to(dtype),
                embeddings(end_ids),
            ),
            dim=1,
        )

    @staticmethod
    def _position_ids(attention_mask: Tensor, width: int | None = None) -> Tensor:
        positions = attention_mask.long().cumsum(dim=-1) - 1
        positions.masked_fill_(attention_mask == 0, 0)
        if width is not None:
            positions = positions[:, -width:]
        return positions.unsqueeze(0).expand(3, -1, -1)

    def forward(
        self,
        first_full: Tensor,
        deltas: Tensor,
        target_ids: Tensor,
        target_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        prefix = self._prefix(first_full, deltas)
        prompt_ids = self.interface.prompt(first_full.shape[0], first_full.device)
        text_ids = torch.cat((prompt_ids, target_ids), dim=1)
        text_embeds = self.thinker.get_input_embeddings()(text_ids)
        inputs_embeds = torch.cat((prefix, text_embeds), dim=1)
        prefix_prompt = prefix.shape[1] + prompt_ids.shape[1]
        attention_mask = torch.cat(
            (
                torch.ones(
                    first_full.shape[0],
                    prefix_prompt,
                    dtype=torch.bool,
                    device=first_full.device,
                ),
                target_mask,
            ),
            dim=1,
        )
        labels = torch.full(
            attention_mask.shape,
            -100,
            dtype=torch.long,
            device=first_full.device,
        )
        labels[:, prefix_prompt:] = target_ids.masked_fill(~target_mask, -100)
        output = self.thinker(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=self._position_ids(attention_mask),
            labels=labels,
            use_cache=False,
        )
        return output.loss, labels.ne(-100).sum()

    @torch.no_grad()
    def generate(
        self,
        first_full: Tensor,
        deltas: Tensor,
        max_new_tokens: int,
    ) -> Tensor:
        prefix = self._prefix(first_full, deltas)
        prompt_ids = self.interface.prompt(first_full.shape[0], first_full.device)
        prompt_embeds = self.thinker.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat((prefix, prompt_embeds), dim=1)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.bool, device=first_full.device
        )
        output = self.thinker(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=self._position_ids(attention_mask),
            use_cache=True,
        )
        next_ids = output.logits[:, -1].argmax(dim=-1)
        generated = []
        finished = torch.zeros(first_full.shape[0], dtype=torch.bool, device=first_full.device)
        past = output.past_key_values
        for step in range(max_new_tokens):
            generated.append(next_ids)
            finished |= next_ids.eq(self.interface.end_token_id)
            if bool(finished.all()):
                break
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(first_full.shape[0], 1, dtype=torch.bool, device=first_full.device),
                ),
                dim=1,
            )
            output = self.thinker(
                input_ids=next_ids.unsqueeze(1),
                attention_mask=attention_mask,
                position_ids=self._position_ids(attention_mask, width=1),
                past_key_values=past,
                use_cache=True,
                cache_position=torch.tensor(
                    [inputs_embeds.shape[1] + step], device=first_full.device
                ),
            )
            past = output.past_key_values
            next_ids = output.logits[:, -1].argmax(dim=-1)
            next_ids = torch.where(
                finished,
                torch.full_like(next_ids, self.interface.end_token_id),
                next_ids,
            )
        return torch.stack(generated, dim=1)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _word_f1(prediction: str, reference: str) -> float:
    predicted = _tokens(prediction)
    target = _tokens(reference)
    if not predicted or not target:
        return float(predicted == target)
    remaining = list(target)
    matched = 0
    for token in predicted:
        if token in remaining:
            remaining.remove(token)
            matched += 1
    precision = matched / len(predicted)
    recall = matched / len(target)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _rouge_l(prediction: str, reference: str) -> float:
    left, right = _tokens(prediction), _tokens(reference)
    if not left or not right:
        return float(left == right)
    row = [0] * (len(right) + 1)
    for left_token in left:
        previous = 0
        for index, right_token in enumerate(right, start=1):
            saved = row[index]
            row[index] = (
                previous + 1 if left_token == right_token else max(row[index], row[index - 1])
            )
            previous = saved
    lcs = row[-1]
    precision, recall = lcs / len(left), lcs / len(right)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _caption_metrics(predictions: list[str], references: list[tuple[str, ...]]) -> dict[str, float]:
    exact = []
    f1 = []
    rouge = []
    for prediction, candidates in zip(predictions, references, strict=True):
        normalized = " ".join(_tokens(prediction))
        exact.append(float(any(normalized == " ".join(_tokens(ref)) for ref in candidates)))
        f1.append(max(_word_f1(prediction, ref) for ref in candidates))
        rouge.append(max(_rouge_l(prediction, ref) for ref in candidates))
    return {
        "exact_match": sum(exact) / len(exact),
        "word_f1": sum(f1) / len(f1),
        "rouge_l": sum(rouge) / len(rouge),
        "examples": float(len(predictions)),
    }


def _load_model(config: CaptionConfig, device: torch.device):
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
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules_regex,
            bias="none",
        ),
    ).to(device)
    interface = CaptionInterface(processor, config.interface)
    model = AudioCaptionModel(
        thinker,
        DeltaPrefixAdapter(config.interface).to(device),
        interface,
        audio_start_token_id=int(base.config.audio_start_token_id),
        audio_end_token_id=int(base.config.audio_end_token_id),
    ).to(device)
    return model, processor


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
    }


def _gather_rng_states(context) -> list[dict[str, Any]] | None:
    local = _rng_state(context.device)
    if context.world_size == 1:
        return [local]
    gathered = [None] * context.world_size if context.is_primary else None
    torch.distributed.gather_object(local, gathered, dst=0)
    return gathered


def _restore_rng(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    torch.random.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], device)


def _checkpoint(path: Path) -> dict[str, Any] | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (EOFError, OSError, RuntimeError):
        return None
    required = {"next_step", "lora", "adapter", "optimizer", "signature", "rng_states"}
    return payload if required <= payload.keys() else None


def _latest_checkpoint(run_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    for path in sorted((run_dir / "checkpoints").glob("step-*.pt"), reverse=True):
        payload = _checkpoint(path)
        if payload is not None:
            return path, payload
    return None


def _prune_checkpoints(run_dir: Path, keep: int) -> None:
    paths = sorted((run_dir / "checkpoints").glob("step-*.pt"))
    for path in paths[:-keep]:
        path.unlink()


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _broadcast_string(value: str | None, context) -> str:
    values = [value]
    if context.world_size > 1:
        torch.distributed.broadcast_object_list(values, src=0)
    if values[0] is None:
        raise RuntimeError("Primary rank did not select a caption run ID")
    return str(values[0])


def _select_run_id(config: CaptionConfig, signature: str) -> str:
    active_path = config.output_root / "active_run.json"
    if config.training.resume == "auto" and active_path.is_file():
        active = json.loads(active_path.read_text(encoding="utf-8"))
        if active.get("signature") == signature and active.get("status") != "complete":
            return str(active["run_id"])
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"audiocaps-caption-{timestamp}-{uuid.uuid4().hex[:8]}"


@torch.no_grad()
def evaluate(
    model: AudioCaptionModel,
    data: PrefixDataset,
    config: CaptionConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    nll = {"normal": 0.0, "zero": 0.0, "shuffled": 0.0}
    tokens = 0
    nll_count = min(config.evaluation.nll_examples, len(data))
    evaluated = 0
    nll_started = time.perf_counter()
    for start in range(0, nll_count, config.evaluation.nll_batch_size):
        indices = torch.arange(start, min(start + config.evaluation.nll_batch_size, nll_count))
        donors = (indices + 1) % len(data)
        first, deltas, captions, _, _ = data.batch(indices)
        _, shuffled, _, _, _ = data.batch(indices, delta_indices=donors)
        first, deltas, shuffled = first.to(device), deltas.to(device), shuffled.to(device)
        target_ids, target_mask = model.interface.targets(captions, device)
        for name, control in (
            ("normal", deltas),
            ("zero", torch.zeros_like(deltas)),
            ("shuffled", shuffled),
        ):
            loss, count = model(first, control, target_ids, target_mask)
            nll[name] += float(loss) * int(count)
        tokens += int(target_mask.sum())
        evaluated += len(indices)
        if evaluated % 50 == 0 or evaluated == nll_count:
            elapsed = time.perf_counter() - nll_started
            eta = elapsed / evaluated * (nll_count - evaluated)
            print(
                f"caption_eval_nll={evaluated}/{nll_count} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    generation_count = min(config.evaluation.generation_examples, len(data))
    generated: dict[str, list[str]] = {"normal": [], "zero": [], "shuffled": []}
    references = []
    sources = []
    generation_started = time.perf_counter()
    for index in range(generation_count):
        donor = (index + 1) % len(data)
        indices = torch.tensor([index])
        first, deltas, _, refs, source = data.batch(indices)
        _, shuffled, _, _, _ = data.batch(indices, delta_indices=torch.tensor([donor]))
        first, deltas, shuffled = first.to(device), deltas.to(device), shuffled.to(device)
        for name, control in (
            ("normal", deltas),
            ("zero", torch.zeros_like(deltas)),
            ("shuffled", shuffled),
        ):
            ids = model.generate(first, control, config.evaluation.max_new_tokens)
            generated[name].extend(model.interface.decode(ids))
        references.extend(refs)
        sources.extend(source)
        completed = index + 1
        elapsed = time.perf_counter() - generation_started
        eta = elapsed / completed * (generation_count - completed)
        print(
            f"caption_eval_generate={completed}/{generation_count} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )
    metrics = {name: _caption_metrics(values, references) for name, values in generated.items()}
    return {
        "nll": {name: value / tokens for name, value in nll.items()},
        "generation": metrics,
        "examples": [
            {
                "source_id": sources[index],
                "references": list(references[index]),
                **{name: values[index] for name, values in generated.items()},
            }
            for index in range(min(10, generation_count))
        ],
    }


def run(
    config_path: Path,
    run_id_override: str | None,
    stop_after_step: int | None,
) -> dict[str, Any]:
    config = load_config(config_path)
    transformers_logging.set_verbosity_error()
    torch.set_num_threads(config.runtime.cpu_threads)
    _set_seed(config.seed)
    manifest = json.loads(config.prefix_manifest.read_text(encoding="utf-8"))
    train = PrefixDataset(manifest, "train", config.runtime.cache_entries)
    validation = PrefixDataset(manifest, "validation", config.runtime.cache_entries)
    with distributed_context(
        config.runtime.device,
        config.runtime.backend,
        nccl_compatibility_mode=config.runtime.nccl_compatibility_mode,
    ) as context:
        combined, _ = _load_model(config, context.device)
        model: nn.Module = combined
        if context.world_size > 1:
            model = DistributedDataParallel(model, device_ids=[context.local_rank])
        core: AudioCaptionModel = unwrap(model)
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
        signature = json.dumps(asdict(config), sort_keys=True, default=str)
        selected = run_id_override
        if context.is_primary and selected is None:
            selected = _select_run_id(config, signature)
        run_id = _broadcast_string(selected, context)
        run_dir = config.output_root / run_id
        log_path = config.log_root / run_id / "metrics.jsonl"
        code_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config_path.resolve().parent.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if context.is_primary:
            run_dir.mkdir(parents=True, exist_ok=True)
            if not (run_dir / "resolved_config.json").is_file():
                _atomic_json(run_dir / "resolved_config.json", asdict(config))
            if not (run_dir / "metadata.json").is_file():
                _atomic_json(
                    run_dir / "metadata.json",
                    {
                        "code_revision": code_revision,
                        "prefix_manifest": str(config.prefix_manifest),
                        "prefix_manifest_sha256": _sha256(config.prefix_manifest),
                        "world_size": context.world_size,
                        "gpu": torch.cuda.get_device_name(context.device),
                        "torch_version": torch.__version__,
                        "cuda_version": torch.version.cuda,
                        "started_at_utc": datetime.now(UTC).isoformat(),
                    },
                )
            _atomic_json(
                config.output_root / "active_run.json",
                {"run_id": run_id, "status": "running", "signature": signature},
            )
        if context.world_size > 1:
            torch.distributed.barrier()
        start_step = 1
        initial = None
        resumed = _latest_checkpoint(run_dir) if config.training.resume == "auto" else None
        if resumed is not None:
            checkpoint_path, payload = resumed
            if payload["signature"] != signature or payload["world_size"] != context.world_size:
                raise ValueError("Caption LoRA checkpoint configuration mismatch")
            if payload.get("code_revision") != code_revision:
                raise ValueError("Exact Caption LoRA resume requires original code revision")
            set_peft_model_state_dict(core.thinker, payload["lora"])
            core.adapter.load_state_dict(payload["adapter"])
            optimizer.load_state_dict(payload["optimizer"])
            _restore_rng(payload["rng_states"][context.rank], context.device)
            start_step = int(payload["next_step"])
            initial = payload.get("initial")
            if context.is_primary:
                print(f"resume={checkpoint_path} next_step={start_step}", flush=True)
        if initial is None:
            if context.is_primary:
                initial = evaluate(core, validation, config, context.device)
            if context.world_size > 1:
                values = [initial]
                torch.distributed.broadcast_object_list(values, src=0)
                initial = values[0]
        final_step = min(config.training.max_steps, stop_after_step or config.training.max_steps)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(context.device)
        for step in range(start_step, final_step + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated = torch.zeros(2, device=context.device)
            for accumulation in range(config.runtime.gradient_accumulation_steps):
                generator = torch.Generator().manual_seed(
                    config.seed * 1_000_003 + step * 101 + accumulation
                )
                global_batch = config.runtime.per_device_batch_size * context.world_size
                indices = torch.randint(0, len(train), (global_batch,), generator=generator)
                local = indices.reshape(context.world_size, -1)[context.rank]
                first, deltas, captions, _, _ = train.batch(local)
                first, deltas = first.to(context.device), deltas.to(context.device)
                target_ids, target_mask = core.interface.targets(captions, context.device)
                synchronize = accumulation == config.runtime.gradient_accumulation_steps - 1
                sync = (
                    nullcontext()
                    if synchronize or not isinstance(model, DistributedDataParallel)
                    else model.no_sync()
                )
                with sync:
                    with torch.autocast(device_type=context.device.type, dtype=torch.bfloat16):
                        loss, token_count = model(first, deltas, target_ids, target_mask)
                        scaled = loss / config.runtime.gradient_accumulation_steps
                    scaled.backward()
                accumulated += torch.stack((scaled.detach(), token_count.detach().float()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            warmup = min(1.0, step / config.training.warmup_steps)
            optimizer.param_groups[0]["lr"] = config.training.lora_learning_rate * warmup
            optimizer.param_groups[1]["lr"] = config.training.interface_learning_rate * warmup
            optimizer.step()
            reduced = reduce_sums({"values": accumulated})["values"] / context.world_size
            if context.is_primary:
                elapsed = time.perf_counter() - started
                completed = step - start_step + 1
                eta = elapsed / completed * (final_step - step)
                record = {"step": step, "loss": float(reduced[0]), "tokens": float(reduced[1])}
                _append_jsonl(log_path, record)
                if step % 10 == 0 or step == final_step:
                    peak = torch.cuda.max_memory_reserved(context.device) / 2**30
                    print(
                        f"caption_step={step}/{final_step} loss={record['loss']:.5f} "
                        f"peak_gib={peak:.2f} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )
            checkpoint_due = (
                step % config.training.checkpoint_interval_steps == 0 or step == final_step
            )
            if checkpoint_due:
                rng_states = _gather_rng_states(context)
                if context.is_primary:
                    _atomic_torch_save(
                        run_dir / "checkpoints" / f"step-{step:06d}.pt",
                        {
                            "next_step": step + 1,
                            "lora": get_peft_model_state_dict(core.thinker),
                            "adapter": core.adapter.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "signature": signature,
                            "rng_states": rng_states,
                            "world_size": context.world_size,
                            "code_revision": code_revision,
                            "initial": initial,
                        },
                    )
                    _prune_checkpoints(run_dir, config.training.keep_last_checkpoints)
        if context.world_size > 1:
            torch.distributed.barrier()
        if final_step < config.training.max_steps:
            result = {"run_id": run_id, "status": "interrupted", "step": final_step}
        elif context.is_primary:
            final = evaluate(core, validation, config, context.device)
            gap = config.evaluation.minimum_control_gap
            checks = {
                "nll_improved": final["nll"]["normal"] < initial["nll"]["normal"],
                "normal_nll_beats_zero": final["nll"]["normal"] < final["nll"]["zero"],
                "normal_nll_beats_shuffled": (final["nll"]["normal"] < final["nll"]["shuffled"]),
                "word_f1_improved": (
                    final["generation"]["normal"]["word_f1"]
                    > initial["generation"]["normal"]["word_f1"]
                ),
                "word_f1_beats_zero": (
                    final["generation"]["normal"]["word_f1"]
                    >= final["generation"]["zero"]["word_f1"] + gap
                ),
                "word_f1_beats_shuffled": (
                    final["generation"]["normal"]["word_f1"]
                    >= final["generation"]["shuffled"]["word_f1"] + gap
                ),
            }
            result = {
                "schema": "deltaomni.audiocaps_caption_lora.v1",
                "run_id": run_id,
                "status": "complete",
                "initial": initial,
                "final": final,
                "checks": checks,
                "passed": all(checks.values()),
                "training_seconds": time.perf_counter() - started,
                "completed_at_utc": datetime.now(UTC).isoformat(),
            }
            _atomic_json(run_dir / "summary.json", result)
        else:
            result = {}
        if context.is_primary:
            _atomic_json(run_dir / "status.json", result)
            _atomic_json(
                config.output_root / "active_run.json",
                {"run_id": run_id, "status": result["status"], "signature": signature},
            )
        if context.world_size > 1:
            torch.distributed.barrier()
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Train AudioCaps caption LoRA from delta prefixes")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/audiocaps_caption_lora.yaml"),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    report = run(args.config, args.run_id, args.stop_after_step)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
