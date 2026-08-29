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
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from deltaomni.audiocaps_caption_lora import (
    AudioCaptionModel,
    CaptionConfig,
    _append_jsonl,
    _atomic_torch_save,
    _broadcast_string,
    _caption_metrics,
    _gather_rng_states,
    _load_model,
    _prune_checkpoints,
    _restore_rng,
)
from deltaomni.audiocaps_caption_lora import (
    load_config as load_caption_config,
)
from deltaomni.continuous_kv import ContinuousKVRunner
from deltaomni.distributed import distributed_context, reduce_sums
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
    cpu_threads: int
    gradient_accumulation_steps: int
    cache_entries: int


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    adapter_learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    checkpoint_interval_steps: int
    keep_last_checkpoints: int
    gradient_clip_norm: float
    zero_ranking_weight: float
    zero_ranking_margin: float
    resume: str
    cross_ranking_weight: float = 0.0
    cross_ranking_margin: float = 0.1
    order_ranking_weight: float = 0.0
    order_ranking_margin: float = 0.1


@dataclass(frozen=True)
class EvaluationConfig:
    windows: int
    max_new_tokens: int
    minimum_delta_gap: float
    minimum_memory_gap: float
    nll_windows: int | None = None
    minimum_cross_gap: float = 0.0
    minimum_order_gap: float = 0.0
    minimum_negative_nll_gap: float = 0.0


@dataclass(frozen=True)
class GoalStepCaptionConfig:
    seed: int
    input_mode: str
    caption_config: Path
    prefix_manifest: Path
    initial_checkpoint: Path
    initial_checkpoint_sha256: str
    runtime: RuntimeConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    output_root: Path
    log_root: Path
    report_path: Path


def load_config(path: Path) -> GoalStepCaptionConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    config = GoalStepCaptionConfig(
        seed=int(raw["seed"]),
        input_mode=str(raw["input_mode"]),
        caption_config=resolve(raw["caption_config"]),
        prefix_manifest=resolve(raw["prefix_manifest"]),
        initial_checkpoint=resolve(raw["initial_checkpoint"]),
        initial_checkpoint_sha256=str(raw["initial_checkpoint_sha256"]),
        runtime=RuntimeConfig(**raw["runtime"]),
        training=TrainingConfig(**raw["training"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        output_root=resolve(raw["output_root"]),
        log_root=resolve(raw["log_root"]),
        report_path=resolve(raw["report_path"]),
    )
    positive = (
        config.runtime.cpu_threads,
        config.runtime.gradient_accumulation_steps,
        config.runtime.cache_entries,
        config.training.learning_rate,
        config.training.adapter_learning_rate,
        config.training.warmup_steps,
        config.training.max_steps,
        config.training.checkpoint_interval_steps,
        config.training.keep_last_checkpoints,
        config.training.gradient_clip_norm,
        config.training.zero_ranking_weight,
        config.training.zero_ranking_margin,
        config.evaluation.windows,
        config.evaluation.max_new_tokens,
        config.evaluation.minimum_delta_gap,
        config.evaluation.minimum_memory_gap,
        *(() if config.evaluation.nll_windows is None else (config.evaluation.nll_windows,)),
    )
    if min(positive) <= 0 or len(config.initial_checkpoint_sha256) != 64:
        raise ValueError("Ego4D GoalStep caption controls must be positive")
    if config.input_mode not in {"delta", "full"}:
        raise ValueError("Ego4D GoalStep caption input mode must be delta or full")
    if config.training.resume not in {"auto", "never"}:
        raise ValueError("Invalid Ego4D GoalStep caption resume mode")
    nonnegative = (
        config.training.cross_ranking_weight,
        config.training.cross_ranking_margin,
        config.training.order_ranking_weight,
        config.training.order_ranking_margin,
        config.evaluation.minimum_cross_gap,
        config.evaluation.minimum_order_gap,
        config.evaluation.minimum_negative_nll_gap,
    )
    if min(nonnegative) < 0:
        raise ValueError("Ego4D GoalStep multi-negative controls must be nonnegative")
    if config.input_mode == "full" and any(
        value > 0
        for value in (
            config.training.cross_ranking_weight,
            config.training.order_ranking_weight,
            config.evaluation.minimum_cross_gap,
            config.evaluation.minimum_order_gap,
            config.evaluation.minimum_negative_nll_gap,
        )
    ):
        raise ValueError("Full-token GoalStep training cannot use delta-negative controls")
    return config


class WindowCache:
    def __init__(self, manifest: dict[str, Any], split: str, cache_entries: int) -> None:
        self.records = list(manifest["splits"][split])
        self.cache_entries = cache_entries
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def source_disjoint_index(self, index: int, offset: int = 1) -> int:
        if not 0 <= index < len(self.records) or offset <= 0:
            raise ValueError("Invalid GoalStep source-disjoint lookup")
        source = self.records[index]["source_group_id"]
        for step in range(len(self.records) - 1):
            candidate = (index + offset + step) % len(self.records)
            if self.records[candidate]["source_group_id"] != source:
                return candidate
        raise ValueError("GoalStep cache has no source-disjoint donor")

    def load(self, index: int) -> dict[str, Any]:
        path = self.records[index]["cache_path"]
        cached = self.cache.get(path)
        if cached is not None:
            self.cache.move_to_end(path)
            return cached
        value = torch.load(path, map_location="cpu", weights_only=False)
        required = {
            "window_id",
            "source_id",
            "first_full",
            "deltas",
            "event_full",
            "events",
        }
        if not required <= value.keys() or value["window_id"] != self.records[index]["window_id"]:
            raise ValueError(f"Invalid Ego4D GoalStep window cache: {path}")
        if len(value["events"]) < 2:
            raise ValueError(f"Ego4D training window has fewer than two commits: {path}")
        self.cache[path] = value
        self.cache.move_to_end(path)
        while len(self.cache) > self.cache_entries:
            self.cache.popitem(last=False)
        return value


def _expanded_adapter_state(
    target: dict[str, Tensor],
    source: dict[str, Tensor],
) -> dict[str, Tensor]:
    result = {name: value.clone() for name, value in target.items()}
    for name, value in source.items():
        if name not in result:
            raise ValueError(f"Unknown adapter checkpoint parameter: {name}")
        if result[name].shape == value.shape:
            result[name] = value
        elif name == "delta_positions" and result[name].shape[1:] == value.shape[1:]:
            if result[name].shape[0] < value.shape[0]:
                raise ValueError("GoalStep adapter has fewer positions than its checkpoint")
            result[name][: value.shape[0]] = value
        else:
            raise ValueError(f"Incompatible adapter checkpoint parameter: {name}")
    return result


def _match_delta_length(deltas: Tensor, length: int) -> Tensor:
    if length < 0 or deltas.shape[0] == 0:
        raise ValueError("Invalid GoalStep delta length match")
    if length == 0:
        return deltas[:0]
    if deltas.shape[0] == length:
        return deltas
    indices = torch.linspace(0, deltas.shape[0] - 1, length).round().long()
    return deltas[indices]


def _permutation_indices(identity: str, length: int) -> Tensor:
    if length <= 1:
        return torch.arange(length)
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little")
    indices = torch.randperm(length, generator=torch.Generator().manual_seed(seed))
    if torch.equal(indices, torch.arange(length)):
        indices = indices.roll(1)
    return indices


class GoalStepCaptionModel(nn.Module):
    def __init__(
        self,
        single: AudioCaptionModel,
        config: CaptionConfig,
        input_mode: str,
    ) -> None:
        super().__init__()
        self.single = single
        self.input_mode = input_mode
        tokenizer = single.interface.tokenizer
        continuation = tokenizer.apply_chat_template(
            [{"role": "user", "content": config.interface.user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        self.continuation_prompt_ids = tuple(
            tokenizer(continuation, add_special_tokens=False)["input_ids"]
        )
        answer_prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": (
                        "What was the last visual event that completed? "
                        "Answer with one short action phrase."
                    ),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        self.answer_prompt_ids = tuple(
            tokenizer(answer_prompt, add_special_tokens=False)["input_ids"]
        )

    def _prompt(self, first: bool, device: torch.device) -> Tensor:
        if first:
            return self.single.interface.prompt(1, device)
        return torch.tensor(self.continuation_prompt_ids, device=device).unsqueeze(0)

    def _project_deltas(self, deltas: Tensor) -> Tensor:
        if deltas.shape[1] == 0:
            return deltas.new_empty((deltas.shape[0], 0, self.single.adapter.delta_type.shape[0]))
        if deltas.shape[1] > self.single.adapter.delta_positions.shape[0]:
            raise ValueError(f"GoalStep delta span is too long: {deltas.shape[1]}")
        squeezed = deltas[:, :, 0]
        projected = self.single.adapter.delta_projection(self.single.adapter.delta_norm(squeezed))
        return (
            projected
            + self.single.adapter.delta_type
            + self.single.adapter.delta_positions[: deltas.shape[1]]
        )

    def _visual_chunk(
        self,
        payload: dict[str, Any],
        event: dict[str, Any],
        event_index: int,
        *,
        include_anchor: bool,
        accumulated: bool,
        control: str,
        donor: dict[str, Any] | None = None,
    ) -> Tensor:
        device = next(self.parameters()).device
        embeddings = self.single.thinker.get_input_embeddings()
        start_ids = torch.full(
            (1, 1), self.single.audio_start_token_id, dtype=torch.long, device=device
        )
        end_ids = torch.full(
            (1, 1), self.single.audio_end_token_id, dtype=torch.long, device=device
        )
        if self.input_mode == "full":
            event_full = payload["event_full"][event_index].float().unsqueeze(0).to(device)
            return torch.cat(
                (
                    embeddings(start_ids),
                    (event_full + self.single.adapter.anchor_type).to(embeddings.weight.dtype),
                    embeddings(end_ids),
                ),
                dim=1,
            )
        full_source = (
            payload["first_full"]
            if event_index == 0
            else payload["event_full"][event_index - 1]
        )
        full = full_source.float().unsqueeze(0).to(device)
        all_deltas = payload["deltas"].float().unsqueeze(0).to(device)
        start = 0 if accumulated else int(event["delta_start"])
        end = int(event["delta_end"])
        deltas = all_deltas[:, start:end]
        if control == "zero":
            deltas = torch.zeros_like(deltas)
        elif control == "cross_video":
            if donor is None or donor["source_id"] == payload["source_id"]:
                raise ValueError("GoalStep cross-video control requires a source-disjoint donor")
            donor_deltas = donor["deltas"].float().to(device)
            deltas = _match_delta_length(donor_deltas, end - start).unsqueeze(0)
        elif control == "permuted":
            indices = _permutation_indices(
                f"{payload['window_id']}:{event_index}:{start}:{end}", end - start
            ).to(device)
            deltas = deltas[:, indices]
        elif control != "normal":
            raise ValueError(f"Unknown GoalStep delta control: {control}")
        pieces = [embeddings(start_ids)]
        if include_anchor:
            pieces.append((full + self.single.adapter.anchor_type).to(embeddings.weight.dtype))
        pieces.append(self._project_deltas(deltas).to(embeddings.weight.dtype))
        pieces.append(embeddings(end_ids))
        return torch.cat(pieces, dim=1)

    def _event_chunk(
        self,
        payload: dict[str, Any],
        event: dict[str, Any],
        event_index: int,
        *,
        first: bool,
        reset_each: bool,
        control: str,
        donor: dict[str, Any] | None = None,
    ) -> Tensor:
        device = next(self.parameters()).device
        visual = self._visual_chunk(
            payload,
            event,
            event_index,
            include_anchor=True,
            accumulated=False,
            control=control,
            donor=donor,
        )
        prompt = self._prompt(first or reset_each, device)
        prompt_embeddings = self.single.thinker.get_input_embeddings()(prompt)
        return torch.cat((visual, prompt_embeddings), dim=1)

    def _caption_loss(
        self,
        payload: dict[str, Any],
        control: str,
        donor: dict[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor]:
        device = next(self.parameters()).device
        embeddings = []
        labels = []
        token_count = 0
        for index, event in enumerate(payload["events"]):
            chunk = self._event_chunk(
                payload,
                event,
                index,
                first=index == 0,
                reset_each=False,
                control=control,
                donor=donor,
            )
            embeddings.append(chunk)
            labels.append(torch.full(chunk.shape[:2], -100, dtype=torch.long, device=device))
            ids = self.single.interface.tokenizer(
                event["text"], add_special_tokens=False
            )["input_ids"]
            ids = ids[: self.single.interface.max_target_tokens - 1]
            ids = ids + [self.single.interface.end_token_id]
            target = torch.tensor(ids, device=device).unsqueeze(0)
            embeddings.append(self.single.thinker.get_input_embeddings()(target))
            labels.append(target)
            token_count += target.shape[1]
        inputs = torch.cat(embeddings, dim=1)
        targets = torch.cat(labels, dim=1)
        attention = torch.ones(inputs.shape[:2], dtype=torch.bool, device=device)
        positions = torch.arange(inputs.shape[1], device=device).view(1, 1, -1)
        positions = positions.expand(3, inputs.shape[0], -1)
        output = self.single.thinker(
            inputs_embeds=inputs,
            attention_mask=attention,
            position_ids=positions,
            labels=targets,
            use_cache=False,
        )
        return output.loss, torch.tensor(token_count, device=device)

    def forward(
        self,
        payload: dict[str, Any],
        control: str = "normal",
        donor: dict[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.input_mode == "full" and control != "normal":
            raise ValueError("Full-token GoalStep model does not support delta controls")
        return self._caption_loss(payload, control, donor)

    @torch.no_grad()
    def generate_window_and_answer(
        self,
        payload: dict[str, Any],
        *,
        control: str,
        reset_each: bool,
        max_new_tokens: int,
        donor: dict[str, Any] | None = None,
    ) -> tuple[list[str], str]:
        self.eval()
        runner = ContinuousKVRunner(self.single.thinker, position_axes=3)
        state = None
        generated = []
        for index, event in enumerate(payload["events"]):
            if reset_each:
                state = None
            chunk = self._event_chunk(
                payload,
                event,
                index,
                first=index == 0,
                reset_each=reset_each,
                control=control,
                donor=donor,
            )
            logits, state = runner.append(state=state, inputs_embeds=chunk)
            ids, state = runner.greedy_append(
                logits,
                state,
                end_token_id=self.single.interface.end_token_id,
                max_new_tokens=max_new_tokens,
            )
            generated.extend(self.single.interface.decode(ids))
        assert state is not None
        answer_prompt = torch.tensor(
            self.answer_prompt_ids,
            device=state.attention_mask.device,
        ).unsqueeze(0)
        logits, state = runner.append(state=state, input_ids=answer_prompt)
        answer_ids, _ = runner.greedy_append(
            logits,
            state,
            end_token_id=self.single.interface.end_token_id,
            max_new_tokens=max_new_tokens,
        )
        answer = self.single.interface.decode(answer_ids)[0]
        return generated, answer

    @torch.no_grad()
    def generate_window(
        self,
        payload: dict[str, Any],
        *,
        control: str,
        reset_each: bool,
        max_new_tokens: int,
    ) -> list[str]:
        captions, _ = self.generate_window_and_answer(
            payload,
            control=control,
            reset_each=reset_each,
            max_new_tokens=max_new_tokens,
        )
        return captions


def _load_goalstep(config: GoalStepCaptionConfig, device: torch.device) -> GoalStepCaptionModel:
    caption_config = load_caption_config(config.caption_config)
    single, _ = _load_model(caption_config, device)
    checkpoint = torch.load(config.initial_checkpoint, map_location="cpu", weights_only=False)
    set_peft_model_state_dict(single.thinker, checkpoint["lora"])
    expanded = _expanded_adapter_state(single.adapter.state_dict(), checkpoint["adapter"])
    single.adapter.load_state_dict(expanded)
    if config.input_mode == "full":
        single.adapter.requires_grad_(False)
    return GoalStepCaptionModel(single, caption_config, config.input_mode).to(device)


@torch.no_grad()
def evaluate(
    model: GoalStepCaptionModel,
    data: WindowCache,
    config: GoalStepCaptionConfig,
) -> dict[str, Any]:
    count = min(config.evaluation.windows, len(data))
    names = ["continuous", "reset_each", "zero"]
    if config.input_mode == "delta":
        names.extend(("cross_video", "permuted"))
    generated = {name: [] for name in names}
    final_answers = {name: [] for name in generated}
    nll_names = ("normal", "zero", "cross_video", "permuted")
    nll_sums = {name: 0.0 for name in nll_names}
    nll_tokens = {name: 0 for name in nll_names}
    nll_count = min(config.evaluation.nll_windows or count, count)
    references = []
    answer_references = []
    examples = []
    started = time.perf_counter()
    for index in range(count):
        payload = data.load(index)
        donor = data.load(data.source_disjoint_index(index))
        outputs = {}
        answers = {}
        controls = [
            ("continuous", "normal", False),
            ("reset_each", "normal", True),
            ("zero", "zero", False),
        ]
        if config.input_mode == "delta":
            controls.extend(
                (("cross_video", "cross_video", False), ("permuted", "permuted", False))
            )
        for name, control, reset_each in controls:
            outputs[name], answers[name] = model.generate_window_and_answer(
                payload,
                control=control,
                reset_each=reset_each,
                max_new_tokens=config.evaluation.max_new_tokens,
                donor=donor if control == "cross_video" else None,
            )
            generated[name].extend(outputs[name])
            final_answers[name].append(answers[name])
        if config.input_mode == "delta" and index < nll_count:
            for control in nll_names:
                loss, tokens = model._caption_loss(
                    payload,
                    control,
                    donor if control == "cross_video" else None,
                )
                token_count = int(tokens)
                nll_sums[control] += float(loss) * token_count
                nll_tokens[control] += token_count
        references.extend((event["text"],) for event in payload["events"])
        answer_references.append((payload["events"][-1]["text"],))
        if len(examples) < 8:
            examples.append(
                {
                    "window_id": payload["window_id"],
                    "source_id": payload["source_id"],
                    "events": payload["events"],
                    "final_answers": answers,
                    **outputs,
                }
            )
        completed = index + 1
        elapsed = time.perf_counter() - started
        eta = elapsed / completed * (count - completed)
        if completed % 5 == 0 or completed == count:
            print(
                f"goalstep_eval={completed}/{count} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    return {
        "metrics": {
            name: _caption_metrics(values, references)
            for name, values in generated.items()
        },
        "caption_events": len(references),
        "final_answer_metrics": {
            name: _caption_metrics(values, answer_references)
            for name, values in final_answers.items()
        },
        "final_answer_probe": "last_completed_event_from_same_kv",
        "nll": (
            {name: nll_sums[name] / nll_tokens[name] for name in nll_names}
            if config.input_mode == "delta"
            else {}
        ),
        "nll_windows": nll_count if config.input_mode == "delta" else 0,
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
        "adapter",
        "optimizer",
        "signature",
        "rng_states",
        "world_size",
        "code_revision",
        "initial",
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
        raise RuntimeError("Ego4D GoalStep captions require a clean Git worktree")
    if sha256_file(config.initial_checkpoint) != config.initial_checkpoint_sha256:
        raise ValueError("Ego4D caption initial checkpoint checksum mismatch")
    manifest = json.loads(config.prefix_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "deltaomni.omni_ego4d_goalstep_manifest.v2":
        raise ValueError("Unexpected Ego4D GoalStep prefix manifest")
    train = WindowCache(manifest, "train", config.runtime.cache_entries)
    validation = WindowCache(manifest, "validation", config.runtime.cache_entries)
    _set_seed(config.seed)
    torch.set_num_threads(config.runtime.cpu_threads)
    signature = resolved_input_signature(
        config,
        {
            "caption_config": config.caption_config,
            "prefix_manifest": config.prefix_manifest,
            "initial_checkpoint": config.initial_checkpoint,
        },
    )
    code_revision = git_revision(root)
    with distributed_context(
        config.runtime.device,
        config.runtime.backend,
        nccl_compatibility_mode=config.runtime.nccl_compatibility_mode,
    ) as context:
        core = _load_goalstep(config, context.device)
        model: nn.Module = core
        if context.world_size > 1:
            model = DistributedDataParallel(model, device_ids=[context.local_rank])
        parameter_groups = [
            {
                "params": [
                    parameter
                    for parameter in core.single.thinker.parameters()
                    if parameter.requires_grad
                ],
                "lr": config.training.learning_rate,
            }
        ]
        if config.input_mode == "delta":
            parameter_groups.append(
                {
                    "params": core.single.adapter.parameters(),
                    "lr": config.training.adapter_learning_rate,
                }
            )
        optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=config.training.weight_decay,
        )
        selected = run_id_override or (
            f"ego4d-goalstep-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        run_id = _broadcast_string(selected if context.is_primary else None, context)
        run_dir = config.output_root / run_id
        log_path = config.log_root / run_id / "metrics.jsonl"
        if context.is_primary:
            run_dir.mkdir(parents=True, exist_ok=True)
            if not (run_dir / "resolved_config.json").is_file():
                _atomic_json(run_dir / "resolved_config.json", asdict(config))
                _atomic_json(
                    run_dir / "metadata.json",
                    {
                        "code_revision": code_revision,
                        "world_size": context.world_size,
                        "gpu": torch.cuda.get_device_name(context.device),
                        "torch_version": torch.__version__,
                        "started_at_utc": datetime.now(UTC).isoformat(),
                    },
                )
        if context.world_size > 1:
            torch.distributed.barrier()
        start_step = 1
        initial = None
        resumed = _latest_checkpoint(run_dir) if config.training.resume == "auto" else None
        if resumed is not None:
            _, value = resumed
            if (
                value["signature"] != signature
                or value["world_size"] != context.world_size
                or value["code_revision"] != code_revision
            ):
                raise ValueError("Ego4D caption checkpoint is incompatible")
            set_peft_model_state_dict(core.single.thinker, value["lora"])
            core.single.adapter.load_state_dict(value["adapter"])
            optimizer.load_state_dict(value["optimizer"])
            _restore_rng(value["rng_states"][context.rank], context.device)
            start_step = int(value["next_step"])
            initial = value["initial"]
        if initial is None:
            if context.is_primary:
                initial = evaluate(core, validation, config)
            if context.world_size > 1:
                values = [initial]
                torch.distributed.broadcast_object_list(values, src=0)
                initial = values[0]
        final_step = min(config.training.max_steps, stop_after_step or config.training.max_steps)
        started = time.perf_counter()
        for step in range(start_step, final_step + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated = torch.zeros(8, device=context.device)
            for accumulation in range(config.runtime.gradient_accumulation_steps):
                generator = torch.Generator().manual_seed(
                    config.seed * 1_000_003 + step * 101 + accumulation
                )
                indices = torch.randint(0, len(train), (context.world_size,), generator=generator)
                selected_index = int(indices[context.rank])
                payload = train.load(selected_index)
                final_accumulation = (
                    accumulation == config.runtime.gradient_accumulation_steps - 1
                )
                normal_sync = (
                    nullcontext()
                    if final_accumulation or not isinstance(model, DistributedDataParallel)
                    else model.no_sync()
                )
                denominator = config.runtime.gradient_accumulation_steps
                if config.input_mode == "full":
                    with normal_sync:
                        with torch.autocast(
                            device_type=context.device.type, dtype=torch.bfloat16
                        ):
                            normal_loss, tokens = model(payload)
                            scaled = normal_loss / denominator
                        scaled.backward()
                    accumulated += torch.stack(
                        (
                            scaled.detach(),
                            normal_loss.detach() / denominator,
                            normal_loss.detach() / denominator,
                            normal_loss.detach() / denominator,
                            normal_loss.detach() / denominator,
                            normal_loss.new_zeros(()),
                            normal_loss.new_zeros(()),
                            normal_loss.new_zeros(()),
                        )
                    )
                    del normal_loss, scaled, tokens
                    continue

                offset = 1 + (
                    step * context.world_size + context.rank + accumulation
                ) % (len(train) - 1)
                donor = train.load(train.source_disjoint_index(selected_index, offset))
                with torch.no_grad(), torch.autocast(
                    device_type=context.device.type, dtype=torch.bfloat16
                ):
                    normal_probe, _ = core(payload, "normal")
                normal_value = normal_probe.detach()
                del normal_probe
                negative_values = {
                    "zero": normal_value,
                    "cross_video": normal_value,
                    "permuted": normal_value,
                }
                rankings = {name: normal_value.new_zeros(()) for name in negative_values}
                active_weight = 0.0
                objective_value = normal_value
                negative_specs = (
                    (
                        "zero",
                        config.training.zero_ranking_weight,
                        config.training.zero_ranking_margin,
                    ),
                    (
                        "cross_video",
                        config.training.cross_ranking_weight,
                        config.training.cross_ranking_margin,
                    ),
                    (
                        "permuted",
                        config.training.order_ranking_weight,
                        config.training.order_ranking_margin,
                    ),
                )
                for control, weight, margin in negative_specs:
                    if weight == 0:
                        continue
                    negative_sync = (
                        model.no_sync()
                        if isinstance(model, DistributedDataParallel)
                        else nullcontext()
                    )
                    with negative_sync:
                        with torch.autocast(
                            device_type=context.device.type, dtype=torch.bfloat16
                        ):
                            negative_loss, _ = model(
                                payload,
                                control,
                                donor if control == "cross_video" else None,
                            )
                            ranking = torch.relu(margin + normal_value - negative_loss.detach())
                            active = bool(ranking > 0)
                            negative_term = (
                                -weight * negative_loss / denominator
                                if active
                                else negative_loss * 0.0
                            )
                        negative_term.backward()
                    negative_values[control] = negative_loss.detach()
                    rankings[control] = ranking
                    objective_value = objective_value + weight * ranking
                    active_weight += weight if active else 0.0
                    del negative_loss, negative_term, ranking
                with normal_sync:
                    with torch.autocast(device_type=context.device.type, dtype=torch.bfloat16):
                        normal_loss, tokens = model(payload, "normal")
                        normal_term = (1.0 + active_weight) * normal_loss / denominator
                    normal_term.backward()
                accumulated += torch.stack(
                    (
                        objective_value / denominator,
                        normal_loss.detach() / denominator,
                        negative_values["zero"] / denominator,
                        negative_values["cross_video"] / denominator,
                        negative_values["permuted"] / denominator,
                        rankings["zero"] / denominator,
                        rankings["cross_video"] / denominator,
                        rankings["permuted"] / denominator,
                    )
                )
                del normal_loss, normal_term, tokens
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            warmup = min(1.0, step / config.training.warmup_steps)
            optimizer.param_groups[0]["lr"] = config.training.learning_rate * warmup
            if config.input_mode == "delta":
                optimizer.param_groups[1]["lr"] = config.training.adapter_learning_rate * warmup
            optimizer.step()
            reduced = reduce_sums({"metrics": accumulated})["metrics"] / context.world_size
            if context.is_primary:
                elapsed = time.perf_counter() - started
                eta = elapsed / (step - start_step + 1) * (final_step - step)
                record = {
                    "step": step,
                    "loss": float(reduced[0]),
                    "normal_nll": float(reduced[1]),
                    "zero_nll": float(reduced[2]),
                    "cross_nll": float(reduced[3]),
                    "permuted_nll": float(reduced[4]),
                    "zero_ranking": float(reduced[5]),
                    "cross_ranking": float(reduced[6]),
                    "order_ranking": float(reduced[7]),
                }
                _append_jsonl(log_path, record)
                if step % 5 == 0 or step == final_step:
                    print(
                        f"goalstep_step={step}/{final_step} loss={record['loss']:.5f} "
                        f"normal={record['normal_nll']:.5f} zero={record['zero_nll']:.5f} "
                        f"cross={record['cross_nll']:.5f} "
                        f"perm={record['permuted_nll']:.5f} "
                        f"rank={record['zero_ranking']:.5f}/"
                        f"{record['cross_ranking']:.5f}/{record['order_ranking']:.5f} "
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
                            "lora": get_peft_model_state_dict(core.single.thinker),
                            "adapter": core.single.adapter.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "signature": signature,
                            "rng_states": states,
                            "world_size": context.world_size,
                            "code_revision": code_revision,
                            "initial": initial,
                        },
                    )
                    _prune_checkpoints(run_dir, config.training.keep_last_checkpoints)
        if context.world_size > 1:
            torch.distributed.barrier()
        result = {}
        if final_step < config.training.max_steps:
            result = {"run_id": run_id, "status": "interrupted", "step": final_step}
        elif context.is_primary:
            final = evaluate(core, validation, config)
            metrics = final["metrics"]
            continuous = metrics["continuous"]["word_f1"]
            checks = {
                "continuous_improves": (
                    continuous > initial["metrics"]["continuous"]["word_f1"]
                ),
                "continuous_beats_reset": (
                    continuous
                    >= metrics["reset_each"]["word_f1"]
                    + config.evaluation.minimum_memory_gap
                ),
            }
            if config.input_mode == "delta":
                checks["continuous_beats_zero"] = (
                    continuous
                    >= metrics["zero"]["word_f1"] + config.evaluation.minimum_delta_gap
                )
                if config.evaluation.minimum_cross_gap > 0:
                    checks["continuous_beats_cross_video"] = (
                        continuous
                        >= metrics["cross_video"]["word_f1"]
                        + config.evaluation.minimum_cross_gap
                    )
                if config.evaluation.minimum_order_gap > 0:
                    checks["continuous_beats_permuted"] = (
                        continuous
                        >= metrics["permuted"]["word_f1"]
                        + config.evaluation.minimum_order_gap
                    )
                if config.evaluation.minimum_negative_nll_gap > 0:
                    normal_nll = final["nll"]["normal"]
                    gap = config.evaluation.minimum_negative_nll_gap
                    checks["normal_nll_beats_zero"] = normal_nll + gap <= final["nll"]["zero"]
                    checks["normal_nll_beats_cross_video"] = (
                        normal_nll + gap <= final["nll"]["cross_video"]
                    )
                    checks["normal_nll_beats_permuted"] = (
                        normal_nll + gap <= final["nll"]["permuted"]
                    )
            result = {
                "schema": "deltaomni.ego4d_goalstep_caption_lora.v1",
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
            _atomic_json(config.report_path, result)
        if context.is_primary:
            _atomic_json(run_dir / "status.json", result)
        if context.world_size > 1:
            torch.distributed.barrier()
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Train natural Ego4D GoalStep caption memory")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/ego4d_goalstep_caption_smoke.yaml")
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
