from __future__ import annotations

import argparse
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
from deltaomni.streaming_sequence import StreamingSequence, build_sequences
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


@dataclass(frozen=True)
class EvaluationConfig:
    sequences: int
    max_new_tokens: int
    minimum_delta_gap: float
    maximum_reset_regression: float


@dataclass(frozen=True)
class ContinuousConfig:
    seed: int
    caption_config: Path
    initial_checkpoint: Path
    initial_checkpoint_sha256: str
    sections_per_sequence: int
    runtime: RuntimeConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    output_root: Path
    log_root: Path
    report_path: Path


def load_config(path: Path) -> ContinuousConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    config = ContinuousConfig(
        seed=int(raw["seed"]),
        caption_config=resolve(raw["caption_config"]),
        initial_checkpoint=resolve(raw["initial_checkpoint"]),
        initial_checkpoint_sha256=str(raw["initial_checkpoint_sha256"]),
        sections_per_sequence=int(raw["sections_per_sequence"]),
        runtime=RuntimeConfig(**raw["runtime"]),
        training=TrainingConfig(**raw["training"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        output_root=resolve(raw["output_root"]),
        log_root=resolve(raw["log_root"]),
        report_path=resolve(raw["report_path"]),
    )
    positive = (
        config.sections_per_sequence,
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
        config.evaluation.sequences,
        config.evaluation.max_new_tokens,
        config.evaluation.minimum_delta_gap,
        config.evaluation.maximum_reset_regression,
    )
    if min(positive) <= 0 or len(config.initial_checkpoint_sha256) != 64:
        raise ValueError("Continuous caption controls must be positive")
    if config.training.resume not in {"auto", "never"}:
        raise ValueError("Invalid continuous caption resume mode")
    return config


class SequenceCache:
    def __init__(self, sequences: list[StreamingSequence], cache_entries: int) -> None:
        self.sequences = sequences
        self.cache_entries = cache_entries
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.sequences)

    def load_path(self, path: Path) -> dict[str, Any]:
        key = str(path)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.move_to_end(key)
            return cached
        value = torch.load(path, map_location="cpu", weights_only=False)
        if not {"source_id", "source_group_id", "first_full", "deltas", "captions"} <= value.keys():
            raise ValueError(f"Incomplete continuous caption cache: {path}")
        self.cache[key] = value
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_entries:
            self.cache.popitem(last=False)
        return value

    def load(self, index: int) -> list[dict[str, Any]]:
        return [self.load_path(section.cache_path) for section in self.sequences[index].sections]


class ContinuousCaptionModel(nn.Module):
    def __init__(self, single: AudioCaptionModel, config: CaptionConfig) -> None:
        super().__init__()
        self.single = single
        tokenizer = single.interface.tokenizer
        continuation = tokenizer.apply_chat_template(
            [{"role": "user", "content": config.interface.user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        self.continuation_prompt_ids = tuple(
            tokenizer(continuation, add_special_tokens=False)["input_ids"]
        )

    def _prompt(self, first: bool, device: torch.device) -> Tensor:
        if first:
            return self.single.interface.prompt(1, device)
        return torch.tensor(self.continuation_prompt_ids, device=device).unsqueeze(0)

    def _chunk(self, payload: dict[str, Any], first: bool, control: str) -> Tensor:
        device = next(self.parameters()).device
        anchor = payload["first_full"].float().unsqueeze(0).to(device)
        deltas = payload["deltas"].float().unsqueeze(0).to(device)
        if control == "zero":
            deltas = torch.zeros_like(deltas)
        prefix = self.single._prefix(anchor, deltas)
        prompt = self._prompt(first, device)
        prompt_embeds = self.single.thinker.get_input_embeddings()(prompt)
        return torch.cat((prefix, prompt_embeds), dim=1)

    def _caption_loss(
        self,
        payloads: list[dict[str, Any]],
        captions: list[str],
        control: str,
    ) -> tuple[Tensor, Tensor]:
        if len(payloads) != len(captions):
            raise ValueError("Continuous caption section/target mismatch")
        device = next(self.parameters()).device
        embeddings = []
        labels = []
        token_count = 0
        for index, (payload, caption) in enumerate(zip(payloads, captions, strict=True)):
            chunk = self._chunk(payload, index == 0, control)
            embeddings.append(chunk)
            labels.append(torch.full(chunk.shape[:2], -100, dtype=torch.long, device=device))
            ids = self.single.interface.tokenizer(caption, add_special_tokens=False)["input_ids"]
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
        payloads: list[dict[str, Any]],
        captions: list[str],
    ) -> tuple[Tensor, Tensor, Tensor]:
        normal, tokens = self._caption_loss(payloads, captions, "normal")
        zero, _ = self._caption_loss(payloads, captions, "zero")
        return normal, zero, tokens

    @torch.no_grad()
    def generate_sequence(
        self,
        payloads: list[dict[str, Any]],
        *,
        control: str,
        reset_each: bool,
        max_new_tokens: int,
    ) -> list[str]:
        self.eval()
        runner = ContinuousKVRunner(self.single.thinker, position_axes=3)
        state = None
        captions = []
        for index, payload in enumerate(payloads):
            if reset_each:
                state = None
            chunk = self._chunk(payload, index == 0 or reset_each, control)
            logits, state = runner.append(state=state, inputs_embeds=chunk)
            ids, state = runner.greedy_append(
                logits,
                state,
                end_token_id=self.single.interface.end_token_id,
                max_new_tokens=max_new_tokens,
            )
            captions.extend(self.single.interface.decode(ids))
        return captions


def _load_continuous(config: ContinuousConfig, device: torch.device) -> ContinuousCaptionModel:
    caption_config = load_caption_config(config.caption_config)
    single, _ = _load_model(caption_config, device)
    value = torch.load(config.initial_checkpoint, map_location="cpu", weights_only=False)
    set_peft_model_state_dict(single.thinker, value["lora"])
    single.adapter.load_state_dict(value["adapter"])
    return ContinuousCaptionModel(single, caption_config).to(device)


@torch.no_grad()
def evaluate(
    model: ContinuousCaptionModel,
    data: SequenceCache,
    config: ContinuousConfig,
) -> dict[str, Any]:
    count = min(config.evaluation.sequences, len(data))
    generated = {name: [] for name in ("continuous", "reset_each", "zero")}
    references = []
    examples = []
    started = time.perf_counter()
    for index in range(count):
        payloads = data.load(index)
        refs = [tuple(payload["captions"]) for payload in payloads]
        outputs = {
            "continuous": model.generate_sequence(
                payloads,
                control="normal",
                reset_each=False,
                max_new_tokens=config.evaluation.max_new_tokens,
            ),
            "reset_each": model.generate_sequence(
                payloads,
                control="normal",
                reset_each=True,
                max_new_tokens=config.evaluation.max_new_tokens,
            ),
            "zero": model.generate_sequence(
                payloads,
                control="zero",
                reset_each=False,
                max_new_tokens=config.evaluation.max_new_tokens,
            ),
        }
        for name, values in outputs.items():
            generated[name].extend(values)
        references.extend(refs)
        if len(examples) < 8:
            examples.append(
                {
                    "source_ids": [payload["source_id"] for payload in payloads],
                    "references": [list(value) for value in refs],
                    **outputs,
                }
            )
        completed = index + 1
        elapsed = time.perf_counter() - started
        eta = elapsed / completed * (count - completed)
        if completed % 5 == 0 or completed == count:
            print(
                f"continuous_eval={completed}/{count} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    return {
        "metrics": {
            name: _caption_metrics(values, references)
            for name, values in generated.items()
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
        raise RuntimeError("Continuous caption runs require a clean Git worktree")
    if sha256_file(config.initial_checkpoint) != config.initial_checkpoint_sha256:
        raise ValueError("Continuous caption initial checkpoint checksum mismatch")
    caption_config = load_caption_config(config.caption_config)
    sequences, discarded = build_sequences(
        caption_config.prefix_manifest,
        sections_per_sequence=config.sections_per_sequence,
        seed=config.seed,
    )
    train = SequenceCache(sequences["train"], config.runtime.cache_entries)
    validation = SequenceCache(sequences["validation"], config.runtime.cache_entries)
    _set_seed(config.seed)
    torch.set_num_threads(config.runtime.cpu_threads)
    signature = resolved_input_signature(
        config,
        {
            "caption_config": config.caption_config,
            "prefix_manifest": caption_config.prefix_manifest,
            "initial_checkpoint": config.initial_checkpoint,
        },
    )
    code_revision = git_revision(root)
    with distributed_context(
        config.runtime.device,
        config.runtime.backend,
        nccl_compatibility_mode=config.runtime.nccl_compatibility_mode,
    ) as context:
        core = _load_continuous(config, context.device)
        model: nn.Module = core
        if context.world_size > 1:
            model = DistributedDataParallel(model, device_ids=[context.local_rank])
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        parameter
                        for parameter in core.single.thinker.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": config.training.learning_rate,
                },
                {
                    "params": core.single.adapter.parameters(),
                    "lr": config.training.adapter_learning_rate,
                },
            ],
            weight_decay=config.training.weight_decay,
        )
        selected = run_id_override or (
            f"msrvtt-continuous-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
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
                raise ValueError("Continuous caption checkpoint is incompatible")
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
            accumulated = torch.zeros(4, device=context.device)
            for accumulation in range(config.runtime.gradient_accumulation_steps):
                generator = torch.Generator().manual_seed(
                    config.seed * 1_000_003 + step * 101 + accumulation
                )
                indices = torch.randint(0, len(train), (context.world_size,), generator=generator)
                index = int(indices[context.rank])
                payloads = train.load(index)
                captions = []
                for payload in payloads:
                    reference = int(
                        torch.randint(0, len(payload["captions"]), (), generator=generator)
                    )
                    captions.append(payload["captions"][reference])
                sync = (
                    nullcontext()
                    if accumulation == config.runtime.gradient_accumulation_steps - 1
                    or not isinstance(model, DistributedDataParallel)
                    else model.no_sync()
                )
                with sync:
                    with torch.autocast(device_type=context.device.type, dtype=torch.bfloat16):
                        normal_loss, zero_loss, tokens = model(payloads, captions)
                        ranking = torch.relu(
                            config.training.zero_ranking_margin + normal_loss - zero_loss
                        )
                        loss = normal_loss + config.training.zero_ranking_weight * ranking
                        scaled = loss / config.runtime.gradient_accumulation_steps
                    scaled.backward()
                    accumulated += torch.stack(
                        (
                            scaled.detach(),
                            normal_loss.detach() / config.runtime.gradient_accumulation_steps,
                            zero_loss.detach() / config.runtime.gradient_accumulation_steps,
                            ranking.detach() / config.runtime.gradient_accumulation_steps,
                        )
                    )
                    del loss, normal_loss, ranking, scaled, tokens, zero_loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            warmup = min(1.0, step / config.training.warmup_steps)
            optimizer.param_groups[0]["lr"] = config.training.learning_rate * warmup
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
                    "zero_ranking": float(reduced[3]),
                }
                _append_jsonl(log_path, record)
                if step % 5 == 0 or step == final_step:
                    print(
                        f"continuous_step={step}/{final_step} loss={record['loss']:.5f} "
                        f"normal={record['normal_nll']:.5f} zero={record['zero_nll']:.5f} "
                        f"rank={record['zero_ranking']:.5f} "
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
            normal = metrics["continuous"]["word_f1"]
            checks = {
                "continuous_improves": normal > initial["metrics"]["continuous"]["word_f1"],
                "continuous_within_reset_tolerance": (
                    normal + config.evaluation.maximum_reset_regression
                    >= metrics["reset_each"]["word_f1"]
                ),
                "continuous_beats_zero": (
                    normal
                    >= metrics["zero"]["word_f1"] + config.evaluation.minimum_delta_gap
                ),
            }
            result = {
                "schema": "deltaomni.msrvtt_continuous_caption_lora.v1",
                "run_id": run_id,
                "status": "complete",
                "sequence_counts": {key: len(value) for key, value in sequences.items()},
                "discarded_sections": discarded,
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
    parser = argparse.ArgumentParser(description="Train continuous-KV MSR-VTT caption streams")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/msrvtt_continuous_caption_smoke.yaml")
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
