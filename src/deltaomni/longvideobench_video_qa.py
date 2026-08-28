from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import set_peft_model_state_dict

from deltaomni.continuous_kv import ContinuousKVRunner
from deltaomni.ego4d_goalstep_caption_lora import (
    GoalStepCaptionModel,
    _expanded_adapter_state,
    _load_goalstep,
)
from deltaomni.ego4d_goalstep_caption_lora import (
    load_config as load_goalstep_config,
)
from deltaomni.run_integrity import git_revision, git_worktree_is_clean, sha256_file
from deltaomni.train_sanity import _atomic_json, _set_seed


@dataclass(frozen=True)
class ModelArm:
    name: str
    mode: str
    model_config: Path
    checkpoint: Path
    checkpoint_sha256: str


@dataclass(frozen=True)
class QAConfig:
    seed: int
    cache_manifest: Path
    device: str
    cpu_threads: int
    maximum_videos: int | None
    maximum_questions: int | None
    caption_max_new_tokens: int
    answer_max_new_tokens: int
    answer_strategy: str
    arms: tuple[ModelArm, ...]
    predictions_path: Path
    report_path: Path


def load_config(path: Path) -> QAConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    config = QAConfig(
        seed=int(raw["seed"]),
        cache_manifest=resolve(raw["cache_manifest"]),
        device=str(raw["device"]),
        cpu_threads=int(raw["cpu_threads"]),
        maximum_videos=(
            None if raw.get("maximum_videos") is None else int(raw["maximum_videos"])
        ),
        maximum_questions=(
            None if raw.get("maximum_questions") is None else int(raw["maximum_questions"])
        ),
        caption_max_new_tokens=int(raw["caption_max_new_tokens"]),
        answer_max_new_tokens=int(raw["answer_max_new_tokens"]),
        answer_strategy=str(raw["answer_strategy"]),
        arms=tuple(
            ModelArm(
                name=str(value["name"]),
                mode=str(value["mode"]),
                model_config=resolve(value["model_config"]),
                checkpoint=resolve(value["checkpoint"]),
                checkpoint_sha256=str(value["checkpoint_sha256"]),
            )
            for value in raw["arms"]
        ),
        predictions_path=resolve(raw["predictions_path"]),
        report_path=resolve(raw["report_path"]),
    )
    positive = (
        config.cpu_threads,
        config.caption_max_new_tokens,
        config.answer_max_new_tokens,
        *(
            value
            for value in (config.maximum_videos, config.maximum_questions)
            if value is not None
        ),
    )
    if min(positive) <= 0 or not config.arms:
        raise ValueError("LongVideoBench QA controls must be positive")
    if len({arm.name for arm in config.arms}) != len(config.arms):
        raise ValueError("LongVideoBench QA arm names must be unique")
    if config.answer_strategy not in {"choice_logit", "greedy"}:
        raise ValueError("Unknown LongVideoBench answer strategy")
    valid_modes = {
        "full",
        "delta",
        "zero",
        "reversed",
        "last_only",
        "cross_video",
        "memory_removed",
    }
    if any(arm.mode not in valid_modes or len(arm.checkpoint_sha256) != 64 for arm in config.arms):
        raise ValueError("Invalid LongVideoBench QA arm")
    return config


class VideoCache:
    def __init__(self, manifest: dict[str, Any], entries: int = 32) -> None:
        if manifest.get("schema") != "deltaomni.omni_longvideobench_manifest.v1":
            raise ValueError("Unexpected LongVideoBench cache manifest")
        self.videos = manifest["videos"]
        self.entries = entries
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def load_window(self, record: dict[str, Any]) -> dict[str, Any]:
        path = str(record["cache_path"])
        value = self.cache.get(path)
        if value is not None:
            self.cache.move_to_end(path)
            return value
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value.get("window_id") != record["window_id"]:
            raise ValueError(f"LongVideoBench cache identity mismatch: {path}")
        self.cache[path] = value
        self.cache.move_to_end(path)
        while len(self.cache) > self.entries:
            self.cache.popitem(last=False)
        return value

    def windows(self, video_id: str) -> list[dict[str, Any]]:
        return [self.load_window(record) for record in self.videos[video_id]["windows"]]


def _load_arm(arm: ModelArm, device: torch.device) -> GoalStepCaptionModel:
    if sha256_file(arm.checkpoint) != arm.checkpoint_sha256:
        raise ValueError(f"LongVideoBench arm checkpoint checksum mismatch: {arm.name}")
    model_config = load_goalstep_config(arm.model_config)
    model = _load_goalstep(model_config, device)
    checkpoint = torch.load(arm.checkpoint, map_location="cpu", weights_only=False)
    set_peft_model_state_dict(model.single.thinker, checkpoint["lora"])
    expanded = _expanded_adapter_state(model.single.adapter.state_dict(), checkpoint["adapter"])
    model.single.adapter.load_state_dict(expanded)
    return model.eval()


def _match_deltas(donor: torch.Tensor, length: int) -> torch.Tensor:
    if donor.shape[0] == length:
        return donor
    indices = torch.linspace(0, donor.shape[0] - 1, length).round().long()
    return donor[indices]


def _payload(
    window: dict[str, Any],
    *,
    mode: str,
    donor: dict[str, Any] | None,
) -> dict[str, Any]:
    deltas = window["deltas"]
    if mode == "zero":
        deltas = torch.zeros_like(deltas)
    elif mode == "reversed":
        deltas = deltas.flip(0)
    elif mode == "last_only":
        retained = torch.zeros_like(deltas)
        retained[-1] = deltas[-1]
        deltas = retained
    elif mode == "cross_video":
        assert donor is not None
        deltas = _match_deltas(donor["deltas"], len(deltas))
    return {
        "window_id": window["window_id"],
        "source_id": window["video_id"],
        "first_full": window["first_full"],
        "deltas": deltas,
        "event_full": window["final_full"].unsqueeze(0),
        "events": [
            {
                "caption_id": f"{window['window_id']}:caption",
                "text": "",
                "start_seconds": window["start_seconds"],
                "end_seconds": window["end_seconds"],
                "commit_seconds": window["end_seconds"],
                "delta_start": 0,
                "delta_end": len(deltas),
            }
        ],
    }


def _answer_prompt(model: GoalStepCaptionModel, question: dict[str, Any]) -> torch.Tensor:
    choices = "\n".join(
        f"{chr(65 + index)}. {choice}" for index, choice in enumerate(question["candidates"])
    )
    text = (
        f"Question: {question['question']}\n{choices}\n"
        "Answer with only the letter of the correct choice."
    )
    tokenizer = model.single.interface.tokenizer
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return torch.tensor(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def _parse_choice(text: str, count: int) -> int | None:
    first = text.strip().splitlines()[0].upper() if text.strip() else ""
    match = re.search(r"(?:^|[^A-Z])([A-Z])(?:[^A-Z]|$)", first)
    if match is None:
        return None
    index = ord(match.group(1)) - ord("A")
    return index if 0 <= index < count else None


@torch.no_grad()
def _predict(
    model: GoalStepCaptionModel,
    windows: list[dict[str, Any]],
    donors: list[dict[str, Any]],
    question: dict[str, Any],
    arm: ModelArm,
    config: QAConfig,
) -> tuple[str, list[str]]:
    runner = ContinuousKVRunner(model.single.thinker, position_axes=3)
    state = None
    captions = []
    for index, window in enumerate(windows):
        if arm.mode == "memory_removed":
            state = None
        payload = _payload(
            window,
            mode=arm.mode,
            donor=donors[index % len(donors)] if donors else None,
        )
        chunk = model._event_chunk(
            payload,
            payload["events"][0],
            0,
            first=index == 0,
            reset_each=arm.mode == "memory_removed",
            control="normal",
        )
        model_device = next(model.parameters()).device
        logits, state = runner.append(state=state, inputs_embeds=chunk.to(model_device))
        ids, state = runner.greedy_append(
            logits,
            state,
            end_token_id=model.single.interface.end_token_id,
            max_new_tokens=config.caption_max_new_tokens,
        )
        captions.extend(model.single.interface.decode(ids))
    assert state is not None
    prompt = _answer_prompt(model, question).to(state.attention_mask.device).unsqueeze(0)
    logits, state = runner.append(state=state, input_ids=prompt)
    if config.answer_strategy == "choice_logit":
        tokenizer = model.single.interface.tokenizer
        choice_ids = []
        for index in range(len(question["candidates"])):
            ids = tokenizer(chr(65 + index), add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ValueError("LongVideoBench choice letter is not one token")
            choice_ids.append(ids[0])
        scores = logits[0, -1, torch.tensor(choice_ids, device=logits.device)]
        selected = int(scores.argmax())
        return chr(65 + selected), captions
    ids, _ = runner.greedy_append(
        logits,
        state,
        end_token_id=model.single.interface.end_token_id,
        max_new_tokens=config.answer_max_new_tokens,
    )
    return model.single.interface.decode(ids)[0], captions


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _select_arms(
    config: QAConfig,
    selected_arms: list[str] | None,
    output_suffix: str | None,
) -> QAConfig:
    if selected_arms is None:
        return config
    requested = set(selected_arms)
    available = {arm.name for arm in config.arms}
    unknown = requested - available
    if unknown:
        raise ValueError(f"Unknown LongVideoBench QA arms: {sorted(unknown)}")
    arms = tuple(arm for arm in config.arms if arm.name in requested)
    if not arms:
        raise ValueError("At least one LongVideoBench QA arm must be selected")
    suffix = output_suffix or "-".join(arm.name for arm in arms)
    if re.fullmatch(r"[A-Za-z0-9_-]+", suffix) is None:
        raise ValueError("LongVideoBench output suffix must be filesystem-safe")

    def suffixed(path: Path) -> Path:
        return path.with_name(f"{path.stem}_{suffix}{path.suffix}")

    return replace(
        config,
        arms=arms,
        predictions_path=suffixed(config.predictions_path),
        report_path=suffixed(config.report_path),
    )


def run(
    config_path: Path,
    *,
    selected_arms: list[str] | None = None,
    output_suffix: str | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    config = _select_arms(config, selected_arms, output_suffix)
    root = config_path.resolve().parent.parent
    if not git_worktree_is_clean(root):
        raise RuntimeError("LongVideoBench QA requires a clean Git worktree")
    _set_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    device = torch.device(config.device)
    manifest = json.loads(config.cache_manifest.read_text(encoding="utf-8"))
    data = VideoCache(manifest)
    video_ids = list(data.videos)
    if config.maximum_videos is not None:
        video_ids = video_ids[: config.maximum_videos]
    rows = []
    arm_reports = {}
    started = time.perf_counter()
    for arm in config.arms:
        model = _load_arm(arm, device)
        correct = 0
        parsed = 0
        completed = 0
        total_questions = sum(len(data.videos[video_id]["questions"]) for video_id in video_ids)
        if config.maximum_questions is not None:
            total_questions = min(total_questions, config.maximum_questions)
        arm_started = time.perf_counter()
        last_progress = arm_started
        print(
            f"longvideobench_arm={arm.name} questions=0/{total_questions} eta=pending",
            flush=True,
        )
        for video_index, video_id in enumerate(video_ids):
            windows = data.windows(video_id)
            donor_id = video_ids[(video_index + 1) % len(video_ids)]
            donors = data.windows(donor_id)
            for question in data.videos[video_id]["questions"]:
                if config.maximum_questions is not None and completed >= config.maximum_questions:
                    break
                prediction, captions = _predict(model, windows, donors, question, arm, config)
                choice = _parse_choice(prediction, len(question["candidates"]))
                expected = int(question["correct_choice"])
                parsed += choice is not None
                correct += choice == expected
                completed += 1
                rows.append(
                    {
                        "arm": arm.name,
                        "id": question["id"],
                        "video_id": video_id,
                        "prediction": prediction,
                        "parsed_choice": choice,
                        "correct_choice": expected,
                        "duration_group": question["duration_group"],
                        "question_category": question["question_category"],
                        "captions": captions,
                    }
                )
                now = time.perf_counter()
                if completed == 1 or now - last_progress >= 180 or completed == total_questions:
                    elapsed = now - arm_started
                    eta = elapsed / completed * (total_questions - completed)
                    print(
                        f"longvideobench_arm={arm.name} "
                        f"questions={completed}/{total_questions} "
                        f"elapsed_seconds={elapsed:.1f} eta_seconds={eta:.1f}",
                        flush=True,
                    )
                    last_progress = now
            if config.maximum_questions is not None and completed >= config.maximum_questions:
                break
        arm_reports[arm.name] = {
            "accuracy": correct / completed,
            "parse_rate": parsed / completed,
            "questions": completed,
        }
        del model
        torch.cuda.empty_cache()
        print(f"longvideobench_arm={arm.name} questions={completed}", flush=True)
    _atomic_jsonl(config.predictions_path, rows)
    report = {
        "schema": "deltaomni.longvideobench_video_qa.v1",
        "arms": arm_reports,
        "videos": len(video_ids),
        "predictions": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "answer_strategy": config.answer_strategy,
        "code_revision": git_revision(root),
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(config.report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate video-only LongVideoBench QA")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/longvideobench_video_qa_smoke.yaml")
    )
    parser.add_argument(
        "--arm",
        action="append",
        dest="arms",
        help="Evaluate only this configured arm; repeat to select multiple arms",
    )
    parser.add_argument(
        "--output-suffix",
        help="Suffix added to prediction and report filenames when --arm is used",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config, selected_arms=args.arms, output_suffix=args.output_suffix),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
