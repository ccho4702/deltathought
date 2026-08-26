from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from deltaomni.config import SanityConfig, load_config
from deltaomni.model import DeltaCodecModel
from deltaomni.synthetic import (
    CLASS_TOKEN_OFFSET,
    SyntheticInterleavedDataset,
    collate_examples,
)
from deltaomni.train_sanity import _atomic_json, _latest_checkpoint, _set_seed


@dataclass
class Collected:
    full_changes: Tensor
    reconstructed_changes: Tensor
    delta_states: Tensor
    labels: Tensor
    final_full_states: Tensor
    caption_histories: Tensor
    last_delta_caption_histories: Tensor
    final_qa_targets: Tensor
    reconstruction_mse: float
    step_reconstruction_mse: float
    commit_reconstruction_mse: float
    last_delta_reconstruction_mse: float
    anchor_mse: float
    shuffled_reconstruction_mse: float
    caption_exact: float
    last_delta_caption_exact: float
    zero_delta_caption_exact: float
    shuffled_delta_caption_exact: float
    length_accuracy: float
    trigger_true: Tensor
    trigger_predicted: Tensor


def _exact_caption(logits: Tensor, targets: Tensor) -> Tensor:
    predicted = logits.argmax(dim=-1)
    expected = targets[:, 1:]
    active = expected.ne(0)
    return ((predicted.eq(expected) | ~active).all(dim=1)).float()


@torch.no_grad()
def collect(
    model: DeltaCodecModel,
    dataset: SyntheticInterleavedDataset,
    config: SanityConfig,
) -> Collected:
    batch = collate_examples([dataset[index] for index in range(len(dataset))])
    full_embeddings = batch["full_embeddings"]
    commit_targets = batch["commit_targets"]
    caption_targets = batch["caption_targets"]
    caption_lengths = batch["caption_lengths"]
    full_changes: list[Tensor] = []
    reconstructed_changes: list[Tensor] = []
    delta_states: list[Tensor] = []
    labels: list[Tensor] = []
    reconstruction_errors: list[Tensor] = []
    step_reconstruction_errors: list[Tensor] = []
    commit_reconstruction_errors: list[Tensor] = []
    last_delta_reconstruction_errors: list[Tensor] = []
    anchor_errors: list[Tensor] = []
    shuffled_errors: list[Tensor] = []
    caption_exact: list[Tensor] = []
    last_caption_exact: list[Tensor] = []
    zero_caption_exact: list[Tensor] = []
    shuffled_caption_exact: list[Tensor] = []
    length_correct: list[Tensor] = []
    trigger_true: list[Tensor] = []
    trigger_predicted: list[Tensor] = []
    final_full_states: list[Tensor] = []
    caption_histories: list[Tensor] = []
    last_caption_histories: list[Tensor] = []
    final_qa_targets: list[Tensor] = []

    model.eval()
    for modality_index, modality in enumerate(config.modalities):
        codec = model.codecs[modality.value]
        anchor = full_embeddings[:, 0, modality_index]
        previous = anchor
        slots = torch.zeros(
            len(dataset),
            config.model.delta_tokens,
            config.model.embedding_dim,
        )
        load = torch.zeros(len(dataset))
        caption_history = torch.zeros(len(dataset), 2, 4)
        last_caption_history = torch.zeros(len(dataset), 2, 4)
        caption_counts = torch.zeros(len(dataset), dtype=torch.long)
        for time_index in range(1, config.training.sequence_steps):
            current = full_embeddings[:, time_index, modality_index]
            delta = codec.delta_encoder(previous, current)
            slots = codec.accumulator(slots, delta)
            load = load + codec.policy.novelty_score(delta)
            trigger_logits, length_logits = codec.policy(slots, load)
            step_reconstructed = codec.reconstructor(previous, delta)
            reconstructed = codec.reconstructor(anchor, slots)
            last_delta_reconstructed = codec.reconstructor(anchor, delta)
            shuffled_slots = slots.roll(1, dims=0)
            shuffled_reconstructed = codec.reconstructor(anchor, shuffled_slots)
            target_commit = commit_targets[:, time_index, modality_index]

            reconstruction_errors.append((reconstructed - current).square().mean(dim=(1, 2)))
            step_reconstruction_errors.append(
                (step_reconstructed - current).square().mean(dim=(1, 2))
            )
            anchor_errors.append((anchor - current).square().mean(dim=(1, 2)))
            shuffled_errors.append(
                (shuffled_reconstructed - current).square().mean(dim=(1, 2))
            )
            trigger_true.append(target_commit)
            trigger_predicted.append(torch.sigmoid(trigger_logits).ge(0.5))

            if target_commit.any():
                targets = caption_targets[target_commit, time_index, modality_index]
                normal_logits = codec.caption_decoder(
                    anchor[target_commit],
                    slots[target_commit],
                    targets[:, :-1],
                )
                zero_logits = codec.caption_decoder(
                    anchor[target_commit],
                    torch.zeros_like(slots[target_commit]),
                    targets[:, :-1],
                )
                shuffled_logits = codec.caption_decoder(
                    anchor[target_commit],
                    shuffled_slots[target_commit],
                    targets[:, :-1],
                )
                last_delta_logits = codec.caption_decoder(
                    anchor[target_commit],
                    delta[target_commit],
                    targets[:, :-1],
                )
                caption_exact.append(_exact_caption(normal_logits, targets))
                last_caption_exact.append(_exact_caption(last_delta_logits, targets))
                zero_caption_exact.append(_exact_caption(zero_logits, targets))
                shuffled_caption_exact.append(_exact_caption(shuffled_logits, targets))
                length_correct.append(
                    length_logits[target_commit]
                    .argmax(dim=-1)
                    .eq(caption_lengths[target_commit, time_index, modality_index])
                    .float()
                )
                full_changes.append((current[target_commit] - anchor[target_commit]).mean(dim=1))
                reconstructed_changes.append(
                    (reconstructed[target_commit] - anchor[target_commit]).mean(dim=1)
                )
                delta_states.append(slots[target_commit].mean(dim=1))
                labels.append(targets[:, 2] - CLASS_TOKEN_OFFSET)
                commit_reconstruction_errors.append(
                    (reconstructed[target_commit] - current[target_commit])
                    .square()
                    .mean(dim=(1, 2))
                )
                last_delta_reconstruction_errors.append(
                    (last_delta_reconstructed[target_commit] - current[target_commit])
                    .square()
                    .mean(dim=(1, 2))
                )
                selected_indices = target_commit.nonzero().flatten()
                class_slice = slice(CLASS_TOKEN_OFFSET, CLASS_TOKEN_OFFSET + 4)
                class_probabilities = normal_logits[:, 1, class_slice]
                class_probabilities = class_probabilities.softmax(dim=-1)
                caption_history[
                    selected_indices,
                    caption_counts[selected_indices],
                ] = class_probabilities
                last_class_probabilities = last_delta_logits[:, 1, class_slice].softmax(dim=-1)
                last_caption_history[
                    selected_indices,
                    caption_counts[selected_indices],
                ] = last_class_probabilities
                caption_counts[selected_indices] += 1

            reset = target_commit[:, None, None]
            anchor = torch.where(reset, current, anchor)
            slots = torch.where(reset, torch.zeros_like(slots), slots)
            load = torch.where(target_commit, torch.zeros_like(load), load)
            previous = current

        if not caption_counts.eq(2).all():
            raise ValueError("Synthetic final-memory task requires two captions per modality")
        final_full_states.append(full_embeddings[:, -1, modality_index].mean(dim=1))
        caption_histories.append(caption_history.flatten(start_dim=1))
        last_caption_histories.append(last_caption_history.flatten(start_dim=1))
        final_qa_targets.append(
            torch.tensor(
                [
                    int(dataset[index].final_qa_targets[modality_index])
                    for index in range(len(dataset))
                ]
            )
        )

    def scalar(values: list[Tensor]) -> float:
        return float(torch.cat([value.flatten() for value in values]).mean())

    return Collected(
        full_changes=torch.cat(full_changes),
        reconstructed_changes=torch.cat(reconstructed_changes),
        delta_states=torch.cat(delta_states),
        labels=torch.cat(labels),
        final_full_states=torch.cat(final_full_states),
        caption_histories=torch.cat(caption_histories),
        last_delta_caption_histories=torch.cat(last_caption_histories),
        final_qa_targets=torch.cat(final_qa_targets),
        reconstruction_mse=scalar(reconstruction_errors),
        step_reconstruction_mse=scalar(step_reconstruction_errors),
        commit_reconstruction_mse=scalar(commit_reconstruction_errors),
        last_delta_reconstruction_mse=scalar(last_delta_reconstruction_errors),
        anchor_mse=scalar(anchor_errors),
        shuffled_reconstruction_mse=scalar(shuffled_errors),
        caption_exact=scalar(caption_exact),
        last_delta_caption_exact=scalar(last_caption_exact),
        zero_delta_caption_exact=scalar(zero_caption_exact),
        shuffled_delta_caption_exact=scalar(shuffled_caption_exact),
        length_accuracy=scalar(length_correct),
        trigger_true=torch.cat(trigger_true),
        trigger_predicted=torch.cat(trigger_predicted),
    )


def _fit_probe(features: Tensor, labels: Tensor, seed: int) -> nn.Linear:
    _set_seed(seed)
    probe = nn.Linear(features.shape[-1], 4)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.05)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(probe(features), labels)
        loss.backward()
        optimizer.step()
    return probe


def _fit_qa_probe(features: Tensor, labels: Tensor, seed: int) -> nn.Module:
    _set_seed(seed)
    probe = nn.Sequential(
        nn.Linear(features.shape[-1], 32),
        nn.GELU(),
        nn.Linear(32, 3),
    )
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.03)
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(probe(features), labels)
        loss.backward()
        optimizer.step()
    return probe


@torch.no_grad()
def _accuracy(probe: nn.Module, features: Tensor, labels: Tensor) -> float:
    return float(probe(features).argmax(dim=-1).eq(labels).float().mean())


def _binary_metrics(targets: Tensor, predictions: Tensor) -> dict[str, float]:
    true_positive = int((targets & predictions).sum())
    false_positive = int((~targets & predictions).sum())
    false_negative = int((targets & ~predictions).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": float(targets.eq(predictions).float().mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def verify(config: SanityConfig, run_id: str | None = None) -> dict[str, Any]:
    if run_id is None:
        runs = sorted(config.training.run_root.glob("delta-sanity-*"))
        if not runs:
            raise FileNotFoundError("No sanity run is available")
        run_dir = runs[-1]
    else:
        run_dir = config.training.run_root / run_id
    checkpoint = _latest_checkpoint(run_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found under {run_dir}")

    model = DeltaCodecModel(config.model, config.modalities)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    train_data = SyntheticInterleavedDataset(config, config.training.examples, split_seed=10_000)
    validation_data = SyntheticInterleavedDataset(
        config,
        config.training.validation_examples,
        split_seed=20_000,
    )
    train = collect(model, train_data, config)
    validation = collect(model, validation_data, config)

    full_probe = _fit_probe(train.full_changes, train.labels, config.seed + 100)
    delta_probe = _fit_probe(train.delta_states, train.labels, config.seed + 200)
    final_full_probe = _fit_qa_probe(
        train.final_full_states,
        train.final_qa_targets,
        config.seed + 300,
    )
    caption_history_probe = _fit_qa_probe(
        train.caption_histories,
        train.final_qa_targets,
        config.seed + 400,
    )
    last_caption_history_probe = _fit_qa_probe(
        train.last_delta_caption_histories,
        train.final_qa_targets,
        config.seed + 450,
    )
    combined_probe = _fit_qa_probe(
        torch.cat((train.final_full_states, train.caption_histories), dim=-1),
        train.final_qa_targets,
        config.seed + 500,
    )
    trigger = _binary_metrics(validation.trigger_true, validation.trigger_predicted)
    metrics = {
        "full_tokens": config.model.embedding_tokens,
        "delta_tokens": config.model.delta_tokens,
        "token_compression_ratio": (
            config.model.embedding_tokens / config.model.delta_tokens
        ),
        "reconstruction_mse": validation.reconstruction_mse,
        "step_reconstruction_mse": validation.step_reconstruction_mse,
        "commit_reconstruction_mse": validation.commit_reconstruction_mse,
        "last_delta_only_reconstruction_mse": validation.last_delta_reconstruction_mse,
        "anchor_only_mse": validation.anchor_mse,
        "shuffled_delta_reconstruction_mse": validation.shuffled_reconstruction_mse,
        "caption_exact": validation.caption_exact,
        "last_delta_only_caption_exact": validation.last_delta_caption_exact,
        "zero_delta_caption_exact": validation.zero_delta_caption_exact,
        "shuffled_delta_caption_exact": validation.shuffled_delta_caption_exact,
        "length_accuracy": validation.length_accuracy,
        "trigger": trigger,
        "full_change_probe_accuracy": _accuracy(
            full_probe, validation.full_changes, validation.labels
        ),
        "reconstructed_change_probe_accuracy": _accuracy(
            full_probe, validation.reconstructed_changes, validation.labels
        ),
        "delta_state_probe_accuracy": _accuracy(
            delta_probe, validation.delta_states, validation.labels
        ),
        "final_full_only_qa_accuracy": _accuracy(
            final_full_probe,
            validation.final_full_states,
            validation.final_qa_targets,
        ),
        "caption_history_qa_accuracy": _accuracy(
            caption_history_probe,
            validation.caption_histories,
            validation.final_qa_targets,
        ),
        "last_delta_caption_history_qa_accuracy": _accuracy(
            last_caption_history_probe,
            validation.last_delta_caption_histories,
            validation.final_qa_targets,
        ),
        "full_plus_caption_qa_accuracy": _accuracy(
            combined_probe,
            torch.cat((validation.final_full_states, validation.caption_histories), dim=-1),
            validation.final_qa_targets,
        ),
    }
    checks = {
        "delta_token_bottleneck": metrics["token_compression_ratio"] >= 2.0,
        "reconstruction_beats_anchor": metrics["reconstruction_mse"] < metrics["anchor_only_mse"],
        "reconstruction_beats_shuffled": (
            metrics["reconstruction_mse"] < metrics["shuffled_delta_reconstruction_mse"]
        ),
        "accumulation_beats_last_delta_reconstruction": (
            metrics["commit_reconstruction_mse"]
            < 0.5 * metrics["last_delta_only_reconstruction_mse"]
        ),
        "caption_uses_delta": metrics["caption_exact"] > metrics["zero_delta_caption_exact"],
        "caption_rejects_shuffled_delta": (
            metrics["caption_exact"] > metrics["shuffled_delta_caption_exact"]
        ),
        "accumulation_beats_last_delta_caption": (
            metrics["caption_exact"] >= metrics["last_delta_only_caption_exact"] + 0.20
        ),
        "trigger_f1": trigger["f1"] >= 0.95,
        "length_accuracy": metrics["length_accuracy"] >= 0.95,
        "full_probe": metrics["full_change_probe_accuracy"] >= 0.95,
        "reconstructed_probe": metrics["reconstructed_change_probe_accuracy"] >= 0.80,
        "delta_probe": metrics["delta_state_probe_accuracy"] >= 0.80,
        "caption_improves_unseen_qa": (
            metrics["full_plus_caption_qa_accuracy"]
            >= metrics["final_full_only_qa_accuracy"] + 0.15
        ),
        "accumulated_caption_improves_unseen_qa": (
            metrics["caption_history_qa_accuracy"]
            >= metrics["last_delta_caption_history_qa_accuracy"] + 0.15
        ),
        "combined_unseen_qa": metrics["full_plus_caption_qa_accuracy"] >= 0.80,
    }
    report = {
        "run_id": run_dir.name,
        "checkpoint": str(checkpoint),
        "qa_contract": {
            "question": (
                "Is the second change class the same as, higher than, or lower than the first?"
            ),
            "answers": ["same", "higher", "lower"],
            "answer_appears_in_caption": False,
        },
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }
    _atomic_json(run_dir / "verification.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify DeltaOmni sanity representation utility")
    parser.add_argument("--config", type=Path, default=Path("configs/sanity.yaml"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    report = verify(load_config(args.config), args.run_id)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
