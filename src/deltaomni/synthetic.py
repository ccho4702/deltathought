from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from deltaomni.config import SanityConfig

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
MODALITY_TOKEN_IDS = {"audio": 3, "video": 4, "image": 11}
CLASS_TOKEN_OFFSET = 5
DETAIL_TOKEN_ID = 9


@dataclass(frozen=True)
class SyntheticExample:
    full_embeddings: Tensor
    commit_targets: Tensor
    caption_targets: Tensor
    caption_lengths: Tensor
    final_qa_targets: Tensor


class SyntheticInterleavedDataset(Dataset[SyntheticExample]):
    """Deterministic embedding streams with independent modality boundaries."""

    def __init__(self, config: SanityConfig, count: int, *, split_seed: int) -> None:
        self.config = config
        self.count = count
        self.split_seed = split_seed
        prototype_generator = torch.Generator().manual_seed(config.seed)
        component_vectors = torch.randn(
            4,
            config.model.embedding_dim,
            generator=prototype_generator,
        )
        component_vectors = torch.nn.functional.normalize(component_vectors, dim=-1) * 0.55
        self.first_component_vectors = component_vectors[:2]
        self.second_component_vectors = component_vectors[2:]
        self.token_weights = torch.linspace(0.65, 1.25, config.model.embedding_tokens)

    def __len__(self) -> int:
        return self.count

    def _boundaries(self, index: int, modality_index: int) -> tuple[int, int]:
        patterns = (
            ((2, 5), (3, 5)),
            ((3, 6), (2, 6)),
            ((2, 6), (4, 6)),
        )
        return patterns[index % len(patterns)][modality_index]

    def _caption(self, modality: str, class_index: int) -> tuple[Tensor, int]:
        tokens = [BOS_TOKEN_ID, MODALITY_TOKEN_IDS[modality], CLASS_TOKEN_OFFSET + class_index]
        if class_index % 2 == 0:
            tokens.append(DETAIL_TOKEN_ID)
        tokens.append(EOS_TOKEN_ID)
        padded = torch.full(
            (self.config.model.max_caption_length,),
            PAD_TOKEN_ID,
            dtype=torch.long,
        )
        padded[: len(tokens)] = torch.tensor(tokens)
        return padded, len(tokens)

    def __getitem__(self, index: int) -> SyntheticExample:
        generator = torch.Generator().manual_seed(self.split_seed + index)
        time_steps = self.config.training.sequence_steps
        modality_count = len(self.config.modalities)
        full = torch.zeros(
            time_steps,
            modality_count,
            self.config.model.embedding_tokens,
            self.config.model.embedding_dim,
        )
        commits = torch.zeros(time_steps, modality_count, dtype=torch.bool)
        captions = torch.zeros(
            time_steps,
            modality_count,
            self.config.model.max_caption_length,
            dtype=torch.long,
        )
        lengths = torch.zeros(time_steps, modality_count, dtype=torch.long)
        final_qa_targets = torch.zeros(modality_count, dtype=torch.long)

        for modality_index, modality in enumerate(self.config.modalities):
            base = torch.randn(
                self.config.model.embedding_tokens,
                self.config.model.embedding_dim,
                generator=generator,
            ) * 0.08
            base = base + modality_index * 0.25
            full[0, modality_index] = base
            previous_boundary = 0
            current = base.clone()
            boundaries = self._boundaries(index, modality_index)
            first_class = 0

            for segment_index, boundary in enumerate(boundaries):
                if segment_index == 0:
                    class_index = (index + modality_index) % 4
                    first_class = class_index
                else:
                    class_index = (index // 4 + 2 * modality_index) % 4
                    final_qa_targets[modality_index] = (
                        0 if class_index == first_class else 1 if class_index > first_class else 2
                    )
                for time_index in range(previous_boundary + 1, boundary + 1):
                    increment = torch.zeros_like(current)
                    if time_index == previous_boundary + 1:
                        first_bit = class_index // 2
                        increment = self.token_weights[:, None] * self.first_component_vectors[
                            first_bit
                        ][None, :]
                    if time_index == boundary:
                        second_bit = class_index % 2
                        increment = increment + self.token_weights[
                            :, None
                        ] * self.second_component_vectors[second_bit][None, :]
                    current = current + increment
                    full[time_index, modality_index] = current
                commits[boundary, modality_index] = True
                caption, length = self._caption(modality.value, class_index)
                captions[boundary, modality_index] = caption
                lengths[boundary, modality_index] = length
                previous_boundary = boundary

            for time_index in range(previous_boundary + 1, time_steps):
                full[time_index, modality_index] = current

        return SyntheticExample(full, commits, captions, lengths, final_qa_targets)


def collate_examples(examples: list[SyntheticExample]) -> dict[str, Tensor]:
    return {
        "full_embeddings": torch.stack([example.full_embeddings for example in examples]),
        "commit_targets": torch.stack([example.commit_targets for example in examples]),
        "caption_targets": torch.stack([example.caption_targets for example in examples]),
        "caption_lengths": torch.stack([example.caption_lengths for example in examples]),
    }


def token_text() -> dict[int, str]:
    return {
        BOS_TOKEN_ID: "<BOS>",
        EOS_TOKEN_ID: "<EOS>",
        MODALITY_TOKEN_IDS["audio"]: "audio",
        MODALITY_TOKEN_IDS["video"]: "video",
        CLASS_TOKEN_OFFSET: "change-zero",
        CLASS_TOKEN_OFFSET + 1: "change-one",
        CLASS_TOKEN_OFFSET + 2: "change-two",
        CLASS_TOKEN_OFFSET + 3: "change-three",
        DETAIL_TOKEN_ID: "detailed",
    }
