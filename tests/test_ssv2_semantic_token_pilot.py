import json
from pathlib import Path

import pytest

from deltaomni.model import ModalityDeltaCodec
from deltaomni.semantic_tokens import SemanticTokenBottleneck
from deltaomni.ssv2_pilot import load_pilot_config
from deltaomni.ssv2_semantic_token_pilot import (
    SemanticTokenModel,
    _assert_evaluation_checkpoint_compatible,
    load_config,
)


def test_a6000_semantic_token_pilot_is_scaled_and_resumable() -> None:
    config = load_config(Path("configs/ssv2_semantic_token_selected_a6000.yaml"))
    source = load_pilot_config(config.ssv2_config)

    assert config.runtime.device == "cuda"
    assert config.initialization == "semantic_checkpoint"
    assert config.runtime.backend == "nccl"
    assert config.runtime.precision == "bfloat16"
    assert config.runtime.nccl_compatibility_mode
    assert config.runtime.per_device_batch_size >= 16
    assert config.resume == "auto"
    assert config.shuffle_repeats >= 8
    assert config.evaluation_split == "validation"
    assert config.minimum_effective_codes >= 3
    assert config.token.usage_entropy_weight == 0.05
    assert config.token.token_count < source.model.delta_tokens
    assert config.token.codebook_size >= len(source.classes)
    assert source.train_per_class >= 512
    assert source.validation_per_class >= 64
    assert source.test_per_class >= 64
    assert source.frames_per_clip >= 8


def test_semantic_token_model_freezes_inactive_caption_and_policy_modules() -> None:
    source = load_pilot_config(Path("configs/ssv2_semantic_scaled.yaml"))
    model = SemanticTokenModel(
        ModalityDeltaCodec(source.model),
        SemanticTokenBottleneck(
            input_dim=source.model.embedding_dim,
            hidden_dim=32,
            token_count=1,
            codebook_size=4,
            classes=4,
            num_heads=4,
        ),
    )

    assert not any(parameter.requires_grad for parameter in model.codec.policy.parameters())
    assert not any(
        parameter.requires_grad for parameter in model.codec.caption_decoder.parameters()
    )


def test_layout_configs_preserve_fixed_state_sizes() -> None:
    balanced = load_config(Path("configs/ssv2_semantic_token_layout17_a6000.yaml"))
    fidelity = load_config(Path("configs/ssv2_semantic_token_layout65_a6000.yaml"))
    balanced_source = load_pilot_config(balanced.ssv2_config)
    fidelity_source = load_pilot_config(fidelity.ssv2_config)

    assert balanced.initialization == fidelity.initialization == "random"
    assert balanced_source.model.delta_tokens == 17
    assert fidelity_source.model.delta_tokens == 65
    assert balanced_source.frames_per_clip == fidelity_source.frames_per_clip == 8


def test_longer_layout_keeps_65_slots_across_fifteen_updates() -> None:
    config = load_config(Path("configs/ssv2_semantic_token_layout65_16frames_a6000.yaml"))
    source = load_pilot_config(config.ssv2_config)

    assert source.frames_per_clip - 1 == 15
    assert source.model.delta_tokens == 65


def test_evaluation_only_allows_only_the_split_to_change() -> None:
    source = json.dumps({"seed": 42, "evaluation_split": "validation", "max_steps": 800})
    test = json.dumps({"seed": 42, "evaluation_split": "test", "max_steps": 800})

    _assert_evaluation_checkpoint_compatible(source, test)

    incompatible = json.dumps(
        {"seed": 314, "evaluation_split": "test", "max_steps": 800}
    )
    with pytest.raises(ValueError, match="training configuration"):
        _assert_evaluation_checkpoint_compatible(source, incompatible)
