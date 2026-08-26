from pathlib import Path

from transformers import Qwen2_5OmniThinkerForConditionalGeneration

from deltaomni.omni_backbones import load_omni_backbone_config
from deltaomni.provenance import audit, require_approved


def test_qwen_omni_backbone_is_pinned_and_required() -> None:
    config = load_omni_backbone_config(Path("configs/qwen2_5_omni.yaml"))
    provenance = audit(Path("configs/provenance.yaml"))

    assert config.model_id == "Qwen/Qwen2.5-Omni-7B"
    assert config.component == "thinker"
    assert config.revision == "ae9e1690543ffd5c0221dc27f79834d0294cba00"
    assert config.sample_rate == 16_000
    assert config.seconds_per_chunk == 2
    assert config.video.min_pixels == config.video.max_pixels == 224 * 224
    assert Qwen2_5OmniThinkerForConditionalGeneration is not None
    require_approved(provenance, [config.resource_name])
