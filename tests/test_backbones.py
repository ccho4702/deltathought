from pathlib import Path

from deltaomni.backbones import load_backbone_config
from deltaomni.provenance import audit, require_approved


def test_backbone_specs_are_pinned_and_provenance_approved() -> None:
    config = load_backbone_config(Path("configs/backbones.yaml"))
    provenance = audit(Path("configs/provenance.yaml"))

    assert len(config.video.revision) == 40
    assert len(config.audio.revision) == 40
    assert config.audio.sample_rate == 48_000
    assert config.video.delta_tokens == 8
    assert config.audio.delta_tokens == 1
    assert config.language.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config.language_smoke_anchor_tokens == 16
    assert config.language_smoke_steps == 8
    require_approved(provenance, [config.video.resource_name, config.audio.resource_name])
