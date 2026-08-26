from pathlib import Path

from deltaomni.config import load_config
from deltaomni.types import Modality


def test_load_sanity_config() -> None:
    config = load_config(Path("configs/sanity.yaml"))

    assert config.modalities == (Modality.AUDIO, Modality.VIDEO)
    assert config.model.delta_tokens == 4
    assert config.model.embedding_tokens > config.model.delta_tokens
    assert config.training.device == "cpu"
    assert config.training.run_root.is_absolute()
