from pathlib import Path

from deltaomni.data.preprocess_nextqa import load_config


def test_nextqa_canonical_preprocessor_is_complete_and_immutable() -> None:
    config = load_config(Path("configs/canonical/nextqa.yaml"))

    assert config.dataset == "nextqa"
    assert set(config.annotations) == {"train", "validation", "test"}
    assert "schema-v2" in config.dataset_revision
    assert config.chunk_seconds == 2.0
    assert config.output_root == Path("intermediates/canonical").resolve()
    assert config.cache_root != config.output_root
