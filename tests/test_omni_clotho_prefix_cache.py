from pathlib import Path

from deltaomni.omni_clotho_prefix_cache import load_config, select_episodes


def test_clotho_prefix_config_preserves_six_to_fourteen_delta_updates() -> None:
    config = load_config(Path("configs/omni_clotho_prefix_cache.yaml"))

    assert config.minimum_blocks - 1 == 6
    assert config.maximum_blocks - 1 == 14
    assert config.train_count == 2048
    assert config.validation_count == 1037
    assert config.test_count == 1045


def test_clotho_prefix_selection_uses_source_disjoint_canonical_splits() -> None:
    config = load_config(Path("configs/omni_clotho_prefix_cache.yaml"))
    selected = select_episodes(config)
    groups = {
        split: {episode.source_group_id for episode in values} for split, values in selected.items()
    }

    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]
