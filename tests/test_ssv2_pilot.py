from pathlib import Path

from deltaomni.ssv2_pilot import load_pilot_config, select_records


def test_ssv2_pilot_config_uses_existing_shared_copy() -> None:
    config = load_pilot_config(Path("configs/ssv2_pilot.yaml"))

    assert config.access_mode == "read_only_existing_shared_copy"
    assert config.media_dir == Path(
        "/mnt/nfs_shared_data/dataset/ssv2/20bn-something-something-v2"
    )
    assert config.model.embedding_tokens > config.model.delta_tokens
    assert config.train_per_class > config.validation_per_class
    assert all("[something]" in template for template in config.classes)
    assert len(config.caption.targets) == len(config.classes)


def test_ssv2_selection_is_balanced_and_deterministic() -> None:
    records = [
        {"id": f"{label}-{index}", "template": label, "label": f"caption {index}"}
        for label in ("up", "down")
        for index in range(5)
    ]

    first = select_records(records, ("up", "down"), 2, seed=42)
    second = select_records(records, ("up", "down"), 2, seed=42)

    assert first == second
    assert [record["class_index"] for record in first] == [0, 0, 1, 1]
    assert len({record["id"] for record in first}) == 4
