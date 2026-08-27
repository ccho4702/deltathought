from pathlib import Path

from deltaomni.data.preprocess_clotho import (
    _deduplicate_sources,
    _read_records,
    load_config,
)


def test_clotho_config_uses_official_zenodo_v2_1_resources() -> None:
    config = load_config(Path("configs/canonical/clotho.yaml"))

    assert config.dataset == "clotho"
    assert config.resource_name == "clotho_v2_1"
    assert set(config.annotations) == {"train", "validation", "test"}
    assert "4783391" in config.dataset_revision
    assert config.license_record.name == "LICENSE"


def test_clotho_annotations_preserve_five_references_and_media_license() -> None:
    config = load_config(Path("configs/canonical/clotho.yaml"))
    records = _read_records(
        config.annotations["validation"],
        config.metadata["validation"],
    )

    assert len(records) == 1045
    assert {len(record.captions) for record in records} == {5}
    assert all(record.media_license.startswith("http") for record in records)


def test_clotho_source_deduplication_preserves_test_and_removes_all_overlap() -> None:
    config = load_config(Path("configs/canonical/clotho.yaml"))
    records = {
        split: _read_records(config.annotations[split], config.metadata[split])
        for split in config.annotations
    }
    kept, excluded = _deduplicate_sources(records)
    sources = {split: {record.sound_id for record in values} for split, values in kept.items()}

    assert len(kept["test"]) == len(records["test"])
    assert len(excluded["validation"]) > 0
    assert len(excluded["train"]) > 0
    assert not sources["train"] & sources["validation"]
    assert not sources["train"] & sources["test"]
    assert not sources["validation"] & sources["test"]
