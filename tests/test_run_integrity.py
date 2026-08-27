from dataclasses import dataclass
from pathlib import Path

import pytest

from deltaomni.run_integrity import require_verified_resource, resolved_input_signature


@dataclass(frozen=True)
class FixtureConfig:
    seed: int
    manifest: Path


def test_signature_changes_when_referenced_input_content_changes(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"revision": 1}', encoding="utf-8")
    config = FixtureConfig(seed=42, manifest=manifest)
    first = resolved_input_signature(config, {"manifest": manifest})

    manifest.write_text('{"revision": 2}', encoding="utf-8")
    second = resolved_input_signature(config, {"manifest": manifest})

    assert first != second


def test_verified_resource_requires_local_license_record(tmp_path: Path) -> None:
    report = {"approved": ["fixture_dataset"]}
    license_record = tmp_path / "accepted.json"

    with pytest.raises(RuntimeError, match="license record"):
        require_verified_resource(report, "fixture_dataset", license_record)

    license_record.write_text(
        '{"dataset":"Fixture","terms_url":"https://example.test/terms",'
        '"accepted_at_utc":"2026-08-28T00:00:00Z","accepted_by":"tester"}',
        encoding="utf-8",
    )
    assert len(require_verified_resource(report, "fixture_dataset", license_record)) == 64
