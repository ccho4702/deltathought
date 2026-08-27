from dataclasses import dataclass
from pathlib import Path

import pytest

from deltaomni.run_integrity import (
    require_media_policy,
    resolved_input_signature,
    source_changes_from_porcelain,
)


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


def test_media_policy_requires_an_approved_resource() -> None:
    policy = Path("configs/nextqa_media_policy.yaml")

    with pytest.raises(RuntimeError, match="provenance gate"):
        require_media_policy({"approved": []}, "nextqa_annotations", policy)

    digest = require_media_policy(
        {"approved": ["nextqa_annotations"]},
        "nextqa_annotations",
        policy,
    )
    assert len(digest) == 64


def test_clean_source_check_ignores_only_generated_artifact_roots() -> None:
    status = " M outputs/reports/result.json\n M src/deltaomni/model.py\n?? temp/debug.txt\n"

    assert source_changes_from_porcelain(status) == ["src/deltaomni/model.py"]
