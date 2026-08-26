from pathlib import Path

import pytest

from deltaomni.provenance import audit, require_approved


def test_provenance_gate_approves_established_models_and_blocks_watchlists() -> None:
    report = audit(Path("configs/provenance.yaml"))

    assert {
        "dinov2",
        "clap",
        "qwen2_5_0_5b_instruct",
        "qwen2_5_omni_7b",
        "something_something_v2",
        "nextqa_annotations",
    } <= set(report["approved"])
    assert report["resources"]["qwen2_5_omni_7b"]["peer_review_exception"]
    assert {"kinetics_geb_plus", "clevr_change", "desed"} <= set(report["blocked"])


def test_require_approved_rejects_unverified_resource() -> None:
    report = audit(Path("configs/provenance.yaml"))

    require_approved(report, ["dinov2", "clap"])
    with pytest.raises(ValueError, match="kinetics_geb_plus"):
        require_approved(report, ["kinetics_geb_plus"])
