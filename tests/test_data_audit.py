import hashlib
import json
from pathlib import Path

from deltaomni.data.audit import audit_source
from deltaomni.provenance import audit


def _config(project_root: Path, enabled: bool) -> dict:
    return {
        "raw_root": "inputs/data/raw",
        "sources": {
            "video": {
                "enabled": enabled,
                "resource_name": "something_something_v2",
                "annotations_dir": "video/annotations",
                "media_dir": "video/media",
                "required_annotations": ["train.json"],
                "media_extensions": [".webm"],
                "license_record": "inputs/licenses/video.accepted.json",
            }
        },
    }


def test_data_audit_stays_blocked_without_user_license_record(tmp_path: Path) -> None:
    report = audit_source(
        tmp_path,
        _config(tmp_path, enabled=True),
        audit(Path("configs/provenance.yaml")),
        "video",
    )

    assert report["status"] == "blocked"
    assert not report["prerequisites"]["license_record_present"]
    assert not report["prerequisites"]["annotations_complete"]
    assert not report["prerequisites"]["media_present"]


def test_data_audit_hashes_ready_official_files(tmp_path: Path) -> None:
    annotation = tmp_path / "inputs/data/raw/video/annotations/train.json"
    media = tmp_path / "inputs/data/raw/video/media/1.webm"
    license_record = tmp_path / "inputs/licenses/video.accepted.json"
    annotation.parent.mkdir(parents=True)
    media.parent.mkdir(parents=True)
    license_record.parent.mkdir(parents=True)
    annotation.write_text("[]", encoding="utf-8")
    media.write_bytes(b"fixture")
    license_record.write_text(
        json.dumps(
            {
                "dataset": "Something-Something V2",
                "terms_url": "https://official.example/terms",
                "accepted_at_utc": "2026-08-26T00:00:00Z",
                "accepted_by": "test-user",
            }
        ),
        encoding="utf-8",
    )

    report = audit_source(
        tmp_path,
        _config(tmp_path, enabled=True),
        audit(Path("configs/provenance.yaml")),
        "video",
    )

    assert report["status"] == "ready"
    assert report["annotation_files"][0]["sha256"] == hashlib.sha256(b"[]").hexdigest()
    assert report["media_count"] == 1
