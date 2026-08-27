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
    assert not report["prerequisites"]["usage_authorization_valid"]
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


def test_data_audit_accepts_explicit_shared_paths(tmp_path: Path) -> None:
    annotation = tmp_path / "shared/annotations/train.json"
    first_media = tmp_path / "shared/media-a"
    second_media = tmp_path / "shared/media-b"
    license_record = tmp_path / "inputs/licenses/video.accepted.json"
    annotation.parent.mkdir(parents=True)
    first_media.mkdir(parents=True)
    second_media.mkdir(parents=True)
    license_record.parent.mkdir(parents=True)
    annotation.write_text("[]", encoding="utf-8")
    (first_media / "one.webm").write_bytes(b"one")
    (second_media / "two.webm").write_bytes(b"two")
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
    config = {
        "raw_root": str(tmp_path / "unused"),
        "sources": {
            "video": {
                "enabled": True,
                "resource_name": "something_something_v2",
                "annotation_files": [str(annotation)],
                "media_dirs": [str(first_media), str(second_media)],
                "media_extensions": [".webm"],
                "license_record": str(license_record),
            }
        },
    }

    report = audit_source(
        tmp_path,
        config,
        audit(Path("configs/provenance.yaml")),
        "video",
    )

    assert report["status"] == "ready"
    assert report["media_count"] == 2


def test_data_audit_accepts_versioned_media_policy(tmp_path: Path) -> None:
    annotation = tmp_path / "nextqa/train.csv"
    media_dir = tmp_path / "nextqa/videos"
    annotation.parent.mkdir(parents=True)
    media_dir.mkdir(parents=True)
    annotation.write_text("header\n", encoding="utf-8")
    (media_dir / "one.mp4").write_bytes(b"fixture")
    config = {
        "raw_root": str(tmp_path),
        "sources": {
            "nextqa": {
                "enabled": True,
                "resource_name": "nextqa_annotations",
                "annotation_files": [str(annotation)],
                "media_dirs": [str(media_dir)],
                "media_extensions": [".mp4"],
                "media_policy": str(Path("configs/nextqa_media_policy.yaml").resolve()),
            }
        },
    }

    report = audit_source(
        tmp_path,
        config,
        audit(Path("configs/provenance.yaml")),
        "nextqa",
    )

    assert report["status"] == "ready"
    assert report["authorization_kind"] == "media_policy"
