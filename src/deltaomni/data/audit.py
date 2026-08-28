from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from deltaomni.provenance import audit as audit_provenance
from deltaomni.run_integrity import validate_media_policy
from deltaomni.train_sanity import _atomic_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _source_paths(
    raw_root: Path,
    source: dict[str, Any],
) -> tuple[list[Path], list[Path]]:
    if "annotation_files" in source:
        annotations = [_resolve(raw_root, value) for value in source["annotation_files"]]
    else:
        annotations_dir = raw_root / source["annotations_dir"]
        annotations = [annotations_dir / name for name in source["required_annotations"]]
    if "media_dirs" in source:
        media_dirs = [_resolve(raw_root, value) for value in source["media_dirs"]]
    else:
        media_dirs = [raw_root / source["media_dir"]]
    if not annotations or not media_dirs:
        raise ValueError("Data sources require annotation files and media directories")
    return annotations, media_dirs


def audit_source(
    project_root: Path,
    data_config: dict[str, Any],
    provenance_report: dict[str, Any],
    source_name: str,
) -> dict[str, Any]:
    source = data_config["sources"][source_name]
    resource_name = source["resource_name"]
    raw_root = _resolve(project_root, data_config["raw_root"])
    required, media_dirs = _source_paths(raw_root, source)
    missing_annotations = [str(path) for path in required if not path.is_file()]
    if "media_policy" in source:
        policy_kind = "media_policy"
        policy_record = _resolve(project_root, source["media_policy"])
        try:
            validate_media_policy(policy_record, resource_name)
            policy_valid = True
        except (FileNotFoundError, ValueError):
            policy_valid = False
    else:
        policy_kind = "existing_shared_read_only"
        policy_record = None
        policy_valid = True
    extensions = {extension.lower() for extension in source["media_extensions"]}
    media_files = sorted(
        {
            path
            for media_dir in media_dirs
            if media_dir.is_dir()
            for path in media_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        }
    )
    prerequisites = {
        "enabled": source.get("enabled") is True,
        "provenance_approved": resource_name in provenance_report["approved"],
        "usage_policy_valid": policy_valid,
        "annotations_complete": not missing_annotations,
        "media_present": bool(media_files),
    }
    status = "ready" if all(prerequisites.values()) else "blocked"
    return {
        "source": source_name,
        "resource_name": resource_name,
        "status": status,
        "prerequisites": prerequisites,
        "usage_policy_kind": policy_kind,
        "usage_policy_record": None if policy_record is None else str(policy_record),
        "missing_annotations": missing_annotations,
        "annotation_files": [
            {"path": str(path), "sha256": _sha256(path)} for path in required if path.is_file()
        ],
        "media_dirs": [str(path) for path in media_dirs],
        "media_count": len(media_files),
        "media_bytes": sum(path.stat().st_size for path in media_files),
    }


def audit_data(
    config_path: Path,
    provenance_path: Path,
    source_names: list[str] | None = None,
) -> dict[str, Any]:
    project_root = config_path.resolve().parent.parent
    data_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data_config, dict) or not isinstance(data_config.get("sources"), dict):
        raise ValueError("Data configuration must contain a sources mapping")
    provenance = audit_provenance(provenance_path)
    selected = source_names or list(data_config["sources"])
    unknown = sorted(set(selected) - set(data_config["sources"]))
    if unknown:
        raise ValueError(f"Unknown data sources: {', '.join(unknown)}")
    sources = {
        name: audit_source(project_root, data_config, provenance, name) for name in selected
    }
    return {
        "ready": sorted(name for name, result in sources.items() if result["status"] == "ready"),
        "blocked": sorted(
            name for name, result in sources.items() if result["status"] != "ready"
        ),
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit official real-data prerequisites")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--provenance", type=Path, default=Path("configs/provenance.yaml"))
    parser.add_argument("--source", action="append")
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/data_readiness.json"))
    parser.add_argument("--allow-not-ready", action="store_true")
    args = parser.parse_args()
    report = audit_data(args.config, args.provenance, args.source)
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if not report["blocked"] or args.allow_not_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
