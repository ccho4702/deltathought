from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_input_signature(
    config: Any,
    input_files: dict[str, Path],
    *,
    extra: dict[str, Any] | None = None,
) -> str:
    if not is_dataclass(config):
        raise TypeError("Run signatures require a dataclass configuration")
    missing = [name for name, path in input_files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing signed inputs: {', '.join(sorted(missing))}")
    payload = {
        "config": asdict(config),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(input_files.items())
        },
        "extra": extra or {},
    }
    return json.dumps(payload, sort_keys=True, default=str)


def git_revision(project_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_worktree_is_clean(project_root: Path) -> bool:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return not status.strip()


def require_verified_resource(
    provenance_report: dict[str, Any],
    resource_name: str,
    license_record: Path,
) -> str:
    if resource_name not in provenance_report.get("approved", []):
        raise RuntimeError(f"Dataset failed provenance gate: {resource_name}")
    try:
        validate_license_record(license_record)
    except (FileNotFoundError, ValueError) as error:
        raise RuntimeError(
            f"Dataset media license record is required before cache or training: {license_record}"
        ) from error
    return sha256_file(license_record)


def validate_license_record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid license record JSON: {path}") from error
    required = ("dataset", "terms_url", "accepted_at_utc", "accepted_by")
    if not isinstance(value, dict) or any(
        not isinstance(value.get(key), str) or not value[key].strip() for key in required
    ):
        raise ValueError(f"License record is missing required fields: {path}")
    if not value["terms_url"].startswith(("https://", "http://")):
        raise ValueError(f"License record terms_url is not an HTTP URL: {path}")
    try:
        datetime.fromisoformat(value["accepted_at_utc"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"License record timestamp is not ISO-8601: {path}") from error
    return {key: value[key] for key in required}
