from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml


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
    return not source_changes_from_porcelain(status)


def source_changes_from_porcelain(status: str) -> list[str]:
    generated_prefixes = ("outputs/", "logs/", "intermediates/", "temp/")
    changed = []
    for line in status.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if not path.startswith(generated_prefixes):
            changed.append(path)
    return changed


def require_media_policy(
    provenance_report: dict[str, Any],
    resource_name: str,
    media_policy: Path,
) -> str:
    if resource_name not in provenance_report.get("approved", []):
        raise RuntimeError(f"Dataset failed provenance gate: {resource_name}")
    try:
        validate_media_policy(media_policy, resource_name)
    except (FileNotFoundError, ValueError) as error:
        raise RuntimeError(
            f"Dataset media policy is required before cache or training: {media_policy}"
        ) from error
    return sha256_file(media_policy)


def validate_media_policy(path: Path, resource_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid media policy YAML: {path}") from error
    if not isinstance(value, dict) or value.get("schema") != "deltaomni.media_policy.v1":
        raise ValueError(f"Unsupported media policy: {path}")
    if value.get("resource_name") != resource_name:
        raise ValueError(f"Media policy resource mismatch: {path}")
    sources = value.get("official_sources")
    if not isinstance(sources, dict) or len(sources) < 2:
        raise ValueError(f"Media policy official sources are incomplete: {path}")
    if any(
        not isinstance(url, str) or not url.startswith(("https://", "http://"))
        for url in sources.values()
    ):
        raise ValueError(f"Media policy contains an invalid official URL: {path}")
    policy = value.get("project_policy")
    required_policy = {
        "use_scope": "internal non-commercial research",
        "raw_media_redistribution": "prohibited",
        "recoverable_embedding_redistribution": "prohibited",
        "publication_requires_per_item_license_audit": True,
    }
    if not isinstance(policy, dict) or any(
        policy.get(key) != expected for key, expected in required_policy.items()
    ):
        raise ValueError(f"Media policy is weaker than the required project policy: {path}")
    return value
