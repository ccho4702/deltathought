from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from deltaomni.train_sanity import _atomic_json


@dataclass(frozen=True)
class ProvenancePolicy:
    minimum_github_stars: int
    minimum_huggingface_downloads: int
    minimum_paper_citations: int
    minimum_adoption_signals: int
    require_official_source: bool
    require_peer_review: bool
    require_verified_license: bool
    peer_review_exceptions: tuple[str, ...]


def _load(path: Path) -> tuple[dict[str, Any], ProvenancePolicy]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("resources"), dict):
        raise ValueError("Provenance configuration must contain a resources mapping")
    policy = raw["policy"]
    return raw, ProvenancePolicy(
        minimum_github_stars=int(policy["minimum_github_stars"]),
        minimum_huggingface_downloads=int(policy["minimum_huggingface_downloads"]),
        minimum_paper_citations=int(policy["minimum_paper_citations"]),
        minimum_adoption_signals=int(policy["minimum_adoption_signals"]),
        require_official_source=bool(policy["require_official_source"]),
        require_peer_review=bool(policy["require_peer_review"]),
        require_verified_license=bool(policy["require_verified_license"]),
        peer_review_exceptions=tuple(
            str(name) for name in policy.get("peer_review_exceptions", [])
        ),
    )


def _meets(value: Any, threshold: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= threshold


def audit_resource(name: str, resource: dict[str, Any], policy: ProvenancePolicy) -> dict[str, Any]:
    evidence = resource.get("evidence", {})
    signals = {
        "recognized_standard": resource.get("recognized_standard") is True,
        "benchmark_adoption": resource.get("benchmark_adoption") is True,
        "github_stars": _meets(evidence.get("github_stars"), policy.minimum_github_stars),
        "huggingface_downloads": _meets(
            evidence.get("huggingface_downloads"),
            policy.minimum_huggingface_downloads,
        ),
        "paper_citations": _meets(
            evidence.get("paper_citations"),
            policy.minimum_paper_citations,
        ),
    }
    peer_review_exception = (
        name in policy.peer_review_exceptions
        and resource.get("required_by_research_question") is True
    )
    prerequisites = {
        "official_source": (
            not policy.require_official_source or resource.get("official_source") is True
        ),
        "peer_reviewed": (
            not policy.require_peer_review
            or resource.get("peer_reviewed") is True
            or peer_review_exception
        ),
        "license_verified": (
            not policy.require_verified_license or resource.get("license_status") == "verified"
        ),
    }
    passed_signals = sum(signals.values())
    approved = all(prerequisites.values()) and passed_signals >= policy.minimum_adoption_signals
    return {
        "name": name,
        "kind": resource.get("kind"),
        "role": resource.get("role"),
        "approved": approved,
        "prerequisites": prerequisites,
        "adoption_signals": signals,
        "passed_adoption_signals": passed_signals,
        "required_adoption_signals": policy.minimum_adoption_signals,
        "peer_review_exception": peer_review_exception,
        "restriction": resource.get("restriction"),
    }


def audit(path: Path) -> dict[str, Any]:
    raw, policy = _load(path)
    results = {
        name: audit_resource(name, resource, policy)
        for name, resource in raw["resources"].items()
    }
    return {
        "checked_at": str(raw["policy"]["checked_at"]),
        "approved": sorted(name for name, result in results.items() if result["approved"]),
        "blocked": sorted(name for name, result in results.items() if not result["approved"]),
        "resources": results,
    }


def require_approved(report: dict[str, Any], names: list[str]) -> None:
    blocked = [name for name in names if name not in report["approved"]]
    if blocked:
        raise ValueError(f"Resources failed provenance gate: {', '.join(blocked)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit model and dataset provenance gates")
    parser.add_argument("--config", type=Path, default=Path("configs/provenance.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/provenance.json"))
    parser.add_argument("--require", nargs="*", default=[])
    args = parser.parse_args()
    report = audit(args.config)
    _atomic_json(args.output, report)
    if args.require:
        require_approved(report, args.require)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
