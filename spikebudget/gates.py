"""Gate ledger loading and validation."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import yaml


REQUIRED_GATE_KEYS = (
    "id",
    "title",
    "status",
    "result",
    "interpretation",
    "artifact_status",
    "artifacts",
)
ARTIFACT_STATUSES = {"included", "summary_only", "external_missing"}
GATE_STATUSES = {"positive", "negative", "caution", "evidence", "inconclusive"}
PRIVATE_PATH_RE = re.compile(
    r"(/tmp/codex_snn_capacity|/Users/[^\s`'\"<>]+|file://|\bomni-[a-z0-9-]+\b|yizhou)",
    re.IGNORECASE,
)


def load_gate_ledger(path: str | Path, repo_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Load and validate the gate ledger YAML file."""

    ledger_path = Path(path)
    with ledger_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    root = Path(repo_root) if repo_root is not None else ledger_path.resolve().parents[1]
    return validate_gate_ledger(data, repo_root=root, source=str(ledger_path))


def validate_gate_ledger(data: Any, repo_root: str | Path, source: str) -> list[dict[str, Any]]:
    """Validate gate records and included artifact paths."""

    if not isinstance(data, list):
        raise ValueError(f"{source}: gate ledger must be a list")

    root = Path(repo_root)
    seen: set[str] = set()
    has_inconclusive = False
    for gate in data:
        if not isinstance(gate, dict):
            raise ValueError(f"{source}: each gate must be a mapping")

        gate_id = str(gate.get("id", "<missing>"))
        for key in REQUIRED_GATE_KEYS:
            if key not in gate:
                raise ValueError(f"{source}: gate {gate_id} missing required key: {key}")

        if gate["id"] in seen:
            raise ValueError(f"{source}: duplicate gate id: {gate['id']}")
        seen.add(gate["id"])

        if gate["status"] not in GATE_STATUSES:
            raise ValueError(f"{source}: gate {gate['id']} has invalid status: {gate['status']}")
        if gate["status"] == "inconclusive":
            has_inconclusive = True

        if gate["artifact_status"] not in ARTIFACT_STATUSES:
            raise ValueError(f"{source}: gate {gate['id']} has invalid artifact_status")

        if not isinstance(gate["artifacts"], list):
            raise ValueError(f"{source}: gate {gate['id']} artifacts must be a list")

        artifact_seen: set[str] = set()
        for artifact in gate["artifacts"]:
            if not isinstance(artifact, str):
                raise ValueError(f"{source}: gate {gate['id']} artifact must be a string")
            if artifact in artifact_seen:
                raise ValueError(f"{source}: gate {gate['id']} has duplicate artifact: {artifact}")
            artifact_seen.add(artifact)
            if Path(artifact).is_absolute() or contains_private_path(artifact):
                raise ValueError(f"{source}: gate {gate['id']} has unsafe artifact path: {artifact}")

        if gate["artifact_status"] == "included":
            for artifact in gate["artifacts"]:
                if not (root / artifact).exists():
                    raise ValueError(f"{source}: gate {gate['id']} missing artifact: {artifact}")

    if not has_inconclusive:
        raise ValueError(f"{source}: gate ledger must include at least one inconclusive gate")

    return data


def contains_private_path(text: str) -> bool:
    """Return true when text contains a local/private path or host marker."""

    return PRIVATE_PATH_RE.search(text) is not None
