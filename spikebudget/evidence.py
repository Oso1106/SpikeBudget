"""Technology milestone evidence loading and validation."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import yaml


REQUIRED_EVIDENCE_KEYS = (
    "id",
    "title",
    "status",
    "result",
    "interpretation",
    "artifact_status",
    "artifacts",
)
ARTIFACT_STATUSES = {"included", "summary_only", "external_missing"}
EVIDENCE_STATUSES = {"positive", "negative", "caution", "evidence", "inconclusive"}
PRIVATE_PATH_RE = re.compile(
    r"(/tmp/codex_snn_capacity|/Users/[^\s`'\"<>]+|file://|\bomni-[a-z0-9-]+\b|yizhou)",
    re.IGNORECASE,
)
_DISALLOWED_BASELINE_LABELS = ("spike" + "llm", "w4" + "a4")
_NUMERIC_MILESTONE_PREFIX = "ga" + "te"
DISALLOWED_PUBLIC_LABEL_RE = re.compile(
    "|".join(
        (
            *_DISALLOWED_BASELINE_LABELS,
            rf"\b{_NUMERIC_MILESTONE_PREFIX}\s*[-_]*\d+",
            rf"\b{_NUMERIC_MILESTONE_PREFIX}[-_]*\d+",
            r"\bG\d+(?:[-_]\d+)?\b",
        )
    ),
    re.IGNORECASE,
)


def load_evidence_ledger(path: str | Path, repo_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Load and validate the technology milestone ledger YAML file."""

    ledger_path = Path(path)
    with ledger_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    root = Path(repo_root) if repo_root is not None else ledger_path.resolve().parents[1]
    return validate_evidence_ledger(data, repo_root=root, source=str(ledger_path))


def validate_evidence_ledger(data: Any, repo_root: str | Path, source: str) -> list[dict[str, Any]]:
    """Validate technology milestone records and included artifact paths."""

    if not isinstance(data, list):
        raise ValueError(f"{source}: evidence ledger must be a list")

    root = Path(repo_root)
    seen: set[str] = set()
    has_inconclusive = False
    for record in data:
        if not isinstance(record, dict):
            raise ValueError(f"{source}: each evidence record must be a mapping")

        record_id = str(record.get("id", "<missing>"))
        for key in REQUIRED_EVIDENCE_KEYS:
            if key not in record:
                raise ValueError(f"{source}: evidence record {record_id} missing required key: {key}")

        if contains_disallowed_public_label(record_id):
            raise ValueError(f"{source}: evidence record has disallowed public label: {record_id}")

        if record["id"] in seen:
            raise ValueError(f"{source}: duplicate evidence id: {record['id']}")
        seen.add(record["id"])

        for key in ("title", "result", "interpretation"):
            if contains_disallowed_public_label(str(record[key])):
                raise ValueError(f"{source}: evidence record {record['id']} has disallowed label in {key}")

        if record["status"] not in EVIDENCE_STATUSES:
            raise ValueError(f"{source}: evidence record {record['id']} has invalid status: {record['status']}")
        if record["status"] == "inconclusive":
            has_inconclusive = True

        if record["artifact_status"] not in ARTIFACT_STATUSES:
            raise ValueError(f"{source}: evidence record {record['id']} has invalid artifact_status")

        if not isinstance(record["artifacts"], list):
            raise ValueError(f"{source}: evidence record {record['id']} artifacts must be a list")

        artifact_seen: set[str] = set()
        for artifact in record["artifacts"]:
            if not isinstance(artifact, str):
                raise ValueError(f"{source}: evidence record {record['id']} artifact must be a string")
            if artifact in artifact_seen:
                raise ValueError(f"{source}: evidence record {record['id']} has duplicate artifact: {artifact}")
            artifact_seen.add(artifact)
            if (
                Path(artifact).is_absolute()
                or contains_private_path(artifact)
                or contains_disallowed_public_label(artifact)
            ):
                raise ValueError(f"{source}: evidence record {record['id']} has unsafe artifact path: {artifact}")

        if record["artifact_status"] == "included":
            for artifact in record["artifacts"]:
                if not (root / artifact).exists():
                    raise ValueError(f"{source}: evidence record {record['id']} missing artifact: {artifact}")

    if not has_inconclusive:
        raise ValueError(f"{source}: evidence ledger must include at least one inconclusive record")

    return data


def contains_private_path(text: str) -> bool:
    """Return true when text contains a local/private path or host marker."""

    return PRIVATE_PATH_RE.search(text) is not None


def contains_disallowed_public_label(text: str) -> bool:
    """Return true when public text uses banned baseline names or numeric milestone labels."""

    return DISALLOWED_PUBLIC_LABEL_RE.search(text) is not None
