"""Validate lightweight public SpikeBudget repository artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from spikebudget.config import load_repro_config
from spikebudget.evidence import (
    contains_disallowed_external_weight_reference,
    contains_disallowed_public_label,
    contains_private_path,
    load_evidence_ledger,
)


REQUIRED_DOCS = (
    "README.md",
    "docs/compute_bound_evidence.md",
    "docs/technology_milestones.md",
    "docs/reproduce_3090.md",
    "docs/limitations.md",
)
PUBLIC_TEXT_GLOBS = (
    "README.md",
    "pyproject.toml",
    "docs/**/*.md",
    "docs/**/*.txt",
    "data/**/*.yaml",
    "configs/**/*.yaml",
    ".github/**/*.yaml",
    ".github/**/*.yml",
    "scripts/**/*.py",
    "spikebudget/**/*.py",
    "tests/**/*.py",
    "artifacts/**/*.md",
    "artifacts/**/*.tsv",
    "artifacts/**/*.csv",
    "artifacts/**/*.txt",
    "artifacts/**/*.json",
    "artifacts/**/*.log",
)
REQUIRED_COMPUTE_BOUND_EVIDENCE = (
    "0.098x",
    "34.14x",
    "128.75x",
    "6.78x",
    "70.7% SM / 0.08% DRAM",
    "22.7% SM / 51.3% DRAM",
    "real CUDA kernels and cudaEvent timing",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def validate(root: Path) -> list[str]:
    errors: list[str] = []

    for doc in REQUIRED_DOCS:
        path = root / doc
        if not path.exists():
            errors.append(f"missing required doc: {doc}")
        elif path.stat().st_size == 0:
            errors.append(f"empty required doc: {doc}")

    compute_bound_doc = root / "docs" / "compute_bound_evidence.md"
    if compute_bound_doc.exists():
        text = compute_bound_doc.read_text(encoding="utf-8")
        for required_text in REQUIRED_COMPUTE_BOUND_EVIDENCE:
            if required_text not in text:
                errors.append(f"missing compute-bound evidence text: {required_text}")

    config_paths = sorted((root / "configs").glob("*.yaml"))
    if not config_paths:
        errors.append("missing reproduction configs under configs/*.yaml")
    for path in config_paths:
        try:
            load_repro_config(path)
        except Exception as exc:  # noqa: BLE001 - report all validation failures.
            errors.append(str(exc))

    ledger_path = root / "data" / "technology_milestones.yaml"
    if not ledger_path.exists():
        errors.append("missing technology milestone ledger: data/technology_milestones.yaml")
    else:
        try:
            load_evidence_ledger(ledger_path, repo_root=root)
        except Exception as exc:  # noqa: BLE001 - report all validation failures.
            errors.append(str(exc))

    errors.extend(validate_public_path_hygiene(root))

    return errors


def validate_public_path_hygiene(root: Path) -> list[str]:
    """Reject private/local machine paths in public docs and lightweight artifacts."""

    errors: list[str] = []
    for path in iter_public_text_files(root):
        relative_path = path.relative_to(root)
        if contains_disallowed_public_label(str(relative_path)):
            errors.append(f"disallowed public label found in path: {relative_path}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if contains_private_path(text):
            errors.append(f"private path found in public file: {relative_path}")
        if contains_disallowed_public_label(text):
            errors.append(f"disallowed public label found in public file: {relative_path}")
        if contains_disallowed_external_weight_reference(text):
            errors.append(f"disallowed external-weight reference found in public file: {relative_path}")
    return errors


def iter_public_text_files(root: Path) -> list[Path]:
    files: set[Path] = set()
    for pattern in PUBLIC_TEXT_GLOBS:
        if any(ch in pattern for ch in "*?["):
            files.update(path for path in root.glob(pattern) if path.is_file())
        else:
            path = root / pattern
            if path.is_file():
                files.add(path)
    return sorted(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repo_root())
    args = parser.parse_args(argv)

    errors = validate(args.root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("SpikeBudget public repository validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
