"""Config loading and validation for public SpikeBudget reproduction recipes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL_KEYS = (
    "name",
    "evidence_lane",
    "hardware",
    "model",
    "optimizer",
    "data",
    "run",
    "claim_scope",
)


def load_repro_config(path: str | Path) -> dict[str, Any]:
    """Load and validate one reproduction config YAML file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return validate_repro_config(data, source=str(config_path))


def validate_repro_config(data: Any, source: str) -> dict[str, Any]:
    """Validate the public config shape and return the config."""

    if not isinstance(data, dict):
        raise ValueError(f"{source}: config must be a mapping")

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in data:
            raise ValueError(f"{source}: missing required key: {key}")

    for key in ("name", "evidence_lane"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"{source}: {key} must be a non-empty string")

    for key in ("hardware", "model", "optimizer", "data", "run", "claim_scope"):
        if not isinstance(data[key], dict):
            raise ValueError(f"{source}: {key} must be a mapping")

    if not isinstance(data["hardware"].get("gpu_class"), str) or not data["hardware"]["gpu_class"].strip():
        raise ValueError(f"{source}: hardware.gpu_class is required")
    require_positive_number(data["hardware"], "min_vram_gb", "hardware.min_vram_gb", source, required=False)

    require_positive_number(data["model"], "parameters", "model.parameters", source, required=True)
    require_positive_number(data["model"], "timesteps", "model.timesteps", source, required=False)
    require_positive_number(data["model"], "sequence_length", "model.sequence_length", source, required=False)

    if not isinstance(data["optimizer"].get("name"), str) or not data["optimizer"]["name"].strip():
        raise ValueError(f"{source}: optimizer.name is required")
    require_positive_number(data["optimizer"], "lr", "optimizer.lr", source, required=False)
    if "lr_grid" in data["optimizer"]:
        require_positive_sequence(data["optimizer"]["lr_grid"], f"{source}: optimizer.lr_grid")

    if not isinstance(data["data"].get("source"), str) or not data["data"]["source"].strip():
        raise ValueError(f"{source}: data.source is required")

    require_positive_run_field(data["run"], "max_steps", source, required=True)
    require_positive_number(data["run"], "batch", "run.batch", source, required=False)
    require_positive_number(data["run"], "eval_iters", "run.eval_iters", source, required=False)
    require_positive_number(data["run"], "eval_every", "run.eval_every", source, required=False)

    if data["claim_scope"].get("scratch_only") is not True:
        raise ValueError(f"{source}: claim_scope.scratch_only must be true")

    if data["claim_scope"].get("eval_energy_win") is not False:
        raise ValueError(f"{source}: claim_scope.eval_energy_win must be false")

    return data


def require_positive_number(
    mapping: dict[str, Any],
    key: str,
    label: str,
    source: str,
    required: bool,
) -> None:
    if key not in mapping:
        if required:
            raise ValueError(f"{source}: {label} is required")
        return
    value = mapping[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{source}: {label} must be positive")


def require_positive_run_field(mapping: dict[str, Any], key: str, source: str, required: bool) -> None:
    if key not in mapping:
        if required:
            raise ValueError(f"{source}: run.{key} is required")
        return
    value = mapping[key]
    if isinstance(value, list):
        require_positive_sequence(value, f"{source}: run.{key}")
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{source}: run.{key} values must be positive")


def require_positive_sequence(values: Any, label: str) -> None:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a non-empty list")
    for value in values:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{label} values must be positive")
