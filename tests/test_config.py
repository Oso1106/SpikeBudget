from pathlib import Path

import pytest

from spikebudget.config import load_repro_config, validate_repro_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_configs_have_required_public_fields():
    for path in sorted((REPO_ROOT / "configs").glob("*.yaml")):
        config = load_repro_config(path)
        assert config["name"]
        assert config["gate"]
        assert config["hardware"]["gpu_class"]
        assert config["claim_scope"]["quality_parity"] is False


def test_config_validation_reports_missing_top_level_key():
    with pytest.raises(ValueError, match="missing required key: optimizer"):
        validate_repro_config(
            {
                "name": "bad",
                "gate": "GateX",
                "hardware": {},
                "model": {},
                "data": {},
                "run": {},
                "claim_scope": {"quality_parity": False},
            },
            source="inline",
        )


def test_config_validation_rejects_quality_parity_claim():
    with pytest.raises(ValueError, match="quality_parity must be false"):
        validate_repro_config(
            {
                "name": "bad",
                "gate": "GateX",
                "hardware": {"gpu_class": "RTX 3090"},
                "model": {"parameters": 1},
                "optimizer": {"name": "adamw"},
                "data": {"source": "wikitext2"},
                "run": {"max_steps": 1},
                "claim_scope": {"quality_parity": True},
            },
            source="inline",
        )


def test_config_validation_rejects_empty_name():
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        validate_repro_config(
            {
                "name": "",
                "gate": "GateX",
                "hardware": {"gpu_class": "RTX 3090"},
                "model": {"parameters": 1},
                "optimizer": {"name": "adamw"},
                "data": {"source": "wikitext2"},
                "run": {"max_steps": 1},
                "claim_scope": {"quality_parity": False, "eval_energy_win": False},
            },
            source="inline",
        )


def test_config_validation_rejects_nonpositive_numeric_fields():
    with pytest.raises(ValueError, match="model.parameters must be positive"):
        validate_repro_config(
            {
                "name": "bad",
                "gate": "GateX",
                "hardware": {"gpu_class": "RTX 3090"},
                "model": {"parameters": 0},
                "optimizer": {"name": "adamw", "lr": 0.001},
                "data": {"source": "wikitext2"},
                "run": {"max_steps": 1},
                "claim_scope": {"quality_parity": False, "eval_energy_win": False},
            },
            source="inline",
        )

    with pytest.raises(ValueError, match="optimizer.lr must be positive"):
        validate_repro_config(
            {
                "name": "bad",
                "gate": "GateX",
                "hardware": {"gpu_class": "RTX 3090"},
                "model": {"parameters": 1},
                "optimizer": {"name": "adamw", "lr": 0},
                "data": {"source": "wikitext2"},
                "run": {"max_steps": 1},
                "claim_scope": {"quality_parity": False, "eval_energy_win": False},
            },
            source="inline",
        )

    with pytest.raises(ValueError, match="run.max_steps values must be positive"):
        validate_repro_config(
            {
                "name": "bad",
                "gate": "GateX",
                "hardware": {"gpu_class": "RTX 3090"},
                "model": {"parameters": 1},
                "optimizer": {"name": "adamw", "lr": 0.001},
                "data": {"source": "wikitext2"},
                "run": {"max_steps": [1, 0]},
                "claim_scope": {"quality_parity": False, "eval_energy_win": False},
            },
            source="inline",
        )
