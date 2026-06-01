from pathlib import Path

import pytest

from spikebudget.gates import contains_private_path, load_gate_ledger, validate_gate_ledger


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_gate_ledger_contains_key_decision_gates():
    ledger = load_gate_ledger(REPO_ROOT / "data" / "gates.yaml")
    gate_ids = {gate["id"] for gate in ledger}

    assert {"gate89", "gate130", "gate133", "gate134", "decision"} <= gate_ids


def test_gate_ledger_preserves_inconclusive_counter_gate():
    ledger = load_gate_ledger(REPO_ROOT / "data" / "gates.yaml")
    statuses = {gate["id"]: gate["status"] for gate in ledger}

    assert statuses["gate103"] == "inconclusive"


def test_included_artifacts_exist():
    ledger = load_gate_ledger(REPO_ROOT / "data" / "gates.yaml")

    for gate in ledger:
        if gate["artifact_status"] == "included":
            for artifact in gate["artifacts"]:
                assert (REPO_ROOT / artifact).exists(), f"{gate['id']} missing {artifact}"


def test_gate_ledger_validation_rejects_missing_status():
    with pytest.raises(ValueError, match="gate bad missing required key: status"):
        validate_gate_ledger(
            [
                {
                    "id": "bad",
                    "title": "Bad Gate",
                    "result": "none",
                    "interpretation": "none",
                    "artifact_status": "summary_only",
                    "artifacts": [],
                }
            ],
            repo_root=REPO_ROOT,
            source="inline",
        )


def test_gate_ledger_validation_rejects_unknown_status():
    with pytest.raises(ValueError, match="gate bad has invalid status"):
        validate_gate_ledger(
            [
                {
                    "id": "bad",
                    "title": "Bad Gate",
                    "status": "maybe",
                    "result": "none",
                    "interpretation": "none",
                    "artifact_status": "summary_only",
                    "artifacts": [],
                }
            ],
            repo_root=REPO_ROOT,
            source="inline",
        )


def test_gate_ledger_validation_rejects_duplicate_artifacts():
    with pytest.raises(ValueError, match="gate bad has duplicate artifact"):
        validate_gate_ledger(
            [
                {
                    "id": "bad",
                    "title": "Bad Gate",
                    "status": "inconclusive",
                    "result": "none",
                    "interpretation": "none",
                    "artifact_status": "summary_only",
                    "artifacts": ["same.txt", "same.txt"],
                }
            ],
            repo_root=REPO_ROOT,
            source="inline",
        )


def test_gate_ledger_validation_requires_inconclusive_status():
    with pytest.raises(ValueError, match="must include at least one inconclusive gate"):
        validate_gate_ledger(
            [
                {
                    "id": "good",
                    "title": "Good Gate",
                    "status": "positive",
                    "result": "none",
                    "interpretation": "none",
                    "artifact_status": "summary_only",
                    "artifacts": [],
                }
            ],
            repo_root=REPO_ROOT,
            source="inline",
        )


def test_private_path_detector_catches_host_markers_and_personal_paths():
    assert contains_private_path("/tmp/codex_snn_capacity/run")
    assert contains_private_path("/Users/kuyue/private")
    assert contains_private_path("omni-lsn-9zcvq")
    assert contains_private_path("Yizhou")
    assert contains_private_path("stream_yizhou_s50000")
    assert not contains_private_path("external-artifact:gate74_dclm_stream_s50000")
