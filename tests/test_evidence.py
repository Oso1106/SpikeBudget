from pathlib import Path

import pytest

from spikebudget.evidence import (
    contains_disallowed_public_label,
    contains_private_path,
    load_evidence_ledger,
    validate_evidence_ledger,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_evidence_ledger_contains_key_decision_records():
    ledger = load_evidence_ledger(REPO_ROOT / "data" / "technology_milestones.yaml")
    record_ids = {record["id"] for record in ledger}

    assert {
        "public_quantized_checkpoint_reproduction",
        "dense_low_lr_scratch_stability",
        "long_horizon_dense_scratch_verification",
        "pretrained_low_lr_horizon_near_miss",
        "decision",
    } <= record_ids


def test_evidence_ledger_preserves_inconclusive_counter_record():
    ledger = load_evidence_ledger(REPO_ROOT / "data" / "technology_milestones.yaml")
    statuses = {record["id"]: record["status"] for record in ledger}

    assert statuses["eval_nsight_counter_parse_check"] == "inconclusive"


def test_included_artifacts_exist():
    ledger = load_evidence_ledger(REPO_ROOT / "data" / "technology_milestones.yaml")

    for record in ledger:
        if record["artifact_status"] == "included":
            for artifact in record["artifacts"]:
                assert (REPO_ROOT / artifact).exists(), f"{record['id']} missing {artifact}"


def test_evidence_ledger_validation_rejects_missing_status():
    with pytest.raises(ValueError, match="evidence record bad missing required key: status"):
        validate_evidence_ledger(
            [
                {
                    "id": "bad",
                    "title": "Bad Record",
                    "result": "none",
                    "interpretation": "none",
                    "artifact_status": "summary_only",
                    "artifacts": [],
                }
            ],
            repo_root=REPO_ROOT,
            source="inline",
        )


def test_evidence_ledger_validation_rejects_unknown_status():
    with pytest.raises(ValueError, match="evidence record bad has invalid status"):
        validate_evidence_ledger(
            [
                {
                    "id": "bad",
                    "title": "Bad Record",
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


def test_evidence_ledger_validation_rejects_duplicate_artifacts():
    with pytest.raises(ValueError, match="evidence record bad has duplicate artifact"):
        validate_evidence_ledger(
            [
                {
                    "id": "bad",
                    "title": "Bad Record",
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


def test_evidence_ledger_validation_requires_inconclusive_status():
    with pytest.raises(ValueError, match="must include at least one inconclusive record"):
        validate_evidence_ledger(
            [
                {
                    "id": "good",
                    "title": "Good Record",
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


def test_public_label_detector_catches_disallowed_baseline_and_numeric_labels():
    assert contains_disallowed_public_label("Spike" + "LLM")
    assert contains_disallowed_public_label("W4" + "A4")
    assert contains_disallowed_public_label("Ga" + "te " + "129")
    assert contains_disallowed_public_label("ga" + "te" + "129")
    assert contains_disallowed_public_label("G" + "129")
    assert not contains_disallowed_public_label("dense low-LR scratch screen")


def test_private_path_detector_catches_host_markers_and_personal_paths():
    assert contains_private_path("/tmp/codex_snn_capacity/run")
    assert contains_private_path("/Users/kuyue/private")
    assert contains_private_path("omni-lsn-9zcvq")
    assert contains_private_path("Yizhou")
    assert contains_private_path("stream_yizhou_s50000")
    assert not contains_private_path("external-artifact:dclm_protocol_stream_s50000")
