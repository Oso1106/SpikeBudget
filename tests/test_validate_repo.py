from pathlib import Path
import subprocess
import sys

from scripts.validate_repo import REQUIRED_COMPUTE_BOUND_EVIDENCE, main, validate


def test_validate_repo_main_reports_success_for_checked_in_repo():
    assert main([]) == 0


def test_validate_repo_rejects_private_paths_in_public_files(tmp_path):
    write_minimal_repo(tmp_path)
    (tmp_path / "docs" / "limitations.md").write_text(
        "| Bad | Path |\n| --- | --- |\n| leak | stream_" + "yi" + "zhou_s50000 |\n",
        encoding="utf-8",
    )

    errors = validate(tmp_path)

    assert any("private path" in error for error in errors)


def test_validate_repo_rejects_external_weight_framing_in_public_files(tmp_path):
    write_minimal_repo(tmp_path)
    (tmp_path / "docs" / "limitations.md").write_text(
        "| Bad | Framing |\n| --- | --- |\n| leak | " + "pre" + "trained control |\n",
        encoding="utf-8",
    )

    errors = validate(tmp_path)

    assert any("external-weight reference" in error for error in errors)


def test_validate_repo_rejects_disallowed_public_labels(tmp_path):
    write_minimal_repo(tmp_path)
    bad_label = "G" + "ate 1"
    (tmp_path / "docs" / "limitations.md").write_text(
        f"| Bad | Label |\n| --- | --- |\n| leak | {bad_label} |\n",
        encoding="utf-8",
    )

    errors = validate(tmp_path)

    assert any("disallowed public label" in error for error in errors)


def test_compute_bound_doc_contains_required_public_evidence():
    text = (Path(__file__).resolve().parents[1] / "docs" / "compute_bound_evidence.md").read_text(
        encoding="utf-8"
    )

    for required in REQUIRED_COMPUTE_BOUND_EVIDENCE:
        assert required in text


def test_full_gpu_runner_example_help_works_without_torch():
    runner = Path(__file__).resolve().parents[1] / "examples" / "full_gpu_runner.py"

    result = subprocess.run(
        [sys.executable, str(runner), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--config" in result.stdout
    assert "--artifact-dir" in result.stdout
    assert "--save-checkpoint" in result.stdout


def test_validate_repo_ignores_generated_run_outputs(tmp_path):
    write_minimal_repo(tmp_path)
    run_dir = tmp_path / "artifacts" / "runs" / "example_gpu_runner"
    run_dir.mkdir(parents=True)
    private_path = "/" + "Users/example/private"
    (run_dir / "summary.json").write_text(f'{{"artifact_dir": "{private_path}"}}\n', encoding="utf-8")

    errors = validate(tmp_path)

    assert not any("artifacts/runs/example_gpu_runner/summary.json" in error for error in errors)


def test_validate_repo_rejects_missing_compute_bound_evidence_text(tmp_path):
    write_minimal_repo(tmp_path)

    errors = validate(tmp_path)

    assert any("missing compute-bound evidence text" in error for error in errors)


def write_minimal_repo(root: Path) -> None:
    (root / "docs").mkdir()
    (root / "configs").mkdir()
    (root / "data").mkdir()
    (root / "README.md").write_text("# SpikeBudget\n", encoding="utf-8")
    (root / "docs" / "compute_bound_evidence.md").write_text("# Compute-Bound Evidence\n", encoding="utf-8")
    (root / "docs" / "technology_milestones.md").write_text("# Technology Milestone Ledger\n", encoding="utf-8")
    (root / "docs" / "reproduce_3090.md").write_text("# Runbook\n", encoding="utf-8")
    (root / "docs" / "limitations.md").write_text("# Limits\n", encoding="utf-8")
    (root / "configs" / "smoke.yaml").write_text(
        """
name: smoke
evidence_lane: Inline validation lane
hardware:
  gpu_class: RTX 3090
model:
  parameters: 1
optimizer:
  name: adamw
data:
  source: tiny
run:
  max_steps: 1
claim_scope:
  scratch_only: true
  eval_energy_win: false
""",
        encoding="utf-8",
    )
    (root / "data" / "technology_milestones.yaml").write_text(
        """
- id: eval_nsight_counter_parse_check
  title: Eval Nsight Counters Inconclusive
  status: inconclusive
  result: none
  interpretation: none
  artifact_status: summary_only
  artifacts: []
""",
        encoding="utf-8",
    )
