from pathlib import Path

from scripts.validate_repo import main, validate


def test_validate_repo_main_reports_success_for_checked_in_repo():
    assert main([]) == 0


def test_validate_repo_rejects_private_paths_in_public_files(tmp_path):
    write_minimal_repo(tmp_path)
    (tmp_path / "docs" / "limitations.md").write_text(
        "| Bad | Path |\n| --- | --- |\n| leak | stream_yizhou_s50000 |\n",
        encoding="utf-8",
    )

    errors = validate(tmp_path)

    assert any("private path" in error for error in errors)


def write_minimal_repo(root: Path) -> None:
    (root / "docs").mkdir()
    (root / "configs").mkdir()
    (root / "data").mkdir()
    (root / "README.md").write_text("# SpikeBudget\n", encoding="utf-8")
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
  quality_parity: false
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
