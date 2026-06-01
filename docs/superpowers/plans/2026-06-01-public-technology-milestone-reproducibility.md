# Public Technology Milestone Reproducibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the minimal SpikeBudget repo into a public, reproducible scaffold for the technology trail behind 7B-class SNN scratch training on RTX 3090-class hardware.

**Architecture:** Keep heavyweight GPU training as documented recipes and make CI validate the lightweight public artifacts. Put deterministic helper code in a small Python package, and keep the technology milestone ledger in checked-in YAML so readers can audit status without running CUDA.

**Tech Stack:** Python 3.10+, PyYAML, pytest, Markdown, GitHub Actions.

---

### Task 1: Public Validation Package

**Files:**
- Create: `spikebudget/__init__.py`
- Create: `spikebudget/config.py`
- Create: `spikebudget/evidence.py`
- Create: `spikebudget/training.py`
- Create: `tests/test_config.py`
- Create: `tests/test_evidence.py`
- Create: `tests/test_training.py`
- Create: `pyproject.toml`

- [ ] **Step 1: Write failing tests for deterministic helpers**

Create tests for `fixed_eval_starts` and `scheduled_lr`, then run `python -m pytest tests/test_training.py -q`.

Expected: fail until `spikebudget.training` exists.

- [ ] **Step 2: Implement public helpers**

Implement deterministic helpers copied in public-safe form from DCLM benchmark-protocol diagnostic.

- [ ] **Step 3: Add config and evidence validators**

Implement YAML loaders with explicit required fields and useful error messages.

- [ ] **Step 4: Run unit tests**

Run: `python -m pytest -q`

Expected: all tests pass.

### Task 2: Public Docs, Configs, and Evidence

**Files:**
- Replace: `README.md`
- Create: `docs/technology_milestones.md`
- Create: `docs/reproduce_3090.md`
- Create: `docs/limitations.md`
- Create: `configs/rtx3090_7b_memory_feasibility_smoke.yaml`
- Create: `configs/dclm_protocol_3090_long.yaml`
- Create: `configs/dense_low_lr_scratch_screen.yaml`
- Create: `data/technology_milestones.yaml`
- Copy: visible DCLM benchmark-protocol diagnostic and dense low-LR scratch screen lightweight artifacts into `artifacts/`

- [ ] **Step 1: Write public docs**

Docs must use tables for results and must not claim broader model-quality comparison or eval energy wins.

- [ ] **Step 2: Add configs**

Configs must include `name`, `evidence_lane`, `hardware`, `model`, `optimizer`, `data`, `run`, and `claim_scope`.

- [ ] **Step 3: Add technology milestone ledger**

`data/technology_milestones.yaml` must include every technology milestone named in the mind map and mark artifact availability as `included`, `summary_only`, or `external_missing`.

- [ ] **Step 4: Copy lightweight evidence**

Copy visible DCLM benchmark-protocol diagnostic files and dense low-LR scratch screen summary/results/log snippets into `artifacts/`.

- [ ] **Step 5: Run tests**

Run: `python -m pytest -q`

Expected: all tests pass.

### Task 3: Repository Validation and CI

**Files:**
- Create: `scripts/validate_repo.py`
- Create: `tests/test_validate_repo.py`
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write failing validation script test**

Test that `scripts.validate_repo.main([])` returns `0` for the checked-in repo state.

- [ ] **Step 2: Implement validation script**

Validate configs, technology milestone ledger, required docs, and included artifact paths.

- [ ] **Step 3: Add CI workflow**

Run `python -m pip install -e .[dev]` and `python -m pytest -q`.

- [ ] **Step 4: Run final verification**

Run:

```bash
python -m pytest -q
python -m scripts.validate_repo
git status --short
```

Expected: tests and validation pass, with only intentional changed files.
