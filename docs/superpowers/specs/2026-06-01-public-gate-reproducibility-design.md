# Public Gate Reproducibility Design

## Goal

Make `Oso1106/SpikeBudget` usable by public GitHub readers who want to understand and reproduce the project path toward 7B-class spiking neural network pretraining on RTX 3090-class hardware.

## Claim Contract

The public repo must present a bounded claim:

| Claim | Public Wording |
|---|---|
| Compute-side SNN training | The studied temporal-gradient schedule and kernel shape can move a matched SNN training path from bandwidth-bound execution toward compute-side GPU execution. |
| 7B-class feasibility | A 6.713B-parameter SNN-transformer shape can be trained from scratch on a single RTX 3090-class GPU with rSVD-GaLore-style optimizer memory control. |
| Quality parity | Not claimed. Gate89 is the fair official SpikeLLM W4A4 control, while local Gate134 nearly closes but does not beat that control. |
| Eval energy | Not claimed. Gate103 counters are inconclusive and Gate104-106 reject the current eval-energy win. |

## Gate Source

The initial gate inventory comes from the supplied SNN project mind map. The visible workspace used for this public release contains only a subset of the raw artifacts named by that mind map, especially Gate74 diagnostic files and Gate129 dense-scaling files.

## Public Repo Structure

| Path | Responsibility |
|---|---|
| `README.md` | Public entrypoint, bounded claims, quickstart table, and reproduction map. |
| `docs/gates.md` | Full gate ledger from the mind map, with status, interpretation, and artifact availability. |
| `docs/reproduce_3090.md` | Hardware assumptions, setup, smoke run, long-run recipes, and expected output shape. |
| `docs/limitations.md` | Negative results and caveats readers must preserve. |
| `spikebudget/config.py` | Parse and validate YAML reproduction configs. |
| `spikebudget/gates.py` | Load and validate the gate ledger. |
| `spikebudget/training.py` | Lightweight public helpers extracted from the Gate74 trainer: fixed eval starts and LR schedule. |
| `scripts/validate_repo.py` | Public validation gate for configs, ledger, docs, and included artifacts. |
| `configs/*.yaml` | Reproduction recipes for smoke, long DCLM, and Gate129 dense screen. |
| `artifacts/` | Public, lightweight included evidence from visible local artifacts. |
| `tests/` | Fast tests for deterministic helpers, config validation, and gate ledger integrity. |
| `.github/workflows/ci.yml` | CI running the fast public validation checks. |

## Data Flow

Readers start in `README.md`, choose a reproduction lane, and run `python -m scripts.validate_repo` before launching any GPU job. The configs provide hardware-aware defaults. `spikebudget.config` validates config shape and catches unsafe public defaults. `spikebudget.gates` validates that each gate status is explicit and each referenced artifact is either present or marked as summary-only.

## Error Handling

Validation failures should be plain `ValueError` messages with the file or gate name included. The repo validation script should aggregate failures and return a non-zero exit code without requiring CUDA, PyTorch, or large datasets.

## Testing

Fast tests cover deterministic evaluation windows, learning-rate scheduling, config validation, gate ledger validation, and the public repository validation script. GPU training itself is documented as a runbook, not executed in CI.
