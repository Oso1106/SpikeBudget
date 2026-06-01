# Public Technology Milestone Reproducibility Design

## Goal

Make `Oso1106/SpikeBudget` usable by public GitHub readers who want to understand and reproduce the project path toward 7B-class spiking neural network pretraining on RTX 3090-class hardware.

## Claim Contract

The public repo must present a bounded claim:

| Claim | Public Wording |
|---|---|
| Compute-side SNN training | The studied temporal-gradient schedule and kernel shape can move a matched SNN training path from bandwidth-bound execution toward compute-side GPU execution. |
| 7B-class feasibility | A 6.713B-parameter SNN-transformer shape can be trained from scratch on a single RTX 3090-class GPU with rSVD-GaLore-style optimizer memory control. |
| Quality parity | Not claimed. Public quantized checkpoint reproduction is the fair public quantized control, while local pretrained low-LR horizon near miss nearly closes but does not beat that control. |
| Eval energy | Not claimed. Eval Nsight counter parse checks are inconclusive and eval energy rejection checks reject the current eval-energy win. |

## Evidence Source

The initial evidence inventory comes from the supplied SNN project mind map. The visible workspace used for this public release contains only a subset of the raw artifacts named by that mind map, especially DCLM benchmark-protocol diagnostic files and dense low-LR scratch screen files.

## Public Repo Structure

| Path | Responsibility |
|---|---|
| `README.md` | Public entrypoint, bounded claims, quickstart table, and reproduction map. |
| `docs/technology_milestones.md` | Full technology milestone ledger from the mind map, with status, interpretation, and artifact availability. |
| `docs/reproduce_3090.md` | Hardware assumptions, setup, smoke run, long-run recipes, and expected output shape. |
| `docs/limitations.md` | Negative results and caveats readers must preserve. |
| `spikebudget/config.py` | Parse and validate YAML reproduction configs. |
| `spikebudget/evidence.py` | Load and validate the technology milestone ledger. |
| `spikebudget/training.py` | Lightweight public helpers extracted from the DCLM benchmark-protocol diagnostic trainer: fixed eval starts and LR schedule. |
| `scripts/validate_repo.py` | Public validation entrypoint for configs, ledger, docs, and included artifacts. |
| `configs/*.yaml` | Reproduction recipes for smoke, long DCLM, and dense low-LR scratch screen. |
| `artifacts/` | Public, lightweight included evidence from visible local artifacts. |
| `tests/` | Fast tests for deterministic helpers, config validation, and technology milestone ledger integrity. |
| `.github/workflows/ci.yml` | CI running the fast public validation checks. |

## Data Flow

Readers start in `README.md`, choose a reproduction lane, and run `python -m scripts.validate_repo` before launching any GPU job. The configs provide hardware-aware defaults. `spikebudget.config` validates config shape and catches unsafe public defaults. `spikebudget.evidence` validates that each evidence status is explicit and each referenced artifact is either present or marked as summary-only.

## Error Handling

Validation failures should be plain `ValueError` messages with the file or evidence record included. The repo validation script should aggregate failures and return a non-zero exit code without requiring CUDA, PyTorch, or large datasets.

## Testing

Fast tests cover deterministic evaluation windows, learning-rate scheduling, config validation, technology milestone ledger validation, and the public repository validation script. GPU training itself is documented as a runbook, not executed in CI.
