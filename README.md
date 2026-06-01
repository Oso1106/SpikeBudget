# SpikeBudget

SpikeBudget is the public documentation, config, and evidence ledger for a
bounded RTX 3090-class SNN pretraining effort. This slice is a reproducibility
scaffold: it ships the gate ledger, public run contracts, lightweight evidence,
and validation code; the full GPU training runner/checkpoints are not included
in this repository.

## Bounded Claim

| Claim area | Public claim | Current evidence | Boundary |
| --- | --- | --- | --- |
| Systems mechanism | Some studied SNN training paths can move from memory-bound temporal state materialization toward compute-side GPU work when the differentiation schedule and gradient kernel shape are chosen correctly. | Gate25 vs Gate27, Gate40, and Gate49-51 summarize the CUDA mechanism and counter evidence. | This is not a claim about all SNN workloads, all frameworks, neuromorphic hardware, or accuracy SOTA. |
| 7B-class memory-feasibility demo | A 6.713B SNN-transformer shape can be launched from scratch on RTX 3090-class hardware using memory-saving methods such as rSVD-GaLore. | Gate77-78 report scratch training with roughly 15-16 GiB memory and Gate78 PPL about 202. | This is a memory-feasibility result at poor language-model quality, not quality parity. |
| Public quality control | Official SpikeLLM W4A4 is the fair public quality control. | Gate89 reproduces official SpikeLLM W4A4 at WikiText2 PPL 9.970943 and C4 PPL 12.820610. | Raw FP16 or stronger checkpoints are not fair W4A4 controls. |
| Best local pretrained result | Dense no-saliency horizon transfer nearly closes the local pretrained gap. | Gate134 reaches WikiText2 PPL 10.027895. | It still misses Gate89 by 0.056951 PPL; do not claim quality parity. |
| Negative gates | Saliency, eval counters, and eval energy branches are kept closed unless new evidence changes the decision rules. | Gate108, Gate111, Gate119-123, Gate128, Gate103, and Gate104-106 are negative or inconclusive. | Do not hide failed hypotheses or convert them into positive claims. |

## Quickstart

| Step | Command or action | Notes |
| --- | --- | --- |
| 1 | `git clone https://github.com/Oso1106/SpikeBudget.git SpikeBudget` | This repository contains the scaffold, configs, validation code, and lightweight evidence. |
| 2 | `cd SpikeBudget` | Keep generated outputs under `artifacts/` or an external run directory. |
| 3 | `python -m venv .venv && . .venv/bin/activate` | Use Python matching the public runner release. |
| 4 | Install a CUDA PyTorch build for your GPU and driver. | RTX 3090-class runs require a CUDA-capable PyTorch install and enough disk for datasets/checkpoints. |
| 5 | Set `SPIKEBUDGET_ARTIFACT_DIR`, `SPIKEBUDGET_DATA_DIR`, `HF_HOME`, and `CUDA_VISIBLE_DEVICES`. | See [docs/reproduce_3090.md](docs/reproduce_3090.md) for the public environment contract. |
| 6 | Run a smoke profile before any long run. | Smoke runs should write configs, logs, hardware metadata, and summaries before long training starts. |
| 7 | Compare against the ledger. | The machine-readable gate ledger is [data/gates.yaml](data/gates.yaml). |

## Runnable Scope

| Component | Included here | Command |
| --- | --- | --- |
| Repo validation | Yes | `python -m scripts.validate_repo` |
| Config and ledger tests | Yes | `python -m pytest -q` |
| 3090 GPU pretraining runner | No | Use the configs and runbook here with the matching external runner release. |
| Checkpoints and datasets | No | Store outside git; `.gitignore` blocks common checkpoint and dataset extensions. |

## Gate Highlights

| Gate | Status | Result | Public interpretation |
| --- | --- | --- | --- |
| Claim / G1-19 | Evidence | Early kernels, event fusion, sequence tasks, and correctness checks established the mechanism vocabulary. | Use as context, not a final performance claim. |
| Gate25 vs Gate27 | Positive | Tiled scalar reduction was 0.098x while CUTLASS GEMM gradient was 34.14x. | Kernel shape, not spiking alone, drives the measured GPU regime. |
| Gate40 | Positive | Matched optimized path was 128.75x memory-cheaper and 6.78x faster on the L4 task. | This is the central task-level CUDA benchmark. |
| Gate41 | Caution | Synthetic reviewer-facing suite was replaced with real CUDA kernels and cudaEvent timing. | The correction is part of the record and must remain visible. |
| Gate74 | Caution | DCLM PPL 113.806 and WT2 continuity PPL 207.136 at step 68100; micro-scores stayed near chance. | Report as diagnostic micro-screens, not official benchmark scores. |
| Gate89 | Positive | Official SpikeLLM W4A4: WikiText2 PPL 9.970943, C4 PPL 12.820610. | This is the fair quality control, recorded here as summary evidence rather than raw run logs. |
| Gate95b | Positive | Clean hidden plus pre-gate affine reached 10.980 PPL. | Best local output-side baseline before longer dense calibration. |
| Gate108 | Negative | No-saliency ranked first at 11.1913 PPL; saliency rules regressed. | Do not continue pretrained saliency from this wrapper. |
| Gate111 | Negative | Fixed timestep reached 3.3490 BPB; learned saliency rows were worse than no-saliency. | Scratch learned-saliency scale-up stays closed. |
| Gate129 | Caution | LR 3e-4 improved both seeds by mean 0.089184 BPB / 2.657603%, below the continuation rule. | Useful screen only; it points toward optimization stability. |
| Gate130 / Gate133 | Positive | Longer low-LR dense no-saliency training produced large scratch BPB gains. | Positive scratch theory is dense optimization horizon, not saliency. |
| Gate134 | Caution | Best pretrained locked PPL 10.027895. | Summary evidence only in this slice; strong near miss, but still not Gate89 parity. |

## What Is Included

| Path | Contents | Evidence policy |
| --- | --- | --- |
| [docs/gates.md](docs/gates.md) | Human-readable full gate table through Gate134 and Decision. | Includes positive, caution, negative, and inconclusive gates. |
| [docs/reproduce_3090.md](docs/reproduce_3090.md) | Public 3090-class setup and runbook. | Uses public env vars and avoids private host assumptions. |
| [docs/limitations.md](docs/limitations.md) | Caveat table for public claims. | Keeps Gate41, saliency, counter, and energy caveats explicit. |
| [data/gates.yaml](data/gates.yaml) | Machine-readable gate ledger. | Each entry has id, title, status, result, interpretation, artifact_status, and artifacts. |
| [artifacts/gate74_diagnostics](artifacts/gate74_diagnostics) | Lightweight Gate74 diagnostic summary. | No checkpoint or large dataset files are included. |
| [artifacts/gate129_dense_scaling](artifacts/gate129_dense_scaling) | Gate129 summary, result tables, and log snippets. | No checkpoints, parquet files, or full long logs are included. |
| [artifacts/external_gate_summaries](artifacts/external_gate_summaries) | Compact summary rows for Gate89, Gate130, Gate133, and Gate134. | Summary evidence from the supplied mind map; raw run artifacts remain external. |

## Reading Rules

| Rule | Reason |
| --- | --- |
| Treat Gate89 as the official W4A4 quality control. | Gate134 is close but still misses 9.970943 WikiText2 PPL. |
| Treat Gate134 as a near miss, not a parity claim. | Its 10.027895 PPL is 0.056951 above Gate89. |
| Treat Gate74 diagnostics as micro-screens. | The reported CORE/MMLU/EXTENDED values are not official benchmark scores. |
| Treat saliency branches as closed. | Gate108, Gate111, Gate119-123, Gate126-128 do not justify scale-up. |
| Treat eval counters and eval energy as boundary-setting only. | Gate103 counters were inconclusive and Gate104-106 rejected the eval energy claim. |
