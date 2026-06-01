# SpikeBudget

SpikeBudget is the public documentation, config, and evidence ledger for a
bounded RTX 3090-class SNN pretraining effort. This slice is a reproducibility
scaffold: it ships the technology milestone ledger, public run contracts,
lightweight evidence, and validation code; the full GPU training runner and
checkpoints are not included in this repository.

## Bounded Claim

| Claim area | Public claim | Current evidence | Boundary |
| --- | --- | --- | --- |
| Systems mechanism | Some studied SNN training paths can move from memory-bound temporal state materialization toward compute-side GPU work when the differentiation schedule and gradient kernel shape are chosen correctly. | Tiled scalar reduction vs CUTLASS GEMM gradient, task-level temporal training CUDA benchmark, and CUDA hardware-counter repeatability summarize the CUDA mechanism and counter evidence. | This is not a claim about all SNN workloads, all frameworks, neuromorphic hardware, or accuracy SOTA. |
| 7B-class memory-feasibility demo | A 6.713B SNN-transformer shape can be launched from scratch on RTX 3090-class hardware using memory-saving methods such as rSVD-GaLore. | The 7B rSVD-GaLore scratch memory-feasibility lane reports training with roughly 15-16 GiB memory and PPL around 202. | This is a memory-feasibility result at poor language-model quality, not quality parity. |
| Public quality control | The public quantized checkpoint reproduction is the fair public quality control. | The reproduction reaches WikiText2 PPL 9.970943 and C4 PPL 12.820610. | Raw FP16 or stronger checkpoints are not matched quantized controls. |
| Best local pretrained result | Dense no-saliency horizon transfer nearly closes the local pretrained gap. | The pretrained low-LR horizon near miss reaches WikiText2 PPL 10.027895. | It still misses the public quantized checkpoint reproduction by 0.056951 PPL; do not claim quality parity. |
| Negative evidence | Saliency, eval counters, and eval energy branches are kept closed unless new evidence changes the decision rules. | Full saliency rule shootout, true scratch saliency shootout, signed subset Taylor saliency guard, multi-seed variance check, eval Nsight counter parse check, and eval energy rejection checks are negative or inconclusive. | Do not hide failed hypotheses or convert them into positive claims. |

## Quickstart

| Step | Command or action | Notes |
| --- | --- | --- |
| 1 | `git clone https://github.com/Oso1106/SpikeBudget.git SpikeBudget` | This repository contains the scaffold, configs, validation code, and lightweight evidence. |
| 2 | `cd SpikeBudget` | Keep generated outputs under `artifacts/` or an external run directory. |
| 3 | `python -m venv .venv && . .venv/bin/activate` | Use Python matching the public runner release. |
| 4 | Install a CUDA PyTorch build for your GPU and driver. | RTX 3090-class runs require a CUDA-capable PyTorch install and enough disk for datasets/checkpoints. |
| 5 | Set `SPIKEBUDGET_ARTIFACT_DIR`, `SPIKEBUDGET_DATA_DIR`, `HF_HOME`, and `CUDA_VISIBLE_DEVICES`. | See [docs/reproduce_3090.md](docs/reproduce_3090.md) for the public environment contract. |
| 6 | Run a smoke profile before any long run. | Smoke runs should write configs, logs, hardware metadata, and summaries before long training starts. |
| 7 | Compare against the ledger. | The machine-readable technology milestone ledger is [data/technology_milestones.yaml](data/technology_milestones.yaml). |

## Runnable Scope

| Component | Included here | Command |
| --- | --- | --- |
| Repo validation | Yes | `python -m scripts.validate_repo` |
| Config and ledger tests | Yes | `python -m pytest -q` |
| 3090 GPU pretraining runner | No | Use the configs and runbook here with the matching external runner release. |
| Checkpoints and datasets | No | Store outside git; `.gitignore` blocks common checkpoint and dataset extensions. |

## Technology Highlights

| Technology lane | Status | Result | Public interpretation |
| --- | --- | --- | --- |
| Claim and early mechanism search | Evidence | Early kernels, event fusion, sequence tasks, and correctness checks established the mechanism vocabulary. | Use as context, not a final performance claim. |
| Tiled scalar reduction vs CUTLASS GEMM gradient | Positive | Tiled scalar reduction was 0.098x while CUTLASS GEMM gradient was 34.14x. | Kernel shape, not spiking alone, drives the measured GPU regime. |
| Task-level temporal training CUDA benchmark | Positive | Matched optimized path was 128.75x memory-cheaper and 6.78x faster on the L4 task. | This is the central task-level CUDA benchmark. |
| Real CUDA timing correction | Caution | Synthetic reviewer-facing timing was replaced with real CUDA kernels and cudaEvent timing. | The correction is part of the record and must remain visible. |
| DCLM benchmark-protocol diagnostic | Caution | DCLM PPL 113.806 and WT2 continuity PPL 207.136 at step 68100; micro-scores stayed near chance. | Report as diagnostic micro-screens, not official benchmark scores. |
| Public quantized checkpoint reproduction | Positive | WikiText2 PPL 9.970943 and C4 PPL 12.820610. | This is the fair quality control, recorded here as summary evidence rather than raw run logs. |
| Clean hidden and pre-activation affine | Positive | Clean hidden plus pre-activation affine reached 10.980 PPL. | Best local output-side baseline before longer dense calibration. |
| Full saliency rule shootout | Negative | No-saliency ranked first at 11.1913 PPL; saliency rules regressed. | Do not continue pretrained saliency from this wrapper. |
| True scratch saliency shootout | Negative | Fixed timestep reached 3.3490 BPB; learned saliency rows were worse than no-saliency. | Scratch learned-saliency scale-up stays closed. |
| Dense low-LR scratch screen | Caution | LR 3e-4 improved both seeds by mean 0.089184 BPB / 2.657603%, below the continuation rule. | Useful screen only; it points toward optimization stability. |
| Dense low-LR scratch stability and long-horizon verification | Positive | Longer low-LR dense no-saliency training produced large scratch BPB gains. | Positive scratch theory is dense optimization horizon, not saliency. |
| Pretrained low-LR horizon near miss | Caution | Best pretrained locked PPL 10.027895. | Summary evidence only in this slice; strong near miss, but still not public-checkpoint parity. |

## What Is Included

| Path | Contents | Evidence policy |
| --- | --- | --- |
| [docs/technology_milestones.md](docs/technology_milestones.md) | Human-readable full technology milestone table through the current decision record. | Includes positive, caution, negative, and inconclusive evidence. |
| [docs/reproduce_3090.md](docs/reproduce_3090.md) | Public 3090-class setup and runbook. | Uses public env vars and avoids private host assumptions. |
| [docs/limitations.md](docs/limitations.md) | Caveat table for public claims. | Keeps real CUDA timing correction, saliency, counter, and energy caveats explicit. |
| [data/technology_milestones.yaml](data/technology_milestones.yaml) | Machine-readable technology milestone ledger. | Each entry has id, title, status, result, interpretation, artifact_status, and artifacts. |
| [artifacts/dclm_protocol_diagnostics](artifacts/dclm_protocol_diagnostics) | Lightweight DCLM benchmark-protocol diagnostic summary. | No checkpoint or large dataset files are included. |
| [artifacts/dense_low_lr_scratch_screen](artifacts/dense_low_lr_scratch_screen) | Dense low-LR scratch screen summary, result tables, and log snippets. | No checkpoints, parquet files, or full long logs are included. |
| [artifacts/external_evidence_summaries](artifacts/external_evidence_summaries) | Compact summary rows for the public quantized checkpoint reproduction, dense low-LR scratch stability verification, long-horizon dense scratch verification, and pretrained low-LR horizon near miss. | Summary evidence from the supplied mind map; raw run artifacts remain external. |

## Reading Rules

| Rule | Reason |
| --- | --- |
| Treat the public quantized checkpoint reproduction as the official quantized quality control. | Pretrained low-LR horizon calibration is close but still misses 9.970943 WikiText2 PPL. |
| Treat pretrained low-LR horizon calibration as a near miss, not a parity claim. | Its 10.027895 PPL is 0.056951 above the public quantized checkpoint reproduction. |
| Treat DCLM benchmark-protocol diagnostics as micro-screens. | The reported CORE/MMLU/EXTENDED values are not official benchmark scores. |
| Treat saliency branches as closed. | Full saliency rule shootout, true scratch saliency shootout, signed subset Taylor saliency guard, and repeat-seed robustness checks do not justify scale-up. |
| Treat eval counters and eval energy as boundary-setting only. | Eval Nsight counters were inconclusive and the eval energy checks rejected the energy-win claim. |
