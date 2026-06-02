# SpikeBudget

SpikeBudget is the public documentation, config, and evidence ledger for a
bounded SNN train-from-scratch effort on GPUs like RTX 3090-class. This slice is
a reproducibility scaffold: it ships the technology milestone ledger, public run
contracts, lightweight scratch evidence, validation code, and a full GPU runner
example. Generated checkpoint bytes remain outside git.

## Compute-Bound Summary

In the studied path, compute-side behavior was achieved by changing the
gradient path from a memory-heavy tiled scalar reduction into a CUTLASS
GEMM-shaped CUDA workload.
That makes the GPU do dense matrix-style work instead of mostly storing and
rereading temporal state. The matched optimized path also avoids much of the
temporal-state materialization, which is why the evidence shows both lower
memory traffic and faster runtime.

For the full evidence table and reading boundary, see
[docs/compute_bound_evidence.md](docs/compute_bound_evidence.md).

| Insight | Implication |
| --- | --- |
| In the studied path, SNN speed is not determined by spiking alone. | The main design question is how the temporal gradient and state path map onto GPU kernels. |
| Kernel shape matters more than the high-level math label. | A scalar-reduction implementation can be slow, while a GEMM-shaped implementation can expose enough dense compute to use the GPU well. |
| Avoiding temporal-state materialization is central. | Memory traffic, not arithmetic, was the bottleneck in the materialized path; reducing it makes larger scratch runs more feasible. |
| Counter evidence matches the mechanism. | High SM use and very low DRAM pressure support the compute-side interpretation for the optimized path. |
| The result is bounded, not universal. | This supports the studied CUDA path and task benchmark; it does not prove all SNN workloads are compute-bound or faster by default. |
| Future work should optimize representation before scaling. | More model size or more saliency machinery is less useful than first ensuring the training path is GPU-friendly and memory-light. |

## Bounded Claim

| Claim area | Public claim | Current evidence | Boundary |
| --- | --- | --- | --- |
| Systems mechanism | Some studied SNN training paths can move from memory-bound temporal state materialization toward compute-side GPU work when the differentiation schedule and gradient kernel shape are chosen correctly. | Tiled scalar reduction was 0.098x while CUTLASS GEMM gradient was 34.14x; the matched optimized path was 128.75x memory-cheaper and 6.78x faster on the NVIDIA L4 task. | This is not a claim about all SNN workloads, all frameworks, neuromorphic hardware, or accuracy SOTA. NVIDIA L4 is not NVIDIA L40. |
| 7B-class memory-feasibility demo | A 6.713B SNN-transformer shape can be launched from scratch on RTX 3090-class hardware using memory-saving methods such as rSVD-GaLore. | The 7B rSVD-GaLore scratch memory-feasibility lane reports training with roughly 15-16 GiB memory and PPL around 202. | This is a memory-feasibility result at poor language-model quality, not a solved quality result. |
| Best scratch checkpoint reference | The best trained-from-scratch checkpoint target in this public slice is the long-horizon dense no-saliency run. | `external-artifact:long_horizon_dense_scratch_lr3e4_steps6000_best_bpb_checkpoint` is selected by mean final BPB 1.804832; see [artifacts/external_scratch_summaries/best_scratch_checkpoint.tsv](artifacts/external_scratch_summaries/best_scratch_checkpoint.tsv). | The generated checkpoint bytes stay outside git; release bundles should provide storage location and checksum. |
| Scratch optimization direction | Dense no-saliency low-LR training is the verified positive scratch direction in this evidence slice. | Dense low-LR scratch stability reaches 2.080363 BPB, and long-horizon dense scratch verification reaches mean final BPB 1.804832. | These are scratch BPB results under the documented profile; they are not claims about broader model-quality evaluation. |
| Negative scratch evidence | Learned timestep saliency, hessian seeding, signed subset scoring, and hard-mask selectors stay closed unless new evidence changes the decision rules. | True scratch saliency shootout, hessian-error saliency guard, signed subset Taylor saliency guard, final subset oracle, repeat-seed robustness, longer-horizon selector, and multi-seed variance check are negative or cautionary. | Do not hide failed hypotheses or convert them into positive claims. |

## Quickstart

| Step | Command or action | Notes |
| --- | --- | --- |
| 1 | `git clone https://github.com/Oso1106/SpikeBudget.git SpikeBudget` | This repository contains the scaffold, configs, validation code, and lightweight scratch evidence. |
| 2 | `cd SpikeBudget` | Keep generated outputs under `artifacts/` or an external run directory. |
| 3 | `python -m venv .venv && . .venv/bin/activate` | Use Python matching the public runner release. |
| 4 | Install a CUDA PyTorch build for your GPU and driver. | RTX 3090-class runs require a CUDA-capable PyTorch install and enough disk for datasets/checkpoints generated by your own runs. |
| 5 | Set `SPIKEBUDGET_ARTIFACT_DIR`, `SPIKEBUDGET_DATA_DIR`, `HF_HOME`, and `CUDA_VISIBLE_DEVICES`. | See [docs/reproduce_3090.md](docs/reproduce_3090.md) for the public environment contract. |
| 6 | Run a smoke profile before any long run. | Smoke runs should write configs, logs, hardware metadata, and summaries before long training starts. |
| 7 | Compare against the ledger. | The machine-readable technology milestone ledger is [data/technology_milestones.yaml](data/technology_milestones.yaml). |

## Runnable Scope

| Component | Included here | Command |
| --- | --- | --- |
| Repo validation | Yes | `python -m scripts.validate_repo` |
| Config and ledger tests | Yes | `python -m pytest -q` |
| Full GPU train-from-scratch runner example | Yes | `python examples/full_gpu_runner.py --config configs/dense_low_lr_scratch_screen.yaml --device cuda --artifact-dir artifacts/runs/example_gpu_runner --max-steps 20` |
| Generated checkpoints and datasets | No | Store outside git; `.gitignore` blocks common checkpoint and dataset extensions. |

## Technology Highlights

| Technology lane | Status | Result | Public interpretation |
| --- | --- | --- | --- |
| Claim and early mechanism search | Evidence | Early kernels, event fusion, sequence tasks, and correctness checks established the mechanism vocabulary. | Use as context, not a final performance claim. |
| Tiled scalar reduction vs CUTLASS GEMM gradient | Positive | Tiled scalar reduction was 0.098x while CUTLASS GEMM gradient was 34.14x. | Kernel shape, not spiking alone, drives the measured GPU regime. |
| Task-level temporal training CUDA benchmark | Positive | Matched optimized path was 128.75x memory-cheaper and 6.78x faster on the NVIDIA L4 task. | NVIDIA L4 is a distinct datacenter GPU model used for this task-level benchmark; it is not NVIDIA L40. |
| Real CUDA timing correction | Caution | Synthetic reviewer-facing timing was replaced with real CUDA kernels and cudaEvent timing. | The correction is part of the record and must remain visible. |
| CUDA hardware-counter repeatability | Positive | Optimized counters reported 70.7% SM / 0.08% DRAM while the materialized reference reported 22.7% SM / 51.3% DRAM. | This is the clearest counter evidence for compute-side execution in the measured optimized path. |
| DCLM benchmark-protocol diagnostic | Caution | DCLM PPL 113.806 and WT2 continuity PPL 207.136 at step 68100; micro-scores stayed near chance. | Report as diagnostic micro-screens, not official benchmark scores. |
| True scratch saliency shootout | Negative | Fixed timestep reached 3.3490 BPB; learned saliency rows were worse than no-saliency. | Scratch learned-saliency scale-up stays closed. |
| Signed subset Taylor saliency guard | Negative | Signed subset Taylor diagnostic gave Spearman -0.2571 and top-k overlap 0.0. | Signed normalization-aware saliency fails oracle alignment. |
| Dense low-LR scratch screen | Caution | LR 3e-4 improved both seeds by mean 0.089184 BPB / 2.657603%, below the continuation rule. | Useful screen only; it points toward optimization stability. |
| Dense low-LR scratch stability and long-horizon verification | Positive | Longer low-LR dense no-saliency training produced large scratch BPB gains. | Positive scratch theory is dense optimization horizon, not saliency. |
| Best trained-from-scratch checkpoint reference | Evidence | `external-artifact:long_horizon_dense_scratch_lr3e4_steps6000_best_bpb_checkpoint` is selected by mean final BPB 1.804832. | The metadata is included; checkpoint bytes are external and need release location plus checksum. |
| Scratch architecture search | Negative | Best tested candidate D768/L4 had mean BPB 2.187050, worse than the optimized baseline. | In architecture labels, `L4` means four layers, not the NVIDIA L4 GPU. |

## What Is Included

| Path | Contents | Evidence policy |
| --- | --- | --- |
| [docs/technology_milestones.md](docs/technology_milestones.md) | Human-readable full technology milestone table through the current scratch-focused decision record. | Includes positive, caution, negative, and inconclusive evidence. |
| [docs/compute_bound_evidence.md](docs/compute_bound_evidence.md) | Explanation of what changed in the CUDA path and why the evidence supports compute-side GPU execution. | Keeps the compute-bound reading tied to measured kernels and task-level evidence. |
| [docs/reproduce_3090.md](docs/reproduce_3090.md) | Public 3090-class scratch setup and runbook. | Uses public env vars and avoids private host assumptions. |
| [docs/limitations.md](docs/limitations.md) | Caveat table for public claims. | Keeps real CUDA timing correction, scratch saliency, counter, and energy caveats explicit. |
| [data/technology_milestones.yaml](data/technology_milestones.yaml) | Machine-readable technology milestone ledger. | Each entry has id, title, status, result, interpretation, artifact_status, and artifacts. |
| [artifacts/dclm_protocol_diagnostics](artifacts/dclm_protocol_diagnostics) | Lightweight DCLM benchmark-protocol diagnostic summary. | No generated checkpoint or large dataset files are included. |
| [artifacts/dense_low_lr_scratch_screen](artifacts/dense_low_lr_scratch_screen) | Dense low-LR scratch screen summary, result tables, and log snippets. | No generated checkpoints, parquet files, or full long logs are included. |
| [artifacts/external_scratch_summaries](artifacts/external_scratch_summaries) | Compact summary rows for dense low-LR scratch stability verification, long-horizon dense scratch verification, and the best trained-from-scratch checkpoint reference. | Summary evidence from the supplied mind map; raw run artifacts remain external. |

## Reading Rules

| Rule | Reason |
| --- | --- |
| Treat this repository as scratch-only evidence. | The public claim is about training SNN profiles from scratch on RTX 3090-class hardware. |
| Treat DCLM benchmark-protocol diagnostics as micro-screens. | The reported CORE/MMLU/EXTENDED values are not official benchmark scores. |
| Treat dense low-LR scratch stability as the current positive scratch direction. | Dense no-saliency low-LR training is the branch with positive scratch BPB evidence. |
| Read L4 labels by context. | `NVIDIA L4` is the datacenter GPU used in the task-level benchmark; architecture labels such as `D768/L4` mean four layers; neither means NVIDIA L40. |
| Treat saliency branches as closed. | True scratch saliency shootout, signed subset Taylor saliency guard, final subset oracle, and repeat-seed robustness checks do not justify scale-up. |
| Treat eval counters and eval energy as boundary-setting only. | Eval Nsight counters were inconclusive and the eval energy checks rejected the energy-win claim. |
