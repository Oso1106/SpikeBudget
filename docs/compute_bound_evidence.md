# Compute-Bound Evidence

This note explains what changed in the studied SpikeBudget CUDA path and why
the public evidence says the optimized path moves work away from memory-bound
temporal state traffic and toward compute-side GPU execution. The claim is
bounded to the measured kernels and task-level benchmark; it is not a claim
about every SNN workload or every framework.

## Short Explanation

Compute-side behavior was achieved by changing the gradient path from a
memory-heavy tiled scalar reduction into a CUTLASS GEMM-shaped CUDA workload.
That makes the GPU do dense matrix-style work instead of mostly storing and
rereading temporal state. The matched optimized path also avoids much of the
temporal-state materialization, which is why the evidence shows both lower
memory traffic and faster runtime.

## What Changed

| Change | What it does | Why it matters for compute-bound behavior |
| --- | --- | --- |
| Differentiation schedule was treated as part of the kernel design. | The backward path was not treated as a fixed scalar-reduction problem. | The same math can be memory-bound or compute-heavy depending on how gradients are shaped for the GPU. |
| Tiled scalar reduction was rejected as the winning gradient shape. | The scalar-reduction implementation did not expose enough dense compute to the GPU. | It measured at 0.098x, showing that a naive tiled reduction can make the path worse rather than compute-bound. |
| CUTLASS GEMM-shaped gradient work was used for the optimized direction. | Gradient work was mapped into dense matrix-multiply-like GPU work. | It measured at 34.14x, showing that the kernel shape can move the same direction into a favorable GPU regime. |
| Temporal state materialization was avoided in the matched optimized path. | The optimized task path reduced the amount of materialized temporal state traffic compared with the reference path. | The measured task path was 128.75x memory-cheaper, which is the central evidence that the path is no longer dominated by storing and rereading temporal state. |
| Task-level timing was measured with real CUDA kernels. | Synthetic reviewer-facing timing was replaced by real CUDA kernels and cudaEvent timing. | This keeps the evidence tied to actual GPU execution rather than a synthetic timing story. |
| Hardware-counter evidence was used as a boundary check. | Optimized and materialized paths were compared with SM and DRAM counter summaries. | High SM use with very low DRAM pressure supports the compute-side interpretation for the optimized path. |

## Evidence

| Evidence lane | Measurement | Interpretation |
| --- | --- | --- |
| Kernel-shape comparison | Tiled scalar reduction was 0.098x while CUTLASS GEMM gradient was 34.14x. | The win comes from shaping the gradient path as GPU-friendly compute, not from spiking alone. |
| Task-level NVIDIA L4 benchmark | Matched optimized path was 128.75x memory-cheaper and 6.78x faster on the NVIDIA L4 task. | The optimized path reduces memory traffic and improves runtime on the measured task. NVIDIA L4 is a datacenter GPU model, not NVIDIA L40. |
| CUDA counter repeatability | Optimized counters reported 70.7% SM / 0.08% DRAM while the materialized reference reported 22.7% SM / 51.3% DRAM. | The optimized path spends far more of the measured profile in compute-side work and far less in DRAM traffic. |
| Real CUDA timing correction | Real CUDA kernels and cudaEvent timing replaced the earlier synthetic timing history. | The correction keeps the public evidence honest and bound to measured kernels. |

## Insights And Implications

| Insight | Implication |
| --- | --- |
| In the studied path, SNN speed is not determined by spiking alone. | The main design question is how the temporal gradient and state path map onto GPU kernels. |
| Kernel shape matters more than the high-level math label. | A scalar-reduction implementation can be slow, while a GEMM-shaped implementation can expose enough dense compute to use the GPU well. |
| Avoiding temporal-state materialization is central. | Memory traffic, not arithmetic, was the bottleneck in the materialized path; reducing it makes larger scratch runs more feasible. |
| Counter evidence matches the mechanism. | High SM use and very low DRAM pressure support the compute-side interpretation for the optimized path. |
| The result is bounded, not universal. | This supports the studied CUDA path and task benchmark; it does not prove all SNN workloads are compute-bound or faster by default. |
| Future work should optimize representation before scaling. | More model size or more saliency machinery is less useful than first ensuring the training path is GPU-friendly and memory-light. |

## Reading Boundary

| Boundary | Public reading |
| --- | --- |
| Scope | The claim applies to the studied temporal-gradient schedules, kernel shapes, and task-level benchmark. |
| What it proves | The optimized path can shift the measured workload from materialized temporal-state memory traffic toward compute-side GPU execution. |
| What it does not prove | It does not prove all SNNs are compute-bound, all sparse variants are better, or language-model quality is solved. |
| L4 wording | `NVIDIA L4` means the GPU used in the task-level benchmark; architecture labels such as `D768/L4` mean four model layers. |
