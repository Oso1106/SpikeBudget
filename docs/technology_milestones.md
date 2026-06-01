# Technology Milestone Ledger

This page is the human-readable evidence ledger for the public SpikeBudget
scratch-training claim. The table intentionally includes failed and
inconclusive checks. Status values match the machine-readable ledger in
[../data/technology_milestones.yaml](../data/technology_milestones.yaml).

| Technology lane | Status | Result | Interpretation | Artifact status |
| --- | --- | --- | --- | --- |
| Bounded claim contract | evidence | SNNs are not slow simply because they spike; differentiation schedule and kernel shape dominate this project. | The strongest claim is about studied small per-step recurrences and GPU execution regime, not arbitrary SNN workloads, neuromorphic devices, or accuracy SOTA. | summary_only |
| Early mechanism search | evidence | Early kernels, event fusion, sequence tasks, and correctness checks established memory traffic, temporal state, fusion, gradient shape, and correctness vocabulary. | Use as mechanism context before the stronger CUDA evidence. | summary_only |
| Tiled scalar reduction vs CUTLASS GEMM gradient | positive | Tiled scalar reduction was 0.098x; CUTLASS GEMM gradient was 34.14x. | The same mathematical direction can fail or win depending on kernel shape. | summary_only |
| Task-level temporal training CUDA benchmark | positive | Matched optimized path was 128.75x memory-cheaper and 6.78x faster on the NVIDIA L4 task. | NVIDIA L4 is the datacenter GPU model used for this task-level benchmark; it is not NVIDIA L40. | summary_only |
| Real CUDA timing correction | caution | Synthetic reviewer-facing timing was replaced with real CUDA kernels and cudaEvent timing. | The correction is a methodology-strength point, but the earlier synthetic-history error must stay visible. | summary_only |
| CUDA hardware-counter repeatability | positive | Optimized hardware counters reported 70.7% SM / 0.08% DRAM while the materialized reference reported 22.7% SM / 51.3% DRAM. | This provides hardware-counter language plus RTX 3090 repeatability context. | summary_only |
| Framework baseline comparison | evidence | SpikingJelly and snnTorch matched supervised CE workloads; BindsNET STDP is not matched CE. | Useful comparison context, not proof that custom kernels dominate all framework workloads. | summary_only |
| Capacity and language-model scaling bridge | evidence | NVIDIA L4 and RTX 3090 experiments show large SNN-transformer training from scratch is feasible, but quality is hard. | This branch bridges systems roofline evidence to language-model quality experiments. | summary_only |
| DCLM benchmark-protocol diagnostic | caution | Step-68100 diagnostic reports DCLM PPL 113.806177 and WT2 continuity PPL 207.135819; 256-example micro-scores stay near chance. | Treat as a benchmark-protocol diagnostic, not official CORE/MMLU/EXTENDED scoring. | included |
| 7B rSVD-GaLore scratch memory-feasibility | caution | Rank-64 rSVD-GaLore reaches 202.127 PPL on the 6.713B SNN-transformer shape. | It proves single-3090 scratch-training feasibility with rSVD-GaLore under the documented memory-saving path. | external_missing |
| True scratch saliency shootout | negative | Fixed timestep reaches 3.3490 BPB; no-saliency reaches 3.3504 BPB; learned saliency reaches 3.3602-3.3674 BPB. | Learned-saliency scratch scale-up stays closed. | external_missing |
| Hessian-error saliency guard | negative | Hessian-seeded joint saliency reaches 3.3371 BPB, but the gain is 0.40% and throughput regresses. | Aligned score machinery is useful instrumentation but misses the quality and efficiency continuation rule. | external_missing |
| Signed subset Taylor saliency guard | negative | Signed subset Taylor diagnostic gives Spearman -0.2571 and top-k overlap 0.0; the follow-up is skipped. | Signed normalization-aware saliency still fails oracle alignment. | external_missing |
| Final subset oracle | negative | Early mask 1100 reaches 3.341844 BPB, only about 0.255% better than no-saliency. | Local Taylor ranking and the late/settling prior fail; later checks show the gain is unstable. | external_missing |
| One-step lookahead | caution | One-step validation predicts 1100 with hard-subset Spearman 0.6000 against the final subset BPB. | Useful single-seed evidence only; repeat-seed robustness rejects it. | external_missing |
| Repeat-seed robustness check | negative | Fresh seed ranks dense no-saliency best at 3.365828 BPB; one-step 1100 becomes worst hard subset. | One-step lookahead is not robust. | external_missing |
| Longer-horizon selector | negative | Horizons 10/30/100 select 0011, 0110, and 1010, missing the seed-4321 final winners. | Cheap short/medium horizon selection is rejected. | external_missing |
| Multi-seed variance check | negative | Best mean hard-mask gain is 0.002098 BPB, below duplicate-eval noise. | Timestep-mask selection is variance dominated and below-noise across seeds. | external_missing |
| Dense low-LR scratch screen | caution | LR 3e-4 improves both seeds by mean 0.089184 BPB / 2.657603%, but no row passes the two-seed >=5% and >=2x-noise rule. | This does not prove a positive theory, but it points away from saliency and toward low-LR optimization stability. | included |
| Dense low-LR scratch stability verification | positive | LR 3e-4 for 2000 dense no-saliency steps reaches 2.108004 and 2.080363 BPB, mean gain 37.815032%. | Positive scratch theory is optimization stability under this profile. | included |
| Scratch architecture search | negative | Best tested candidate D768/L4 has mean BPB 2.187050, worse than the optimized baseline. | In architecture labels, `L4` means four layers, not the NVIDIA L4 GPU. | external_missing |
| Long-horizon dense scratch verification | positive | LR 3e-4 for 6000 scratch steps reaches mean final BPB 1.804832 and improves 13.815176%. | Dense no-saliency low-LR training remains horizon-limited beyond 2000 steps. | included |
| Best trained-from-scratch checkpoint reference | evidence | `external-artifact:long_horizon_dense_scratch_lr3e4_steps6000_best_bpb_checkpoint` is the current best scratch checkpoint target, selected by mean final BPB 1.804832. | Summary metadata is included; checkpoint bytes are not stored in git, and public artifact bundles should attach storage location and checksum to this reference. | included |
| Eval Nsight counter parse check | inconclusive | RTX 3090 eval profiling produced zero parsed counter samples. | Eval comparisons remain functional evidence only, not counter-backed kernel attribution. | external_missing |
| Eval energy rejection checks | negative | The C4 quality run uses +2.824% total energy/token and sparse variants fail quality. | No eval energy-win claim is valid under the C4 quality bound. | external_missing |
| Decision | evidence | Paper core is solid; scratch saliency is closed; dense low-LR scratch optimization is the current positive direction. | Trust the bounded systems claim, the 7B scratch memory-feasibility result as a launch proof, dense low-LR scratch training as the positive optimization direction, and negative scratch evidence as closed. | summary_only |

## Artifact Coverage

| Artifact status | Meaning | Public paths |
| --- | --- | --- |
| included | Lightweight evidence is copied into this repository. | `artifacts/dclm_protocol_diagnostics`, `artifacts/dense_low_lr_scratch_screen`, `artifacts/external_scratch_summaries` |
| summary_only | The public ledger records the source mind-map interpretation, but no per-run evidence file is included here. | `data/technology_milestones.yaml`, this page |
| external_missing | The source mind map names or implies detailed artifacts that are not present in the current public evidence slice. | Future public releases should add small summaries, checksums, or result tables before changing this status. |
