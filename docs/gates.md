# Gate Ledger

This page is the human-readable gate ledger for the public SpikeBudget claim.
The table intentionally includes failed and inconclusive gates. Status values
match the machine-readable ledger in [../data/gates.yaml](../data/gates.yaml).

| Gate | Title | Status | Result | Interpretation | Artifact status |
| --- | --- | --- | --- | --- | --- |
| Claim | Bounded Claim Contract | evidence | SNNs are not slow simply because they spike; differentiation schedule and kernel shape dominate this project. | The strongest claim is about studied small per-step recurrences and GPU execution regime, not arbitrary SNN workloads, neuromorphic devices, or accuracy SOTA. | summary_only |
| G1-19 | Early Mechanism Search | evidence | Early kernels, event fusion, sequence tasks, and correctness checks established memory traffic, temporal state, fusion, gradient shape, and correctness gates. | Use as mechanism context before the stronger CUDA gates. | summary_only |
| G25 vs G27 | Kernel Shape Decides | positive | Tiled scalar reduction was 0.098x; CUTLASS GEMM gradient was 34.14x. | The same mathematical direction can fail or win depending on kernel shape. | summary_only |
| G40 | Task-Level Temporal Training | positive | Matched optimized path was 128.75x memory-cheaper and 6.78x faster on the L4 task. | This is the core task-level benchmark tying optimized work to the materialized reference. | summary_only |
| G41 | Real Baseline Correction | caution | Synthetic reviewer-facing suite was replaced with real CUDA kernels and cudaEvent timing. | The correction is a methodology-strength point, but the earlier synthetic-history error must stay visible. | summary_only |
| G49-51 | Counters and Repeatability | positive | Gate49 reports optimized 70.7% SM / 0.08% DRAM and materialized 22.7% SM / 51.3% DRAM. | These gates provide hardware-counter language plus 3090 and Gate41 repeatability context. | summary_only |
| G53-54 | Framework Baselines | evidence | SpikingJelly and snnTorch matched supervised CE workloads; BindsNET STDP is not matched CE. | Useful comparison context, not proof that custom kernels dominate all framework workloads. | summary_only |
| G55-76 | Capacity, LM, and Spike-Native Scaling | evidence | L4/3090 experiments show large SNN-transformer training is feasible, but quality is hard. | This branch bridges systems roofline evidence to language-model quality experiments. | summary_only |
| Gate74 | DCLM Benchmark Protocol Diagnostic | caution | Step 68100 diagnostic reports DCLM PPL 113.806177 and WT2 continuity PPL 207.135819; 256-example micro-scores stay near chance. | Treat as a benchmark-protocol diagnostic, not official CORE/MMLU/EXTENDED scoring. | included |
| G77-78 | 7B Scratch Training | caution | Gate78 rank64 reaches 202.127 PPL on the 6.713B SNN-transformer shape. | It proves single-3090 training feasibility with rSVD-GaLore, not SpikeLLM parity. | external_missing |
| G80-83 | Gate80 Teacher and Distillation | negative | Gate83 completes 1000-step top-k distillation at PPL 217.724. | Alpha 0.2 improves over high-alpha distillation but still misses the Gate78 scratch best. | external_missing |
| G89 | Official SpikeLLM Reproduction | positive | Official SpikeLLM W4A4 reaches WikiText2 PPL 9.970943 and C4 PPL 12.820610. | This is the fair target control; Gate90 stronger-checkpoint chase was stopped. | included summary |
| G84-88 | Locked Metric and Calibration Push | evidence | Locked seq2048/eight-window metric makes Gate80 harder: 12.137 baseline, Gate85 best 11.962. | The short-window result is fragile; locked long-context evaluation is the comparison point. | external_missing |
| G91-95 | Output-Side Ladder | positive | Gate93 hidden-channel affine reaches 11.050; soft clipping produces 250.898 and 202.168 PPL. | Output-side repair is real, but the tested soft-clip path is destructive. | external_missing |
| G95b | Best Local Output-Side Result | positive | Clean hidden plus pre-gate affine reaches 10.980 PPL. | Best local output-side baseline for later saliency and dense calibration, still 1.101x Gate89. | external_missing |
| G96 | Uniform Deeper Wrapping Fails | negative | Last-four residual bridge regresses to 12.583 PPL. | More wrapped layers are not automatically better. | external_missing |
| G97 | Trainable Saliency Smoke | caution | Short seq512 / eval_iters=2 saliency smoke reports 8.519 PPL. | Strong smoke result only; it is not comparable to Gate89 or Gate95b locked metrics. | external_missing |
| G98 | Locked Soft Saliency Misses | negative | Best soft hidden saliency reaches 11.6916 PPL, worse than Gate95b 10.9803. | Locked saliency does not improve quality. | external_missing |
| G99-102 | Hard Mask, Bridges, Robustness | negative | Hard top-k reaches 11.8336/11.8589; bridge variants reach 14.0803/14.1415; offset robustness reaches 16.6474. | Hard masking, naive bridge expansion, and offset validation do not rescue saliency. | external_missing |
| G108 | Full Saliency Rule Shootout | negative | Locked full run ranks no-saliency first at 11.1913 PPL; best saliency is 11.6513. | Do not continue to Gate109 from this wrapper. | external_missing |
| G111 | True Scratch Saliency Shootout | negative | Fixed timestep reaches 3.3490 BPB; no-saliency 3.3504; learned saliency 3.3602-3.3674. | Gate110 and Gate114 learned-saliency scale-up stay closed. | external_missing |
| G115-118 | Hessian-Error Saliency Guard | negative | Gate117 hessian-seeded joint reaches 3.3371 BPB, but the gain is 0.40% and throughput regresses. | Aligned score machinery is useful instrumentation but misses the quality and efficiency continuation rule. | external_missing |
| G119-123 | Signed Subset Saliency Guard | negative | Signed subset Taylor diagnostic gives Spearman -0.2571 and top-k overlap 0.0; Gate123 is skipped. | Signed normalization-aware saliency still fails oracle alignment. | external_missing |
| G124 | Final Subset Oracle | negative | Early mask 1100 reaches 3.341844 BPB, only about 0.255% better than no-saliency. | Local Taylor ranking and the late/settling prior fail; later gates show the gain is unstable. | external_missing |
| G125 | One-Step Lookahead | caution | One-step validation predicts 1100 with hard-subset Spearman 0.6000 vs Gate124 final BPB. | Useful single-seed evidence only; Gate126 rejects robustness. | external_missing |
| G126 | Real-Training Repeat | negative | Fresh seed ranks dense no-saliency best at 3.365828 BPB; one-step 1100 becomes worst hard subset. | One-step lookahead is not robust. | external_missing |
| G127 | Longer-Horizon Selector | negative | Horizons 10/30/100 select 0011, 0110, and 1010, missing the seed-4321 final winners. | Cheap short/medium horizon selection is rejected. | external_missing |
| G128 | Multi-Seed Variance Check | negative | Best mean hard-mask gain is 0.002098 BPB, below duplicate-eval noise. | Timestep-mask selection is variance dominated and below-noise across seeds. | external_missing |
| G129 | Dense Scale Screen | caution | LR 3e-4 improves both seeds by mean 0.089184 BPB / 2.657603%, but no row passes the two-seed >=5% and >=2x-noise rule. | This does not prove a positive theory, but it points away from saliency and toward low-LR optimization stability. | included |
| G130 | Optimization Stability Verified | positive | LR 3e-4 for 2000 dense no-saliency steps reaches 2.108004 and 2.080363 BPB, mean gain 37.815032%. | Positive scratch theory is optimization stability under this profile. | included summary |
| G131 | Pretrained Optimization Transfer | caution | Gate95b wrapper with LR 7.5e-5 for 1400 steps reaches 10.889457 PPL. | The low-LR horizon theory transfers, but it remains 0.918514 PPL above Gate89. | external_missing |
| G132 | Scratch Architecture Search | negative | Best tested candidate D768/L4 has mean BPB 2.187050, worse than the optimized baseline. | Tested architecture changes do not create a verified positive theory. | external_missing |
| G133 | Longer Horizon Verified | positive | LR 3e-4 for 6000 scratch steps reaches mean final BPB 1.804832 and improves 13.815176%. | Dense no-saliency low-LR training remains horizon-limited beyond 2000 steps. | included summary |
| G134 | Pretrained Horizon Near Miss | caution | LR 7.5e-5 for 6000 pretrained calibration reaches 10.027895 PPL. | Strong local pretrained lever, but still 0.056951 PPL above official Gate89 W4A4. | included summary |
| G103 | Eval Nsight Counters Inconclusive | inconclusive | RTX 3090 eval profiling produced zero parsed counter samples. | Eval comparisons remain functional evidence only, not counter-backed kernel attribution. | external_missing |
| G104-106 | Eval Energy Claim Rejected | negative | Gate91 passes C4 quality but uses +2.824% total energy/token; sparse variants fail quality. | No eval energy-win claim is valid under the C4 quality bound. | external_missing |
| Decision | What To Trust Today | evidence | Paper core is solid; saliency is closed; dense optimization nearly reaches Gate89 but not parity. | Trust the bounded systems claim, Gate89 as the fair control, Gate134 as a near miss, and negative gates as closed. | summary_only |

## Artifact Coverage

| Artifact status | Meaning | Public paths |
| --- | --- | --- |
| included | Lightweight evidence is copied into this repository. | `artifacts/gate74_diagnostics`, `artifacts/gate129_dense_scaling` |
| summary_only | The public ledger records the source mind-map interpretation, but no per-run evidence file is included here. | `data/gates.yaml`, this page |
| external_missing | The source mind map names or implies detailed artifacts that are not present in the current public evidence slice. | Future public releases should add small summaries, checksums, or result tables before changing this status. |
