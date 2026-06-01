# Technology Milestone Ledger

This page is the human-readable evidence ledger for the public SpikeBudget claim.
The table intentionally includes failed and inconclusive checks. Status values
match the machine-readable ledger in
[../data/technology_milestones.yaml](../data/technology_milestones.yaml).

| Technology lane | Status | Result | Interpretation | Artifact status |
| --- | --- | --- | --- | --- |
| Bounded claim contract | evidence | SNNs are not slow simply because they spike; differentiation schedule and kernel shape dominate this project. | The strongest claim is about studied small per-step recurrences and GPU execution regime, not arbitrary SNN workloads, neuromorphic devices, or accuracy SOTA. | summary_only |
| Early mechanism search | evidence | Early kernels, event fusion, sequence tasks, and correctness checks established memory traffic, temporal state, fusion, gradient shape, and correctness vocabulary. | Use as mechanism context before the stronger CUDA evidence. | summary_only |
| Tiled scalar reduction vs CUTLASS GEMM gradient | positive | Tiled scalar reduction was 0.098x; CUTLASS GEMM gradient was 34.14x. | The same mathematical direction can fail or win depending on kernel shape. | summary_only |
| Task-level temporal training CUDA benchmark | positive | Matched optimized path was 128.75x memory-cheaper and 6.78x faster on the L4 task. | This is the core task-level benchmark tying optimized work to the materialized reference. | summary_only |
| Real CUDA timing correction | caution | Synthetic reviewer-facing timing was replaced with real CUDA kernels and cudaEvent timing. | The correction is a methodology-strength point, but the earlier synthetic-history error must stay visible. | summary_only |
| CUDA hardware-counter repeatability | positive | Optimized hardware counters reported 70.7% SM / 0.08% DRAM while the materialized reference reported 22.7% SM / 51.3% DRAM. | This provides hardware-counter language plus RTX 3090 repeatability context. | summary_only |
| Framework baseline comparison | evidence | SpikingJelly and snnTorch matched supervised CE workloads; BindsNET STDP is not matched CE. | Useful comparison context, not proof that custom kernels dominate all framework workloads. | summary_only |
| Capacity and language-model scaling bridge | evidence | L4 and RTX 3090 experiments show large SNN-transformer training is feasible, but quality is hard. | This branch bridges systems roofline evidence to language-model quality experiments. | summary_only |
| DCLM benchmark-protocol diagnostic | caution | Step-68100 diagnostic reports DCLM PPL 113.806177 and WT2 continuity PPL 207.135819; 256-example micro-scores stay near chance. | Treat as a benchmark-protocol diagnostic, not official CORE/MMLU/EXTENDED scoring. | included |
| 7B rSVD-GaLore scratch memory-feasibility | caution | Rank-64 rSVD-GaLore reaches 202.127 PPL on the 6.713B SNN-transformer shape. | It proves single-3090 training feasibility with rSVD-GaLore, not public checkpoint parity. | external_missing |
| Teacher and distillation calibration | negative | A 1000-step top-k distillation run reaches PPL 217.724. | Alpha 0.2 improves over high-alpha distillation but still misses the best scratch memory-feasibility result. | external_missing |
| Public quantized checkpoint reproduction | positive | The public quantized checkpoint reproduction reaches WikiText2 PPL 9.970943 and C4 PPL 12.820610. | This is the fair public quality-control target; the stronger-checkpoint chase was stopped. | included summary |
| Locked metric and calibration push | evidence | Locked seq2048/eight-window metric makes the teacher/distillation branch harder: 12.137 baseline, 11.962 best calibration result. | The short-window result is fragile; locked long-context evaluation is the comparison point. | external_missing |
| Output-side repair ladder | positive | Hidden-channel affine output repair reaches 11.050 PPL; soft clipping produces 250.898 and 202.168 PPL. | Output-side repair is real, but the tested soft-clip path is destructive. | external_missing |
| Clean hidden and pre-activation affine | positive | Clean hidden plus pre-activation affine reaches 10.980 PPL. | Best local output-side baseline for later saliency and dense calibration, still 1.101x the public quantized checkpoint result. | external_missing |
| Uniform deeper wrapping failure | negative | Last-four residual bridge regresses to 12.583 PPL. | More wrapped layers are not automatically better. | external_missing |
| Trainable saliency smoke | caution | Short seq512 / eval_iters=2 saliency smoke reports 8.519 PPL. | Strong smoke result only; it is not comparable to locked public quality-control or clean hidden affine metrics. | external_missing |
| Locked soft saliency miss | negative | Best soft hidden saliency reaches 11.6916 PPL, worse than the clean hidden affine baseline at 10.9803 PPL. | Locked saliency does not improve quality. | external_missing |
| Hard-mask bridge robustness branch | negative | Hard top-k reaches 11.8336/11.8589; bridge variants reach 14.0803/14.1415; offset robustness reaches 16.6474. | Hard masking, naive bridge expansion, and offset validation do not rescue saliency. | external_missing |
| Full saliency rule shootout | negative | Locked full run ranks no-saliency first at 11.1913 PPL; best saliency is 11.6513 PPL. | Do not continue pretrained saliency scale-up from this wrapper. | external_missing |
| True scratch saliency shootout | negative | Fixed timestep reaches 3.3490 BPB; no-saliency reaches 3.3504 BPB; learned saliency reaches 3.3602-3.3674 BPB. | Learned-saliency scratch scale-up stays closed. | external_missing |
| Hessian-error saliency guard | negative | Hessian-seeded joint saliency reaches 3.3371 BPB, but the gain is 0.40% and throughput regresses. | Aligned score machinery is useful instrumentation but misses the quality and efficiency continuation rule. | external_missing |
| Signed subset Taylor saliency guard | negative | Signed subset Taylor diagnostic gives Spearman -0.2571 and top-k overlap 0.0; the follow-up is skipped. | Signed normalization-aware saliency still fails oracle alignment. | external_missing |
| Final subset oracle | negative | Early mask 1100 reaches 3.341844 BPB, only about 0.255% better than no-saliency. | Local Taylor ranking and the late/settling prior fail; later checks show the gain is unstable. | external_missing |
| One-step lookahead | caution | One-step validation predicts 1100 with hard-subset Spearman 0.6000 against the final subset BPB. | Useful single-seed evidence only; repeat-seed robustness rejects it. | external_missing |
| Repeat-seed robustness check | negative | Fresh seed ranks dense no-saliency best at 3.365828 BPB; one-step 1100 becomes worst hard subset. | One-step lookahead is not robust. | external_missing |
| Longer-horizon selector | negative | Horizons 10/30/100 select 0011, 0110, and 1010, missing the seed-4321 final winners. | Cheap short/medium horizon selection is rejected. | external_missing |
| Multi-seed variance check | negative | Best mean hard-mask gain is 0.002098 BPB, below duplicate-eval noise. | Timestep-mask selection is variance dominated and below-noise across seeds. | external_missing |
| Dense low-LR scratch screen | caution | LR 3e-4 improves both seeds by mean 0.089184 BPB / 2.657603%, but no row passes the two-seed >=5% and >=2x-noise rule. | This does not prove a positive theory, but it points away from saliency and toward low-LR optimization stability. | included |
| Dense low-LR scratch stability verification | positive | LR 3e-4 for 2000 dense no-saliency steps reaches 2.108004 and 2.080363 BPB, mean gain 37.815032%. | Positive scratch theory is optimization stability under this profile. | included summary |
| Pretrained low-LR transfer check | caution | Clean hidden affine wrapper with LR 7.5e-5 for 1400 steps reaches 10.889457 PPL. | The low-LR horizon theory transfers, but it remains 0.918514 PPL above the public quantized checkpoint reproduction. | external_missing |
| Scratch architecture search | negative | Best tested candidate D768/L4 has mean BPB 2.187050, worse than the optimized baseline. | Tested architecture changes do not create a verified positive theory. | external_missing |
| Long-horizon dense scratch verification | positive | LR 3e-4 for 6000 scratch steps reaches mean final BPB 1.804832 and improves 13.815176%. | Dense no-saliency low-LR training remains horizon-limited beyond 2000 steps. | included summary |
| Pretrained low-LR horizon near miss | caution | LR 7.5e-5 for 6000 pretrained calibration reaches 10.027895 PPL. | Strong local pretrained lever, but still 0.056951 PPL above the public quantized checkpoint reproduction. | included summary |
| Eval Nsight counter parse check | inconclusive | RTX 3090 eval profiling produced zero parsed counter samples. | Eval comparisons remain functional evidence only, not counter-backed kernel attribution. | external_missing |
| Eval energy rejection checks | negative | The C4 quality run passes C4 quality but uses +2.824% total energy/token; sparse variants fail quality. | No eval energy-win claim is valid under the C4 quality bound. | external_missing |
| Decision | evidence | Paper core is solid; saliency is closed; dense optimization nearly reaches the public quantized checkpoint reproduction but not parity. | Trust the bounded systems claim, public quantized checkpoint reproduction as the fair control, pretrained low-LR calibration as a near miss, and negative evidence as closed. | summary_only |

## Artifact Coverage

| Artifact status | Meaning | Public paths |
| --- | --- | --- |
| included | Lightweight evidence is copied into this repository. | `artifacts/dclm_protocol_diagnostics`, `artifacts/dense_low_lr_scratch_screen` |
| summary_only | The public ledger records the source mind-map interpretation, but no per-run evidence file is included here. | `data/technology_milestones.yaml`, this page |
| external_missing | The source mind map names or implies detailed artifacts that are not present in the current public evidence slice. | Future public releases should add small summaries, checksums, or result tables before changing this status. |
