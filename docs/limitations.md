# Limitations And Closed Gates

This file records the boundaries that must travel with any public SpikeBudget
claim. The negative and inconclusive gates are part of the result, not cleanup
items.

| Area | Gate evidence | Public limitation |
| --- | --- | --- |
| Claim scope | Claim, G25/G27, G40 | The project studies specific temporal-gradient schedules and kernel shapes. It does not prove that every SNN workload is compute-bound on GPUs. |
| Synthetic-history correction | G41 | Earlier synthetic reviewer-facing timing history was replaced with real CUDA kernels and cudaEvent timing. Keep the correction visible. |
| Framework comparisons | G53-54 | SpikingJelly, snnTorch, and BindsNET comparisons are context, not a broad framework benchmark. |
| 7B-class training | G77-78, Gate74 | Single-GPU 7B-class training is memory-feasible evidence. It is not evidence of SpikeLLM quality parity. |
| Distillation | G80-83 | Top-k distillation learned but did not beat the scratch best or approach Gate89 quality. |
| Official quality control | G89 | Gate89 official SpikeLLM W4A4 remains the fair public control at WikiText2 PPL 9.970943. |
| Local pretrained repair | G95b, G131, G134 | Gate134 reaches 10.027895 PPL and is a near miss, not a parity result. |
| Pretrained saliency | G97-102, G108 | Smoke saliency did not survive locked metrics or the full saliency-rule shootout. |
| Scratch saliency | G111, G115-128 | Learned timestep saliency, hessian seeding, signed subset scoring, and hard-mask selectors do not pass continuation rules. |
| Architecture search | G132 | Tested tied head, RMSNorm, SwiGLU, seq256, and D768/L4 rows regress against the optimized baseline. |
| Eval counters | G103 | RTX 3090 eval Nsight profiles produced zero parsed counter samples, so eval claims are not counter-backed. |
| Eval energy | G104-106 | Sparse eval variants fail quality and Gate91 uses more total energy/token than raw FP16. No eval energy-win claim is valid. |
| Public artifact coverage | Current repository | Only lightweight Gate74 and Gate129 artifacts are included. Other gates are summary-only or marked external-missing until small public evidence files are added. |
| Runner coverage | Current repository | This slice includes configs, run contracts, validation code, and lightweight evidence. The full 3090 GPU training runner and checkpoints are external release items. |
