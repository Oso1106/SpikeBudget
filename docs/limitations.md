# Limitations And Closed Evidence

This file records the boundaries that must travel with any public SpikeBudget
claim. The negative and inconclusive checks are part of the result, not cleanup
items.

| Area | Evidence lane | Public limitation |
| --- | --- | --- |
| Claim scope | Bounded claim contract, kernel-shape comparison, task-level temporal training CUDA benchmark | The project studies specific temporal-gradient schedules and kernel shapes. It does not prove that every SNN workload is compute-bound on GPUs. |
| Synthetic-history correction | Real CUDA timing correction | Earlier synthetic reviewer-facing timing history was replaced with real CUDA kernels and cudaEvent timing. Keep the correction visible. |
| Framework comparisons | Framework baseline comparison | SpikingJelly, snnTorch, and BindsNET comparisons are context, not a broad framework benchmark. |
| 7B-class training | 7B rSVD-GaLore scratch memory-feasibility, DCLM benchmark-protocol diagnostic | Single-GPU 7B-class training is memory-feasible evidence. It is not evidence of public quantized checkpoint quality parity. |
| Distillation | Teacher and distillation calibration | Top-k distillation learned but did not beat the scratch best or approach public quantized checkpoint reproduction quality. |
| Official quality control | Public quantized checkpoint reproduction | The public quantized checkpoint reproduction remains the fair public control at WikiText2 PPL 9.970943. |
| Local pretrained repair | Clean hidden and pre-activation affine, pretrained low-LR transfer, pretrained low-LR horizon near miss | Pretrained low-LR horizon calibration reaches 10.027895 PPL and is a near miss, not a parity result. |
| Pretrained saliency | Trainable saliency smoke, locked soft saliency miss, hard-mask bridge robustness, full saliency rule shootout | Smoke saliency did not survive locked metrics or the full saliency-rule shootout. |
| Scratch saliency | True scratch saliency shootout, hessian-error saliency guard, signed subset Taylor saliency guard, final subset oracle, one-step lookahead, repeat-seed robustness, longer-horizon selector, multi-seed variance check | Learned timestep saliency, hessian seeding, signed subset scoring, and hard-mask selectors do not pass continuation rules. |
| Architecture search | Scratch architecture search | Tested tied head, RMSNorm, SwiGLU, seq256, and D768/L4 rows regress against the optimized baseline. |
| Eval counters | Eval Nsight counter parse check | RTX 3090 eval Nsight profiles produced zero parsed counter samples, so eval claims are not counter-backed. |
| Eval energy | Eval energy rejection checks | Sparse eval variants fail quality and the C4 quality run uses more total energy/token than raw FP16. No eval energy-win claim is valid. |
| Public artifact coverage | Current repository | Only lightweight DCLM benchmark-protocol diagnostic and dense low-LR scratch screen artifacts are included. Other evidence lanes are summary-only or marked external-missing until small public evidence files are added. |
| Runner coverage | Current repository | This slice includes configs, run contracts, validation code, and lightweight evidence. The full 3090 GPU training runner and checkpoints are external release items. |
