# Limitations And Closed Evidence

This file records the boundaries that must travel with any public SpikeBudget
scratch-training claim. The negative and inconclusive checks are part of the
result, not cleanup items.

| Area | Evidence lane | Public limitation |
| --- | --- | --- |
| Claim scope | Bounded claim contract, kernel-shape comparison, task-level temporal training CUDA benchmark | The project studies specific temporal-gradient schedules and kernel shapes. It does not prove that every SNN workload is compute-bound on GPUs. |
| Synthetic-history correction | Real CUDA timing correction | Earlier synthetic reviewer-facing timing history was replaced with real CUDA kernels and cudaEvent timing. Keep the correction visible. |
| Framework comparisons | Framework baseline comparison | SpikingJelly, snnTorch, and BindsNET comparisons are context, not a broad framework benchmark. |
| 7B-class scratch training | 7B rSVD-GaLore scratch memory-feasibility, DCLM benchmark-protocol diagnostic | Single-GPU 7B-class training from scratch is memory-feasible evidence. It is not evidence that language-model quality is solved. |
| Scratch saliency | True scratch saliency shootout, hessian-error saliency guard, signed subset Taylor saliency guard, final subset oracle, one-step lookahead, repeat-seed robustness, longer-horizon selector, multi-seed variance check | Learned timestep saliency, hessian seeding, signed subset scoring, and hard-mask selectors do not pass continuation rules. |
| Dense low-LR scratch optimization | Dense low-LR scratch screen, dense low-LR scratch stability verification, long-horizon dense scratch verification | This is the current positive scratch direction, but it is bounded to the documented BPB profile and must not be generalized beyond that evidence. |
| Architecture search | Scratch architecture search | Tested tied head, RMSNorm, SwiGLU, seq256, and D768/L4 rows regress against the optimized baseline; in D768/L4, L4 means four layers, not the NVIDIA L4 GPU. |
| Eval counters | Eval Nsight counter parse check | RTX 3090 eval Nsight profiles produced zero parsed counter samples, so eval claims are not counter-backed. |
| Eval energy | Eval energy rejection checks | Sparse eval variants fail quality and the C4 quality run uses more total energy/token than the dense reference. No eval energy-win claim is valid. |
| Public artifact coverage | Current repository | Lightweight DCLM benchmark-protocol diagnostics, dense low-LR scratch artifacts, external scratch summaries, and the best trained-from-scratch checkpoint reference are included. Other evidence lanes are summary-only or marked external-missing until small public evidence files are added. |
| Runner coverage | Current repository | This slice includes configs, run contracts, validation code, and lightweight evidence. The full 3090 GPU training runner and generated checkpoints are external release items. |
