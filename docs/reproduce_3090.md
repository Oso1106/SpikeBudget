# RTX 3090-Class Reproduction Runbook

This runbook defines the public environment and evidence contract for
3090-class SpikeBudget train-from-scratch runs. It avoids private host names,
private scratch paths, and external weight assumptions. This repository
contains configs and validation code; the full GPU training runner is an
external release item.

## Hardware And Software Contract

| Requirement | Public expectation | Notes |
| --- | --- | --- |
| GPU | RTX 3090-class CUDA GPU with 24 GiB VRAM | Adjacent consumer or workstation GPUs can run smoke jobs, but long-run memory/timing claims should be labeled by GPU model. |
| Driver | NVIDIA driver compatible with the selected PyTorch CUDA build | Record `nvidia-smi` output in each run artifact bundle. |
| Python | Python version supported by the public runner release | Use a clean virtual environment per release. |
| PyTorch | CUDA-enabled PyTorch build | Record `python -c "import torch; print(torch.__version__, torch.version.cuda)"`. |
| Datasets | Publicly fetchable datasets only | Put caches under `SPIKEBUDGET_DATA_DIR` or `HF_HOME`; do not commit datasets. |
| Models | Scratch model definitions from the config | Public evidence lanes instantiate model weights from scratch. |

## Environment Variables

| Variable | Example | Purpose |
| --- | --- | --- |
| `SPIKEBUDGET_ARTIFACT_DIR` | `$PWD/artifacts/runs/dense_low_lr_scratch_smoke` | Destination for configs, logs, summaries, and small result tables. |
| `SPIKEBUDGET_DATA_DIR` | `$HOME/.cache/spikebudget/data` | Dataset cache root. |
| `HF_HOME` | `$HOME/.cache/huggingface` | Hugging Face cache root. |
| `CUDA_VISIBLE_DEVICES` | `0` | Select the 3090-class GPU. |
| `SPIKEBUDGET_SEED` | `1234` | Seed used in smoke or long runs. |
| `SPIKEBUDGET_PROFILE` | `smoke` or `long_3090` | Selects smoke or long-run configuration in the public runner release. |

## Source Checkout Commands

| Check | Command | Expected result |
| --- | --- | --- |
| Install scaffold | `python -m pip install -e '.[dev]'` | Installs validators and test dependencies. |
| Validate public artifacts | `python -m scripts.validate_repo` | Confirms docs, configs, ledger, and included artifacts are internally consistent. |
| Run fast tests | `python -m pytest -q` | Exercises config parsing, evidence validation, and deterministic helper functions. |

## Smoke Run Contract

| Phase | Action | Required evidence |
| --- | --- | --- |
| Hardware capture | Save `nvidia-smi`, Python, CUDA, and PyTorch metadata. | `hardware.txt` or equivalent metadata table. |
| Dataset probe | Load a tiny public dataset slice and verify tokenization/cache paths. | Dataset id, split, cache root, and sample count. |
| Model probe | Instantiate the scratch SNN shape at reduced size. | Parameter count, dtype, sequence length, batch size, timesteps. |
| Short train/eval | Run a bounded smoke profile, usually minutes rather than hours. | Config, stdout/stderr log, final metric, peak memory, tokens/s. |
| Ledger mapping | Label the smoke as diagnostic unless it matches a published technology-lane protocol. | Evidence id, status, and whether the result is comparable to a locked milestone. |

## Long 3090-Class Run Contract

| Phase | Action | Required evidence |
| --- | --- | --- |
| Preflight | Run the smoke profile successfully on the same environment. | Smoke artifact path and timestamp. |
| Config freeze | Save the exact config before launch. | Config file, git revision, package version, command line. |
| Runtime logging | Emit periodic step metrics and memory/timing summaries. | JSONL or TSV logs with step, train metric, eval metric, tokens/s, peak memory. |
| Evaluation | Use the technology-lane locked metric and validation window. | Metric name, dataset, sequence length, eval iterations, seed/window ids. |
| Decision rule | Apply the published continuation or rejection rule before making claims. | Small summary table with pass/fail and reason. |
| Artifact pruning | Keep summaries, result tables, and log snippets in git; keep checkpoints and datasets outside git. | Public artifact bundle with no large parquet files or checkpoints. |
| Best scratch checkpoint handoff | Preserve the best long-horizon dense scratch checkpoint outside git and record its release location and checksum. | The public reference is `external-artifact:long_horizon_dense_scratch_lr3e4_steps6000_best_bpb_checkpoint`; metadata lives in `artifacts/external_scratch_summaries/best_scratch_checkpoint.tsv`. |

## Command Template

| Run type | Template | What to replace |
| --- | --- | --- |
| Smoke | `SPIKEBUDGET_PROFILE=smoke SPIKEBUDGET_ARTIFACT_DIR=$PWD/artifacts/runs/smoke CUDA_VISIBLE_DEVICES=0 $SPIKEBUDGET_RUNNER` | Replace `$SPIKEBUDGET_RUNNER` with the external GPU runner command for the release you are testing. |
| Dense low-LR scratch smoke | `SPIKEBUDGET_PROFILE=dense_low_lr_scratch_smoke SPIKEBUDGET_SEED=1234 SPIKEBUDGET_ARTIFACT_DIR=$PWD/artifacts/runs/dense_low_lr_scratch_s1234 CUDA_VISIBLE_DEVICES=0 $SPIKEBUDGET_RUNNER` | Use the public config that matches d_model, layer count, LR, steps, and seed. |
| Long dense horizon | `SPIKEBUDGET_PROFILE=long_3090 SPIKEBUDGET_SEED=1234 SPIKEBUDGET_ARTIFACT_DIR=$PWD/artifacts/runs/long_s1234 CUDA_VISIBLE_DEVICES=0 $SPIKEBUDGET_RUNNER` | Confirm disk, time budget, and checkpoint storage before launch. |

## Expected Public Artifact Layout

| Path | Contents | Commit policy |
| --- | --- | --- |
| `artifacts/runs/<run>/config.*` | Frozen run config | Include if small and free of private paths. |
| `artifacts/runs/<run>/hardware.txt` | GPU, driver, Python, CUDA, PyTorch metadata | Include. |
| `artifacts/runs/<run>/summary.md` | Evidence result and decision rule | Include. |
| `artifacts/runs/<run>/results.tsv` | Compact result table | Include. |
| `artifacts/runs/<run>/log_snippets.md` | Start, warning, eval, and final summary snippets | Include. |
| `artifacts/runs/<run>/*.jsonl` | Full step logs | Include only when lightweight; otherwise publish a summary and checksum. |
| `checkpoints/**` | Model checkpoints | Do not commit. |
| `data/**/*.parquet` | Dataset shards | Do not commit. |

## Technology-Lane Notes

| Technology lane | Reproduction rule | Caveat |
| --- | --- | --- |
| DCLM benchmark-protocol diagnostic | Treat the provided diagnostics as micro-screens. | Do not label CORE/MMLU/EXTENDED diagnostic scores as official benchmark scores. |
| full saliency rule shootout / true scratch saliency shootout | Compare saliency rules against matched no-saliency controls. | Existing evidence closes saliency scale-up; a new run needs a predeclared continuation rule. |
| dense low-LR scratch screen | Use two seeds and compare against same-seed 300-step dense baselines. | dense low-LR scratch screen is screen-only because no row passed >=5% and >=2x-noise. |
| dense low-LR scratch stability and long-horizon verification | Longer low-LR dense no-saliency training is the verified positive scratch direction. | Keep the claim to scratch BPB horizon, not broader model-quality comparisons. |
| task-level NVIDIA L4 benchmark | Report the matched optimized path as 128.75x memory-cheaper and 6.78x faster on NVIDIA L4. | NVIDIA L4 is a separate datacenter GPU model name, not NVIDIA L40. |
| architecture labels such as D768/L4 | Interpret D768 as d_model 768 and L4 as four model layers. | This L4 suffix is an architecture abbreviation, not a GPU model. |

## External Artifact References

| Convention | Meaning |
| --- | --- |
| `external-artifact:<name>` | A raw log, generated checkpoint, or generated artifact that existed in the experiment workspace but is not included in this lightweight public slice. |
| `artifacts/external_scratch_summaries/*` | Compact scratch summary evidence included in this repository. These rows are not raw logs and should not be treated as full reproduction bundles. |
