# DCLM benchmark-protocol diagnostic Step 68100 Benchmark Protocol Debug

Scratch run artifact: `external-artifact:dclm_protocol_stream_s50000_best_ppl_model.pt`

Step: local `46500`, global `68100`

Validation: DCLM PPL `113.8061772770991`, WT2 continuity PPL `207.13581907254456`

## Results

| Scoring mode | CORE | MMLU | EXTENDED |
|---|---:|---:|---:|
| Original 32-example smoke, raw answer text | `0.375` | `0.25` | `0.21875` |
| 256-example raw answer text | `0.3359375` | `0.25390625` | `0.17578125` |
| 256-example length-normalized answer text | `0.265625` | `0.265625` | `0.2421875` |
| 256-example label scoring with choices shown | `0.28515625` | `0.26953125` | `0.22265625` |
| Expected random chance for 256 sample | `0.3328125` | `0.25` | `0.2338541666666671` |
| Majority-answer baseline for 256 sample | `0.39453125` | `0.265625` | `0.26953125` |

## Interpretation

| Finding | Evidence | Interpretation |
|---|---|---|
| The original 32-example result was noisy but directionally real | 256-example raw mode stayed near chance: CORE `0.3359`, MMLU `0.2539`, EXTENDED `0.1758` | Low scores were not just a 32-example sampling accident |
| Length normalization does not explain the issue | Length-normalized results: CORE `0.2656`, MMLU `0.2656`, EXTENDED `0.2422` | Raw continuation length bias is present, but fixing it does not recover meaningful accuracy |
| Showing answer options and scoring labels does not rescue performance | Label results: CORE `0.2852`, MMLU `0.2695`, EXTENDED `0.2227` | Hidden options are not the sole root cause |
| Label scoring exposed a strong A-label bias | Label scorer predicted answer index `0` for most examples | The scratch run artifact does not yet robustly condition on MCQ option semantics |
| These are diagnostic micro-scores, not official benchmark scores | Artifacts use `built_in_dclm_protocol_diagnostic; not official DCLM centered CORE/EXTENDED` | Report as micro screens, not official CORE/MMLU/EXTENDED |

## Artifact Paths

| Mode | External artifact prefix |
|---|---|
| Raw text | `external-artifact:dclm_step68100_diag256_cachefix_raw_text_*` |
| Length-normalized text | `external-artifact:dclm_step68100_diag256_cachefix_len_norm_text_*` |
| Label scoring | `external-artifact:dclm_step68100_diag256_cachefix_label_*` |

The archival diagnostic runner is intentionally not included in this public
evidence slice. Public runs should use the release runner and configs in this
repository rather than unpublished scratch-host scripts.
