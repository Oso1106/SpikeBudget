# Artifact Index

| Artifact set | Contents | Exclusions |
| --- | --- | --- |
| `dclm_protocol_diagnostics` | DCLM benchmark-protocol diagnostic summary. | Checkpoints, generated benchmark caches, and archival runner source. |
| `dense_low_lr_scratch_screen` | Dense low-LR scratch summary, results, theory summary, best summary, sweep log, and compact log snippets. | Full checkpoints, dataset shards, and large scratch directories. |
| `external_scratch_summaries` | Compact summary rows for dense low-LR scratch stability verification, long-horizon dense scratch verification, and the best trained-from-scratch checkpoint reference extracted from the supplied evidence map. | Raw logs, raw JSONL files, generated checkpoint bytes, and datasets. |

These files are lightweight evidence for the public ledger. Large parquet files,
checkpoints, and private scratch paths are intentionally not copied into this
repository.

`external-artifact:<name>` references point to raw experiment artifacts that are
not included in this lightweight public slice.
