# Artifact Index

| Artifact set | Contents | Exclusions |
| --- | --- | --- |
| `gate74_diagnostics` | Gate74 step-68100 benchmark protocol diagnostic summary. | Checkpoints, generated benchmark caches, and archival runner source. |
| `gate129_dense_scaling` | Gate129 dense-scaling summary, results, theory summary, best summary, sweep log, and compact log snippets. | Full checkpoints, dataset shards, and large scratch directories. |
| `external_gate_summaries` | Compact summary rows for Gate89, Gate130, Gate133, and Gate134 extracted from the supplied gate map. | Raw logs, raw JSONL files, checkpoints, and datasets. |

These files are lightweight evidence for the public ledger. Large parquet files,
checkpoints, and private scratch paths are intentionally not copied into this
repository.

`external-artifact:<name>` references point to raw experiment artifacts that are
not included in this lightweight public slice.
