# Issue #40 — Host and GPU trace correlation qualification

- **Execution class**: `bounded_contract_harness`
- **Physical hardware gate**: `unavailable_hardware_runtime`

## Positive and negative control outcomes

| Control | Causal path substantiated | Status | Errors | Unresolved metadata | Unsupported conclusions |
| --- | --- | --- | --- | --- | --- |
| `positive-known-kernel-copy-sync` | `true` | `qualified_positive_control` | 0 | 0 | 1 |
| `neg-wrong-run-and-module-identity` | `false` | `incomplete_cannot_substantiate` | 4 | 0 | 1 |
| `neg-unmapped-clock-domain` | `false` | `incomplete_cannot_substantiate` | 4 | 0 | 1 |
| `neg-swapped-stream-and-device-ids` | `false` | `incomplete_cannot_substantiate` | 2 | 0 | 1 |
| `neg-duplicate-out-of-order-and-collision` | `false` | `incomplete_cannot_substantiate` | 1 | 0 | 1 |
| `neg-dropped-events-and-overflow` | `false` | `incomplete_cannot_substantiate` | 2 | 0 | 1 |
| `neg-missing-drop-counter-not-zero` | `false` | `incomplete_cannot_substantiate` | 1 | 0 | 1 |
| `neg-late-metadata-unresolved` | `false` | `incomplete_cannot_substantiate` | 0 | 1 | 1 |

Only the calibrated, lossless positive control substantiates a causal path, while its multi-request batched kernel still keeps per-request attribution explicitly `unavailable`. Performance ranking remains restricted to uninstrumented trials.
