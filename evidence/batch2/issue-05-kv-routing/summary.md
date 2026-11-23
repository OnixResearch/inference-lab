# Issue #5: Session KV Routing vs Host-RDMA Cache Migration vs Recomputation

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`multi_replica_gpu_rdma_fabric_and_tt_to_nvidia_hosts`)
- Decision: `INCONCLUSIVE_UNAVAILABLE_HARDWARE_RUNTIME`

## Snapshot Compatibility & Fencing Controls
| Case | Compatible | Quant Collapse | Disposition |
| --- | --- | --- | --- |
| `positive-exact-same-variant` | `true` | `false` | `snapshot_admitted_exact_reuse` |
| `observation-quantization-collapse-same-effective-weights` | `true` | `true` | `snapshot_admitted_exact_reuse` |
| `negative-rsi86-same-config-different-weights` | `false` | `false` | `refused_cross_variant_reuse_declared_recompute_required` |
| `negative-adapter-replacement` | `false` | `false` | `refused_cross_variant_reuse_declared_recompute_required` |
| `negative-stale-deployment-generation` | `false` | `false` | `fenced_rejected_late_generation_response` |
| `negative-heterogeneous-sm100a-to-sm121` | `false` | `false` | `refused_incompatible_snapshot_declared_recompute` |
| `negative-tt46-translated-state-parity-fail-despite-task-pass` | `false` | `false` | `refused_incompatible_snapshot_declared_recompute` |

## ChaosControl Transfer & Fencing Faults
- `chaos-donor-evicted-before-transfer`: `safe_cache_miss_declared_recompute` -> `accepted` (safe cache-miss fallback: `true`)
- `chaos-redis-ownership-fence-violated`: `fatal_consistency_fence_violation` -> `invalid` (safe cache-miss fallback: `false`)
- `chaos-torn-rdma-payload-digest-mismatch`: `fatal_transport_integrity_failure` -> `internal_failure` (safe cache-miss fallback: `false`)
- `chaos-receiver-kv-oom-during-restore`: `receiver_capacity_exhausted` -> `oom` (safe cache-miss fallback: `false`)

> Bounded contract harness only. No multi-replica RDMA fabric or physical TT-to-NVIDIA transfer ran locally.
