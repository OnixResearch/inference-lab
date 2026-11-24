# Issue #76: Blink CPU-Free RDMA Serving & Host-CPU Contention Resilience

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Artifact gate: `blocked_on_artifacts` (`rdma_gate_outcome: blocked_on_artifacts`)

## Host-Scheduled vs GPU-Resident Scheduling Under Host-CPU Contention
| Scheduler | Interference | P50 TTFT (ms) | P99 TTFT (ms) | P99 TTFT / Idle | ITL Jitter (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| `host_scheduled_vllm_sglang` | `idle` | `18.05` | `20.55` | `1.000` | `0.45` |
| `host_scheduled_vllm_sglang` | `moderate` | `21.65` | `33.25` | `1.618` | `3.10` |
| `host_scheduled_vllm_sglang` | `heavy` | `28.35` | `72.45` | `3.526` | `10.70` |
| `gpu_resident_persistent` | `idle` | `17.31` | `18.64` | `1.000` | `0.15` |
| `gpu_resident_persistent` | `moderate` | `17.31` | `18.64` | `1.000` | `0.15` |
| `gpu_resident_persistent` | `heavy` | `17.31` | `18.64` | `1.000` | `0.15` |

## Negative Controls (8 / 8 rejected)
- `nc1_unpinned_blink_artifacts_blocked` -> `blocked_on_artifacts` (missing public Blink RDMA/persistent-kernel code returns blocked_on_artifacts)
- `nc2_silent_tcp_fallback_rejected` -> `invalid` (silently falling back from GPUDirect RDMA to host TCP invalidates CPU-free test)
- `nc3_missing_smartnic_gpudirect_unsupported` -> `unsupported` (host without GPUDirect RDMA SmartNIC returns unsupported)
- `nc4_cross_pcie_root_topology_gated` -> `unavailable_hardware_runtime` (NIC and GPU on separate PCIe root complexes cannot claim zero-hop RDMA)
- `nc5_empty_latency_population_never_zero` -> `invalid` (empty TTFT/ITL sample returns None rather than collapsing to 0.0 ms)
- `nc6_nan_latency_sample_rejected` -> `invalid` (non-finite latency measurement rejected by percentile reducer)
- `nc7_oob_percentile_rejected` -> `invalid` (percentile outside [0, 100] rejected)
- `nc8_unverified_8_47x_claim_non_gating` -> `invalid` (Blink's 8.47x P99 TTFT claim stays unverified related work, never a gate)
