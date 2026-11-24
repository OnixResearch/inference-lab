# Issue #91: Perseus Decoupled RDMA Signaling & MPK Persistent Kernel Related Work

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (2-node DGX Spark RDMA decode cluster (Perseus / MPK / ForgeMegakernel))

## 2-Node RDMA Signaling Mode Comparison
- `naive_proxy_put_signal`: `parity_err=0.000e0`, `ordering_ok=true`, `fence_stall=1280.0us` (`65.8%`), `step_total=1944.0us`, `timing_admitted=true`
- `perseus_decoupled_signaling`: `parity_err=0.000e0`, `ordering_ok=true`, `fence_stall=48.0us` (`11.1%`), `step_total=432.0us`, `timing_admitted=true`
- `mpk_sm_persistent_graph`: `parity_err=0.000e0`, `ordering_ok=true`, `fence_stall=96.0us` (`20.0%`), `step_total=480.0us`, `timing_admitted=true`

## Negative Controls (8 / 8 rejected)
- `nc1_out_of_order_nic_completion_rejected` -> `invalid` (missing NIC ordering fence produces collective parity failure and blocks timing)
- `nc2_silent_tcp_fallback_rejected` -> `invalid` (silent TCP socket fallback refused when RDMA NIC absent)
- `nc3_single_node_mislabeled_two_node_rejected` -> `invalid` (node_count < 2 rejected by 2-node RDMA qualification gate)
- `nc4_unpinned_perseus_mpk_artifacts_blocked` -> `blocked_on_artifacts` (unpinned Perseus/MPK/ForgeMegakernel artifacts return BlockedOnArtifacts)
- `nc5_missing_gpudirect_rdma_hardware_unavailable` -> `unavailable_hardware_runtime` (absent GPUDirect RDMA NIC returns UnavailableHardwareRuntime)
- `nc6_nonfinite_wire_latency_rejected` -> `invalid` (NaN wire latency rejected cleanly)
- `nc7_negative_fence_penalty_rejected` -> `invalid` (negative NIC fence stall duration rejected)
- `nc8_unverified_10x_regression_claim_non_gating` -> `invalid` (authors' unverified 10x naive proxy RDMA regression figure is non-gating related work)
