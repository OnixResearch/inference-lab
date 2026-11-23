# Issue #34: Phoenix Nano Training, Checkpoint Resume & gRPC Serving

- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (One-node qualified GPU (A100/H100/H200/GB200/GB300) JAX/CUDA + Rust gRPC Phoenix serve)
- Resume parity: `max_err = 0.000e0`, Direct-vs-gRPC parity: `max_err = 0.000e0` (`4 items ranked`)

## GPU Cohort Qualification (`gpu_arch()`)
- `GB300` -> `unavailable_hardware_runtime` (recognized GPU architecture but no physical JAX/CUDA device executed locally)
- `GB10` -> `unsupported` (rejected by upstream gpu_arch() cohort (only A100/H100/H200/GB200/GB300 recognized))
- `RTX 5080` -> `unsupported` (rejected by upstream gpu_arch() cohort (only A100/H100/H200/GB200/GB300 recognized))

## Negative Controls (9 / 9 rejected)
- `nc1_shifted_wire_vs_model_sid_detected` -> `invalid` (wire SID 0 maps to model SID 1; model SID 0 stays strictly reserved for missing)
- `nc2_out_of_range_sid_rejected` -> `invalid` (wire SID >= codebook_size fails codec validation)
- `nc3_codebook_digest_mismatch_rejected` -> `invalid` (mismatched codebook digest refused at fresh-process restore)
- `nc4a_dropped_optimizer_state_only_on_resume` -> `invalid` (dropping AdaGrad/AdamW optimizer state with intact cursor caused 1.039e-1 drift)
- `nc4b_stale_data_cursor_on_resume` -> `invalid` (resetting data cursor on checkpoint resume fails continuation check)
- `nc5_gpu_arch_name_override_rejected` -> `invalid` (device-name override rejected: overriding gpu_arch() is not kernel-correctness evidence)
- `nc6_candidate_attention_isolation_leak` -> `invalid` (cross-candidate leakage shifted scores by 3.239e-2)
- `nc7_direct_vs_served_score_mismatch` -> `invalid` (served gRPC score discrepancy against direct inference retained and rejected)
- `nc8_readiness_only_without_retrieve_rank` -> `invalid` (process readiness alone cannot pass without completed retrieve->rank request)
