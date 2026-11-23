# Issue #29: RustingStrata Expert Placement & Host Residency vs llama.cpp

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`discrete_nvidia_node_flint_ud_q5_k_xl_rustingstrata_vs_llamacpp`)
- Decision: `INCONCLUSIVE_PHYSICAL_DISCRETE_NVIDIA_EXECUTION_REQUIRED`

## Negative controls
- `mismatched_gguf_identity_rejected` -> `invalid` (mismatched GGUF digest on trial `rep-1`)
- `mislabeled_cold_state_persisted_profile` -> `invalid` (mislabeled cold state: routing_profile=persisted_from_prior_run, kv=reset_empty, adaptive=reset_empty)
- `duplicate_paired_trial_rejected` -> `invalid` (duplicate trial_id `rep-1` detected)
- `missing_paired_repetitions_rejected` -> `invalid` (at least 3 paired repetitions are required)
- `incorrect_free_gen_task_output_rejected` -> `invalid` (independent checker rejects subtly wrong subtraction implementation)
- `unobserved_router_equivalence_stays_unverified` -> `invalid` (routing_equivalence=unverified_router_telemetry_absent)
- `madvise_and_unified_memory_non_freeing_guards` -> `invalid` (madvise_rejected=true, unified_swap_rejected=true)
- `custom_license_vendoring_gate_enforced` -> `unsupported` (Custom LICENSE.md: commercial apps allowed; engine-resale restriction, attribution, and same-license terms for substantial portions (evaluate externally before vendoring))
