# Issue #28: KV Compression Diagnostics & Grounded End-Task Contract

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`physical_gpu_transformers_kvpress_long_context`)
- Identity/no-compression control max error: `0.000e0`
- Decision: `INCONCLUSIVE_PHYSICAL_GPU_KVPRESS_EVAL_REQUIRED`

## Negative controls
- `removed_decisive_evidence_fails_end_task` -> `invalid` (decisive_pos_retained=false)
- `corrupted_position_mapping_detected` -> `invalid` (corrupted_pos_attention_err=0.043771)
- `stale_cache_from_other_request_rejected` -> `invalid` (expected=req-64, observed=req-other)
- `superficial_substring_decoy_rejected` -> `invalid` (fluent text containing target substring rejected by structured answer+citation checker)
- `silent_input_truncation_workload_mismatch` -> `invalid` (silent input truncation detected: observed_tokens=48 != planned_tokens=64)
- `claimed_ratio_exceeds_total_peak_memory_reduction` -> `invalid` (claimed_ratio_pct=25.0%, actual_peak_with_temp_pct=-56.25%)
- `end_task_leakage_into_press_tuning_rejected` -> `invalid` (end-task checker ID `corrected_policy_limit` leaked into press tuning set)
