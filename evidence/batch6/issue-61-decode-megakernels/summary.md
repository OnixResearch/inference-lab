# Issue #61: Agent-Written Per-Model Decode Megakernels vs Multi-Launch Baseline

- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`, `pin_status: repo_url_only_commit_unpinned_offline`)
- Hardware gate: `unavailable_hardware_runtime` (Physical GPU with pinned vLLM/SGLang baseline and persistent CUDA megakernel compilation toolchain)
- Gain survives `batch > 1` (synthetic contract): `true`

## Batch Scaling & Equivalence Gate (`batch = 1, 4, 8`)
- `batch=1`: `token_exact=true`, `max_logit_err=0.000e0`, `timing_permitted=true`, `launches=288->1`, `host_syncs=12->1`, `speedup=Some(1.334870042127017)`, `outcome=accepted`
- `batch=4`: `token_exact=true`, `max_logit_err=0.000e0`, `timing_permitted=true`, `launches=288->1`, `host_syncs=12->1`, `speedup=Some(1.1988387317128708)`, `outcome=accepted`
- `batch=8`: `token_exact=true`, `max_logit_err=0.000e0`, `timing_permitted=true`, `launches=288->1`, `host_syncs=12->1`, `speedup=Some(1.1052565967172243)`, `outcome=accepted`

## Agent Kernel-Build Ledger
- `attempt_01_shared_kv_bug` (`mini_causal_2l_synthetic_contract`): `wall_s=142.0`, `compile=true`, `equiv=false`, `speedup=None`, `failures=["cross_batch_kv_cache_index_collision", "equivalence_gate_blocked_timing"]`, `outcome=invalid`
- `attempt_02_persistent_fused_verified` (`mini_causal_2l_synthetic_contract`): `wall_s=215.0`, `compile=true`, `equiv=true`, `speedup=Some(1.334870042127017)`, `failures=[]`, `outcome=accepted`

## Negative Controls (8 / 8 rejected)
- `nc1_logit_drift_blocks_timing_and_speedup` -> `invalid` (logit drift 1.000e-4 forced timing_permitted=false and speedup=None)
- `nc2_token_mismatch_blocks_timing_and_speedup` -> `invalid` (argmax feedback mismatch blocked timing and suppressed speedup)
- `nc3_batch_gt_1_kv_contamination_caught` -> `invalid` (batch=4 KV cross-talk diverged 1.281e-1 and blocked timing)
- `nc4_skipped_final_rmsnorm_caught` -> `invalid` (omitted final RMSNorm diverged 1.041e0 and blocked timing)
- `nc5_zero_duration_div_by_zero_guarded` -> `invalid` (zero elapsed duration returned None instead of Inf/NaN)
- `nc6_non_finite_latency_sample_rejected` -> `invalid` (NaN latency sample refused before percentile sort)
- `nc7_batch_gt_1_regression_flags_non_survival` -> `invalid` (batch=8 regression (0.91x) flipped gain_survives_batch_gt_1 to false)
- `nc8_fabricated_commit_sha_rejected` -> `invalid` (all-zero fabricated commit SHA refused; offline repo stays honestly unpinned)
