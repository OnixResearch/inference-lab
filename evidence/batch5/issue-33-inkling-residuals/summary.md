# Issue #33: Inkling Block Attention Residuals & Dynamic Sinks

- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`)
- Artifact/hardware gate: `blocked_on_artifacts` (Inkling-2-Small 594B & matched-budget small-model training runs)
- Extreme sink stability (`-50.0` / `+50.0`): `true` / `true`
- Behavioral bounds: `training_superiority_established = false`, `full_594b_claim_supported = false`

## Matched Ablation Arms (12 layers = 2 x 6-layer blocks)
- `ordinary_residual_no_sink`: `chunked_err = 0.000e0`, `decode_err = 0.000e0`, `l2 = 1.566703`, `outcome = accepted`
- `ordinary_residual_static_sink`: `chunked_err = 0.000e0`, `decode_err = 0.000e0`, `l2 = 1.566148`, `outcome = accepted`
- `block_residual_only`: `chunked_err = 0.000e0`, `decode_err = 0.000e0`, `l2 = 0.724515`, `outcome = accepted`
- `dynamic_sink_only`: `chunked_err = 0.000e0`, `decode_err = 0.000e0`, `l2 = 1.536122`, `outcome = accepted`
- `combined_block_res_dynamic_sink_gate_fixed`: `chunked_err = 0.000e0`, `decode_err = 0.000e0`, `l2 = 0.615221`, `outcome = accepted`
- `combined_block_res_dynamic_sink_gate_ablated`: `chunked_err = 0.000e0`, `decode_err = 0.000e0`, `l2 = 1.099583`, `outcome = accepted`

## Negative Controls (7 / 7 rejected)
- `nc1_stale_cache_across_reset` -> `invalid` (stale KV/conv cache caused 2.312e-1 divergence)
- `nc2_crossed_sequence_state` -> `invalid` (crossed sequence prefix caused 3.784e-1 divergence)
- `nc3_missing_block_candidate` -> `invalid` (omitting completed 6-layer block caused 2.471e-1 divergence)
- `nc4_skipped_block_reset` -> `invalid` (skipping 6-layer reset caused 1.842e-1 divergence)
- `nc5_nonfinite_sink_logit_rejected` -> `invalid` (NaN sink logit fails explicit finiteness check)
- `nc6_causal_mask_future_leak` -> `invalid` (future-token leak caused 1.537e-1 divergence)
- `nc7_disabling_only_not_training_superiority` -> `invalid` (disabling pretrained components proves sensitivity, not training superiority)
