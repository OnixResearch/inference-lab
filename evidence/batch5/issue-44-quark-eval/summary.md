# Issue #44: Quark Base/Zero Cached-Decode Parity & Context Costs

- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`)
- Artifact/hardware gate: `blocked_on_artifacts` (empero-ai/Quark-150M-128K-Base & Zero HF checkpoints + Transformers 5.23+/CUDA)

## Per-Checkpoint Parity & Held-Out Summary
- `Quark-150M-128K-Base`: `rollover_err=0.000e0`, `changed_tool_err=0.000e0`, `verified=0/5`, `cost_per_success=None`
- `Quark-150M-128K-Zero`: `rollover_err=0.000e0`, `changed_tool_err=0.000e0`, `verified=4/5`, `cost_per_success=Some(31.5)`

## Negative Controls (8 / 8 rejected)
- `nc1_transformers_version_below_5_23` -> `unsupported` (Transformers < 5.23 rejected as unsupported for Quark custom code)
- `nc2_local_window_rollover_corruption` -> `invalid` (wrong local-window eviction diverged at token Some(4) (max_err=1.717e-1))
- `nc3_wrong_position_after_chunk` -> `invalid` (shifted post-chunk position ID diverged at token Some(4) (max_err=7.731e-2))
- `nc4_hallucinated_tool_result_rejected` -> `invalid` (generated text resembling a tool result is never an executed effect)
- `nc5_missing_assistant_stop_delimiter` -> `invalid` (unterminated assistant turn rejected by rendering checker)
- `nc6_recompute_recovery_mislabeled_native` -> `invalid` (recomputation recovery must stay labeled separately from native state support)
- `nc7_base_vs_zero_not_numerical_parity` -> `invalid` (Base vs Zero diverge by 1.288e0; comparison is quality, not parity)
- `nc8_zero_success_undefined_cost` -> `invalid` (zero verified successes keeps cost_per_verified_success explicitly None/null)
