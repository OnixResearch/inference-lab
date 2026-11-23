# Issue #37: Native Recurrent-State Benchmark (RWKV-7, Mamba-3, xLSTM)

- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (Official RWKV-7, Mamba-3 SISO, and xLSTM-7b checkpoints on accelerator hardware)
- Milestone A (`RWKV-7` + `Mamba-3`): `true`, Milestone B (`xLSTM` gated): `true`

## Architecture Matrix (vs Independent Scalar Reference)
- `rwkv7` (`milestone_a`): `scalar_vs_chunk=0.000e0`, `restore_err=0.000e0`, `fork_err=0.000e0`, `batch_err=0.000e0`, `edges_ok=true`, `state_bytes=168`, `rollback=unsupported`, `outcome=TINY_REF_QUALIFIED_HARDWARE_UNAVAILABLE`
- `mamba3_siso` (`milestone_a`): `scalar_vs_chunk=0.000e0`, `restore_err=0.000e0`, `fork_err=0.000e0`, `batch_err=0.000e0`, `edges_ok=true`, `state_bytes=168`, `rollback=unsupported`, `outcome=TINY_REF_QUALIFIED_HARDWARE_UNAVAILABLE`
- `xlstm_mlstm` (`milestone_b_conditional`): `scalar_vs_chunk=0.000e0`, `restore_err=0.000e0`, `fork_err=0.000e0`, `batch_err=0.000e0`, `edges_ok=true`, `state_bytes=168`, `rollback=unsupported`, `outcome=TINY_REF_QUALIFIED_HARDWARE_UNAVAILABLE`

## Negative Controls (10 / 10 rejected)
- `nc1_rwkv7_wrong_decay_transition` -> `invalid` (wrong RWKV-7 decay sign diverged at token Some(1) (max_err=1.100e0))
- `nc2_mamba3_wrong_decay_transition` -> `invalid` (wrong Mamba-3 decay sign diverged at token Some(1) (max_err=2.047e-1))
- `nc3_ignored_position_clock` -> `invalid` (ignoring token clock diverged at token Some(1) (max_err=3.584e-3))
- `nc4_cross_family_state_restore_rejected` -> `invalid` (RWKV-7 serialized state refused by Mamba-3 profile)
- `nc5_stale_worker_generation_rejected` -> `invalid` (older worker generation state rejected after restart)
- `nc6_cross_session_restore_rejected` -> `invalid` (sess-a snapshot rejected when restoring sess-b)
- `nc7_capacity_overflow_resource_stop` -> `exhausted` (exceeding session capacity stops cleanly with explicit resource_stop)
- `nc8_recompute_mislabeled_as_native_state` -> `invalid` (full-prefix recomputation wrapper cannot be labeled native state support)
- `nc9_corrupted_nan_state_bytes_rejected` -> `invalid` (NaN in serialized state payload rejected on restore)
- `nc10_xlstm_nan_stabilizer_not_swallowed` -> `invalid` (NaN log_stab_m triggers explicit numerical_failure instead of being swallowed)
