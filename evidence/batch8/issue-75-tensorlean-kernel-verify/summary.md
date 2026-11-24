# Issue #75: TensorLean Formal-Verification Boundary & Seeded Kernel-Bug Coverage

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Artifact gate: `blocked_on_artifacts` (TensorLean Lean4 formal verification framework + pinned GPU kernel)
- Catch rates on 5 seeded bugs: formal `1.00`, aligned-diff (`seq=4`) `0.80`, unaligned-diff (`seq=5`) `0.80`

## Formal Obligations vs Differential Testing
| Seeded Bug | Formal Caught | Aligned seq=4 Err | Aligned Caught | Unaligned seq=5 Err | Unaligned Caught |
| --- | --- | ---: | --- | ---: | --- |
| `none_reference` | `false` | `0.000e0` | `false` | `0.000e0` | `false` |
| `tile_tail_boundary_oob` | `true` | `0.000e0` | `false` | `4.163e-17` | `false` |
| `missing_reduction_barrier_carry` | `true` | `1.681e-2` | `true` | `1.681e-2` | `true` |
| `wrong_softmax_scale` | `true` | `4.209e-2` | `true` | `4.209e-2` | `true` |
| `fp16_accumulator_truncation` | `true` | `7.630e-5` | `true` | `1.553e-4` | `true` |
| `causal_mask_off_by_one` | `true` | `4.317e-1` | `true` | `4.317e-1` | `true` |

## Negative Controls (8 / 8 rejected)
- `nc1_missing_public_tensorlean_artifacts_blocked` -> `blocked_on_artifacts` (live Lean4 TensorLean verification stays blocked until public repo/commit is pinned)
- `nc2_incomplete_tcb_axiom_boundary_rejected` -> `invalid` (proof with unrecorded/incomplete hardware axiom TCB rejected)
- `nc3_aligned_only_diff_misses_tail_oob` -> `invalid` (aligned seq=4 differential test missed tile_tail_boundary_oob caught by formal check)
- `nc4_zero_tile_dimension_rejected` -> `invalid` (degenerate zero tile_k dimension rejected by obligation checker)
- `nc5_zero_kernel_dimensions_refused` -> `invalid` (zero sequence length rejected before kernel execution)
- `nc6_nan_kernel_output_rejected` -> `invalid` (NaN kernel output fails differential comparator cleanly)
- `nc7_mismatched_output_shape_rejected` -> `invalid` (mismatched tensor output shape rejected by comparator)
- `nc8_formal_ir_proof_does_not_bypass_silicon_check` -> `invalid` (IR-level formal proof cannot substitute for ptxas/silicon differential validation)
