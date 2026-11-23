# Issue #23: FlashTopK Sparse-Indexer Reference & Hardware Gate

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`cuda_sm90_sm100_flashtopk_gpu`)
- Scoped decisions: SM120/SM121 = `NO_GO_UNSUPPORTED_AT_PINNED_REVISION`; SM90/SM100 = `INCONCLUSIVE_GPU_EXECUTION_REQUIRED`

## Compatibility matrix (`FlashTopK@090902ab`)
| Target | Outcome | Reason |
| --- | --- | --- |
| `h100_sm90_glm51_32h_fp8` | `accepted` | matches pinned FlashTopK@090902ab contract (physical CUDA execution still required) |
| `b200_sm100_glm51_32h_fp8` | `accepted` | matches pinned FlashTopK@090902ab contract (physical CUDA execution still required) |
| `rtx5080_sm120_blackwell_consumer` | `unsupported` | CUDA SM120 rejected by pinned supported() gate (major 9 or 10 only; SM120/SM121 excluded) |
| `dgx_spark_gb10_sm121` | `unsupported` | CUDA SM121 rejected by pinned supported() gate (major 9 or 10 only; SM120/SM121 excluded) |
| `deepseek_v32_64head_on_sm100` | `unsupported` | query shape [rows, 64, 128] outside released [rows, 32, 128] contract (e.g. 64-head DeepSeek-V3.2 unsupported) |
| `block_level_qsa_selector_sm100` | `unsupported` | paged KV, multi-window masks, and block-level CSA/QSA remain unsupported upstream |
| `tenstorrent_p150_non_cuda` | `unsupported` | backend `tenstorrent` requires separate implementation (CUDA kernels are not portable) |

## Independent correctness & recovery checks
| Case | Exact Match | Max Err | Repeat Identical | Partition Inv | Underflow Rec | Overflow Rec | Fault Injected |
| --- | --- | ---: | --- | --- | ---: | ---: | --- |
| `fp8_causal_and_nonzero_offset_windows` | `true` | `0.000e0` | `true` | `true` | 0 | 1 | `false` |
| `bf16_short_and_empty_windows` | `true` | `0.000e0` | `true` | `true` | 0 | 0 | `false` |
| `bf16_k1_boundary` | `true` | `0.000e0` | `true` | `true` | 0 | 0 | `false` |
| `signed_zero_exact_ties_and_near_kth_boundary` | `true` | `0.000e0` | `true` | `true` | 0 | 0 | `false` |
| `fault_injected_underflow_recovery` | `true` | `0.000e0` | `true` | `true` | 4 | 0 | `true` |
| `fault_injected_overflow_recovery` | `true` | `0.000e0` | `true` | `true` | 0 | 4 | `true` |

## Negative controls
- `omitted_fp8_per_key_scales` -> `invalid` (max_score_err=0.361922, differing_indices=11)
- `incorrect_window_mask` -> `invalid` (differing_indices=13)
- `wrong_tie_order_descending_index` -> `invalid` (differing_indices=8)
- `skipped_underflow_recovery` -> `invalid` (differing_indices=40)
- `non_finite_query_rejected` -> `invalid` (non-finite or mismatched query vector)
- `invalid_window_bounds_rejected` -> `invalid` (invalid window [8, 4) at row 0 for num_keys=12)
- `sm121_and_64head_capability_gate_enforced` -> `unsupported` (sm121_refused=true, deepseek_64head_refused=true)
