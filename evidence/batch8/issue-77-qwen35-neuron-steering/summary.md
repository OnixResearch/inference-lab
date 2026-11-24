# Issue #77: Qwen3.5-4B Two-Neuron `down_proj` Steering Regression Fixture

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (Local Qwen3.5-4B pinned checkpoint serving host (reusing #45 intervention-replay))
- Recipe: `L14 N8949 (+1)`, `L13 N7956 (-1)`, `delta = sign * dose * median(||resid||) / ||W_down[:, n]||`

## Dose-Response vs Unsteered & Matched-Norm Random Control
| Dose | Targeted KL | Targeted L1 | Top1 | Random Ctrl KL | Random Ctrl L1 | Exact Replay |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `0.00` | `0.000000` | `0.000000` | 3 | `0.000000` | `0.000000` | `true` |
| `0.25` | `0.003155` | `0.064935` | 3 | `0.001216` | `0.045116` | `true` |
| `0.50` | `0.012419` | `0.128827` | 3 | `0.004974` | `0.091193` | `true` |
| `1.00` | `0.054138` | `0.266085` | 3 | `0.021260` | `0.187314` | `true` |
| `1.50` | `0.134040` | `0.408543` | 3 | `0.053854` | `0.292772` | `true` |
| `2.00` | `0.260715` | `0.547635` | 3 | `0.113965` | `0.416066` | `true` |

## Negative Controls (8 / 8 rejected)
- `nc1_negative_steering_dose_rejected` -> `invalid` (negative steering dose refused; sign lives on explicit +1/-1 target)
- `nc2_nan_steering_dose_rejected` -> `invalid` (non-finite NaN dose rejected cleanly)
- `nc3_cross_arch_qwen35_neuron_transfer_rejected` -> `invalid` (Qwen3.5-4B neuron IDs (L14 N8949, L13 N7956) refused on non-Qwen arch)
- `nc4_zero_w_down_col_norm_divisor_rejected` -> `invalid` (zero ||W_down[:, n]|| divisor refused instead of producing Inf/NaN)
- `nc5_mutable_latest_model_revision_rejected` -> `invalid` (floating `latest` model revision fails immutable provenance check)
- `nc6_tampered_replay_receipt_detected` -> `invalid` (re-execution against tampered SHA-256 receipt returns Invalid)
- `nc7_non_unit_steering_sign_rejected` -> `invalid` (steering sign outside {-1, +1} rejected to preserve exact recipe formula)
- `nc8_empty_residual_stream_median_rejected` -> `invalid` (empty residual stream cannot produce a silent 0.0 median norm)
