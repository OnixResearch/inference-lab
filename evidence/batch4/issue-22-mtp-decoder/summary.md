# Issue #22: Small MTP Draft-and-Verify Decoder Conformance

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`serving_scale_gpu_mtp_runtime`)
- Decision: `GO_CPU_CONFORMANCE_ONLY_NO_SERVING_SPEEDUP_CLAIMED` (no serving speedup assumed)
- Target cached vs full recompute max error: `0.000e0`
- Prefix-causal no-future-leak max error: `0.000e0`
- Multi-request reset isolation max error: `0.000e0`

## Greedy conformance fixtures
| Fixture | Exact Agreement | Max Cache Err | Max Logit Err | Proposed | Accepted | Fallbacks | Stop |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `greedy_natural_mixed_accept_reject` | `true` | `0.000e0` | `0.000e0` | 6 | 2 | 0 | `budget_reached` |
| `greedy_first_position_rejection` | `true` | `0.000e0` | `0.000e0` | 4 | 1 | 0 | `budget_reached` |
| `greedy_later_position_rejection` | `true` | `0.000e0` | `0.000e0` | 3 | 1 | 0 | `budget_reached` |
| `greedy_all_accepted_plus_bonus` | `true` | `0.000e0` | `0.000e0` | 2 | 2 | 0 | `budget_reached` |
| `greedy_explicit_target_only_fallback` | `true` | `0.000e0` | `0.000e0` | 0 | 0 | 4 | `budget_reached` |
| `greedy_cancel_stop` | `true` | `0.000e0` | `0.000e0` | 2 | 0 | 0 | `cancelled` |
| `greedy_immediate_continuation` | `true` | `0.000e0` | `0.000e0` | 2 | 1 | 0 | `budget_reached` |
| `greedy_context_capacity_edge` | `true` | `0.000e0` | `0.000e0` | 2 | 0 | 0 | `context_capacity_reached` |
| `greedy_eos_boundary_no_tail_emit` | `true` | `0.000e0` | `0.000e0` | 2 | 1 | 0 | `eos` |

## Stochastic rejection-sampling conformance
- Exact small-vocab marginal max error: `0.000e0`
- Calibrated empirical check (N=4000): max abs error `0.00825` <= `0.03` -> `true` (finite empirical check does not prove universal preservation)
- Impossible zero-mass residual rejected: `true`; bonus-token rule verified: `true`

## Negative controls
- `swapped_p_q_probabilities` -> `invalid` (marginal_l_inf_err=0.200000 > 1e-6)
- `incorrect_residual_correction_mass` -> `invalid` (biased_marginal_err=0.200000 > 1e-6)
- `off_by_one_target_alignment` -> `invalid` (shifted_logit_err=1.010996 > 1e-9)
- `missing_cache_rollback_poison_read` -> `invalid` (poisoned_logit_err=694823.769128 > 1e-6)
- `emitted_rejected_speculative_token` -> `invalid` (exact_token_agreement=false)
- `future_ground_truth_leak_and_bias_mutation` -> `invalid` (future_leak_err=0.589463, bias_drift=0.161133)
