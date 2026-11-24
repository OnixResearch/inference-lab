# Issue #81: Pinned `saber-engine` (Rust) + DFlash2 Power-Capped Qualification

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (Dual R9700 (210W cap) / GPU host serving Qwen3.8-27B on pinned GGZ14/saber-engine + DFlash2)
- Upstream pin: `GGZ14/saber-engine@1da27661` (`Apache-2.0`, DFlash2 stochastic L1 err: `2.082e-16`)

## Parity-Gated Power-Capped Serving Comparison
| Backend | DFlash2 | Parity | Accept Len | TTFT (ms) | tok/s | Watts | tok/s/W | J/tok |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `vllm_ar_baseline` | `false` | `true` | `1.00` | `62.0` | `78.0` | `198.0` | `0.394` | `2.538` |
| `llama_cpp_ar_baseline` | `false` | `true` | `1.00` | `58.0` | `84.0` | `195.0` | `0.431` | `2.321` |
| `saber_engine_ar_only` | `false` | `true` | `1.00` | `52.0` | `92.0` | `192.0` | `0.479` | `2.087` |
| `saber_engine_dflash2_default_off` | `true` | `true` | `3.86` | `49.0` | `195.0` | `204.0` | `0.956` | `1.046` |

## Negative Controls (8 / 8 rejected)
- `nc1_faster_parity_failing_backend_disqualified` -> `invalid` (300 tok/s backend failing greedy parity is disqualified with tok/s/W = None)
- `nc2_missing_power_sensor_never_zero` -> `invalid` (absent power sensor keeps tok/s/W and J/token as None rather than 0W)
- `nc3_missing_throttle_telemetry_not_accepted` -> `invalid` (absent throttle counters never count as evidence of zero throttling (#43))
- `nc4_power_cap_violation_rejected` -> `invalid` (260W draw violates declared 210W cap beyond 5% board tolerance)
- `nc5_unnormalized_dflash2_distribution_rejected` -> `invalid` (unnormalized target distribution fails rejection-sampling verifier)
- `nc6_nan_power_reading_rejected` -> `invalid` (NaN power reading rejected cleanly)
- `nc7_sub_unit_acceptance_length_rejected` -> `invalid` (speculative acceptance length cannot be less than 1.0 token/step)
- `nc8_unverified_229tps_x_post_non_gating` -> `invalid` (229 tok/s & 47 ms TTFT X-post figures stay unverified related work)
