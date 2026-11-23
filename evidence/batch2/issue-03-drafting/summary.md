# Issue #3: Diffusion Drafting Conformance & Target-Model Parity

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`dgx_spark_gb10_or_h100_target_and_drafter_serving`)
- Offline retained-output recheck matches recorded verdicts: `true` (2 tasks)
- Decision: `INCONCLUSIVE_UNAVAILABLE_HARDWARE_RUNTIME`

## Stochastic Rejection Sampling Checks
| Fixture | Residual Correction | Valid Dist | Max Abs Error | Restores Target |
| --- | --- | --- | ---: | --- |
| `positive-exact-residual-rejection` | `true` | `true` | 0.0 | `true` |
| `negative-broken-sampler-no-residual` | `false` | `true` | 0.14 | `false` |
| `negative-unnormalized-distribution` | `true` | `false` | null | `false` |
| `negative-nan-probability` | `true` | `false` | null | `false` |

## Arm Conditions & Silent-Fallback Disqualification
| Arm | Conc | Draft Passes | Mean Acc/Verify | Fallback Frac | Accepted Tasks | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `no-draft-baseline-c1` | 1 | 0 | 0.0 | 0.0 | 7/8 | `accepted` |
| `draft-assisted-contract-c1` | 1 | 14 | 2.285714285714 | 0.071428571429 | 7/8 | `accepted` |
| `no-draft-baseline-c4` | 4 | 0 | 0.0 | 0.0 | 7/8 | `accepted` |
| `draft-assisted-contract-c4` | 4 | 15 | 2.0 | 0.133333333333 | 7/8 | `accepted` |
| `negative-silent-fallback-arm` | 4 | 0 | 0.0 | 1.0 | 7/8 | `invalid` |
| `negative-fast-but-zero-accepted-tasks` | 1 | 10 | 3.0 | 0.0 | 0/8 | `invalid` |

> External claims (JRB_AI DFlash2 6.6x, DNeo 1.3x / 701 tok/s) are logged strictly as unverified non-gating references. Peak resident memory and energy remain `null`/unknown because no GPU ran locally.
