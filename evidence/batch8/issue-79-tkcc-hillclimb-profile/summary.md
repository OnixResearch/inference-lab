# Issue #79: Together Compiler (TKCC) Related Work & Local Hill-Climb Cycle Profiler

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (2-node continual GPU compilation & serving hill-climb cluster (beside #53 & #61))

## Hill-Climb Experiment-Cycle Bottlenecks
| Profile | Trials (Acc/Total) | Wall (s) | Trials/hr | Accepted/hr | Dominant Phase |
| --- | --- | ---: | ---: | ---: | --- |
| `local_cold_rebuild_hillclimb` | `2/5` | `373.5` | `48.2` | `19.3` | `compile` |
| `local_cached_persistent_hillclimb` | `2/5` | `149.4` | `120.5` | `48.2` | `measurement` |

## Negative Controls (8 / 8 rejected)
- `nc1_skipped_parity_gate_on_accepted_rejected` -> `invalid` (inflating experiment throughput by skipping parity checks is refused)
- `nc2_negative_phase_duration_rejected` -> `invalid` (negative compile duration rejected cleanly)
- `nc3_nan_phase_duration_rejected` -> `invalid` (NaN measurement duration rejected)
- `nc4_empty_trial_log_rejected` -> `invalid` (empty hill-climb ledger refused instead of zero division)
- `nc5_zero_accepted_reports_none_rate` -> `exhausted` (zero accepted candidates returns accepted_trials_per_hour = None)
- `nc6_failed_candidates_retained_in_denominator` -> `invalid` (omitting compile-failed/parity-failed/timeout trials from wall-time denominator is invalid)
- `nc7_zero_total_trial_duration_rejected` -> `invalid` (zero-duration trial refused before throughput rate computation)
- `nc8_unverified_252x_tkcc_claim_non_gating` -> `invalid` (TKCC 252x experiment-throughput claim stays unverified related work)
