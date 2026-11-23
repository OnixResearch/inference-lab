# Issue #11: Centralized Model-Lifecycle Measurements (Spark Adapter Contract)

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`spark_nixos_gb10_disposable_adapter_and_cuda_completion`)
- A/B/A exclusive elapsed-time partition exact: `true` (wall=`1500ms`, raw overlapping sum=`1880ms`)
- RSI #86 parent->variant->parent exclusive partition exact: `true` (wall=`1700ms`, raw overlapping sum=`1960ms`)
- Decision: `INCONCLUSIVE_UNAVAILABLE_HARDWARE_RUNTIME`

## Lifecycle & Animus #17 Controls (13/13)
| Control | Ready | False-Ready | Outcome | Reason |
| --- | --- | --- | --- | --- |
| `01-positive-aba-first-checked-completion` | `true` | `false` | `accepted` | none |
| `02-negative-wrong-model-loaded` | `false` | `true` | `invalid` | wrong_model_or_variant_loaded |
| `03-negative-animus17-model-listed-before-ready` | `false` | `true` | `invalid` | false_ready_model_listed_before_first_checked_completion |
| `04-negative-animus17-stale-healthy-after-shutdown` | `false` | `true` | `invalid` | stale_healthy_observation_after_shutdown |
| `05-negative-late-completion-previous-generation` | `false` | `true` | `invalid` | late_completion_from_previous_deployment_generation |
| `06-negative-unsupported-native-sleep-no-restart-fallback` | `false` | `false` | `unsupported` | native_sleep_unsupported_no_silent_process_restart |
| `07-negative-failed-wake-or-parent-restore` | `false` | `false` | `internal_failure` | failed_wake_or_parent_restoration |
| `08-negative-interrupted-transition-timeout` | `false` | `false` | `timeout` | interrupted_or_timed_out_transition_unresolved |
| `09-negative-cleanup-failure-retained` | `false` | `false` | `internal_failure` | cleanup_failure_retained_in_denominator |
| `10-negative-refuse-foreign-resource-cleanup` | `false` | `false` | `invalid` | refused_cleanup_of_non_experiment_owned_resource |
| `11-negative-animus17-two-clients-one-slot-rejected` | `true` | `false` | `exhausted` | second_client_admission_rejected_for_single_slot |
| `12-control-unavailable-telemetry-stays-unknown` | `true` | `false` | `accepted` | none |
| `13-negative-delayed-readiness-unlisted-not-ready` | `false` | `false` | `timeout` | delayed_readiness_first_checked_completion_not_observed |

> Device-free contract harness only. Physical Spark NixOS #4 GB10 execution and CUDA completion remain required external gates.
