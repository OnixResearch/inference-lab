# Issue #43 — Power and throttle cadence qualification report

- **Execution class**: `bounded_contract_harness`
- **Physical hardware gate**: `unavailable_hardware_runtime` (`physical_acceptance_complete = false`)
- **External reference rule**: Ayaka TPU v7x 25.9%/26.4% claims (x.com/ayaka14732/status/2122033206719639812) are unverified author reports and NOT lab targets.

## Protocol control outcomes

| Control arm | Verdict | Causal mechanism | Causal allowed | Observer perturbation (%) | Energy status |
| --- | --- | --- | --- | --- | --- |
| `control-known-equivalent` | `no_benefit` | `unresolved_mechanism` | `true` | 4.390243902439 | `unavailable_no_sensor_observation` |
| `control-wrong-but-fast` | `disqualified_numerical_mismatch` | `unresolved_mechanism` | `false` | 4.390243902439 | `unavailable_no_sensor_observation` |
| `control-order-drift` | `inconclusive_order_drift` | `chip_thermal_or_power_throttle` | `true` | 3.658536585366 | `measured_sensor:synthetic_contract_fixture_board_rail:joules` |
| `control-host-cadence-cannot-claim-instruction-spacing` | `invalid_axis_claim` | `unresolved_mechanism` | `false` | 3.658536585366 | `unavailable_no_sensor_observation` |
| `control-corrupted-telemetry-blocks-causal-claim` | `causal_claim_blocked_by_telemetry` | `unresolved_mechanism` | `false` | 4.878048780488 | `unavailable_no_sensor_observation` |

Physical-device acceptance stays explicitly incomplete until a real accelerator run with qualified raw timing and sensor/throttle telemetry is executed.
