# Issue #84: ML-and-Silicon Co-Design Report (`arXiv:2611.18859`) Assumption Audit

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Artifact gate: `blocked_on_artifacts` (Fabricated physical silicon for arXiv:2611.18859 co-design & sensor-calibrated lab hardware baselines (#43))

## Pre-Silicon Projections vs Lab Hardware Baseline Envelopes (27B Active Params, Batch=1)
| Platform | Maturity | Roofline tok/s | Bottleneck | Implied J/tok | Silicon Validated |
| --- | --- | ---: | --- | ---: | --- |
| `arxiv_2611_18859_presilicon_sram_projection` | `pre_silicon_simulation_only` | `29629.6` | `memory_bandwidth_bound` | `0.0051` | `false` |
| `nvidia_b300_hbm3e_envelope` | `lab_hardware_baseline_envelope` | `592.6` | `memory_bandwidth_bound` | `2.0250` | `false` |
| `dgx_spark_gb10_lpddr5x_envelope` | `lab_hardware_baseline_envelope` | `20.2` | `memory_bandwidth_bound` | `8.4066` | `false` |
| `apple_m2_ultra_unified_envelope` | `lab_hardware_baseline_envelope` | `59.3` | `memory_bandwidth_bound` | `2.0250` | `false` |
| `tenstorrent_p150_gddr6_envelope` | `lab_hardware_baseline_envelope` | `37.9` | `memory_bandwidth_bound` | `7.9102` | `false` |

## Negative Controls (8 / 8 rejected)
- `nc1_presilicon_marked_silicon_validated_rejected` -> `invalid` (pre-silicon simulation cannot be tagged physical_silicon_validated)
- `nc2_presilicon_never_eligible_as_measured_lab_result` -> `blocked_on_artifacts` (arXiv:2611.18859 pre-silicon projection blocked on fabricated silicon)
- `nc3_zero_board_power_rejected` -> `invalid` (zero board power refused instead of reporting 0.0 J/token)
- `nc4_zero_memory_bandwidth_rejected` -> `invalid` (zero memory bandwidth rejected cleanly)
- `nc5_nan_parameter_count_rejected` -> `invalid` (NaN model parameter count rejected cleanly)
- `nc6_negative_weight_bits_rejected` -> `invalid` (negative weight bitwidth rejected)
- `nc7_infinite_peak_tflops_rejected` -> `invalid` (non-finite infinite TFLOPS rejected)
- `nc8_unverified_110x_690x_presilicon_claims_non_gating` -> `invalid` (110x speed and 690x energy claims remain unverified pre-silicon related work)
