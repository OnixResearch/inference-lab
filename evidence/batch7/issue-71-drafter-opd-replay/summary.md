# Issue #71: On-Policy Speculative Drafter Adaptation (`Draft-OPD` & `D-PACE`) on Open-Loop Replay Harness

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (GPU speculative-drafter training & production open-loop replay cluster (DFlash2 / Draft-OPD / D-PACE))

## Open-Loop Replay Arms (Noise-Floor Gated & Lossless-Verified)
- `frozen_baseline_drafter` (`none_frozen`): `accept_len=3.315`, `drift_l1=2.359e-16`, `tok_s=119.34` (`gain=+0.0%`, `noise_floor=1.34`), `verdict=lossless_accepted`, `outcome=accepted`
- `draft_opd_on_policy` (`uniform_on_policy_kl`): `accept_len=4.332`, `drift_l1=3.574e-16`, `tok_s=155.94` (`gain=+30.7%`, `noise_floor=1.34`), `verdict=lossless_accepted`, `outcome=accepted`
- `d_pace_rejection_weighted` (`rejection_point_weighted_kl`): `accept_len=4.517`, `drift_l1=1.665e-16`, `tok_s=162.62` (`gain=+36.3%`, `noise_floor=1.34`), `verdict=lossless_accepted`, `outcome=accepted`

## Negative Controls (8 / 8 rejected)
- `nc1_within_noise_floor_gain_not_accepted` -> `exhausted` (+0.3 tok/s inside 2-sigma noise floor yielded lossless_within_noise_floor)
- `nc2_target_drift_on_lossless_arm_rejected` -> `invalid` (target drift L1=0.3106 on declared lossless arm returned rejected_invalid)
- `nc3_lossy_arm_segregated_from_lossless` -> `unsupported` (lossy arm separated into lossy_separate_verdict)
- `nc4_single_sample_noise_floor_rejected` -> `invalid` (N=1 sample refused because repeat variance cannot be estimated)
- `nc5_nan_replay_sample_rejected` -> `invalid` (NaN throughput sample refused before noise-floor calculation)
- `nc6_empty_replay_trace_rejected` -> `invalid` (empty open-loop replay trace refused)
- `nc7_negative_learning_rate_rejected` -> `invalid` (negative drafter learning rate refused)
- `nc8_default_on_drafter_mutation_refused` -> `unsupported` (on-policy drafter adaptation remains strictly default-off)
