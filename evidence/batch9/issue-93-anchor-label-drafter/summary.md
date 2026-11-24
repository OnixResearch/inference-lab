# Issue #93: Off-Policy Anchor-Label Relabelling vs Draft-OPD & D-PACE (#71)

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (GPU speculative drafter training & open-loop replay beside Draft-OPD/D-PACE (#71))

## Matched-Budget Drafter Supervision Recovery
- `naive_off_policy_truncated`: `sup=24/64` (`0.375`), `mean_len=3.1931` (`gain=+0.0000`, `mde=0.0800`), `drift=3.608e-16`, `replay_tok/s=1850.0`, `outcome=accepted`
- `anchor_label_relabelling_default_off`: `sup=58/64` (`0.90625`), `mean_len=3.5859` (`gain=+0.3928`, `mde=0.0800`), `drift=3.608e-16`, `replay_tok/s=1720.0`, `outcome=accepted`
- `draft_opd_on_policy`: `sup=64/64` (`1.0`), `mean_len=3.6579` (`gain=+0.4647`, `mde=0.0800`), `drift=3.886e-16`, `replay_tok/s=1180.0`, `outcome=accepted`
- `d_pace_rejection_weighted`: `sup=64/64` (`1.0`), `mean_len=3.9232` (`gain=+0.7301`, `mde=0.0800`), `drift=2.914e-16`, `replay_tok/s=1140.0`, `outcome=accepted`

## Negative Controls (8 / 8 rejected)
- `nc1_unmatched_training_budget_rejected` -> `invalid` (2x extra gradient steps vs matched budget rejected)
- `nc2_lossy_verification_drift_rejected` -> `invalid` (lossy verification causing non-zero target distribution L1 drift fails lossless gate)
- `nc3_stale_target_digest_anchor_rejected` -> `invalid` (relabelling anchors against mismatched target policy digest rejected)
- `nc4_gain_within_noise_floor_unproven` -> `exhausted` (acceptance-length gain below 2-sigma MDE fails noise-floor gate)
- `nc5_empty_rollouts_rejected` -> `invalid` (empty rollout batch rejected cleanly)
- `nc6_nonfinite_noise_sigma_rejected` -> `invalid` (NaN repeat_sigma_len rejected)
- `nc7_zero_grad_steps_rejected` -> `invalid` (zero gradient-step budget rejected)
- `nc8_target_weight_mutation_prohibited` -> `invalid` (target model weights must remain strictly frozen during off-policy drafter recovery)
