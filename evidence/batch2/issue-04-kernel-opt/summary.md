# Issue #4: Reference-Assisted vs Spec-Only Agent Kernel Optimization

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`sm120_rtx5080_sm121_gb10_or_b200_gpu_kernel_execution`)
- Local noise floor: `0.1594 ms` (1.59% of `10.0100 ms` baseline; foreign roofline reused: `false`)
- Decision: `INCONCLUSIVE_UNAVAILABLE_HARDWARE_RUNTIME`

| Submission | Arm | Timing Eligible | Cleared Noise | Outcome | Rejection Stage |
| --- | --- | --- | --- | --- | --- |
| `sub-01-valid-clears-soak-and-noise` | `reference_assisted` | `true` | `true` | `accepted` | none |
| `sub-02-identity-noop-baseline` | `spec_only` | `true` | `false` | `invalid` | identity_noop_baseline_not_an_optimization |
| `sub-03-wrong-but-fast-logit-drift` | `spec_only` | `false` | `false` | `invalid` | gate_2a_fp64_relative_tolerance_exceeded_or_nonfinite |
| `sub-04-fast-batch-reorder-drift` | `reference_assisted` | `false` | `false` | `invalid` | gate_2c_batch_reorder_not_bit_identical |
| `sub-05-correct-but-overbudget-schedule` | `reference_plus_profiler` | `false` | `false` | `invalid` | gate_2d_cachet_schedule_scratch_overbudget |
| `sub-06-passes-dev-fails-soak` | `reviewer_guided_continuation` | `false` | `false` | `invalid` | gate_3_soak_stage_failed |
| `sub-07-within-local-noise-floor` | `compileiq_acf_separate` | `true` | `false` | `invalid` | gate_4_within_local_noise_floor_no_improvement |
| `sub-08-forged-timing-tampered-eval` | `persistent_tiered_evaluator_38` | `false` | `false` | `invalid` | gate_1_integrity_or_forged_timing_rejected |
| `sub-09-nan-logits-must-fail-gate2a` | `spec_only` | `false` | `false` | `invalid` | gate_2a_fp64_relative_tolerance_exceeded_or_nonfinite |
| `sub-10-zero-latency-invalid-timing` | `spec_only` | `false` | `false` | `invalid` | gate_4_invalid_nonpositive_or_nonfinite_latency |

> Bounded contract harness only. Upstream megakernel-zoo/Snarkify/Han Xiao numbers remain unverified non-gating references.
