# Issue #38: Tiered Persistent-Engine Evaluation with Cold Final Replay

- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (Physical GPU persistent forward-pass engine with CUDA/HIP graph capture)
- `A -> B -> A` restored vs fresh-process A: `max_err = 0.000e0`

## Tiered Evaluation Arms
- `arm_a_restart_per_candidate`: `valid=4/8`, `restarts=1`, `false_rejects_audited=0`, `winner=Some("cand_1_valid_faster")`, `cold_replay=accepted`
- `arm_b_full_fidelity_persistent`: `valid=4/8`, `restarts=1`, `false_rejects_audited=0`, `winner=Some("cand_1_valid_faster")`, `cold_replay=accepted`
- `arm_c_fast_screen_plus_full_persistent`: `valid=3/8`, `restarts=1`, `false_rejects_audited=1`, `winner=Some("cand_1_valid_faster")`, `cold_replay=accepted`

## Negative Controls (8 / 8 rejected)
- `nc1_weight_mutation_detected_and_restarted` -> `invalid` (weight mutation caught by immutable digest and forced worker restart)
- `nc2_aba_skipped_reset_contamination` -> `invalid` (skipped reset leaked 5.000e-1 error into restored A)
- `nc3_stale_graph_capture_divergence` -> `invalid` (stale graph caused 1.349e0 numerical mismatch)
- `nc4_unwritten_output_buffer_leak` -> `invalid` (unwritten output buffer leaked 4.218e1 stale values)
- `nc5_wrong_but_fast_disqualified` -> `invalid` (wrong-but-fast candidate failed numerical gate (3.500e-1))
- `nc6_fast_tier_cannot_bypass_cold_replay` -> `invalid` (fast-tier pass refused final acceptance without fresh cold held-out replay)
- `nc7_candidate_exception_retained` -> `internal_failure` (candidate runtime exception captured as explicit internal_failure)
- `nc8_unverified_worker_termination` -> `invalid` (stop request alone without verified termination cannot establish cleanliness)
