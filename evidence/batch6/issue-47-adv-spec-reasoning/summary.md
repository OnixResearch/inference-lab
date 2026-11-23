# Issue #47: Adversarial Speculative Reasoning Drafts vs Unbiased Drafts

- Extends drafting issues: `#3`, `#6`, `#22` (non-gating reference: `arXiv:2611.05345`)
- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (GPU speculative decoding runtime with slot-indexed batched verifier and trained adversarial drafter)

## Per-Slot Numerical Noise (vs Independent Slot-0 Reference)
- Slot `0` (`samples=57`): `max_abs_err=0.000000`, `l2_err=0.000000`
- Slot `1` (`samples=57`): `max_abs_err=0.011893`, `l2_err=0.158768`
- Slot `2` (`samples=57`): `max_abs_err=0.023352`, `l2_err=0.305849`
- Slot `3` (`samples=57`): `max_abs_err=0.035934`, `l2_err=0.482644`

## Matched-Seed Draft Comparison (`seed=47202611`)
- `unbiased_draft`: `accepted=1/6`, `exact_slot0=1/6`, `decision_at_slot0=8/18`, `target_calls=20`, `wall_proxy_ms=38.720`
- `adversarial_decision_draft`: `accepted=4/6`, `exact_slot0=4/6`, `decision_at_slot0=16/18`, `target_calls=25`, `wall_proxy_ms=48.490`

## Negative Controls (8 / 8 rejected)
- `nc1_mismatched_seed_rejected` -> `invalid` (comparing unbiased and adversarial drafts under mismatched seeds is invalid)
- `nc2_nan_logits_rejected` -> `invalid` (non-finite logits fail finite check rather than being swallowed)
- `nc3_zero_planned_tasks_guarded` -> `invalid` (zero planned tasks yields None rates (no div-by-zero) and fails comparison)
- `nc4_non_slot0_claiming_exactness_rejected` -> `invalid` (accepting a decision token at slot k=2 cannot claim slot-0 numerical exactness)
- `nc5_silent_verification_bypass_rejected` -> `invalid` (bypassing rejection-sampling verification is refused)
- `nc6_negative_wall_time_rejected` -> `invalid` (negative wall-time measurement is rejected)
- `nc7_paper_claim_as_gate_rejected` -> `invalid` (arXiv:2611.05345 claims are unverified related work, never lab pass gates)
- `nc8_decision_flip_fails_independent_checker` -> `invalid` (flipping a decision-point token fails independent task verification)
