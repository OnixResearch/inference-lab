# Issue #46: Biological-Recording Adapters vs Matched Synthetic & Shuffled Controls

- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`)
- Artifact gate: `blocked_on_artifacts` (Paired CL1 biological recordings & BioLLM-8B adapter checkpoints)
- Assumes BioLLM-8B benchmarks exist: `false`; Fixture criterion verdict: `FALSIFIED_OR_NULL`

## Equal-Budget 5-Arm Evaluation (Synthetic Contract Fixture)
- `base_model` (`steps=0`): `acc=0.625`, `ret=0.920`, `ECE=0.0131`, `Brier=0.1855`, `lat=12.0ms`
- `text_only_lora` (`steps=1000`): `acc=0.875`, `ret=0.900`, `ECE=0.1775`, `Brier=0.1080`, `lat=12.8ms`
- `bio_paired_adapter` (`steps=1000`): `acc=0.875`, `ret=0.890`, `ECE=0.1898`, `Brier=0.1204`, `lat=16.5ms`
- `bio_shuffled_control` (`steps=1000`): `acc=0.875`, `ret=0.880`, `ECE=0.2115`, `Brier=0.1286`, `lat=16.5ms`
- `matched_synthetic_control` (`steps=1000`): `acc=0.875`, `ret=0.890`, `ECE=0.1929`, `Brier=0.1170`, `lat=16.5ms`

## Negative Controls (8 / 8 rejected)
- `nc1_unequal_training_budget_rejected` -> `invalid` (giving bio_paired_adapter 2x training steps rejected)
- `nc2_culture_session_leakage_rejected` -> `invalid` (shared culture/session between train and eval splits rejected)
- `nc3_task_family_leakage_rejected` -> `invalid` (shared task family across train and held-out splits rejected)
- `nc4_reconstructed_timing_mislabeled_measured` -> `invalid` (reconstructed/interpolated timing refused from claiming measured hardware timing)
- `nc5_unmatched_synthetic_moments_rejected` -> `invalid` (synthetic control failing variance match rejected)
- `nc6_tied_synthetic_control_falsifies_bio_claim` -> `invalid` (paired bio tying matched synthetic & text LoRA fails falsifiable criterion)
- `nc7_nonfinite_calibration_prob_rejected` -> `invalid` (NaN predicted probability rejected cleanly in ECE/Brier reducer)
- `nc8_unverified_biollm8b_benchmark_assumption` -> `blocked_on_artifacts` (do not assume current BioLLM-8B benchmarks exist; stays blocked_on_artifacts)
