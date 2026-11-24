# Issue #86: MIT QDA Mixed-Precision Quantization Recipe vs Pinned Baselines

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Hardware / artifact gate: `blocked_on_artifacts` (MIT QDA public repository & physical GPU decode-cost benchmark)
- Proxy sensitivity vs true layer KL Spearman rho: `0.885714285714`

## Pinned Recipe Comparison (Matched Budget <= 4.5 bpw)
- `fp64_reference`: `bpw=64.000`, `weight_mse=0.000e0`, `logit_kl=0.000e0`, `acc=12/12`, `cost/accepted=8.333333333333`, `outcome=accepted`
- `manual_gguf_int4_uniform`: `bpw=4.500`, `weight_mse=5.965e-3`, `logit_kl=3.674e-2`, `acc=11/12`, `cost/accepted=3.090909090909`, `outcome=accepted`
- `nvfp4_e2m1_uniform`: `bpw=4.500`, `weight_mse=7.061e-3`, `logit_kl=2.647e-2`, `acc=9/12`, `cost/accepted=3.333333333333`, `outcome=accepted`
- `qda_mixed_precision_default_off`: `bpw=4.500`, `weight_mse=4.010e-3`, `logit_kl=1.747e-2`, `acc=9/12`, `cost/accepted=3.555555555556`, `outcome=accepted`

## Negative Controls (8 / 8 rejected)
- `nc1_unlocated_qda_repo_blocked` -> `blocked_on_artifacts` (MIT QDA public repo/code commit not verified locally -> blocked_on_artifacts)
- `nc2_recipe_exceeding_bit_budget_rejected` -> `invalid` (8.5-bpw recipe exceeding 4.5-bpw budget rejected)
- `nc3_unpinned_recipe_digest_rejected` -> `invalid` (floating `latest` recipe digest rejected)
- `nc4_calibration_eval_overlap_rejected` -> `invalid` (non-disjoint calibration and held-out eval split rejected)
- `nc5_nonfinite_weight_rejected` -> `invalid` (NaN weight tensor rejected cleanly)
- `nc6_zero_accepted_tasks_non_collapsing` -> `exhausted` (zero accepted tasks maps to cost_per_accepted_task=None and Exhausted (never 0.0))
- `nc7_nonfinite_spearman_input_rejected` -> `invalid` (NaN sensitivity metric rejected by Spearman rank correlation)
- `nc8_nonpositive_decode_cost_rejected` -> `invalid` (non-positive decode_cost_units rejected cleanly)
