# Issue #50: Latent Feedback vs Matched Extra-Epoch Control (Data-Limited)

- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (GPU small-model training cluster (60M–778M parameter scale) for data-limited Transformer vs latent-feedback runs)
- Large-scale stability: `open_untested` | FD gradient check: `max_abs_err=3.349e-10` | Causal invariance: `0.000e0` | FLOP ratio: `1.0122`

## Matched-FLOP Miniature Training Arms
- `standard_extra_epoch_control`: `epochs=31`, `unique_tokens=100`, `total_flops=3548880`, `train_loss=2.0882->1.7776`, `heldout_loss=1.9849`, `heldout_acc=0.111`
- `latent_feedback`: `epochs=24`, `unique_tokens=100`, `total_flops=3592320`, `train_loss=2.0861->1.2643`, `heldout_loss=1.6432`, `heldout_acc=0.278`

## Negative Controls (8 / 8 rejected)
- `nc1_unmatched_flop_budget_rejected` -> `invalid` (2x FLOP disparity between arms violates matched-FLOP control)
- `nc2_train_heldout_overlap_rejected` -> `invalid` (duplicate sequence across train and held-out splits is refused)
- `nc3_future_token_leakage_caught` -> `invalid` (injected future-token leak caused 1.172e-2 causal perturbation)
- `nc4_corrupted_gradient_caught_by_fd` -> `invalid` (corrupted analytical gradient diverged by 2.500e-1 vs FD)
- `nc5_large_scale_stability_claim_rejected` -> `invalid` (large-scale stability must remain explicitly open_untested)
- `nc6_nan_learning_rate_rejected` -> `invalid` (non-finite learning rate or loss fails finite check)
- `nc7_zero_flop_budget_guarded` -> `invalid` (zero FLOP count is rejected before ratio division)
- `nc8_paper_savings_not_pass_gate` -> `invalid` (arXiv:2611.20404 48% compute / 47% data claims are unverified, never pass gates)
