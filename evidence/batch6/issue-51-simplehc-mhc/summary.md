# Issue #51: SimpleHC vs DeepSeek mHC Beside Block Attention Residuals (#33)

- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`)
- Hardware gate: `blocked_on_artifacts` (PyTorch/GPU training comparison against pinned SimpleHC primary source)
- Primary source pin status: `post_only_unlocated_paper_repo` (post `2119613538515960212`; 59-100% overhead claim is non-gating)
- Matched params & tokens: `accepted`; mHC step-FLOP overhead vs SimpleHC: `240.380348652932%`

## Hyper-Connection & Attention-Residual Arms
- `deepseek_mhc`: `params=462`, `tokens=375`, `sinkhorn_ops/step=15616`, `step_flops=21478`, `loss=1.8404->1.3956`, `heldout=3.5477`
- `simple_hc`: `params=462`, `tokens=375`, `sinkhorn_ops/step=0`, `step_flops=6310`, `loss=1.8403->1.4295`, `heldout=3.5434`
- `block_attn_res_ref_33`: `params=462`, `tokens=375`, `sinkhorn_ops/step=0`, `step_flops=3868`, `loss=1.7336->0.2739`, `heldout=4.1622`

## Negative Controls (8 / 8 rejected)
- `nc1_claiming_primary_repo_pinned_rejected` -> `invalid` (claiming unlocated SimpleHC primary repo is pinned rejected)
- `nc2_sinkhorn_doubly_stochastic_violation` -> `invalid` (non-doubly-stochastic matrix rejected (max_err=7.000e-1))
- `nc3_mismatched_params_rejected` -> `invalid` (arms with mismatched parameter counts rejected)
- `nc4_mismatched_tokens_rejected` -> `invalid` (arms with mismatched training token budgets rejected)
- `nc5_nan_in_sinkhorn_or_weights_rejected` -> `invalid` (NaN in Sinkhorn-Knopp logits rejected cleanly)
- `nc6_faulty_candidate_caught_by_scalar_ref` -> `invalid` (faulty stream-mix candidate caught by scalar ref (err=2.278e-2))
- `nc7_zero_flop_divisor_guard` -> `invalid` (zero SimpleHC FLOP divisor returns None and refuses division by zero)
- `nc8_importing_post_59_100_overhead_as_gate_rejected` -> `invalid` (post's 59-100% overhead claim cannot be imported as a pass/fail gate)
