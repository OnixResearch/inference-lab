# Issue #68: LinQuant Algebraic Recurrent Prefill Rewrite on GDN & Mamba-2

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `pin_status: post_only_unlocated_paper_repo`, cross-repo: `OnixResearch/tt-onix#34`)
- Hardware / artifact gate: `blocked_on_artifacts` (`grants_blackhole_acceptance: false`)
- Exact FP64 algebraic invariance gate (verified before quantization): `gdn_max_err=6.069e-16`, `mamba2_max_err=1.388e-16` (tol `<= 1e-12`)

## Low-Precision Naive vs LinQuant Rewrite Error (Outlier Fixture)
- `gdn` (`bf16`): `naive_state_l2=0.00546` vs `linquant_state_l2=0.00690` (`reduction=Some(0.790654550539389)`), `naive_y_max=0.00720` vs `linquant_y_max=0.00446`
- `mamba2` (`bf16`): `naive_state_l2=0.01748` vs `linquant_state_l2=0.00678` (`reduction=Some(2.5790586431092355)`), `naive_y_max=0.00113` vs `linquant_y_max=0.00101`
- `gdn` (`fp8_e4m3`): `naive_state_l2=0.16426` vs `linquant_state_l2=0.09954` (`reduction=Some(1.6501442554410528)`), `naive_y_max=0.10102` vs `linquant_y_max=0.07771`
- `mamba2` (`fp8_e4m3`): `naive_state_l2=0.12307` vs `linquant_state_l2=0.06628` (`reduction=Some(1.8567036919557958)`), `naive_y_max=0.41219` vs `linquant_y_max=0.02070`
- `gdn` (`int8_per_vec`): `naive_state_l2=0.03022` vs `linquant_state_l2=0.01461` (`reduction=Some(2.068011505913663)`), `naive_y_max=0.01967` vs `linquant_y_max=0.01634`
- `mamba2` (`int8_per_vec`): `naive_state_l2=0.02676` vs `linquant_state_l2=0.00595` (`reduction=Some(4.500138343074442)`), `naive_y_max=0.05543` vs `linquant_y_max=0.00294`

## Negative Controls (8 / 8 rejected)
- `nc1_non_orthogonal_rewrite_breaks_fp64_invariance` -> `invalid` (perturbed rewrite broke FP64 invariance by 1.701e-1 (> 1e-12))
- `nc2_unstable_gdn_alpha_rejected` -> `invalid` (alpha=1.25 > 1.0 rejected before recurrence)
- `nc3_negative_mamba2_decay_rejected` -> `invalid` (negative Mamba-2 decay -0.1 rejected)
- `nc4_nan_value_vector_rejected` -> `invalid` (NaN value vector refused in FP8 LinQuant rewrite)
- `nc5_infinite_mamba2_scalar_rejected` -> `invalid` (Inf scalar input rejected in Mamba-2 BF16 prefill)
- `nc6_zero_error_divisor_guarded` -> `invalid` (0.0 linquant error yields error_reduction_ratio=None instead of Inf)
- `nc7_portable_cpu_does_not_grant_blackhole_acceptance` -> `unsupported` (physical Blackhole fused GDN kernel verification remains tracked in tt-onix#34)
- `nc8_unlocated_linquant_reference_blocked` -> `blocked_on_artifacts` (public LinQuant reference repo/paper formulation unlocated offline)
