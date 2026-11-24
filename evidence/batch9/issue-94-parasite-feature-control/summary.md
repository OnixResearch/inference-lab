# Issue #94: Subspace Interchange, Necessity Ablation & Planted Parasite-Feature Control

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Paper & reference: `arXiv:2611.26711` / `dounick/parasite-features@1df61727bb2f502b882d334eb7f2ca1a38e05712` (`missing_license_reference_only_no_code_copied`)
- Coordination note: Coordinate with inference-lab#77 and rsi-lab#172: report necessity_passed and classification next to any steering or behavioral-specificity metric.

## Sufficiency vs Necessity Controls (16 Matched-Rank Random Subspaces)
- `planted_parasite_rank2_span_e0_e1` (rank `2`): `sufficiency=true` (`rate=1.00`), `ablation_drop=0.00` vs `random_mean=0.062 +/- 0.171` (`max=0.50`) -> `necessity=false`, `classification=sufficient_not_necessary`
- `positive_control_true_task_dir_e2` (rank `1`): `sufficiency=true` (`rate=1.00`), `ablation_drop=0.50` vs `random_mean=0.039 +/- 0.127` (`max=0.50`) -> `necessity=true`, `classification=found_mechanism`
- `#45` tiny Qwen subspace replay exact: `interchange=true`, `ablation=true`, `sufficient=true`, `necessary=true`

## Negative Controls (9 / 9 rejected)
- `nc1_planted_parasite_caught_sufficient_not_necessary` -> `invalid` (rank-2 parasite span(e0,e1) passes interchange (1.0) but fails necessity (0.0 drop vs random) -> sufficient_not_necessary)
- `nc2_rank_mismatch_rejected` -> `invalid` (declared rank=2 with 1 basis vector rejected)
- `nc3_non_orthonormal_basis_rejected` -> `invalid` (non-orthonormal basis (dot != 0 / norm != 1) rejected)
- `nc4_random_subspace_wrong_rank_rejected` -> `invalid` (rank-1 random subspaces supplied for rank-2 necessity check rejected)
- `nc5_fewer_than_10_random_subspaces_rejected` -> `invalid` (only 5 random subspaces (< 10 required) rejected)
- `nc6_tampered_subspace_replay_digest_rejected` -> `invalid` (re-execution against tampered SHA-256 trace digest fails replay check)
- `nc7_nonfinite_basis_element_rejected` -> `invalid` (NaN subspace basis coordinate rejected)
- `nc8_mutable_model_revision_rejected` -> `invalid` (floating `:latest` revision rejected in subspace manifest)
- `nc9_unlicensed_reference_repo_no_code_copied` -> `blocked_on_artifacts` (dounick/parasite-features@1df61727 has no license file; reference reading only, zero code copied)
