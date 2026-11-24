# Issue #72: K2 Horizon `Uno` Draftless Lossless Decoding & MoVA/Softplus Attention Gating

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `pin_status: arxiv_id_only_artifacts_unverified_offline`, paper `arXiv:2611.21003`)
- Hardware / artifact gate: `blocked_on_artifacts` (GPU decode host evaluating K2 Horizon (arXiv:2611.21003) Uno adapter & MoVA/softplus checkpoints)
- MoVA + Softplus vs Standard Softmax: `softmax_entropy=2.0674`, `mova_entropy=2.0727`, `l2_delta=0.3245`, `softplus(+50)=50.00`, `softplus(-50)=1.929e-22`
- Draftless `Uno` lossless check: `mismatches_vs_ar=0`, `mean_acceptance_length=2.400`

## Batch Scaling & Memory Overhead (`batch = 1, 4, 8`)
- `standard_autoregressive` (`batch=1`): `extra_mem=0 MB`, `tok_s=Some(54.0)`, `speedup=Some(1.0)`, `outcome=accepted`
- `k2_uno_draftless_adapter` (`batch=1`): `extra_mem=140 MB`, `tok_s=Some(103.14)`, `speedup=Some(1.91)`, `outcome=accepted`
- `mtp_head_control` (`batch=1`): `extra_mem=280 MB`, `tok_s=Some(100.872)`, `speedup=Some(1.8679999999999999)`, `outcome=accepted`
- `separate_draft_model_control` (`batch=1`): `extra_mem=2400 MB`, `tok_s=Some(97.84799999999998)`, `speedup=Some(1.8119999999999998)`, `outcome=accepted`
- `standard_autoregressive` (`batch=4`): `extra_mem=0 MB`, `tok_s=Some(174.19354838709677)`, `speedup=Some(1.0)`, `outcome=accepted`
- `k2_uno_draftless_adapter` (`batch=4`): `extra_mem=140 MB`, `tok_s=Some(296.1290322580645)`, `speedup=Some(1.7)`, `outcome=accepted`
- `mtp_head_control` (`batch=4`): `extra_mem=280 MB`, `tok_s=Some(285.3700189753321)`, `speedup=Some(1.638235294117647)`, `outcome=accepted`
- `separate_draft_model_control` (`batch=4`): `extra_mem=2400 MB`, `tok_s=Some(266.0410557184751)`, `speedup=Some(1.5272727272727273)`, `outcome=accepted`
- `standard_autoregressive` (`batch=8`): `extra_mem=0 MB`, `tok_s=Some(276.9230769230769)`, `speedup=Some(1.0)`, `outcome=accepted`
- `k2_uno_draftless_adapter` (`batch=8`): `extra_mem=140 MB`, `tok_s=Some(425.1583710407239)`, `speedup=Some(1.5352941176470587)`, `outcome=accepted`
- `mtp_head_control` (`batch=8`): `extra_mem=280 MB`, `tok_s=Some(407.55852842809355)`, `speedup=Some(1.4717391304347824)`, `outcome=accepted`
- `separate_draft_model_control` (`batch=8`): `extra_mem=2400 MB`, `tok_s=Some(376.41933287950985)`, `speedup=Some(1.35929203539823)`, `outcome=accepted`

## Negative Controls (8 / 8 rejected)
- `nc1_unverified_uno_draft_causes_token_mismatch` -> `invalid` (skipping Uno verification produced 11 mismatches vs AR)
- `nc2_nan_softplus_input_rejected` -> `invalid` (NaN logit refused by overflow-safe softplus gate)
- `nc3_inf_softplus_input_rejected` -> `invalid` (+Inf logit refused before softplus attention normalization)
- `nc4_empty_kv_mova_rejected` -> `invalid` (empty key/value sequence refused in MoVA+softplus attention)
- `nc5_mismatched_mova_value_banks_rejected` -> `invalid` (ragged MoVA value banks refused before routing)
- `nc6_nan_query_softmax_rejected` -> `invalid` (NaN query refused in standard softmax attention reference)
- `nc7_unverified_k2_checkpoints_blocked` -> `blocked_on_artifacts` (arXiv:2611.21003 Uno adapter & checkpoint digests unverified offline)
- `nc8_unverified_2_2x_claim_not_acceptance_gate` -> `unsupported` (2.2x X-post speedup claim stays unverified related work, never a pass threshold)
