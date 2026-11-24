# Issue #92: Sparse-MoE Speculative Verification Roofline & Temporal Expert Overlap

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (30B–1T sparse MoE GPU speculative verification (arXiv:2610.38981, QuantSpec, MoE-SpeQ))
- Governor recommended draft length: FP16 KV `Some(8)`, QuantSpec FP8 KV `Some(8)`

## Temporal Expert-Routing Overlap & Batched Parity
- `gamma=1`: `unique_experts=2` vs `naive_bound=2` (`overlap=0.0%`), `parity_err=0.000e0`
- `gamma=2`: `unique_experts=2` vs `naive_bound=4` (`overlap=50.0%`), `parity_err=0.000e0`
- `gamma=4`: `unique_experts=4` vs `naive_bound=8` (`overlap=50.0%`), `parity_err=0.000e0`
- `gamma=8`: `unique_experts=4` vs `naive_bound=16` (`overlap=75.0%`), `parity_err=0.000e0`

## Negative Controls (8 / 8 rejected)
- `nc1_zero_top_k_rejected` -> `invalid` (top_k = 0 rejected cleanly)
- `nc2_oob_top_k_rejected` -> `invalid` (top_k > NUM_EXPERTS rejected)
- `nc3_zero_draft_length_rejected` -> `invalid` (draft_len_gamma = 0 rejected)
- `nc4_invalid_acceptance_alpha_rejected` -> `invalid` (alpha > 1.0 rejected cleanly)
- `nc5_kv_exceeding_vram_returns_oom` -> `oom` (2M-context FP16 KV exceeding 112 GiB budget returns Oom)
- `nc6_failed_kv_parity_disqualified` -> `invalid` (QuantSpec quantized KV failing parity gate is disqualified)
- `nc7_nonfinite_bandwidth_rejected` -> `invalid` (NaN memory bandwidth rejected)
- `nc8_empty_draft_block_rejected` -> `invalid` (empty draft token slice rejected)
