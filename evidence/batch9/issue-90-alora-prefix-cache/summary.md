# Issue #90: Activated LoRA (aLoRA) vs Hypertune Soft-Prefill KV (#66) & Standard LoRA

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (Multi-adapter GPU serving comparing aLoRA (arXiv:2512.17910), Hypertune (#66), and S-LoRA/Punica)

## Pure-Rust Prefix-KV Invariance & Delayed-Suffix Math
- `unadapted_base`: `prefix_kv_diff=0.000e0`, `reusable=true`, `cached_vs_full_err=0.000e0`, `suffix_delta_l2=0.0000`
- `standard_lora_full_sequence`: `prefix_kv_diff=3.277e-1`, `reusable=false`, `cached_vs_full_err=3.277e-1`, `suffix_delta_l2=0.1340`
- `alora_delayed_suffix_default_off`: `prefix_kv_diff=0.000e0`, `reusable=true`, `cached_vs_full_err=0.000e0`, `suffix_delta_l2=0.0463`
- `hypertune_soft_kv_adapter`: `prefix_kv_diff=0.000e0`, `reusable=true`, `cached_vs_full_err=0.000e0`, `suffix_delta_l2=0.1091`

## Negative Controls (8 / 8 rejected)
- `nc1_standard_lora_breaks_prefix_kv_invariance` -> `invalid` (standard LoRA perturbs prefix KV by 3.277e-1 (> 1e-12); cannot share base prefix cache)
- `nc2_early_activation_inside_prefix_rejected` -> `invalid` (activating aLoRA before prefix boundary breaks shared prefix KV cache)
- `nc3_zero_lora_rank_rejected` -> `invalid` (LoRA rank 0 rejected (zero-divisor guard))
- `nc4_nonfinite_lora_alpha_rejected` -> `invalid` (NaN LoRA alpha rejected cleanly)
- `nc5_empty_sequence_rejected` -> `invalid` (empty token sequence rejected)
- `nc6_zero_concurrent_adapters_rejected` -> `invalid` (zero concurrent adapters rejected cleanly)
- `nc7_mismatched_kv_cache_len_rejected` -> `invalid` (mismatched prefix KV lengths rejected)
- `nc8_mismatched_base_model_digest_prefix_reuse_rejected` -> `invalid` (shared prefix KV reuse requires identical base-model and tokenizer digests)
