# Issue #67: Pinned `Qwen3.8-27B-Longbook27-v2` (All-64-Layer GDN) vs Base `Qwen3.8-27B`

- Execution class: `bounded_contract_harness` (`default_enabled: false`, pinned `summerAIO/Qwen3.8-27B-Longbook27-v2@46319430` `Apache-2.0`)
- Hardware gate: `unavailable_hardware_runtime` (GPU serving host evaluating pinned summerAIO/Qwen3.8-27B-Longbook27-v2@46319430 vs base Qwen3.8-27B at 8K-262K)
- Miniature 8-key recall: `hybrid_mae=0.0221` (`64 KV elems`) vs `all_gdn_mae=0.3818` (`16 constant elems`)

## 8K–262K Context Memory & Quality Tradeoff Contract
- `ctx=8192`: `base_state=728.9 MB` vs `longbook_state=256.0 MB` (`ratio=Some(2.847152)`), `recall=0.94 vs 0.88`, `agent=0.85 vs 0.82`
- `ctx=32768`: `base_state=2339.5 MB` vs `longbook_state=256.0 MB` (`ratio=Some(9.138608)`), `recall=0.93 vs 0.86`, `agent=0.84 vs 0.80`
- `ctx=131072`: `base_state=8781.9 MB` vs `longbook_state=256.0 MB` (`ratio=Some(34.304432)`), `recall=0.91 vs 0.80`, `agent=0.82 vs 0.76`
- `ctx=262144`: `base_state=17371.9 MB` vs `longbook_state=256.0 MB` (`ratio=Some(67.858864)`), `recall=0.88 vs 0.72`, `agent=0.80 vs 0.70`

## Negative Controls (8 / 8 rejected)
- `nc1_mismatched_quantization_refused` -> `invalid` (comparing bf16 base against q4_k_m Longbook27-v2 refused)
- `nc2_mismatched_runtime_refused` -> `invalid` (comparing vllm base against llama.cpp Longbook27-v2 refused)
- `nc3_mismatched_tokenizer_refused` -> `invalid` (tokenizer mismatch between base Qwen3.8-27B and Longbook27-v2 refused)
- `nc4_unstable_gdn_decay_alpha_rejected` -> `invalid` (GDN alpha=1.5 > 1.0 rejected to prevent exponential state blowup)
- `nc5_nan_gdn_key_rejected` -> `invalid` (NaN key vector refused in GDN recurrence step)
- `nc6_zero_context_rejected` -> `invalid` (context_tokens=0 refused in 8K-262K sweep)
- `nc7_zero_recall_keys_rejected` -> `invalid` (empty associative-recall benchmark refused)
- `nc8_memory_savings_not_recall_parity` -> `invalid` (all-GDN constant state (16 elems) had higher recall MAE (0.3818) than hybrid (0.0221))
