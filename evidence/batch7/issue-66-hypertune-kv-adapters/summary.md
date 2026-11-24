# Issue #66: Pinned Hypertune Soft-Prefill KV Adapters vs Heterogeneous Batched LoRA

- Execution class: `bounded_contract_harness` (`default_enabled: false`, pinned `Hyperstitioncc/Hypertune@7aaaf8c2` `MIT`)
- Hardware gate: `unavailable_hardware_runtime` (GPU serving host running pinned Hyperstitioncc/Hypertune@7aaaf8c2 soft-prefill KV adapters vs batched LoRA)
- Paged soft-KV vs unpaged reference parity: `max_abs_err=0.000e0`, `composed_l2_delta=0.105532`, `lora_l2_delta=0.060192`

## Multi-Adapter Concurrency Scaling (Matched Batch = 16)
- `adapters=1`: `hypertune_kv=8.39 MB`, `lora_bw=48.0 MB`, `hyper_tok_s=Some(219.9932416)`, `lora_tok_s=Some(210.5263157894737)`, `outcome=accepted`
- `adapters=4`: `hypertune_kv=33.55 MB`, `lora_bw=192.0 MB`, `hyper_tok_s=Some(219.9729664)`, `lora_tok_s=Some(186.4406779661017)`, `outcome=accepted`
- `adapters=8`: `hypertune_kv=67.11 MB`, `lora_bw=384.0 MB`, `hyper_tok_s=Some(219.9459328)`, `lora_tok_s=Some(161.76470588235296)`, `outcome=accepted`
- `adapters=16`: `hypertune_kv=134.22 MB`, `lora_bw=768.0 MB`, `hyper_tok_s=Some(219.89186560000002)`, `lora_tok_s=Some(127.90697674418605)`, `outcome=accepted`

## Negative Controls (8 / 8 rejected)
- `nc1_mismatched_soft_kv_block_lengths_rejected` -> `invalid` (prepended K length != V length refused before attention)
- `nc2_nan_soft_kv_entry_rejected` -> `invalid` (NaN in prepended soft-prefill key block caught)
- `nc3_soft_kv_budget_overflow_returns_oom` -> `oom` (64 adapters x 2048 soft tokens exceeded 4096 MB KV budget -> oom)
- `nc4_zero_active_adapters_rejected` -> `invalid` (active_distinct_adapters=0 returned invalid)
- `nc5_nan_lora_alpha_rejected` -> `invalid` (non-finite LoRA scaling alpha refused)
- `nc6_cross_tenant_adapter_swap_detected` -> `invalid` (swapping adapter_code with adapter_sql diverged by 2.060e-1)
- `nc7_zero_task_denominator_returns_none` -> `invalid` (empty task fixture returned pass_rate=None instead of 0/0 NaN)
- `nc8_unmatched_concurrency_comparison_refused` -> `unsupported` (comparing Hypertune batch=16 against LoRA batch=1 refused as unmatched)
