# Issue #95: Ling-3.2-air NVFP4 Single-Spark 2x2 FP8 KV vs Torch Allocator Cap Separation

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (Single DGX Spark (GB10) serving pinned Ling-3.2-air NVFP4 at 262,144 context)
- Pinned recipe: `tonyd2wild/Ling-3.2-air-NVFP4-DGX-Spark@d2548801e02c3bfb337a35a9fd9b0d16d71df3f2` (`MIT`); NVFP4 `e7461aad4dfaedf0e4064d3db0e08d53ce641711`; base `1613b1a4dcc6aa0d8f346606628cd9a92061a525`
- 6-axis profile digest sensitivity verified: `true`
- Pure-Rust MLA parity: isolated 1-stage BF16 max err `0.000e0` (`top1=1.00`), FP8-E4M3 KV `top1=1.00`, `mean_lp_drift=0.0055`, `max_lp_drift=0.0191`

## 2x2 KV-Dtype x Allocator-Policy Matrix (MemAvailable Brake = 8192 MiB)
- `bf16_kv_stock_alloc` (`bf16`, `stock`): `brake_tripped=true`, `outcome=oom`, `full_pool_outcome=oom`, `counters=4`
- `bf16_kv_capped_alloc` (`bf16`, `capped_FP8FIX_TORCH_GROWTH_GIB=8`): `brake_tripped=false`, `outcome=accepted`, `full_pool_outcome=accepted`, `counters=4`
- `fp8_kv_stock_alloc` (`fp8_e4m3`, `stock`): `brake_tripped=true`, `outcome=oom`, `full_pool_outcome=oom`, `counters=4`
- `fp8_kv_capped_alloc` (`fp8_e4m3`, `capped_FP8FIX_TORCH_GROWTH_GIB=8`): `brake_tripped=false`, `outcome=accepted`, `full_pool_outcome=accepted`, `counters=4`

## Negative Controls (9 / 9 rejected)
- `nc1_floating_vllm_image_tag_rejected` -> `invalid` (floating `:nightly` vLLM image tag rejected)
- `nc2_unpinned_weight_revision_rejected` -> `invalid` (non-40-hex `main` weight revision rejected)
- `nc3_fp8_missing_mla_patch_returns_unsupported` -> `unsupported` (FP8 KV arm without 1-stage MLA decode patch recorded as Unsupported)
- `nc4_capped_alloc_missing_patch_returns_unsupported` -> `unsupported` (capped allocator arm without torch growth-cap patch recorded as Unsupported)
- `nc5_mem_available_brake_retained_without_retry` -> `oom` (bf16_kv_stock_alloc tripping 8192 MiB MemAvailable brake retained as Oom with silent_retry_performed=false)
- `nc6_full_pool_stress_brake_retained` -> `oom` (fp8_kv_stock_alloc tripping MemAvailable brake under full-pool stress retained as Oom)
- `nc7_position_zero_probe_pressure_only` -> `invalid` (bench/longprompt.py depth=0.0 probe segregated as memory_pressure_check_only)
- `nc8_nonfinite_gpu_mem_utilization_rejected` -> `invalid` (NaN gpu_memory_utilization rejected cleanly)
- `nc9_single_repeat_no_noise_floor_rejected` -> `invalid` (single throughput sample cannot establish a local noise floor)
