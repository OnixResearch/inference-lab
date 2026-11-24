# Issue #70: Pinned `ttf-home` 1.6-bpw Ternary Qwen3.8-27B vs 4-bit GGUF Controls

- Execution class: `bounded_contract_harness` (`default_enabled: false`, pinned `oguzeldereli/ttf-home@2f470627` license `NOASSERTION`)
- Hardware gate: `unavailable_hardware_runtime` (12 GB VRAM GPU host running pinned oguzeldereli/ttf-home@2f470627 modified llama.cpp ternary Qwen3.8-27B vs 4-bit GGUF & FP16/BF16)
- Ternary quantizer check: `pack_roundtrip_exact=true`, `matvec_max_abs_err=2.220e-16`, `trit_bpw=1.60`

## Quantization Arms (Numeric Parity, VRAM, Task Pass & Cost)
- `qwen38_27b_fp16_bf16_baseline` (`16.00 bpw`): `mse=0.00000`, `kl=0.00000`, `vram=56.0 GB`, `math=0.91`, `code=0.92` (`delta=+0.00`), `agent=0.86`, `cost/acc=Some(1.0)`
- `qwen38_27b_q4_k_m_gguf_control` (`4.50 bpw`): `mse=0.00090`, `kl=0.00148`, `vram=18.5 GB`, `math=0.89`, `code=0.90` (`delta=-0.02`), `agent=0.83`, `cost/acc=Some(0.75)`
- `qwen38_27b_ttf_home_1_6bpw_ternary` (`1.60 bpw`): `mse=0.03942`, `kl=0.03788`, `vram=8.2 GB`, `math=0.88`, `code=0.81` (`delta=-0.11`), `agent=0.76`, `cost/acc=Some(0.625)`

## Negative Controls (8 / 8 rejected)
- `nc1_invalid_base3_packed_byte_rejected` -> `invalid` (byte 243 (>= 3^5) rejected by 5-trits-per-byte unpacker)
- `nc2_out_of_domain_trit_rejected` -> `invalid` (trit value +2 outside {-1, 0, +1} rejected before packing)
- `nc3_nan_weight_rejected` -> `invalid` (NaN weight refused in absmean ternary quantizer)
- `nc4_zero_accepted_tasks_returns_none_cost` -> `exhausted` (0 accepted tasks produced cost_per_accepted_task=None and exhausted)
- `nc5_code_regression_surfaced_not_masked_by_math` -> `invalid` (code_delta_vs_fp16=-0.110 flagged explicit code regression)
- `nc6_corrupted_packed_trit_caught_by_parity` -> `invalid` (1-byte packed trit flip diverged matvec by 1.677e-1)
- `nc7_zero_q4_block_size_rejected` -> `invalid` (Q4_0 block_size=0 rejected)
- `nc8_unverified_license_blocks_default_admission` -> `unsupported` (ttf-home license is NOASSERTION; default-off experimental profile only)
