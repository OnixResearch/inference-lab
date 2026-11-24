# Issue #89: Cross-Loop KV Cache Compression (ResidualQuant INT2 & LoopVQ)

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (Pinned looped-transformer GPU checkpoint (TML-d24-varloop-research, #69))
- Zero-residual identity max error: `0.000e0` (`<= 1e-12`)

## Cross-Loop KV Compression Comparison
- `full_per_loop_fp16_kv`: `payload_bpw=16.00`, `bytes=4096/4096` (`1.00x`, `0.0%` saved incl. metadata), `mse=0.000e0`, `attn_err=0.000e0`, `pack_exact=true`
- `residual_quant_int2_cross_loop`: `payload_bpw=5.50`, `bytes=1432/4096` (`2.86x`, `65.0%` saved incl. metadata), `mse=3.387e-4`, `attn_err=1.276e-2`, `pack_exact=true`
- `loop_vq_0p5_bpw_joint`: `payload_bpw=0.50`, `bytes=640/4096` (`6.40x`, `84.4%` saved incl. metadata), `mse=5.078e-2`, `attn_err=8.109e-2`, `pack_exact=true`
- `loop_vq_0p25_bpw_joint`: `payload_bpw=0.25`, `bytes=576/4096` (`7.11x`, `85.9%` saved incl. metadata), `mse=1.267e-1`, `attn_err=1.199e-1`, `pack_exact=true`

## Negative Controls (8 / 8 rejected)
- `nc1_int2_code_out_of_range_rejected` -> `invalid` (INT2 symbol 4 (>3) rejected by bit-exact packer)
- `nc2_int2_unpack_length_mismatch_rejected` -> `invalid` (truncated packed buffer rejected cleanly)
- `nc3_nonfinite_residual_rejected` -> `invalid` (NaN cross-loop residual rejected before quantization)
- `nc4_zero_norm_cosine_guard` -> `invalid` (zero-norm KV vector returns None instead of NaN)
- `nc5_loop_vq_oob_index_rejected` -> `invalid` (out-of-range LoopVQ codebook index rejected)
- `nc6_negative_residual_scale_rejected` -> `invalid` (negative dequantization scale rejected)
- `nc7_loop_vq_dim_mismatch_rejected` -> `invalid` (mismatched cross-loop vector group length rejected)
- `nc8_unverified_80pct_claim_non_gating` -> `invalid` (authors' unverified >80% compression figure is non-gating; codebook/scale metadata always counted)
