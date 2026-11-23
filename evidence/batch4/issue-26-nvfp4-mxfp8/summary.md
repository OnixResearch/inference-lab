# Issue #26: NVFP4 Forward / MXFP8 Backward Numerics & Native Gate

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`rtx5080_sm120_and_spark_sm121_native_nvfp4_mxfp8`)
- Decision: `NUMERICAL_REFERENCE_GO_NATIVE_SM120_SM121_MXFP8_UNSUPPORTED_AT_PINNED_TE`
- Fixed-operand surrogate FD gradient max error: `2.202e-9`
- Arm 3 stochastic MXFP8 backward (`S=64`): dX max |mean err| `0.00924`, dW max |mean err| `0.02911`, operand cov `1.736e-4` (Arm 4 deterministic bias: dX `0.32335`, dW `0.49301`)

## Transformer Engine / CUTLASS per-recipe matrix
| GPU | CC | Arm | Outcome | Reason |
| --- | --- | --- | --- | --- |
| `B200_datacenter` | `sm_100` | `nvfp4_fwd_mxfp8_bwd` | `unavailable_hardware_runtime` | gate allows CC 10.0; physical SM100 hardware not attached locally |
| `GB300_datacenter` | `sm_103` | `nvfp4_fwd_mxfp8_bwd` | `unavailable_hardware_runtime` | gate allows CC 10.3; physical SM103 hardware not attached locally |
| `RTX_5080` | `sm_120` | `te_all_gemm_layout_mxfp8_bwd` | `unsupported` | inspected TE gate explicitly rejects all-GEMM-layout MXFP8 on CC >= 12.0; retained as valid unsupported result (no monkey-patch) |
| `DGX_Spark_GB10` | `sm_121` | `te_all_gemm_layout_mxfp8_bwd` | `unsupported` | inspected TE gate explicitly rejects all-GEMM-layout MXFP8 on CC >= 12.0; SM121 row kept distinct from SM120 |
| `RTX_5080` | `sm_120` | `nvfp4_forward_only_separate_arm` | `unavailable_hardware_runtime` | capability check alone does not prove working linear layer; separate arm from mixed MXFP8 backward |

## Negative controls
- `wrong_or_stale_scale_detected` -> `invalid` (stale_scale_recon_err=1.404115)
- `transposed_block_layout_mismatch` -> `invalid` (transposed_layout_err=6.743411)
- `omitted_saved_forward_reconstruction` -> `invalid` (arm4_dx_err=0.3233, arm4_dw_err=0.4930)
- `correlated_rng_stream_detected` -> `invalid` (reused_stream_cov=8.153915e-4 vs independent_cov=1.735687e-4)
- `non_finite_tensor_rejected` -> `invalid` (non-finite input rejected in encode_nvfp4)
- `te_sm120_sm121_mxfp8_refusal_and_no_bf16_relabel` -> `unsupported` (sm120_unsupported=true, sm121_unsupported=true)
- `modelopt_format_and_calibration_mismatch` -> `invalid` (format_rejected=true, calib_rejected=true, version_rejected=true)
- `fabric_codec_incompatible_transform_and_duplicate_contrib` -> `invalid` (incompatible_rejected=true, dup_rejected=true)
