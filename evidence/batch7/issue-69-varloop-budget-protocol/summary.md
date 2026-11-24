# Issue #69: Pinned `TML-d24-varloop-research` Loop Scaling, 2DSD Lossless Verification & Equal-Cost Protocol

- Execution class: `bounded_contract_harness` (`default_enabled: false`, pinned `TML-d24-varloop-research@657b53ba`, `SqueezeAILab/2DSD@962874f3` `feat/inkling-dspark`, drafter `@57f09bbe`)
- Hardware gate: `unavailable_hardware_runtime` (Linux x86_64 + CUDA 13.0 + BF16 NVIDIA GPU (H100/L40S/A100 >= 48 GB) for TML-d24-varloop-research & SqueezeAILab/2DSD)

## Lossless 2DSD & Autoregressive Cells (`R = 1, 3, 4, 5`)
- `r1_autoregressive` (`R=1`, `autoregressive`): `mismatches=0`, `timing_ok=true`, `loop1_agree=Some(1.0)`, `acc_len=Some(1.0)`, `tok_s=Some(176.88679245283018)`
- `r3_autoregressive` (`R=3`, `autoregressive`): `mismatches=0`, `timing_ok=true`, `loop1_agree=Some(0.875)`, `acc_len=Some(1.0)`, `tok_s=Some(86.00917431192661)`
- `r3_bundled_mtp` (`R=3`, `mtp_drafter`): `mismatches=0`, `timing_ok=true`, `loop1_agree=Some(0.875)`, `acc_len=Some(4.0)`, `tok_s=Some(404.749055585537)`
- `r3_2dsd` (`R=3`, `2dsd`): `mismatches=0`, `timing_ok=true`, `loop1_agree=Some(0.875)`, `acc_len=Some(4.0)`, `tok_s=Some(477.8287461773701)`
- `r4_2dsd_r3_drafter_transfer` (`R=4`, `2dsd`): `mismatches=0`, `timing_ok=true`, `loop1_agree=Some(0.875)`, `acc_len=Some(4.0)`, `tok_s=Some(380.17031630170317)`
- `r5_2dsd_r3_drafter_transfer` (`R=5`, `2dsd`): `mismatches=0`, `timing_ok=true`, `loop1_agree=Some(0.875)`, `acc_len=Some(3.2)`, `tok_s=Some(252.5252525252525)`

## Equal-Cost Loop vs Thinking-Token Comparison (Unfiltered & Repetition-Filtered Side-by-Side)
- `r1_more_thinking_tokens` (`long_context_reasoning_graphwalks_mrcr`, `R=1`, `budget_tok=452`): `unfiltered=0.58`, `rep_filtered=0.62`
- `r3_2dsd_equal_cost` (`long_context_reasoning_graphwalks_mrcr`, `R=3`, `budget_tok=1223`): `unfiltered=0.72`, `rep_filtered=0.76`
- `r1_closed_book` (`closed_book_knowledge`, `R=1`, `budget_tok=452`): `unfiltered=0.66`, `rep_filtered=0.67`
- `r3_2dsd_closed_book` (`closed_book_knowledge`, `R=3`, `budget_tok=1223`): `unfiltered=0.65`, `rep_filtered=0.66`

## Negative Controls (8 / 8 rejected)
- `nc1_arm64_dgx_spark_refused_for_2dsd` -> `unavailable_hardware_runtime` (ARM64 DGX Spark returned unavailable_hardware_runtime for x86_64 2DSD arm)
- `nc2_24gb_host_refused_for_bf16_2dsd` -> `unavailable_hardware_runtime` (24 GB VRAM host below BF16 floor returned unavailable_hardware_runtime)
- `nc3_spark_or_quantized_substitution_rejected` -> `invalid` (attempt to substitute Spark or quantized numbers into BF16 2DSD arm refused)
- `nc4_2dsd_main_branch_without_inkling_rejected` -> `unsupported` (SqueezeAILab/2DSD main@431aade2 has no Inkling entry; requires feat/inkling-dspark)
- `nc5_lossy_2dsd_mismatch_blocks_timing` -> `invalid` (lossy 2DSD produced 4 mismatches vs AR and suppressed tok/s to None)
- `nc6_zero_loop_count_rejected` -> `invalid` (loop_count=0 refused before forward pass)
- `nc7_cuda_12_host_rejected_for_cuda_13_gate` -> `unavailable_hardware_runtime` (CUDA 12.8 host fails 2DSD CUDA 13.0 requirement)
- `nc8_default_on_2dsd_vllm_fork_refused` -> `unsupported` (2DSD & vLLM fork remain strictly default-off; KV sharing stays in #89)
