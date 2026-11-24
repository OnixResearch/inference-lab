# Issue #63: 2-Node DGX Spark RDMA Tensor Parallelism & 4-Bit Attention (8K–1M Context)

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `pin_status: post_only_unlocated_paper_repo`)
- Hardware / artifact gate: `blocked_on_artifacts` (2x DGX Spark cluster over raw RDMA running RLX with 4-bit attention out to 1M context)
- Reference parity: `fp64_tp2_max_abs_err=1.110e-16`, `q4_tp2_max_abs_err=0.003857` (gate `<= 0.080`)

## Matched-Context Topology Comparison (`8K`, `128K`, `1M`)
- `rdma_tp2_int4_attn` @ `8192` (`int4_kv_fp16_compute`): `parity_ok=true`, `task_pass=0.88`, `decode_tok_s=Some(30.0)`, `prefill_tok_s=Some(1800.0)`, `outcome=accepted`
- `replicated_single_node` @ `8192` (`bf16_kv`): `parity_ok=true`, `task_pass=0.92`, `decode_tok_s=Some(16.5)`, `prefill_tok_s=Some(950.0)`, `outcome=accepted`
- `layer_split_pp2` @ `8192` (`bf16_kv`): `parity_ok=true`, `task_pass=0.92`, `decode_tok_s=Some(18.2)`, `prefill_tok_s=Some(1020.0)`, `outcome=accepted`
- `rdma_tp2_int4_attn` @ `131072` (`int4_kv_fp16_compute`): `parity_ok=true`, `task_pass=0.86`, `decode_tok_s=Some(27.6)`, `prefill_tok_s=Some(1656.0)`, `outcome=accepted`
- `replicated_single_node` @ `131072` (`bf16_kv`): `parity_ok=true`, `task_pass=0.91`, `decode_tok_s=Some(15.180000000000001)`, `prefill_tok_s=Some(874.0)`, `outcome=accepted`
- `layer_split_pp2` @ `131072` (`bf16_kv`): `parity_ok=true`, `task_pass=0.91`, `decode_tok_s=Some(16.744)`, `prefill_tok_s=Some(938.4000000000001)`, `outcome=accepted`
- `rdma_tp2_int4_attn` @ `1048576` (`int4_kv_fp16_compute`): `parity_ok=true`, `task_pass=0.84`, `decode_tok_s=Some(25.2)`, `prefill_tok_s=Some(1512.0)`, `outcome=accepted`
- `replicated_single_node` @ `1048576` (`bf16_kv`): `parity_ok=true`, `task_pass=0.90`, `decode_tok_s=Some(13.86)`, `prefill_tok_s=Some(798.0)`, `outcome=accepted`
- `layer_split_pp2` @ `1048576` (`bf16_kv`): `parity_ok=true`, `task_pass=0.90`, `decode_tok_s=Some(15.287999999999998)`, `prefill_tok_s=Some(856.8)`, `outcome=accepted`

## Negative Controls (8 / 8 rejected)
- `nc1_rdma_checksum_corruption_rejected` -> `invalid` (corrupted rank-1 RDMA payload checksum aborted all-reduce)
- `nc2_q4_numerical_drift_blocks_throughput` -> `invalid` (q4 error 0.25 > gate 0.08 forced throughput_permitted=false and tok/s=None)
- `nc3_long_context_task_failure_blocks_throughput` -> `invalid` (task pass rate 0.50 < 0.75 gate suppressed reported throughput)
- `nc4_fabricated_rlx_commit_sha_rejected` -> `invalid` (all-zero fabricated RLX SHA rejected; unlocated repo stays honestly unpinned)
- `nc5_nan_in_4bit_quantizer_rejected` -> `invalid` (NaN K/V activation refused by 4-bit quantizer)
- `nc6_zero_context_length_rejected` -> `invalid` (context_tokens=0 refused before matrix admission)
- `nc7_out_of_range_pass_rate_rejected` -> `invalid` (task_pass_rate=1.25 outside [0, 1] refused)
- `nc8_unlocated_rlx_artifact_blocked` -> `blocked_on_artifacts` (RLX Rust engine & RDMA crate artifacts unlocated offline; physical run blocked)
