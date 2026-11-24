# Issue #64: Two-Spark KV Sharding & Disk-Persisted KV Checkpoint Resume (GLM-5.3-Flash)

- Execution class: `bounded_contract_harness` (`default_enabled: false`, pinned `MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104@6c55c20d` `Apache-2.0`)
- Hardware gate: `unavailable_hardware_runtime` (2x DGX Spark cluster serving pinned MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104@6c55c20d with NVMe KV checkpoints)

## Sharded Disk-Resume vs Fresh Recomputation Parity
- `chat_resume_8_plus_3` (`cached=8`, `tail=3`, `steps=5`): `bytes=2064`, `exact=true`, `max_logit_err=0.000e0`, `outcome=accepted`
- `chat_resume_12_plus_4` (`cached=12`, `tail=4`, `steps=5`): `bytes=3088`, `exact=true`, `max_logit_err=0.000e0`, `outcome=accepted`

## KV Capacity Headroom & Save/Load I/O vs Recomputation
- `2spark_101k_chat_98k_cached` (`cap=3148148`, `cached=98000`, `tail=3000`): `headroom=Some(0.9679176455490657)`, `load_s=Some(0.9621818181818181)`, `resume_ttft_s=Some(2.212181818181818)` vs `recompute_ttft_s=42.08`, `outcome=accepted`
- `2spark_512k_session_490k_cached` (`cap=3148148`, `cached=490000`, `tail=22000`): `headroom=Some(0.837364698228927)`, `load_s=Some(4.8109090909090915)`, `resume_ttft_s=Some(14.810909090909092)` vs `recompute_ttft_s=232.73`, `outcome=accepted`
- `2spark_1m_session_980k_cached` (`cap=3148148`, `cached=980000`, `tail=20000`): `headroom=Some(0.682352926228373)`, `load_s=Some(9.621818181818183)`, `resume_ttft_s=Some(20.14813397129187)` vs `recompute_ttft_s=526.32`, `outcome=accepted`

## Negative Controls (8 / 8 rejected)
- `nc1_corrupted_shard1_checksum_rejected` -> `invalid` (1-bit corruption in Spark-1 KV shard refused by SHA-256 verification)
- `nc2_prefix_token_drift_rejected` -> `invalid` (modified prefix token failed prefix_tokens_sha256 match)
- `nc3_weight_digest_mismatch_rejected` -> `invalid` (checkpoint produced by different weight digest refused)
- `nc4_unpinned_commit_mismatch_rejected` -> `invalid` (unpinned/mismatched GLM-5.3-Flash-DGX-Sparks commit refused)
- `nc5_truncated_shard0_rejected` -> `invalid` (truncated Spark-0 shard refused even with recomputed digest)
- `nc6_nan_kv_entry_blocked_on_save` -> `invalid` (NaN KV shard entry refused before disk checkpoint serialization)
- `nc7_kv_capacity_overflow_returns_oom` -> `oom` (4.1M tokens > 2-Spark capacity returned explicit oom without silent eviction)
- `nc8_zero_nvme_bandwidth_guarded` -> `invalid` (0 GB/s NVMe throughput returned invalid with save/load_io_s=None)
