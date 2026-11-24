# Issue #62: Disaggregated Spark-CUDA Prefill to Apple-Metal Decode KV Prefix Transfer (10GbE)

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (DGX Spark CUDA prefill host + Apple Silicon Mac Studio Metal decode host over 10GbE)
- Wire contract: `disagg-kv-v1`

## Single-Host vs Disaggregated KV Wire Parity
- `frozen_short_8tok` (`prefix=8`, `steps=6`): `wire_bytes=2056`, `exact=true`, `max_logit_err=0.000e0`, `outcome=accepted`
- `frozen_medium_16tok` (`prefix=16`, `steps=6`): `wire_bytes=4104`, `exact=true`, `max_logit_err=0.000e0`, `outcome=accepted`
- `frozen_repeat_12tok` (`prefix=12`, `steps=6`): `wire_bytes=3080`, `exact=true`, `max_logit_err=0.000e0`, `outcome=accepted`

## Synthetic 10GbE Transfer & Cold/Warm TTFT Contract Profiles
- `ctx_32k_10gbe` (`ctx=32768`, `B/tok=10240`): `single_cold_ttft_s=92.00`, `wire_s=Some(0.29826161777777777)`, `disagg_cold_ttft_s=Some(22.29826161777778)`, `disagg_warm_ttft_s=Some(2.498261617777777)`, `outcome=accepted`
- `ctx_128k_10gbe` (`ctx=131072`, `B/tok=10240`): `single_cold_ttft_s=380.00`, `wire_s=Some(1.193046471111111)`, `disagg_cold_ttft_s=Some(91.19304647111112)`, `disagg_warm_ttft_s=Some(10.19304647111111)`, `outcome=accepted`
- `ctx_241k_10gbe` (`ctx=241000`, `B/tok=10240`): `single_cold_ttft_s=720.00`, `wire_s=Some(2.1936355555555553)`, `disagg_cold_ttft_s=Some(167.19363555555555)`, `disagg_warm_ttft_s=Some(15.39363555555555)`, `outcome=accepted`

## Negative Controls (8 / 8 rejected)
- `nc1_corrupted_wire_checksum_rejected` -> `invalid` (1-bit payload corruption caught by SHA-256 digest verification)
- `nc2_weight_digest_mismatch_rejected` -> `invalid` (prefill KV produced with mismatched decoder weight digest refused)
- `nc3_truncated_wire_payload_rejected` -> `invalid` (truncated KV byte stream refused even with recomputed payload digest)
- `nc4_unsupported_dtype_rejected` -> `unsupported` (unknown wire KV dtype tag refused with unsupported)
- `nc5_prefix_token_count_mismatch_rejected` -> `invalid` (expected 5 prefix tokens vs 4 in wire header refused)
- `nc6_non_finite_kv_entry_rejected` -> `invalid` (NaN KV tensor element refused before wire serialization)
- `nc7_zero_link_bandwidth_guarded` -> `invalid` (0 Gbps link returned invalid with wire_transfer_s=None instead of Inf)
- `nc8_silent_cpu_fallback_refused` -> `invalid` (attempt to substitute CPU execution for Spark-CUDA + Apple-Metal refused)
