# Issue #48: NVMe Interrupt Coalescing Controlled Host Variable (DGX Spark)

- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (Physical NVIDIA DGX Spark NVMe (/dev/nvme0) with nvidia-nvme-interrupt-coalescing.service and SSD-paged Engram/PLE tables)
- Probe command: `nvme get-feature /dev/nvme0 -f 8 -H` | Unit: `nvidia-nvme-interrupt-coalescing.service` | `mutates_host_service: false`

## 2x2 Controlled Factor Matrix (Storage 4K QD1 Separated from Decode)
- `cold_coalescing_off` (`coalescing=off`, `cache=cold`): `storage_p50_us=61.5`, `mean_io_ms=0.846`, `mean_decode_ms=15.098`, `decode_tok_s=66.23`
- `cold_coalescing_on` (`coalescing=on`, `cache=cold`): `storage_p50_us=194.0`, `mean_io_ms=2.579`, `mean_decode_ms=16.831`, `decode_tok_s=59.41`
- `warm_coalescing_off` (`coalescing=off`, `cache=warm`): `storage_p50_us=61.5`, `mean_io_ms=0.097`, `mean_decode_ms=14.350`, `decode_tok_s=69.69`
- `warm_coalescing_on` (`coalescing=on`, `cache=warm`): `storage_p50_us=194.0`, `mean_io_ms=0.297`, `mean_decode_ms=14.549`, `decode_tok_s=68.73`

## Negative Controls (8 / 8 rejected)
- `nc1_unknown_coalescing_never_defaults_off` -> `invalid` (unparseable nvme feature 8 output classifies as unknown and is refused)
- `nc2_missing_drive_model_rejected` -> `invalid` (drive model is a required controlled host factor)
- `nc3_cold_vs_warm_cache_conflation_rejected` -> `invalid` (paired coalescing comparison cannot mix cold and warm page-cache runs)
- `nc4_storage_latency_not_tok_s_claim` -> `invalid` (4K QD1 storage latency is strictly separated from decode tok/s claims)
- `nc5_nan_latency_sample_rejected` -> `invalid` (non-finite SSD I/O latency is rejected instead of swallowed)
- `nc6_zero_tokens_guarded` -> `invalid` (zero decoded tokens yields None decode latency and tok/s)
- `nc7_host_service_mutation_refused` -> `invalid` (harness records controlled host variables without mutating systemd units)
- `nc8_post_latency_claims_not_pass_gates` -> `invalid` (unverified X post 55-57 us vs 200 us figures are never lab acceptance thresholds)
