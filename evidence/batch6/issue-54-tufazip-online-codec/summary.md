# Issue #54: Bounded tufazip-Style Online Compressor (Small Prefix & B200 Gate)

- Pin: `Tufalabs/tufazip@b9d678db3a0f726e117c8cab51927975830395ac` (`MIT`) | `default_enabled: false` | `enwik9_full_run_attempted: false`
- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (Linux x86-64 + 2x NVIDIA B200 GPUs + CUDA 13.0 for official Tufalabs/tufazip enwik9-v2.bin)

## Lossless Round-Trip Comparison (`input_bytes=2223`, `sha256=67a98f6e7c6b949a0682c9806f0d4463a59bacd9efa7d3a92a14634aa0415c7f`)
- `miniature_online_fixed_point_range`: `payload=1284`, `prior=16`, `total=1300`, `bpb=4.678`, `online_weights_transferred=false`, `sha256_match=true`
- `conventional_lz77_baseline`: `payload=700`, `prior=0`, `total=700`, `bpb=2.519`, `online_weights_transferred=false`, `sha256_match=true`

## Negative Controls (8 / 8 rejected)
- `nc1_decoder_skipping_online_updates_fails_roundtrip` -> `invalid` (disabling decoder-side online weight relearning breaks bit-exact reconstruction (finding, not silent pass))
- `nc2_single_bit_payload_corruption_detected` -> `invalid` (flipping one payload bit breaks SHA-256 round-trip verification)
- `nc3_omitting_prior_bytes_rejected` -> `invalid` (total size must include required offline prior/decoder bytes)
- `nc4_full_enwik9_first_arm_refused` -> `unsupported` (full 1 GB enwik9 job is refused by bounded-prefix policy)
- `nc5_non_2xb200_hardware_gate` -> `unavailable_hardware_runtime` (official tufazip decoder requires Linux x86-64 + 2x NVIDIA B200 + CUDA 13.0)
- `nc6_unpinned_tufazip_commit_rejected` -> `invalid` (floating commit 'HEAD' is rejected; must pin exact tufazip SHA)
- `nc7_author_enwik9_rank_not_pass_gate` -> `invalid` (author's enwik9 total, rank, and B200 timings are unverified related work, never pass gates)
- `nc8_empty_input_zero_divisor_guarded` -> `invalid` (empty input yields None bpb and compression ratio (no division by zero))
