# Issue #80: EXL3 2.25-bpw Long-Context Quantization vs FP4/FP64 Baselines

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (Strix Halo APU + RX 7900 XTX / GPU host serving DeepSeek-V4.1-Flash-1017 EXL3 2.25-bpw at 128K-1M)
- Upstream pin: `pepuscz/strix-halo-llm@52993549` (`MIT`, exact packed bpw: `2.25`, roundtrip: `true`)

## Pure-Rust 2.25-bpw Trellis Quantizer vs FP4 & FP64
| Arm | Effective BPW | Weight MSE | Weight SNR (dB) | Logit KL | Pack Parity Err |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fp64_reference` | `64.00` | `0.000000` | `n/a` | `0.000000` | `0.000e0` |
| `fp4_e2m1_block16` | `4.50` | `0.002868` | `20.02` | `0.010747` | `0.000e0` |
| `exl3_2_25bpw_wht16_trellis` | `2.25` | `0.029091` | `9.96` | `0.048174` | `0.000e0` |

## Negative Controls (8 / 8 rejected)
- `nc1_nan_weight_in_exl3_tile_rejected` -> `invalid` (NaN weight rejected before WHT-16 rotation and trellis search)
- `nc2_odd_tile_count_packer_rejected` -> `invalid` (odd 36-bit tile count cannot pack to whole bytes without explicit padding)
- `nc3_oob_4bit_scale_index_rejected` -> `invalid` (scale index >= 16 outside 4-bit EXL3 codebook rejected)
- `nc4_oob_2bit_trellis_symbol_rejected` -> `invalid` (trellis symbol >= 4 outside 2-bit alphabet rejected)
- `nc5_truncated_packed_buffer_rejected` -> `invalid` (truncated 2.25-bpw packed byte buffer rejected)
- `nc6_needle_only_pass_cannot_substitute_end_task_eval` -> `invalid` (100% needle retrieval hides long-context coding/reasoning accuracy drop)
- `nc7_zero_accepted_tasks_cost_is_none` -> `exhausted` (zero accepted reasoning/coding tasks returns None instead of Inf/0)
- `nc8_unverified_strix_halo_x_post_tps_non_gating` -> `invalid` (718/51.1 tok/s X-post figures stay unverified non-gating related work)
