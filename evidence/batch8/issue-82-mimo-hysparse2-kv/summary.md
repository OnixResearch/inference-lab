# Issue #82: MiMo HySparse2 Causal Encoder-Decoder KV Projection

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Artifact gate: `blocked_on_artifacts` (Xiaomi MiMo HySparse2 public checkpoint & reference runtime at 8K-1M context)
- Pure-Rust parity: `cached_vs_full = 0.000e0`, `causal_non_leakage = 0.000e0`

## Analytical Prefill-Compute & KV-Size Ablation
| Context | Dense 49L KV (GiB) | HySparse2 KV (GiB) | KV Ratio | Prefill Ratio | Dense Acc | HySparse2 Acc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | `0.77` | `0.17` | `4.45x` | `2.01x` | `0.92` | `0.91` |
| 131072 | `12.25` | `2.75` | `4.45x` | `2.67x` | `0.89` | `0.87` |
| 524288 | `49.00` | `11.00` | `4.45x` | `4.05x` | `0.85` | `0.83` |
| 1048576 | `98.00` | `22.00` | `4.45x` | `5.04x` | `0.82` | `0.79` |

## Negative Controls (8 / 8 rejected)
- `nc1_missing_public_mimo_hysparse2_weights_blocked` -> `blocked_on_artifacts` (live HySparse2 evaluation stays blocked until public weights/code appear)
- `nc2_bidirectional_encoder_future_leak_detected` -> `invalid` (non-causal encoder leaks future token into position 0 and is rejected)
- `nc3_encoder_exceeds_total_layers_rejected` -> `invalid` (encoder_layers > dense_layers rejected cleanly)
- `nc4_zero_context_length_rejected` -> `invalid` (zero context length rejected before KV/FLOP ratio division)
- `nc5_zero_sparse_fraction_rejected` -> `invalid` (sparse_top_frac = 0.0 rejected)
- `nc6_nan_sparse_fraction_rejected` -> `invalid` (NaN sparse attention fraction rejected)
- `nc7_empty_prompt_rejected` -> `invalid` (empty prompt sequence refused by cached decoder)
- `nc8_unverified_5x_prefill_4_5x_kv_claims_non_gating` -> `invalid` (claimed 5x prefill and 4.5x KV ratios stay unverified related work)
