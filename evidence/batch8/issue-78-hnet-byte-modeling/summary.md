# Issue #78: H-Net v2 Byte-Level Hierarchical Modeling & Tokenizer-Free Edge Cases

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Artifact gate: `blocked_on_artifacts` (H-Net v2 public byte-level model weights & runtime vs BPE baseline)
- Mean bytes per dynamic chunk: `1.734`

## Byte-Level Dynamic Chunking & Parity on Edge Cases
| Fixture | Bytes | Chunks | Bytes/Chunk | Exact Roundtrip | Upsample Err | Raw Byte/BPE Step Ratio |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `utf8_multibyte_cjk_latin` | 28 | 14 | `2.00` | `true` | `0.000e0` | `2.80x` |
| `emoji_zwj_sequence` | 40 | 16 | `2.50` | `true` | `0.000e0` | `3.64x` |
| `python_indentation_whitespace` | 30 | 23 | `1.30` | `true` | `0.000e0` | `2.50x` |
| `multi_digit_arithmetic` | 28 | 22 | `1.27` | `true` | `0.000e0` | `2.55x` |
| `code_punctuation_glitch` | 49 | 26 | `1.88` | `true` | `0.000e0` | `3.06x` |
| `role_injection_delimiter` | 47 | 27 | `1.74` | `true` | `0.000e0` | `3.36x` |

## Negative Controls (8 / 8 rejected)
- `nc1_missing_public_hnet_v2_weights_blocked` -> `blocked_on_artifacts` (H-Net v2 model execution remains blocked until public weights and code are pinned)
- `nc2_invalid_utf8_stream_rejected` -> `invalid` (malformed non-UTF-8 byte sequence fails closed)
- `nc3_empty_byte_input_rejected` -> `invalid` (empty byte input refused instead of dividing by zero chunks)
- `nc4_nan_boundary_threshold_rejected` -> `invalid` (NaN routing boundary threshold rejected)
- `nc5_oob_boundary_threshold_rejected` -> `invalid` (cosine boundary threshold outside [0, 2] rejected)
- `nc6_corrupted_chunk_byte_fails_roundtrip` -> `invalid` (single-bit corruption in chunk payload detected by byte round-trip check)
- `nc7_truncated_chunk_coverage_rejected` -> `invalid` (chunk sequence failing to cover full input byte span rejected)
- `nc8_unverified_scaling_claim_non_gating` -> `invalid` (H-Net v2 3-5x / 10x efficiency claims stay unverified related work)
