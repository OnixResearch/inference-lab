# Issue #41 — Versioned transformer stage contract report

- **Execution class**: `bounded_contract_harness`
- **Model**: `frozen-dense-4l-h8-v16-v1` (`sha256:79098833a3fedb0f`)
- **Apple M2 Ultra / Tenstorrent P150**: `unavailable_hardware_runtime` (zero silent CPU fallback)

## Split vs unsplit numerical parity (reference `f64`)

| Placement | Mode | Max activation err | Max logit err | Passed | Matches reference tokens |
| --- | --- | --- | --- | --- | --- |
| `split-2stage-0..2-2..4` | `full_prefill_plus_forced_decode` | 0.0 | 0.0 | `true` | `true` |
| `split-2stage-0..2-2..4` | `chunked_prefill_2+2_plus_forced_decode` | 0.0 | 0.0 | `true` | `true` |
| `split-3stage-0..1-1..3-3..4` | `full_prefill_plus_forced_decode` | 0.0 | 0.0 | `true` | `true` |
| `split-2stage-reset-reuse` | `populated_kv_gen1_reset_then_gen2_full_prefill_plus_decode` | 0.0 | 0.0 | `true` | `true` |

## Adversarial negative controls

| Control | Rejected | Outcome | Reason |
| --- | --- | --- | --- |
| `neg-overlapping-layer-interval` | `true` | `invalid` | `non_contiguous_or_overlapping_layer_interval:expected=2:got=1` |
| `neg-multiple-sampling-owners` | `true` | `invalid` | `invalid_single_owner_contract:embed=1:logits=1:sampler=2` |
| `neg-checkpoint-digest-mismatch` | `true` | `invalid` | `checkpoint_or_effective_weight_digest_mismatch` |
| `neg-dtype-mismatch` | `true` | `invalid` | `dtype_or_byte_order_mismatch` |
| `neg-stale-request-generation` | `true` | `invalid` | `stale_request_generation:expected=1:got=99` |
| `neg-wrong-token-position` | `true` | `invalid` | `wrong_token_position:expected=0:got=3` |
| `neg-truncated-buffer` | `true` | `invalid` | `truncated_tensor_buffer` |
| `neg-incompatible-stride-layout` | `true` | `invalid` | `incompatible_stride_layout` |
| `neg-payload-digest-integrity-mismatch` | `true` | `invalid` | `payload_digest_integrity_mismatch` |
| `neg-timeout-poisoned-buffer-and-stale-kv` | `true` | `invalid` | `poisoned_buffer_cannot_be_reused_after_timeout_or_cancel` |
| `neg-no-silent-cpu-fallback-for-apple-tt-hardware` | `true` | `unavailable_hardware_runtime` | `physical_backend_unavailable_no_silent_cpu_fallback:apple_gpu_metal_mlx` |

A split CPU reference establishes contract and numerical equivalence only; it does not qualify physical Apple M2 Ultra or Tenstorrent P150 execution.
