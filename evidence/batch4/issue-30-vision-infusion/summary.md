# Issue #30: Separate Vision Encoders vs Early In-Fusion at Matched Capacity

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`physical_gpu_eager_and_compiled_multimodal_serving`)
- Quality status: `quality_unestablished`
- Principal arms visual budget matched (`1024`): `true`

| Architecture | Total Params | Vision Params | FLOPs (img4+txt4) | FLOPs (img16+txt4) | Packed vs Single Err | Text-Only Inv Err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `encoder_free` | 2048 | 0 | 36864 | 107520 | `0.000e0` | `0.000e0` |
| `separate_encoder` | 3072 | 1024 | 45568 | 148480 | `0.000e0` | `0.000e0` |
| `early_ffn_infusion` | 3072 | 1024 | 45056 | 140288 | `0.000e0` | `0.000e0` |
| `early_full_infusion` | 3072 | 1024 | 45056 | 140288 | `0.000e0` | `0.000e0` |

## Negative controls
- `misrouted_text_token_to_vision_branch` -> `invalid` (text_output_drift=1.511626)
- `altered_causal_attention_mask` -> `invalid` (mask_drift=0.301776)
- `cross_request_mixing_in_packed_batch` -> `invalid` (text_req_contamination_err=0.557293)
- `gather_scatter_token_permutation_bug` -> `invalid` (gather_scatter_err=1.490423)
- `empty_declared_visual_input_rejected` -> `invalid` (empty visual input rejected when image is declared)
- `unmatched_visual_budget_encoder_free_separated` -> `invalid` (unmatched visual parameter budget: `separate_encoder` has 1024 vs `encoder_free` has 0)
- `untrained_weights_cannot_establish_quality` -> `blocked_on_artifacts` (quality_unestablished: compatible trained variants and training-exposure records unavailable)
