# Issue #45: Reproducible Model Intervention & Replay Harness

- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (Pinned local Qwen transformer serve on admitted host)
- Matched L2 norm (`inject` vs `random_control`): `1.913113` vs `1.913113`

## Paired Baseline vs Intervention & Random-Direction Runs
- `exp-scale:scale`: `KL=0.000203`, `L1=0.016052`, `top1=6->6`, `task_pass=true->true`, `bit_identical_replay=true`
- `exp-ablate:ablate`: `KL=0.000127`, `L1=0.012810`, `top1=6->6`, `task_pass=true->true`, `bit_identical_replay=true`
- `exp-inject-targeted:inject`: `KL=0.001585`, `L1=0.044277`, `top1=6->6`, `task_pass=true->true`, `bit_identical_replay=true`
- `exp-inject-random-direction-control:inject`: `KL=0.002632`, `L1=0.062407`, `top1=6->6`, `task_pass=true->true`, `bit_identical_replay=true`

## Negative Controls (8 / 8 rejected)
- `nc1_cross_arch_neuron_id_transfer_rejected` -> `invalid` (Qwen neuron IDs and layer indices refused on recurrent architecture)
- `nc2_looped_backend_future_only_not_first_slice` -> `unsupported` (Goodfire/NDIF looped-model backend stays deferred related work)
- `nc3_mutable_model_revision_rejected` -> `invalid` (floating `latest` model revision fails immutable manifest check)
- `nc4_nonfinite_inject_vector_rejected` -> `invalid` (NaN activation injection rejected cleanly)
- `nc5_oob_token_position_rejected` -> `invalid` (out-of-bounds target_token_pos rejected)
- `nc6_oob_layer_index_rejected` -> `invalid` (out-of-bounds target_layer rejected)
- `nc7_unmatched_random_direction_norm` -> `invalid` (random-direction control with 2x mismatched L2 norm detected)
- `nc8_tampered_replay_digest_detected` -> `invalid` (re-executing spec against tampered replay digest fails provenance receipt)
