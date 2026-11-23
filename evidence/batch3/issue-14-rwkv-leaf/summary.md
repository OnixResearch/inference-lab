# Issue #14: Conditional Native RWKV Numerical Leaf Gate

- Execution class: `blocked_on_artifacts`
- Conditional gate: `blocked_on_artifacts` (`second_concrete_native_rwkv_consumer_and_pinned_leaf_crate`)
- Reusable workspace reallocations across 6 steps: `0`
- Byte-prefix trie roundtrip exact: `true` (generic whitespace diverges: `true`)

## Trigger Gate Verdicts
- `01-current-state-deferred-single-rsi-consumer`: satisfied=`false` (paper_planned_benchmark_or_mock_cannot_satisfy_second_consumer_trigger)
- `02-negative-second-mock-adapter-rejected`: satisfied=`false` (paper_planned_benchmark_or_mock_cannot_satisfy_second_consumer_trigger)
- `03-positive-contract-two-concrete-consumers-satisfy-trigger`: satisfied=`true` (passed)

## Safetensors Admission & RSI Wrapper Controls
- `01-positive-valid-safetensors-admission`: `accepted` — admitted
- `02-negative-malformed-empty-header`: `invalid` — malformed_or_oversized_safetensors_header
- `03-negative-shape-byte-span-mismatch`: `invalid` — tensor_shape_and_byte_span_mismatch
- `04-negative-overlapping-tensor-ranges`: `invalid` — overlapping_tensor_byte_ranges
- `05-negative-tampered-weight-digest`: `invalid` — pinned_weight_digest_mismatch
- `06-negative-missing-immutability-safety-contract`: `invalid` — mapped_file_immutability_safety_contract_missing
- `07-control-rsi-wrapper-per-agent-isolation-and-offset-refusal`: `accepted` — Agent epochs/offsets, immutable snapshots, fork/restore, and staged atomic rollback stay owned by RSI rwkv7_backend.rs.
- `08-negative-silent-checkpoint-version-bump-rejected`: `invalid` — Behavioral extraction cannot silently alter checkpoint bytes; any version change requires explicit declaration.
- `09-negative-fp32-cpu-claimed-equivalent-to-fla-bf16-cuda-rejected`: `invalid` — FP32 CPU architectural behavior is explicitly non-interchangeable with FLA BF16 CUDA policy parity.

> Explicitly `blocked_on_artifacts` (deferred) until a second concrete non-mock native-RWKV consumer demonstrates the leaf API.
