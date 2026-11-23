# Issue #16: `ggml` RWKV-7 Recurrence & State-Continuation Parity

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`compiled_ggml_cpu_and_cuda_wkv7_kernels`)
- State layout: `row_major_[n_heads,head_dim,head_dim]` (`f32=128 B`, `f64=256 B`)
- Exported t=24 state SHA-256: `df336ef583f90f8bed34761764522fd260c41e88ad172f1a4c85e065551a0ba7`
- Asserts RSI agent-level persistence correctness: `false`

## Long-Horizon Drift Checkpoints
| Step | Max Output Abs Err | Max State Abs Err |
| ---: | ---: | ---: |
| 1 | `0.000000001009` | `0.00000000163` |
| 8 | `0.000000005164` | `0.000000006715` |
| 16 | `0.000000005522` | `0.000000007654` |
| 32 | `0.000000007182` | `0.000000008776` |
| 64 | `0.000000007182` | `0.000000006394` |

## Parity & Negative Controls
- `01-positive-token-and-full-parity-f32-vs-f64-ref`: `accepted` (err=`0.000000007182`) — Independent FP32 ggml-style wkv7 matches FP64 native reference across 64 steps.
- `02-positive-export-import-state-continuation-24-plus-40`: `accepted` (err=`0.000000007182`) — Binary LE export/import at t=24 reproduces uninterrupted 64-step outputs and final state.
- `03-negative-altered-exported-state-detected`: `invalid` (err=`0.11730210946`) — Perturbing exported state[0] before continuation breaks final-state parity.
- `04-negative-sequence-boundary-duplicate-token-detected`: `invalid` (err=`0.041397475491`) — Off-by-one chunk boundary duplication at t=23..64 fails state parity.
- `05-negative-bf16-precision-drift-exceeds-f32-tolerance`: `invalid` (err=`0.000329214335`) — BF16-truncated recurrence drifts beyond 1e-5 F32 tolerance over 64 steps.
- `06-negative-backward-training-claim-unsupported`: `unsupported` (err=`null`) — ggml wkv7 inference qualification explicitly rejects unsupported backward/training claims.
- `07-control-ggml-cuda-wkv-unavailable-no-cpu-fallback`: `unavailable_hardware_runtime` (err=`null`) — Physical ggml-cuda wkv.cu is unavailable locally and never falls back silently to CPU.

> Kernel numerical state parity never asserts durable RSI agent epochs, snapshots, or policy commits.
