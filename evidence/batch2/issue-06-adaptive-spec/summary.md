# Issue #6: Adaptive Speculative Graph & Concurrency Characterization

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`cuda_graph_capture_replay_and_concurrent_gpu_serving`)
- Decision: `INCONCLUSIVE_UNAVAILABLE_HARDWARE_RUNTIME`

| Condition | Policy | Concurrency | Active/Replica | Outcome | Violation |
| --- | --- | ---: | ---: | --- | --- |
| `fixed-block-4-single-stream` | `fixed_block_4` | 1 | 1 | `accepted` | none |
| `adaptive-frozen-c4-cpu-overlap-labeled` | `adaptive_1..6_thr_0.800` | 4 | 1 | `accepted` | none |
| `negative-cpu-overlap-mislabeled-continuous-batching` | `adaptive_1..6_thr_0.800` | 4 | 1 | `invalid` | cpu_preparation_overlap_mislabeled_as_continuous_batching |
| `negative-sm100a-b300-extrapolation` | `adaptive_1..6_thr_0.800` | 1 | 1 | `unsupported` | unsupported_arch_extrapolation:requested=b300_sm100a_sm103a allowed=gb10_sm121_contract |

> Reuses Issue #3 verification primitives and Issue #7 workload identity. No CUDA graphs or GPU serving executed locally.
