# Issue #74: PDL Kernel Overlap & Breakable Prefill CUDA Graphs

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (NVIDIA Blackwell / DGX Spark GPU running SGLang flash-MoE prefill with PDL & CUDA graphs)
- Related work: Baseten DeepSeek-V4.1-Flash write-up (`post_only_article_no_public_repo`, unverified non-gating)

## Toggle Matrix & Correlated Host/GPU Trace Summary
| Arm | Max Abs Err | Parity | GPU Busy (us) | Gap Bubble (us) | PDL Saved (us) | Graph Breaks | TTFT (us) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `standard_launches` | `0.000e0` | `true` | `204.00` | `52.50` | `0.00` | 0 | `259.50` |
| `pdl_overlap_only` | `0.000e0` | `true` | `204.00` | `24.50` | `28.00` | 0 | `231.50` |
| `breakable_prefill_graph_only` | `0.000e0` | `true` | `204.00` | `6.10` | `0.00` | 1 | `213.10` |
| `pdl_plus_breakable_prefill_graph` | `0.000e0` | `true` | `204.00` | `0.00` | `28.00` | 1 | `207.00` |

## Negative Controls (8 / 8 rejected)
- `nc1_uncalibrated_clock_domain_rejected` -> `invalid` (cross-domain timestamp subtraction refused without calibrated clock mapping)
- `nc2_dropped_trace_events_refused` -> `invalid` (incomplete host/GPU trace with dropped events refused)
- `nc3_causality_violation_rejected` -> `invalid` (GPU kernel start earlier than host enqueue beyond uncertainty rejected)
- `nc4_impossible_pdl_overlap_exceeds_kernel_duration` -> `invalid` (PDL overlap claiming more saved time than kernel duration rejected)
- `nc5_nonfinite_trace_timestamp_rejected` -> `invalid` (NaN GPU trace timestamp rejected cleanly)
- `nc6_zero_graph_chunk_size_rejected` -> `invalid` (breakable prefill CUDA graph with zero chunk size rejected)
- `nc7_numerical_drift_disqualifies_pdl_speedup` -> `invalid` (faster PDL/graph candidate with 1e-3 numerical drift is disqualified)
- `nc8_pre_hopper_sm80_pdl_unsupported` -> `unsupported` (PDL overlap on pre-Hopper/Blackwell architecture returns unsupported)
