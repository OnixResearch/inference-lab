# Issue #7: Portable Serving Benchmark Extraction & Trace Profiles

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`live_http_sse_serving_endpoint_and_tt_p150x2`)
- Donor pin: `OnixResearch/tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a` (`LGPL-2.1-only`, schema `onix.qwen38-benchmark.v1`)

## Offline Reducer & Tamper Controls
| Fixture | Valid | Planned | Accepted | Failed/Unfinished | Reasons |
| --- | --- | ---: | ---: | ---: | --- |
| `positive-legacy-v1-replay` | `true` | 3 | 2 | 1 | none |
| `negative-tampered-summary-and-dropped-timeout` | `false` | 3 | 2 | 1 | tampered_accepted_count: observed=2 claimed=3, tampered_completion_tokens: observed=102 claimed=150, tampered_p95_call_mean_itl |
| `negative-missing-usage-refuses-throughput` | `false` | 3 | 2 | 1 | missing_usage_or_invalid_timing:call-01, tampered_p50_call_mean_itl |
| `negative-qwen18-task-pass-cannot-override-parity-fail` | `false` | 3 | 2 | 1 | task_pass_cannot_override_failed_greedy_parity |
| `negative-credential-and-url-leak` | `false` | 3 | 2 | 1 | credential_or_url_validation_failed:call-01 |

## Branch-Aware Agent Trace & OASYS RLM Controls
- Branch-trace controls evaluated: 7 (1 positive, 6 negative).
- Eunomia transport controls evaluated: 2 (`Connection: close` preserved, BPF kernel loop not justified).
- OASYS RLM trace controls evaluated: 7 (2 shapes: `externally_retained_results` vs `appended_child_results`, 5 negative controls).

> Device-free contract verification only. No live HTTP/SSE model server or Tenstorrent P150 hardware was invoked.
