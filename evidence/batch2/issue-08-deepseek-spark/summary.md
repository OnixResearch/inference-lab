# Issue #8: DeepSeek/Spark Migration into Shared Serving Benchmark

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`dgx_spark_gb10_deepseek_endpoint`)
- Burst vs sustained cross-comparison rejected: `true`

## Device-Free Fixtures (9/9)
| Fixture | Handled Honestly | Outcome |
| --- | --- | --- |
| `01-multi-token-sse-chunks` | `true` | `accepted` |
| `02-missing-usage-refuses-throughput` | `true` | `invalid` |
| `03-missing-spec-counters-unavailable` | `true` | `unsupported` |
| `04-multiple-metric-label-series` | `true` | `invalid` |
| `05-counter-reset-detected` | `true` | `invalid` |
| `06-empty-or-truncated-output` | `true` | `invalid` |
| `07-malformed-sse-stream` | `true` | `internal_failure` |
| `08-unsupported-tokenize-or-ignore-eos` | `true` | `unsupported` |
| `09-delayed-and-failed-requests-retained` | `true` | `timeout` |

## Legacy Substring vs Independent Executable Semantic Quality Check
- `cand-correct-clamp-alt-expr`: legacy substring=`true` vs semantic=`true` (failed inputs: `[]`)
- `cand-wrong-abs-passes-legacy-substring`: legacy substring=`true` vs semantic=`false` (failed inputs: `[-5, -1]`)

> Device-free contract verification only. Private donor logs/weights and deployment scripts remain excluded; no live Spark endpoint ran locally.
