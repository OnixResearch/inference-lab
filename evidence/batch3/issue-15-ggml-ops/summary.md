# Issue #15: Pinned `ggml` Backend-Ops Qualification Harness

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`pinned_ggml_test_backend_ops_accelerator_binary`)
- Pinned `ggml` commit: `417a024b4fb6bc3116d41af7c067166a039bdf12`
- Reference block-32 dot (`F32`): `10.237247889145` (`Q8_0` abs err: `0.003216211898`, `Q4_0` abs err: `0.35677856694`)
- Grants placement authority: `false` (Kiln `#60` retains scheduling)

## 4-Axis Qualification Verdicts
| Case | Dtype | Req->Act | Adv | Exec | Num | Perf | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `01-positive-cpu-ref-mul-mat-f32` | `F32` | `CPU->CPU` | `true` | `true` | `true` | `false` | `accepted` |
| `02-positive-cpu-ref-mul-mat-q8_0` | `Q8_0` | `CPU->CPU` | `true` | `true` | `true` | `false` | `accepted` |
| `03-positive-cpu-ref-mul-mat-q4_0` | `Q4_0` | `CPU->CPU` | `true` | `true` | `true` | `false` | `accepted` |
| `04-control-unsupported-mxfp4-explicit` | `MXFP4` | `CPU->CPU` | `false` | `false` | `false` | `false` | `unsupported` |
| `05-control-unsupported-nvfp4-explicit` | `NVFP4` | `CUDA0->none` | `false` | `false` | `false` | `false` | `unsupported` |
| `06-negative-silent-cpu-fallback-disqualified` | `Q8_0` | `CUDA0->CPU` | `true` | `true` | `true` | `false` | `invalid` |
| `07-negative-numerical-mismatch-disqualified` | `F32` | `CUDA0->CUDA0` | `true` | `true` | `false` | `false` | `invalid` |
| `08-negative-nonfinite-nan-output-disqualified` | `Q4_0` | `CUDA0->CUDA0` | `true` | `true` | `false` | `false` | `invalid` |
| `09-negative-advertised-supported-but-exec-failed` | `Q8_0` | `CUDA0->CUDA0` | `true` | `false` | `false` | `false` | `internal_failure` |
| `10-control-explicit-skip-distinct-from-pass-or-fail` | `Q4_0` | `CPU->CPU` | `true` | `false` | `false` | `false` | `unsupported` |

> Pure-Rust quantized reference kernels and qualification controls executed locally; no fabricated accelerator performance or compiled `test-backend-ops` GPU run is claimed.
