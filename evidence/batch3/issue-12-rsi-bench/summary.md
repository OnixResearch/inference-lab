# Issue #12: Standalone RSI Inference Benchmark Consolidation

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`cuda_gpu_torch_fla_linquant_runtime`)
- Donor pin: `OnixResearch/rsi-lab@01e47e268d71bc1c53cc36e11c179f2b8f22f5c5`
- Fake-quantization max abs error: `0.000238417851` (`implies_int8_speedup=false`)

## RSI `#82` Sweep Qualification Verdicts
- `01-positive-repaired-rsi82-contract`: qualified=`true` (`accepted` — passed)
- `02-negative-legacy-unfixed-rng-in-timed-region`: qualified=`false` (`invalid` — inputs_and_gates_not_precomputed_equivalently)
- `03-negative-altered-final-state-fails-rsi82`: qualified=`false` (`invalid` — final_recurrent_state_parity_exceeds_tolerance)
- `04-negative-overwriting-superseded-evidence-rejected`: qualified=`false` (`invalid` — historical_broken_sweep_overwritten_instead_of_superseded)

## Consolidation & Recurrence Controls
| Control | Category | Outcome | Max Abs Err |
| --- | --- | --- | ---: |
| `01-positive-rwkv7-chunked-output-and-state-parity` | `rwkv7_invariances` | `accepted` | 0.0 |
| `02-negative-altered-transform-fails-parity` | `rwkv7_invariances` | `invalid` | 0.029921370327 |
| `03-negative-perturbed-final-state-fails-parity` | `rwkv7_invariances` | `invalid` | 0.25 |
| `04-negative-encoder-compile-capture-conflation-rejected` | `encoder_bench` | `invalid` | null |
| `05-negative-fake-quant-claimed-as-int8-speedup-rejected` | `quantization_probe` | `invalid` | 0.000238417851 |
| `06-negative-overwrite-closed-rsi18-receipt-rejected` | `provenance_immutability` | `invalid` | null |
| `07-negative-private-trace-or-model-asset-leak-rejected` | `public_publication_boundary` | `invalid` | null |
| `08-control-missing-gpu-or-fla-stays-unavailable` | `optional_runtime_boundary` | `unavailable_hardware_runtime` | null |

> Pure-Rust CPU recurrence and contract checks executed locally. Optional GPU Torch/FLA/Transformers/LinQuant sweeps remain `unavailable_hardware_runtime`; RSI `#11` and stateful/decision owners stay in RSI Lab.
