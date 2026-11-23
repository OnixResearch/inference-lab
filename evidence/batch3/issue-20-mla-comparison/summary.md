# Issue #20: Explicit vs Absorbed MLA Parity & Memory Accounting

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`physical_gpu_jit_mla_latency_and_peak_vram`)
- Absorbed vs uncached max abs err: `0.0` (gradient FD diff: `0.0`)
- Lowered materialized per-head K/V expansion elems: absorbed=`0` (`absorbed_prep_bytes=192` B), explicit=`1472`
- Default-dim persistent payload: `144` vs `480` (`70.0%` element reduction; `is_measured_speedup=false`)
- Scoped decision: `GO_FOR_CPU_ABSORBED_MLA_PARITY_INCONCLUSIVE_FOR_GPU_SPEEDUP`

## Cache Prefix Strategy & Recompilation Accounting
| Strategy | Recompiles | Explicit KV Elems | Absorbed KV Elems | Explicit Peak B | Absorbed Peak B |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_allocated_cache` | 1 | 3072 | 0 | 4992 | 2304 |
| `valid_prefix_only` | 4 | 1472 | 0 | 4224 | 2304 |
| `bucketed_prefix` | 3 | 2048 | 0 | 4992 | 2304 |

## Parity & Sentinel Negative Controls
- `01-positive-absorbed-vs-explicit-vs-uncached-poisoned-cache`: `accepted` (err=`0.0`, first_div_pos=`None`) — Absorbed MLA matches explicit & uncached paths across batch=2, chunks [3,2,1,3], reset/reuse, and 1e6 sentinel-poisoned unused slots with 0 per-head K/V expansion intermediates.
- `02-negative-wrong-causal-mask-detected`: `invalid` (err=`0.119083031164`, first_div_pos=`Some(0)`) — Future token leakage from disabled causal mask caught at earliest position.
- `03-negative-wrong-rope-offset-detected`: `invalid` (err=`0.000955998726`, first_div_pos=`Some(1)`) — Shifted RoPE position offset caught immediately.
- `04-negative-wrong-attention-scale-detected`: `invalid` (err=`0.002947591826`, first_div_pos=`Some(1)`) — 2x attention scaling mismatch detected.
- `05-negative-wrong-absorbed-projection-layout-detected`: `invalid` (err=`0.001890499194`, first_div_pos=`Some(1)`) — Misaligned head projection in absorbed W_UK caught immediately.
- `06-negative-unused-1e6-sentinel-leak-detected`: `invalid` (err=`479978.318573297525`, first_div_pos=`Some(0)`) — Reading one unused 1e6 sentinel cache slot causes catastrophic output divergence.
- `07-negative-stale-cache-and-capacity-overflow-detected`: `invalid` (err=`0.067861964992`, first_div_pos=`Some(0)`) — Unreset stale cache diverges from clean sequence and capacity overflow is explicitly rejected.

> Local cache-layout comparison does not establish cross-replica transport or heterogeneous cache compatibility (`#5`).
