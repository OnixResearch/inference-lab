# Issue #17: Discrete Graphical Sampling Benchmark & Exact Oracle Controls

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`physical_accelerator_discrete_graphical_sampler`)
- Exact oracle (`16` states): `Z=374.53892448`, `E[energy]=-3.797167707182`, conditional max err `0.0`
- Matched schedule+RNG bit-identical trajectory (serial vs color-phase): `true`
- Scoped decision: `INCONCLUSIVE_FOR_ACCELERATOR_GO_FOR_CPU_EXACT_ORACLE_CONTRACT`

## CPU Samplers & Adversarial Controls
| Variant | Valid Sched | Max Marginal Err | Energy Err | ESS (`ess-ips-v1`) | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| `sequential_reference` | `true` | `0.022232734751` | `0.019961706954` | `778.944972752274` | `accepted` |
| `serial_color_schedule` | `true` | `0.008233588326` | `0.008487993207` | `848.581918414125` | `accepted` |
| `color_phase_parallel_model` | `true` | `0.008233588326` | `0.008487993207` | `848.581918414125` | `accepted` |
| `faulty_stale_neighbor_cache` | `true` | `0.045405789294` | `0.069711843205` | `1019.560857180406` | `invalid` |
| `faulty_invalid_one_coloring` | `false` | `0.003706380763` | `1.192829586319` | `270.650911142674` | `invalid` |
| `faulty_omitted_commit` | `true` | `0.155096489799` | `0.796230281249` | `null` | `invalid` |
| `faulty_swapped_pairwise_index` | `true` | `0.105780083549` | `0.040318653785` | `698.829103218863` | `invalid` |
| `faulty_reused_rng_stream` | `true` | `0.118963677299` | `0.552272161571` | `1024.0` | `invalid` |

> AMARA (`arXiv:2611.15167v1`) and default-off CRAM (`arXiv:2611.17309`) claims remain strictly non-gating; stuck or biased samplers never win on raw updates/s.
