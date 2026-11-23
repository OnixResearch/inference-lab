# Issue #36: Compact Hybrid Recurrent Language Models Comparison Profile

- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (DGX Spark / RTX 5080 / Strix Halo serving of Granite 4.0 H-Tiny, Jamba2 3B, Falcon H1 3B)
- Zero-accepted cost undefined (`None`): `true`

## Candidate Profiles
- `ibm-granite-4.0-h-tiny-spark-v1` (`IBM Granite 4.0 H-Tiny`): `arch = hybrid_mamba_attention`, `visibility = opaque_endpoint`, `components = 5`, `ops = 2`
- `ai21-jamba2-3b-spark-v1` (`AI21 Jamba2 3B`): `arch = hybrid_mamba_attention`, `visibility = opaque_endpoint`, `components = 4`, `ops = 2`
- `tii-falcon-h1-3b-spark-v1` (`TII Falcon H1 3B`): `arch = hybrid_mamba_attention`, `visibility = opaque_endpoint`, `components = 4`, `ops = 2`

## Negative Controls (9 / 9 rejected)
- `nc1_opaque_endpoint_claims_serialize` -> `invalid` (opaque_endpoint refused from claiming serialize/fork support from theory)
- `nc2_dev_heldout_source_leakage` -> `invalid` (shared logical_source_id across dev and held_out rejected)
- `nc3_unvalidated_tool_call_not_accepted` -> `invalid` (model-emitted tool call without consumer validation yields 0 accepted tasks)
- `nc4_trace_replay_not_quality_pass` -> `invalid` (serving-mechanics trace replay cannot pass quality decision rule)
- `nc5_weight_file_as_resident_memory` -> `invalid` (reporting weight file size as resident memory rejected)
- `nc6_unified_memory_overlap_summed` -> `invalid` (summing overlapping CPU/GPU allocations on unified-memory host rejected)
- `nc7_unmatched_profile_claims_arch_speedup` -> `invalid` (unmatched quant/engine supports complete-profile comparison only)
- `nc8_nonfinite_wall_time_rejected` -> `invalid` (NaN wall_time_ms fails explicit finiteness check)
- `nc9_noninferior_without_cost_improvement_nogo` -> `invalid` (noninferior rate without material cost/density improvement yields NO-GO)
