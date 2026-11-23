# Issue #57: Default-Off Local Serve Qualification for Pinned Inkling 2 Mini

- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`, `default_enabled: false`)
- Hardware gate: `unavailable_hardware_runtime` (Admitted host (DGX Spark / >=32GB Apple Silicon / >=32GB VRAM GPU) with pinned Inkling 2 Mini checkpoint and an authors-listed runtime)
- Topology: `24 stored layers` (`20 NVFP4`, `4 MXFP8`), `48 -> 48 applications` (6 entry + 12x3 middle + 6 exit), `8-of-128 MoE`
- Unrolled 48-layer reference vs 3x shared-loop parity: `max_abs_err = 0.000e0` (`accepted`)

## Optional Research Varloop Sweep (`TML-d24-varloop-research`, research-only)
- `loop_count=1`: `layer_apps=24`, `active_experts=192`, `l2_norm=1.105050`, `max_diff_vs_3x=1.492e-1`
- `loop_count=2`: `layer_apps=36`, `active_experts=288`, `l2_norm=1.067771`, `max_diff_vs_3x=7.536e-2`
- `loop_count=3`: `layer_apps=48`, `active_experts=384`, `l2_norm=1.042173`, `max_diff_vs_3x=0.000e0`
- `loop_count=4`: `layer_apps=60`, `active_experts=480`, `l2_norm=1.035742`, `max_diff_vs_3x=8.317e-2`

## Negative Controls (8 / 8 rejected)
- `nc1_gpu_below_32gb_floor_unsupported` -> `unsupported` (host rtx-3090-24gb below #57 floor (requires DGX Spark, >=32GB Apple Silicon, or >=32GB VRAM GPU))
- `nc2_apple_silicon_below_32gb_unsupported` -> `unsupported` (host mac-16gb below #57 floor (requires DGX Spark, >=32GB Apple Silicon, or >=32GB VRAM GPU))
- `nc3_maya_ik2_unlisted_for_issue_57` -> `unsupported` (runtime "maya-ik2" is not on Thinking Machines' runtime list for #57 (maya-ik2 belongs strictly to #59))
- `nc4_unpinned_hf_tip_rejected` -> `invalid` (checkpoint thinkingmachines/Inkling-2-Mini@main does not match pinned Inkling 2 Mini or varloop SHA)
- `nc5_missing_local_runtime_reports_unavailable` -> `unavailable_hardware_runtime` (host dgx-spark-offline meets floor and runtime sglang is admitted, but local runtime/weights are not present)
- `nc6_loop_count_mismatch_detected` -> `invalid` (loop_count=2 (36 apps) diverged 7.536e-2 from 48-app reference)
- `nc7_wrong_moe_topk_routing_detected` -> `invalid` (4-of-128 MoE routing diverged 4.720e-2 from 8-of-128 reference)
- `nc8_non_finite_activation_rejected` -> `invalid` (NaN activation caught before MoE routing)
