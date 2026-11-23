# Issue #49: Harness-Specific LoRA vs Base, Prompt-Only, and Harness-Only Arms

- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`)
- Artifact gate: `blocked_on_artifacts` (harness-hub/Qwen3.8-27B-pi-adapted LoRA weights, base Qwen3.8-27B checkpoint, and adapter training-set manifest)
- Benchmark overlap audit: `1` / `5` tasks flagged contaminated and excluded from clean scoring

## 4-Arm Clean Held-Out Results (Tool Costs Included)
- `unadapted_base`: `accepted=1/4`, `tool_calls=38`, `total_cost_usd=0.1180`, `cost_per_accepted=Some(0.11800000000000001)`, `self_report_disagreements=1`
- `prompt_only_change`: `accepted=2/4`, `tool_calls=28`, `total_cost_usd=0.1020`, `cost_per_accepted=Some(0.051000000000000004)`, `self_report_disagreements=1`
- `harness_only_change`: `accepted=3/4`, `tool_calls=19`, `total_cost_usd=0.0820`, `cost_per_accepted=Some(0.027333333333333334)`, `self_report_disagreements=0`
- `harness_specific_lora`: `accepted=3/4`, `tool_calls=15`, `total_cost_usd=0.0740`, `cost_per_accepted=Some(0.024666666666666667)`, `self_report_disagreements=0`

## Negative Controls (8 / 8 rejected)
- `nc1_training_overlap_task_excluded` -> `invalid` (task overlapping adapter training fingerprint is rejected from clean held-out scoring)
- `nc2_zero_accepted_cost_is_none` -> `invalid` (zero accepted tasks returns None cost_per_accepted_task (zero divisor guarded))
- `nc3_omitting_tool_cost_rejected` -> `invalid` (total cost accounting must include tool-call costs alongside model tokens)
- `nc4_false_self_report_overridden` -> `invalid` (self_reported_pass=true is overridden to invalid when independent checker fails)
- `nc5_mismatched_task_population_rejected` -> `invalid` (evaluating an arm on a subset of the frozen task set is rejected)
- `nc6_nan_cost_rejected` -> `invalid` (non-finite model or tool cost is rejected rather than swallowed)
- `nc7_post_tool_reduction_claim_not_gate` -> `invalid` (authors' 55-62% tool-call reduction claim is unverified related work, not a gate)
- `nc8_unpinned_adapter_tip_rejected` -> `invalid` (floating adapter tip 'main' is refused; pinned revision required)
