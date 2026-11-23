# Issue #24: MACaQ-R Native MoE GGUF Refinement Contract

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`physical_node_qwen36_35b_macaq_r_llamacpp`)
- Decision: `INCONCLUSIVE_PHYSICAL_NODE_CREATION_AND_SERVE_REQUIRED`
- Synthetic K-quant MoE refinement (`Q4_K`/`Q5_K`/`Q6_K`, 3 tensors): calib KL `0.018246` -> `0.000653`; held-out KL `0.009427` -> `0.001496`
- Byte separation: tensor payload `94` vs `94` bytes (unchanged); total GGUF `606` vs `734` bytes (metadata receipt added)

## Negative controls
- `mutated_block_scale_rejected` -> `invalid` (block scale or activation basis mutated on `blk.0.ffn_gate_exps.weight` (MACaQ-R must preserve native scales/basis))
- `modified_out_of_scope_tensor_rejected` -> `invalid` (out-of-scope tensor `token_embd.weight` was modified)
- `unsupported_split_gguf_and_iq4xs_rejected` -> `unsupported` (split_err=true, iq4xs_err=true)
- `calibration_heldout_contamination_rejected` -> `invalid` (held-out contamination detected in calibration split)
- `stale_resume_receipt_rejected` -> `invalid` (stale or mismatched resume state identity rejected)
- `mismatched_serving_mtp_setting_rejected` -> `invalid` (MTP must stay off for matched serving arms (violation on t1))
- `creation_time_dependency_at_serve_rejected` -> `invalid` (refined artifact cannot require creation-time HF/teacher/calibration deps at serve time)
- `zero_accepted_tasks_cost_undefined` -> `invalid` (cost_per_accepted_task=null when accepted=0)
