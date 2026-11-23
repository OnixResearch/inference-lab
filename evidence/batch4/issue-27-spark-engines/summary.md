# Issue #27: Hyperflare & TensorRT Edge-LLM DGX Spark Qualification

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`dgx_spark_gb10_aarch64_sm121_hyperflare_and_trt_edge_llm`)
- Decision: `INCONCLUSIVE_SINGLE_SPARK_PHYSICAL_QUALIFICATION_REQUIRED`

## Hybrid KV + Recurrent Continuation Parity
- `append_resume_and_tool_pauses`: logit_err=`0.000e0`, recurrent_err=`0.000e0`, passed=`true`
- `sibling_branch_isolation`: logit_err=`0.000e0`, recurrent_err=`0.000e0`, passed=`true`
- `rollback_and_continue`: logit_err=`0.000e0`, recurrent_err=`0.000e0`, passed=`true`
- `cancellation_state_integrity`: logit_err=`0.000e0`, recurrent_err=`0.000e0`, passed=`true`

## Negative controls
- `incompatible_runtime_core_digest` -> `invalid` (incompatible model or runtime-core identity)
- `stale_recurrent_state_after_kv_rollback` -> `invalid` (stale_recurrent_logit_err=0.766043)
- `sibling_branch_contamination` -> `invalid` (branch_contamination_err=0.740199)
- `context_capacity_exhaustion` -> `exhausted` (context capacity exhausted)
- `unsupported_hyperflare_adapter_and_edge_tp2_context_reuse` -> `unsupported` (hyperflare=(Hyperflare core-design.md excludes adapters); edge_tp2=(Edge-LLM multi-device.md excludes context reuse when TP>1))
- `incomplete_output_missing_finish_reason` -> `invalid` (incomplete output: missing finish_reason remains unknown)
- `three_rank_baseline_not_relabeled_as_tp2` -> `invalid` (observed 3 ranks cannot be relabeled as Edge-LLM TP2)
- `host_rdma_does_not_qualify_gpu_nccl_collectives` -> `unsupported` (host-memory RDMA alone does not qualify GPU NCCL collectives (requires Spark #7 receipt))
- `independent_end_task_checker_rejects_wrong_answer` -> `invalid` (wrong end-task answer rejected independently of replay trace mechanics)
