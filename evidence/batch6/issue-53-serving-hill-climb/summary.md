# Issue #53: Ceiling-First Interleaved-Control Serving Hill-Climb

- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`, `default_enabled: false`)
- Hardware gate: `unavailable_hardware_runtime` (Admitted local LLM server host with OpenAI-compatible engine and pinned checkpoint)
- Pinned skill: `sybil-solutions/inference-optimisation-skill@fda968e7fc1996aa23c67139c755fe541c9d4be0` (`MIT`; authors' B390/B70/3090/H200/2.1x figures are non-gating)
- Pre-search ceilings: decode `practical=56.67 tok/s` (`theoretical=66.67 tok/s`), prefill `practical=9000.00 tok/s`, target feasibility `accepted`

## Hill-Climb Ledger (All Accepted & Rejected Runs Retained)
- `cand-01-fused-kv-page-layout` (`harness=inference-lab-issue7-serving-runner`, `thinking=false`, `warmup=true`): `ctrl_before=44.15`, `cand=47.95`, `ctrl_after=44.15`, `quality_trade=false`, `headline_eligible=true`, `outcome=accepted`
- `cand-02-aggressive-batch-coalesce` (`harness=inference-lab-issue7-serving-runner`, `thinking=false`, `warmup=true`): `ctrl_before=44.15`, `cand=49.2`, `ctrl_after=44.15`, `quality_trade=false`, `headline_eligible=true`, `outcome=invalid`
- `cand-03-lossy-prefill-quant` (`harness=inference-lab-issue7-serving-runner`, `thinking=false`, `warmup=true`): `ctrl_before=44.15`, `cand=47.95`, `ctrl_after=44.15`, `quality_trade=true`, `headline_eligible=false`, `outcome=accepted`

## Clean-Checkout Rebuild Recheck: `cand-01-fused-kv-page-layout` -> `accepted` (harness `inference-lab-issue7-clean-rebuild-runner`)

## Negative Controls (8 / 8 rejected)
- `nc1_target_above_practical_ceiling_refused` -> `unsupported` (requested target 71.7 tok/s exceeds practical ceiling 56.7 tok/s)
- `nc2_streamed_chunks_token_accounting_rejected` -> `invalid` (streamed SSE chunk counts rejected in favor of server usage / tokenizer)
- `nc3_unified_memory_missing_residency_rejected` -> `invalid` (unified-memory host requires residency measurement beyond cgroups)
- `nc4_beats_only_one_neighbor_control_rejected` -> `invalid` (candidate failing to beat control_after rejected as drift artifact)
- `nc5_secondary_metric_regression_over_3pct_rejected` -> `invalid` (secondary metric regressed 6.043956043956% (> 3.0% bound))
- `nc6_fast_looping_or_empty_soak_run_rejected` -> `invalid` (suspiciously fast looping/empty concurrent KV soak run rejected)
- `nc7_clean_rebuild_disappearing_win_rejected` -> `invalid` (candidate win that disappears on clean pinned rebuild is not accepted)
- `nc8_importing_b390_b70_3090_h200_2_1x_claims_rejected` -> `invalid` (authors' B390/B70/3090/H200/2.1x figures cannot be imported as pass/fail gates)
