# Issue #87: Knowledgator System 1 vs Clairic (#55) & Structured-Output LLM Extraction

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Hardware / artifact gate: `blocked_on_artifacts` (Pinned Knowledgator System 1 checkpoint alongside Clairic (#55) on local serving host)
- Fixture SHA-256: `977a1abd3c4a7582c1504d038947834732c250caa8d3bf503687967512d620f0` (`6` frozen hierarchical-extraction documents)

## Extraction Arm Metrics
- `knowledgator_system1_small_default_off`: `F1=0.944444444444`, `schema=6/6`, `hallucinations=0`, `docs/s=83.333333333333`, `outcome=accepted`
- `clairic_flash_9b_slot_ar`: `F1=1.0`, `schema=6/6`, `hallucinations=0`, `docs/s=22.222222222222`, `outcome=accepted`
- `conventional_json_schema_structured_output_baseline_v1`: `F1=1.0`, `schema=6/6`, `hallucinations=0`, `docs/s=5.555555555556`, `outcome=accepted`

## Negative Controls (8 / 8 rejected)
- `nc1_knowledgator_checkpoint_blocked_on_artifacts` -> `blocked_on_artifacts` (real Knowledgator System 1 checkpoint digest requires external pin before live serving)
- `nc2_floating_checkpoint_tag_rejected` -> `invalid` (floating `:latest` checkpoint tag rejected)
- `nc3_high_confidence_hallucination_disqualified` -> `invalid` (0.99 informational confidence never overrides ungrounded hallucinated field)
- `nc4_schema_invalid_doc_disqualified` -> `invalid` (schema-invalid output retains missed gold fields as FN and fails admission)
- `nc5_dropped_population_document_rejected` -> `invalid` (omitting a failed document from population denominator rejected)
- `nc6_clairic_preview_single_pass_rejected` -> `invalid` (outdated 19 Oct 2026 single-forward-pass Clairic preview path refused)
- `nc7_nonfinite_latency_rejected` -> `invalid` (NaN per-document latency rejected)
- `nc8_unverified_100x_claim_non_gating` -> `invalid` (authors' unverified ~100x speedup claim is strictly non-gating related work)
