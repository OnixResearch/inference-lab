# Issue #55: Pinned Open-Weight Clairic Typed Extraction vs Structured-Output Baseline

- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`, `default_enabled: false`, `starts_issue_10_extraction: false`)
- Hardware gate: `unavailable_hardware_runtime` (PyTorch + Transformers + xgrammar GPU host for Cloudflare/clairic-flash (9B preferred) or Cloudflare/clairic (27B))
- Preferred variant: `Cloudflare/clairic-flash@36ad68b999560f56b57ef16a22180f8cb83c6d77` | Observed path: `parallel_slot_autoregressive_2026_11_18`
- Frozen fixture digest (`sha256`): `54df8b7c9f1edbfd09ce9677388f29215a7fb45fa93d18b91e48eb5264c141cf`

## Extraction Arms
- `arm_clairic_flash_slot_ar` (`parallel_slot_autoregressive_2026_11_18`): `schema_valid=8/8`, `value_agreed=8/8`, `spans_grounded=2/2`, `well_typed_wrong=0`, `steps=25`, `wall_ms_synth=184.5`, `outcome=accepted`
- `arm_structured_output_baseline` (`sequential_json_schema_autoregressive`): `schema_valid=8/8`, `value_agreed=8/8`, `spans_grounded=2/2`, `well_typed_wrong=0`, `steps=233`, `wall_ms_synth=933.4`, `outcome=accepted`

## Negative Controls (8 / 8 rejected)
- `nc1_preview_single_forward_pass_rejected` -> `unsupported` (outdated preview single-forward-pass architecture rejected in favor of 18 Nov parallel slot-AR)
- `nc2_well_typed_wrong_value_high_confidence` -> `invalid` (well-typed wrong value: got "499.99" vs expected "450.00" (confidence=0.999, p_null=0.000 are non-gating scores))
- `nc3_text_span_mismatch_ungrounded` -> `invalid` (span [0, 10) extracted "InvoiceRef", does not match returned "Northwind Optical Supply")
- `nc4_out_of_bounds_span_rejected` -> `invalid` (span [900, 920) out of bounds for doc char len 195)
- `nc5_schema_type_violation_rejected` -> `invalid` (failed to parse finite number from "twelve_units")
- `nc6_non_finite_confidence_rejected` -> `invalid` (non-finite or out-of-range p_null/confidence score)
- `nc7_unpinned_model_revision_rejected` -> `invalid` (floating branch ref 'main' refused; exact pinned SHA required)
- `nc8_unavailable_official_runtime_hardware_gate` -> `unavailable_hardware_runtime` (official runtime (pytorch+transformers+xgrammar) not installed on host local-cpu-no-xgrammar)
