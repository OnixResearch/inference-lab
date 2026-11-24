# Issue #88: Unlimited-OCR (3B) Multi-Page PDF Qualification vs Page-Chunked Baselines

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Hardware / artifact gate: `blocked_on_artifacts` (Local vLLM/SGLang serve of pinned MIT Unlimited-OCR (3B) on multi-page PDF corpus)

## Multi-Page PDF Fixture Observations
- `pdf_12p_spec_sheet` / `unlimited_ocr_3b_single_pass_default_off` (`12p`): `NED=0.0`, `tables=2/2`, `headings=4/4`, `mem=7184.0MiB`, `pages/s=3.77358490566`, `outcome=accepted`
- `pdf_12p_spec_sheet` / `page_chunked_ocr_baseline` (`12p`): `NED=0.246913580247`, `tables=1/2`, `headings=2/4`, `mem=4200.0MiB`, `pages/s=2.4`, `outcome=accepted`
- `pdf_40p_technical_report` / `unlimited_ocr_3b_single_pass_default_off` (`40p`): `NED=0.010989010989`, `tables=6/6`, `headings=12/12`, `mem=8080.0MiB`, `pages/s=5.633802816901`, `outcome=accepted`
- `pdf_40p_technical_report` / `page_chunked_ocr_baseline` (`40p`): `NED=0.183486238532`, `tables=3/6`, `headings=8/12`, `mem=4200.0MiB`, `pages/s=3.333333333333`, `outcome=accepted`
- `pdf_64p_manual` / `unlimited_ocr_3b_single_pass_default_off` (`64p`): `NED=0.0`, `tables=10/10`, `headings=18/18`, `mem=8848.0MiB`, `pages/s=6.118546845124`, `outcome=accepted`
- `pdf_64p_manual` / `page_chunked_ocr_baseline` (`64p`): `NED=0.178571428571`, `tables=5/10`, `headings=12/18`, `mem=4200.0MiB`, `pages/s=3.555555555556`, `outcome=accepted`
- `pdf_100p_long_filing` / `unlimited_ocr_3b_single_pass_default_off` (`100p`): `NED=0.0`, `tables=16/16`, `headings=28/28`, `mem=10000.0MiB`, `pages/s=6.451612903226`, `outcome=accepted`
- `pdf_100p_long_filing` / `page_chunked_ocr_baseline` (`100p`): `NED=0.188679245283`, `tables=8/16`, `headings=18/28`, `mem=4200.0MiB`, `pages/s=3.703703703704`, `outcome=accepted`

## Negative Controls (8 / 8 rejected)
- `nc1_unlimited_ocr_checkpoint_blocked` -> `blocked_on_artifacts` (real Unlimited-OCR 3B checkpoint & vLLM/SGLang serve not executed locally)
- `nc2_32k_context_overflow_not_silently_truncated` -> `exhausted` (40960-token document exceeding 32768-token window returns Exhausted)
- `nc3_missing_pages_rejected` -> `invalid` (emitting 38 of 40 pages fails page-completeness check)
- `nc4_empty_edit_distance_denominator_guard` -> `invalid` (empty reference/hypothesis pair yields None instead of NaN/0.0)
- `nc5_floating_checkpoint_tag_rejected` -> `invalid` (floating checkpoint tag rejected)
- `nc6_preserved_count_exceeding_total_rejected` -> `invalid` (cross_page_tables_preserved > total rejected)
- `nc7_nonfinite_memory_rejected` -> `invalid` (NaN peak_memory_mib rejected)
- `nc8_unverified_omnidocbench_claim_non_gating` -> `invalid` (authors' unverified 93% OmniDocBench v1.5 and <0.11 edit-distance claims are non-gating)
