# Issue #32: QSA Cache Reuse & Target/Indexer/GDN/MTP State Coherence

- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (DGX Spark GB10 ARM64 / Qwen3.8-27B QSA live single-slot server)
- Service qualification verdict: `INCONCLUSIVE` (toy fixtures do not qualify a 27B service)
- Tested limits: `slots = 1`, `context_tokens = 32`, `block_size = 4`

## Path Coherence Checks
- `qsa_chunked_block_and_sparse_boundary` (`qsa_block_sparse`): `max_err = 0.000e0`, `reuse = verified_against_fresh_recompute`, `outcome = accepted` — chunks [3,2,4,4] cross 4-token block and 8-token sparse-selection boundaries
- `dense_control_append_resume` (`dense_control`): `max_err = 0.000e0`, `reuse = verified_against_fresh_recompute`, `outcome = accepted` — fixed dense-mode control checked independently of QSA approximation quality
- `qsa_checkpoint_restore_and_changed_suffix` (`qsa_block_sparse`): `max_err = 0.000e0`, `reuse = verified_against_fresh_recompute`, `outcome = accepted` — rollback to mid-block pos 7 rebuilt pooled index keys + poisoned tail
- `mtp_draft2_reject_pos0` (`qsa_block_sparse`): `max_err = 0.000e0`, `reuse = verified_against_fresh_recompute`, `outcome = accepted` — committed 1 tokens cleanly matching fresh recompute
- `mtp_draft2_reject_pos1` (`qsa_block_sparse`): `max_err = 0.000e0`, `reuse = verified_against_fresh_recompute`, `outcome = accepted` — committed 2 tokens cleanly matching fresh recompute
- `mtp_draft2_accept_both_plus_bonus` (`qsa_block_sparse`): `max_err = 0.000e0`, `reuse = verified_against_fresh_recompute`, `outcome = accepted` — committed 3 tokens cleanly matching fresh recompute
- `single_slot_two_session_explicit_reset` (`qsa_block_sparse`): `max_err = 0.000e0`, `reuse = verified_against_fresh_recompute`, `outcome = accepted` — sess-1 state completely cleared on reset; zero influence on sess-2

## Negative Controls (8 / 8 rejected)
- `nc1_stale_pooled_index_keys` -> `invalid` (pooled index keys not rebuilt on rollback fail independent checker)
- `nc2_wrong_token_positions` -> `invalid` (shifted token position encoding diverges and fails position/state check)
- `nc3_omitted_gdn_rollback` -> `invalid` (omitting GDN recurrent rollback corrupts matrix state and output logits)
- `nc4_retained_rejected_draft` -> `invalid` (retained rejected draft slots break valid_len and accepted-prefix agreement)
- `nc5_mismatched_indexer_artifact_identity` -> `invalid` (swapped indexer hash refuses checkpoint continuation)
- `nc6_cross_session_prefix_reuse` -> `invalid` (session-b cannot inherit session-a live slot state without isolation breach)
- `nc7_missing_reuse_telemetry_client_overlap_only` -> `invalid` (client-known prefix overlap without server reuse telemetry cannot pass)
- `nc8_stale_deployment_generation_after_restart` -> `invalid` (deployment generation bump after restart invalidates pre-restart state)
