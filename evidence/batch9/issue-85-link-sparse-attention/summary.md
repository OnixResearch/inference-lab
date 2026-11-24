# Issue #85: Constructor AGI Link Learned Sparse Attention Related Work & Long-Context Boundary

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Hardware / artifact gate: `blocked_on_artifacts` (Constructor AGI Link public code/weights & 4M-context GPU serve)
- Upstream reference: https://x.com/michaeltgao/status/2123204373773983992 (`post_only_unlocated_paper_repo`; unverified claims: +53% retrieval at 4M, 9.9x GPU efficiency)

## Pure-Rust Learned Sparse-Read vs Exact Dense Attention
- `dense_exact_attention`: `16/16` blocks (`1.000` KV reads), `max_abs_err=0.000e0`, `l2_err=0.000e0`, `needle_recalled=true`, `parity_gate=true`
- `link_learned_full_read_k16`: `16/16` blocks (`1.000` KV reads), `max_abs_err=0.000e0`, `l2_err=0.000e0`, `needle_recalled=true`, `parity_gate=true`
- `link_learned_sparse_k4`: `4/16` blocks (`0.250` KV reads), `max_abs_err=1.221e-1`, `l2_err=2.109e-1`, `needle_recalled=true`, `parity_gate=true`
- `link_learned_sparse_k2`: `2/16` blocks (`0.125` KV reads), `max_abs_err=1.874e-1`, `l2_err=2.957e-1`, `needle_recalled=true`, `parity_gate=true`

## Negative Controls (8 / 8 rejected)
- `nc1_unlocated_link_artifacts_blocked` -> `blocked_on_artifacts` (Link public code/weights not pinned -> blocked_on_artifacts)
- `nc2_zero_block_selection_rejected` -> `invalid` (top_k_blocks = 0 rejected cleanly)
- `nc3_oob_block_count_rejected` -> `invalid` (top_k_blocks exceeding block count rejected)
- `nc4_nonfinite_query_rejected` -> `invalid` (NaN query vector rejected before block scoring)
- `nc5_faster_but_wrong_end_task_disqualified` -> `invalid` (sparse candidate with 10x fewer KV reads but 0.62 < 0.80 end-task accuracy disqualified)
- `nc6_needle_only_without_end_task_rejected` -> `invalid` (retrieval-only check without independent end-task evaluation rejected)
- `nc7_oob_block_index_attention_rejected` -> `invalid` (out-of-range block index rejected by attention reference)
- `nc8_unverified_x_claim_as_gate_rejected` -> `invalid` (authors' unverified +53% retrieval / 9.9x GPU efficiency claims are non-gating related work only)
