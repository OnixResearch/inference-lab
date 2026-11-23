# Issue #21: Sparse MoE Dispatch vs Dense Reference & Gradient Parity

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`physical_gpu_sparse_grouped_gemm_moe_timing`)
- Dense vs tiny max err: `0.0`; dense vs sparse forward err: `0.0`; grad err: `0.0`; analytical vs FD grad err: `0.000000000032`
- 3-step matched update parity max err: `0.0`; BF16 underflow assignment count preserved: `true`
- Scoped decision: `GO_FOR_CPU_SPARSE_MOE_PARITY_INCONCLUSIVE_FOR_GPU_SPEEDUP`

## Differential Fixtures
- `01-skewed-bias-reversal-empty-expert-and-overflow-preserved`: `accepted` (pairs `12/24`, empty_skipped=2, fwd_err=`0.0`)
- `02-top-k-1-and-top-k-all-parity`: `accepted` (pairs `6/24`, empty_skipped=3, fwd_err=`0.0`)
- `03-extreme-finite-logits-bf16-gate-underflow-load-count-safe`: `accepted` (pairs `2/4`, empty_skipped=2, fwd_err=`0.0`)
- `04-exact-topk-tie-deterministic-selection`: `accepted` (pairs `2/4`, empty_skipped=2, fwd_err=`0.0`)
- `05-three-step-matched-weight-and-bias-update-parity`: `accepted` (pairs `12/24`, empty_skipped=2, fwd_err=`0.0`)

## Adversarial Negative Controls
- `01-negative-wrong-bias-added-to-output-weights`: `invalid` (err=`0.701941087579`) — Adding load-balancing bias into output gate weights fails numerical parity.
- `02-negative-missing-affinity-normalization`: `invalid` (err=`0.036479841997`) — Omitting normalization over selected raw affinities fails numerical parity.
- `03-negative-duplicate-scatter-contribution`: `invalid` (err=`0.285502538451`) — Duplicating scatter-add contribution fails output parity.
- `04-negative-silent-token-drop-at-capacity-boundary`: `invalid` (err=`0.397286737673`) — Silently dropping overflow assignments at capacity boundary fails output parity.
- `05-negative-incorrect-shared-expert-multi-execution`: `invalid` (err=`0.333178847092`) — Running shared expert top_k times instead of once per token fails output parity.
- `06-negative-nonfinite-logit-rejected`: `invalid` (err=`0.0`) — Non-finite NaN input/logit explicitly rejected rather than propagated.
- `07-negative-dense-fallback-cannot-be-mislabeled-sparse`: `invalid` (err=`0.0`) — Dense path executes 24/24 pairs with dense_fallback_triggered=true; cannot be reported as sparse.

> Selected-parameter count alone is not evidence of saved compute; GPU grouped-GEMM timing remains `unavailable_hardware_runtime`.
