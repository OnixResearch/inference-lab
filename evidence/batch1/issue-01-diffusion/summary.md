# Issue #1 — Bounded diffusion proposal contract report

- **Execution class**: `bounded_contract_harness`
- **Physical hardware gate**: `unavailable_hardware_runtime`
- **Noble deterministic control pin**: `OnixResearch/noble-lang@8698dca070e2d89e0bc23ea4050fc79b9fa230b6`
- **Real-model comparison decision**: `UNAVAILABLE_HARDWARE_RUNTIME_FOR_REAL_MODEL_ARMS`

## Sampler capability probes

| Backend | Qualified diffusion sampler | Mismatches |
| --- | --- | --- |
| `ar-chat-baseline-adapter` | `false` | 6 |
| `diffusion-contract-reference` | `true` | 0 |

## Noble admission and adversarial guard outcomes

| Candidate | Guard passed | Parse/type/effect | Bounded test status | Proof status | Rejection reason |
| --- | --- | --- | --- | --- | --- |
| `cand-00-root-valid` | `true` | `accepted` | `passed_bounded_tests_not_universal_proof` | `run_only` | `none` |
| `cand-01-type-ok-test-fail` | `true` | `accepted` | `failed_bounded_tests` | `unproven` | `none` |
| `cand-02-frozen-edit` | `false` | `invalid` | `not_evaluated` | `unproven` | `frozen_prefix_edited` |
| `cand-03-stale-base` | `false` | `invalid` | `not_evaluated` | `unproven` | `stale_base_digest` |
| `cand-04-malformed` | `false` | `invalid` | `not_evaluated` | `unproven` | `malformed_quotation_syntax` |
| `cand-05-late-result` | `false` | `exhausted` | `not_evaluated` | `unproven` | `late_result_after_deadline` |
| `cand-06-excess-budget` | `false` | `exhausted` | `not_evaluated` | `unproven` | `excessive_candidates_budget_exceeded` |

## Default-off candidate/repair tree interchange comparison

| Format | Serialized bytes | Nodes | Activation/bookmarks | Selection label | Field losses |
| --- | --- | --- | --- | --- | --- |
| `dependent-weave-v1` | 1256 | 7 | `true` | `false` | selection_procedure_label, candidate_base_and_diagnostic_digests, external_consumer_verdict_tag |
| `jsonl-parent-stream-v1` | 4475 | 7 | `true` | `true` | none |
| `socketteer-nested-json-v1` | 718 | 7 | `false` | `false` | bookmark_bits, multi_node_active_flags, selection_procedure_label, candidate_base_and_diagnostic_digests, external_consumer_verdict_tag |

Hard consumer admission remains outside both soft scoring and tree serialization. Offline pool rescoring does not establish real-model closed-loop performance.
