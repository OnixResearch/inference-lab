# Issue #13: Reusable `djev` Protocol Client Split

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`live_djev_http_server_and_rsi_cutover`)
- Wire version: `djev.systemone.v1`
- Golden ASCII SHA-256: `f7b2abd023e5252a634933b508a8eb31eeba3e04f17ad6038f25d2ef50f6ddd3`
- Golden non-ASCII SHA-256: `ed926086b94c59247151efba947b65d991f995e5df52ea5631f0437233aadda5`

## Closed RSI `#42` Effective Cache-Identity Regressions
- `00-identical-effective-read-reuses-cache`: invalidated_prior_cache=`false`
- `01-changed-prompt-rendering`: invalidated_prior_cache=`true`
- `02-reordered-candidates`: invalidated_prior_cache=`true`
- `03-changed-template-revision`: invalidated_prior_cache=`true`
- `04-changed-sample-count`: invalidated_prior_cache=`true`
- `05-changed-seed`: invalidated_prior_cache=`true`
- `06-changed-max-options-tournament`: invalidated_prior_cache=`true`
- `07-changed-canvas`: invalidated_prior_cache=`true`
- `08-changed-served-revision`: invalidated_prior_cache=`true`
- `09-missing-immutable-model-digest-non-reusable`: invalidated_prior_cache=`true`

## Transport, Capability & Deadline Controls
| Case | Capability | Outcome | Elapsed ms | Reason |
| --- | --- | --- | ---: | --- |
| `01-positive-bounded-structured-read-scoring` | `structured_read_scoring` | `accepted` | 200 | none |
| `02-negative-generative-completion-unsupported` | `generative_masked_completion` | `unsupported` | 200 | requested_capability_unsupported_no_silent_chat_or_proof_fallback |
| `03-negative-noble-proof-acceptance-unsupported` | `noble_proof_acceptance` | `unsupported` | 200 | requested_capability_unsupported_no_silent_chat_or_proof_fallback |
| `04-negative-schema-rejection` | `structured_read_scoring` | `invalid` | 200 | protocol_schema_rejection |
| `05-negative-malformed-response-json` | `structured_read_scoring` | `invalid` | 200 | malformed_response_json |
| `06-negative-http-transport-failure` | `structured_read_scoring` | `internal_failure` | 200 | http_transport_failure |
| `07-negative-bounded-response-oversize` | `structured_read_scoring` | `exhausted` | 200 | bounded_response_byte_limit_exceeded |
| `08-negative-whole-decision-deadline-not-reset-by-transport` | `structured_read_scoring` | `timeout` | 550 | whole_decision_queue_plus_read_deadline_exceeded |
| `09-negative-cancelled-decision` | `structured_read_scoring` | `cancelled` | 200 | caller_cancelled_before_decision_commit |
| `10-negative-late-stale-epoch-refused` | `structured_read_scoring` | `cancelled` | 200 | late_stale_epoch_result_refused |
| `11-negative-saturated-queue-refused` | `structured_read_scoring` | `exhausted` | 200 | bounded_async_queue_saturated_refused |

> Model-free protocol conformance only; structured-read scoring never grants generative masked completion or Noble proof acceptance.
