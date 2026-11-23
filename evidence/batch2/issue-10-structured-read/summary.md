# Issue #10: Conditional Shared Structured-Read Contracts

- Execution class: `blocked_on_artifacts`
- Conditional gate: `blocked_on_artifacts` (`two_distinct_real_structured_read_backends_and_second_consumer`)
- Donor pin: `OnixResearch/tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a (structured_read_core.py, check.py)` (`LGPL-2.1-only`)

## Dependency Gate Checks
- `current-lab-state-blocked-on-live-backends`: satisfied=`false` (mocks_only_cannot_satisfy_extraction_gate)
- `negative-two-urls-same-adapter`: satisfied=`false` (fewer_than_two_distinct_real_backend_implementations)
- `negative-two-chat-only-models`: satisfied=`false` (chat_only_models_cannot_satisfy_extraction_gate)

## Adversarial Readout Fixtures
| Fixture | Op | Outcome | Reason |
| --- | --- | --- | --- |
| `00a-positive-ar-label-read-explicit-scope` | `ar_label_read` | `accepted` | none |
| `00b-positive-diffusion-canvas-read` | `diffusion_canvas_read` | `accepted` | none |
| `00c-positive-clef-joint-head-read` | `clef_joint_head_read` | `accepted` | none |
| `01-negative-missing-label` | `ar_label_read` | `invalid` | missing_permitted_label |
| `02-negative-multi-token-label` | `ar_label_read` | `invalid` | multi_token_or_empty_label_unsupported_in_single_slot_read |
| `03-negative-label-order-drift` | `ar_label_read` | `invalid` | label_order_drift_detected |
| `04-negative-nonfinite-distribution` | `ar_label_read` | `invalid` | malformed_or_nonfinite_probability_distribution |
| `05-negative-invalid-probability-mass` | `ar_label_read` | `invalid` | permitted_labels_distribution_not_normalized_to_one |
| `06-negative-exceeded-budget` | `ar_label_read` | `exhausted` | read_or_token_budget_exceeded |
| `07-negative-missing-diffusion-capability` | `diffusion_canvas_read` | `unsupported` | missing_required_capability_no_silent_chat_degradation |
| `08-negative-stale-calibration-identity` | `ar_label_read` | `invalid` | stale_identity_calibration_invalidated |
| `09-negative-cancelled-or-late-result` | `ar_label_read` | `cancelled` | cancelled_or_late_result_discarded |
| `10-negative-renormalized-claimed-as-full-vocab` | `ar_label_read` | `invalid` | restricted_label_renormalization_presented_as_full_vocab |

> Explicitly `blocked_on_artifacts`. No shared runtime library was manufactured ahead of two live distinct backends and a second concrete consumer.
