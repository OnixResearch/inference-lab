# Issue #35: Nemotron 3.5 Lightning & Bolt DGX Spark Serving Profiles

- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (DGX Spark GB10 live Nemotron 3.5 Lightning (30B-A3B NVFP4) & Bolt (74B-A9B 24GB) serve)
- Positive device-free conformance: `4/4`; Lifecycle reducer: `3/5` accepted (`warm_mean_ttft=Some(21.5)ms`, `cost_per_accepted=Some(256.3333333333333)ms`)

## Staged Serving Profiles
- `nemotron-3.5-lightning-30b-a3b-nvfp4-thinking-off` (`sha256:d9731a835020...`): device-free=`accepted`, Spark=`INCONCLUSIVE_UNAVAILABLE_SPARK`
- `nemotron-3.5-lightning-30b-a3b-nvfp4-thinking-on` (`sha256:61c65eb36f30...`): device-free=`accepted`, Spark=`INCONCLUSIVE_UNAVAILABLE_SPARK`
- `nemotron-3.5-lightning-30b-a3b-nvfp4-thinking-off-dspark` (`sha256:28ba3104427c...`): device-free=`accepted`, Spark=`INCONCLUSIVE_UNAVAILABLE_SPARK`
- `nemotron-3.5-bolt-74b-a9b-24gb-target-only` (`sha256:b54c5b1d04e9...`): device-free=`accepted`, Spark=`INCONCLUSIVE_UNAVAILABLE_SPARK`

## Negative Controls (12 / 12 rejected)
- `nc1_floating_checkpoint_alias_rejected` -> `invalid` (floating `main` alias without sha256 digest rejected)
- `nc2_silent_parser_fallback` -> `invalid` (parser/template mismatch or silent parser fallback detected)
- `nc3_missing_usage_not_sse_chunks` -> `invalid` (missing usage accounting (SSE chunks never substituted for tokens))
- `nc4_malformed_sse_stream` -> `invalid` (malformed SSE stream chunk)
- `nc5_stale_endpoint_incarnation` -> `invalid` (stale endpoint incarnation or server restart detected)
- `nc6_silent_drafter_fallback_to_target` -> `invalid` (requested drafter without observed activation fails profile gate)
- `nc7_unquoted_malformed_json_rejected` -> `invalid` (unquoted/malformed JSON rejected independently of model confidence)
- `nc8_plausible_wrong_answer_rejected` -> `invalid` (well-formed JSON with wrong answer rejected independently of confidence)
- `nc9_request_timeout_retained` -> `timeout` (request timed out)
- `nc10_unsupported_field_silently_ignored` -> `unsupported` (unsupported request field silently ignored)
- `nc11_client_prefix_overlap_not_cache_hit` -> `invalid` (client-known prefix overlap cannot be counted as server cache hit)
- `nc12_wrong_served_model_returned` -> `invalid` (wrong served model returned by endpoint)
