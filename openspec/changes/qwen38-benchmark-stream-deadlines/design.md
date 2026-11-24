# Design: Stream Deadlines, Stall Detection, and Prefill-Interference Mode (`tt-onix#17`)

## Schema & backward-compatibility boundaries

- Legacy schemas remain intact and verifiable:
  - `run`: `onix.qwen38-benchmark.v1` (retaining identical default `workload_sha256` and all legacy summary/record metrics; additive `schema_extensions: ["onix.qwen38-stream-deadlines.v1"]`, per-record `stream_observation` + `outcome`, and mode summary `stream_deadlines` + `population_outcomes`).
  - `agentic`: `onix.qwen38-agentic-benchmark.v1` (retaining identical default `workload_sha256`, required `--ttft-p95`/`--itl-p95`/`--task-p95`, and legacy `itl_seconds_mean` curve fields; adding optional `--deadline-ttft-slo`, `--deadline-tbt-slo`, `--max-gap-slo`, `--max-trace-events`, additive `stream_observation`, `population_outcomes`, `stream_deadlines`, and fail-closed `run_status`).
  - `compare`: continues validating and comparing `run` results (including legacy v1 artifacts without stream-deadline extensions).
- New subcommand and schema:
  - `interference`: `onix.qwen38-prefill-interference.v1`, backed by frozen `pkgs/qwen38-benchmark/prefill_interference.json` and validated by `validate_interference_result`.

## Pure reducer & deterministic clock injection

1. **Pure timestamp reducer (`evaluate_stream_timestamps`)**:
   - Inputs: monotonic client arrival offsets `content_event_offsets_seconds`, optional explicit `token_timestamps` + `token_mapping_provenance`, `max_trace_events`, `trace_truncated`, `timing_valid`, `ttft_slo_seconds`, `tbt_slo_seconds`, `max_gap_slo_seconds`.
   - Floating-point deadline tolerance: `DEADLINE_EPS = 1e-12` so exact arithmetic boundaries like `0.20 + 3 * 0.30` vs `1.10` do not suffer IEEE-754 representation drift while genuine misses fail cleanly.
   - Outputs:
     - `content_event_gaps`: `status` (`observed` | `single_content_event` | `no_content_events` | `invalid_truncated` | `invalid_timing`), `gap_count`, `event_gap_seconds_p50`, `event_gap_seconds_p95`, `event_gap_seconds_p99`, `event_gap_seconds_max`, `event_gap_seconds_mean`.
     - `max_gap_compliance`: strictly evaluates `max_gap <= max_gap_slo_seconds` (distinct from cumulative token deadlines). Invalidated (`status = "invalid_truncated"` or `"invalid_timing"`, `max_gap_within_budget = False`) on overflow or invalid timing.
     - `token_deadline_compliance`: strictly evaluates `t_{i-1} <= ttft_slo + (i - 1) * tbt_slo` for `i = 1..N` ONLY when `token_mapping_provenance` is explicit and validated (`endpoint_token_ids`, `endpoint_cumulative_tokens`, or `explicit_timestamps` for direct reducer tests). Reports `status = "unavailable"` and `every_token_meets_deadline = None` whenever no explicit token-to-event mapping is present; reports `"invalid_truncated"` / `"invalid_timing"` with `every_token_meets_deadline = False` on overflow or invalid timing.

2. **Explicit token-to-event mapping extraction**:
   - `endpoint_token_ids`: `choices[0].token_ids` or `choices[0].delta.token_ids` (non-empty list of ints on every content event; total token IDs across stream must equal `usage.completion_tokens`). Every token in event `k` receives client delivery timestamp `t_k` (never spread backward).
   - `endpoint_cumulative_tokens`: `choices[0].cumulative_completion_tokens` or per-content-event `usage.completion_tokens` (strictly increasing positive ints on every content event; final value must equal `usage.completion_tokens`). Newly delivered tokens in event `k` receive client delivery timestamp `t_k`.
   - Any missing, partial, non-integer, non-monotonic, or count-mismatched mapping produces `status = "unavailable"` with an explicit reason.

3. **SSE event classification & bounded trace capture (`observe_stream_completion`)**:
   - Injectable `clock: Callable[[], float] = time.perf_counter` and optional `cancel_event: threading.Event | None = None`.
   - Separates SSE lines into:
     - `heartbeat`: SSE comment lines (`:...`)
     - `done`: `data: [DONE]`
     - `content`: JSON payload with non-empty `content_text(decoded)`
     - `role_only`: JSON payload with empty content and `delta.role`
     - `usage_only`: JSON payload with empty content, no `delta.role`, and `usage` object
     - `empty`: other empty-content JSON payloads (including finish-reason-only frames)
   - Incremental SHA-256 (`hashlib.sha256()`) and `text_characters` avoid retaining raw response text.
   - `content_event_offsets_seconds` is capped at `max_trace_events`; additional content events increment `dropped_trace_events` and set `trace_truncated = True`, `overflow = True`.

## Population accounting & fail-closed run status

Every request dispatched in `run`, `agentic`, and `interference` receives a non-collapsing `outcome` in `{"completed", "rejected", "timed_out", "incomplete", "unobservable"}` plus `outcome_reason`:
- `completed`: HTTP 200, valid content, positive `usage.completion_tokens`, valid stream termination (`finish_reason_seen` and `done_seen` for streaming), `not trace_truncated`, `timing_valid`.
- `rejected`: HTTP non-200 or JSON `error` payload.
- `timed_out`: `TimeoutError` / `socket.timeout`.
- `incomplete`: EOF without valid completion (`finish_reason` or `[DONE]` missing), missing `usage.completion_tokens`, empty content, connection drop, or client cancellation (`outcome_reason = "cancelled"`, also counted in `cancelled` field).
- `unobservable`: completed payload whose timing/trace observation is invalidated (`trace_truncated`, `not timing_valid`, or non-stream mode where stream gaps are unobservable).

Invariants:
- `completed + rejected + timed_out + incomplete + unobservable == offered`
- `admitted = "unknown"` for endpoint-only client runs
- All SLO compliance denominators use `offered` (never dropping failed/truncated/missing-timing records)
- `run_status` is `"passed"` only when `completed == offered`, no failures/unobservables exist, and every configured SLO is evaluated and satisfied.

## Prefill-interference fixture & mode

- `pkgs/qwen38-benchmark/prefill_interference.json` defines:
  - Fixed seed (`20261124`), system prompt, bounded victim streams (`victim-0`, `victim-1`) with deterministic dispatch offsets and `max_tokens`, and a burst schedule (`burst_offset_seconds`, frozen `long_prompts`, and matched `short_control_prompts`).
- `run_prefill_interference_benchmark` runs matched control (`short_prompt` or `idle`) and `interference_burst` phases on the same endpoint and settings, records per-stream planned/actual dispatch offsets, TTFT, task latency, event gaps, max-gap and token-deadline compliance, population outcomes, side-by-side interference-vs-control comparisons, and `server_side_phase_attribution: {"status": "unavailable", ...}`.

## Provenance & packaging

- `corpus.json`, `agentic_trace.json`, and `LICENSES/LGPL-2.1-only.txt` remain byte-identical to `OnixResearch/tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a`.
- `pkgs/qwen38-benchmark/PROVENANCE.json` keeps all donor baseline fields and adds `derived_extensions.stream_deadline_extension` documenting `OnixResearch/tt-onix#17`, new schemas, and SHA-256 digests.
- `pkgs/qwen38-benchmark/default.nix` installs `prefill_interference.json` and exposes `QWEN38_BENCH_INTERFERENCE_WORKLOAD` in the wrapped binary; `checkPhase` and `installCheckPhase` run `--self-test`.
- `tests/qwen38_benchmark_import.rs` continues pinning exact donor SHA-256 on `corpus.json`, `agentic_trace.json`, and `LICENSES/LGPL-2.1-only.txt`, verifies `PROVENANCE.json` donor + `#17` extension fields and `prefill_interference.json` digest, and enforces zero local home-directory path leaks.
