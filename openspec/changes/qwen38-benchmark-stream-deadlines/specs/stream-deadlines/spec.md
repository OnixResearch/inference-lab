# Spec: Per-Token Deadlines, Stream-Stall Checks, and Prefill-Interference Mode (`tt-onix#17`)

## Requirements

1. **Bounded streaming observations and overflow invalidation**:
   - `stream_completion` / `observe_stream_completion` MUST record monotonic relative arrival offsets (from request dispatch) for content-bearing SSE events separately from heartbeat comments, role-only deltas, usage-only chunks, empty chunks, and `[DONE]` markers.
   - Per-stream trace storage MUST be explicitly bounded (`max_trace_events`, default `4096`). Exceeding the bound MUST set `trace_truncated = true` and `overflow = true`, increment `dropped_trace_events`, and invalidate gap and deadline compliance (`status = "invalid_truncated"`, never passing).
   - Output records MUST retain content SHA-256 hashes and character counts without publishing raw response text or credentials.

2. **Chunk-vs-token separation and explicit mapping provenance**:
   - Client-observed content-event gaps MUST be labeled explicitly (`content_event_gaps`, `event_gap_seconds_*`) and distinguished from device generation time (`timestamp_domain`, `generation_time_attribution`).
   - Cumulative per-token deadline evaluation (`TTFT_SLO + (i - 1) * TBT_SLO`) MUST run ONLY when an explicit, validated token-to-event mapping is present (`endpoint_token_ids` or `endpoint_cumulative_tokens`).
   - Multi-token chunks MUST NEVER have per-token timestamps fabricated by uniform spreading, and SSE delta counts MUST NEVER be equated with token counts. Without an explicit mapping, `token_deadline_compliance.status` MUST be `"unavailable"` and `every_token_meets_deadline` MUST be `null` (never `true` or `0`).

3. **Versioned additive metric semantics**:
   - Legacy `onix.qwen38-benchmark.v1` and `onix.qwen38-agentic-benchmark.v1` mean-ITL fields, default workload digests, and `run`/`agentic`/`compare` workflows MUST remain backward compatible.
   - Stream summaries MUST report within-stream content-event-gap `p50`/`p95`/`p99`/`max`, per-request maximum-gap quantiles, explicit `event_weighted` vs `request_weighted` weighting with sample counts, and strictly distinct `max_gap_within_budget` vs `every_token_meets_deadline` compliance metrics.

4. **Failure retention in the evaluation population**:
   - Run and mode summaries MUST record `offered`, `admitted` (`"unknown"` when endpoint-only), `completed`, `rejected`, `timed_out`, `incomplete` (including explicit `cancelled` count), and `unobservable` counts with `completed + rejected + timed_out + incomplete + unobservable == offered`.
   - No missing-timing, truncated, failed, cancelled, or premature-EOF stream MAY be dropped to become compliant. Failure and unobservable rates MUST be reported alongside goodput, arrival mode (`closed_loop` vs `interference_burst`), duration, offered load, warmup, SLO denominators, and fail-closed `run_status`.

5. **Reproducible prefill-interference fixture and mode**:
   - `pkgs/qwen38-benchmark/corpus.json` and `pkgs/qwen38-benchmark/agentic_trace.json` MUST remain byte-identical to `OnixResearch/tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a`.
   - A frozen, deterministic prefill-interference fixture (`pkgs/qwen38-benchmark/prefill_interference.json`) and `interference` subcommand (`onix.qwen38-prefill-interference.v1`) MUST run bounded long-lived victim streams against a seeded long-prompt burst and a matched short-prompt or idle control on the same endpoint and settings, reporting dispatch offsets, TTFT, event/token deadline observations, task latency, full population outcomes, and `server_side_phase_attribution.status = "unavailable"`.

6. **Deterministic device-free verification**:
   - Pure timestamp-reducer tests MUST verify the exact `#17` smooth-vs-burst witness (`[0.10, 0.35, 0.60, 0.85, 1.10]` vs `[0.10, 1.07, 1.08, 1.09, 1.10]` with `TTFT_SLO = 0.20, TBT_SLO = 0.30`) and cumulative-deadline slack separation from max-gap SLO.
   - Injected-clock loopback SSE tests MUST verify smooth delivery, stalled delivery, coalesced/unmapped deltas (`unavailable`), heartbeat/role/usage/empty events, one-token outputs, premature EOF, cancellation, trace overflow, invalid/non-monotonic timing, mixed failure populations, and prefill-interference mode without submillisecond OS scheduling assumptions.
   - `PROVENANCE.json`, `tests/qwen38_benchmark_import.rs`, and `nix flake check` MUST preserve donor fixture hashes, verify `#17` extension provenance, keep `evidence/` byte-identical, and enforce zero local home-directory path leaks.
