# Portable Qwen serving benchmark (`qwen38-benchmark`)

> [!NOTE]
> Imported from [`OnixResearch/tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a`](https://github.com/OnixResearch/tt-onix/tree/5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a/pkgs/qwen38-benchmark) under `LGPL-2.1-only` (see [`LICENSES/LGPL-2.1-only.txt`](../LICENSES/LGPL-2.1-only.txt) and [`pkgs/qwen38-benchmark/PROVENANCE.json`](../pkgs/qwen38-benchmark/PROVENANCE.json)). Tracked under [`inference-lab#7`](https://github.com/OnixResearch/inference-lab/issues/7). Tenstorrent-specific P150x2 deployment, vLLM plugin preflight, and hardware admission evidence remain in [`OnixResearch/tt-onix`](https://github.com/OnixResearch/tt-onix/blob/main/docs/qwen38-benchmark.md).

`qwen38-benchmark` measures the client-visible behavior of any OpenAI-compatible
Qwen endpoint on a frozen corpus. It exists so distinct serving stacks (for
example a Tenstorrent onix service, vLLM, or llama.cpp) can be compared with the
same prompts, the same greedy settings, and the same accounting.

It is a benchmark, not an admission receipt. It never starts a server, touches a
device, reads server-private evidence, or claims that a result is a production
throughput measurement.

## Build and self-test

```sh
nix run .#qwen38-benchmark -- --self-test
nix build .#checks.x86_64-linux.qwen38-benchmark --no-update-lock-file --option builders ''
```

The self-test starts a loopback fake OpenAI server using only the Python
standard library and exercises streaming, batch release, summaries, parity,
agentic SLO sweeps, and tamper rejection. It needs no accelerator driver,
hardware device, model weights, or credentials.

## Frozen workload

The packaged corpus is `pkgs/qwen38-benchmark/corpus.json`: the eight prompts
from the admitted Qwen3.8-27B P150x2 B=8 batch profile. The default run uses
64 generated tokens, greedy decoding, streaming, one warmup, three measured
runs, and a batch width of 8.

Changing the corpus, prompt order, `--max-tokens`, concurrency, stream mode, or
run counts changes the workload digest. `compare` refuses results whose digests
differ.

## Measure one endpoint

```sh
nix run .#qwen38-benchmark -- run \
  --label onix \
  --base-url http://127.0.0.1:8000 \
  --output onix.json

nix run .#qwen38-benchmark -- run \
  --label vllm \
  --base-url http://127.0.0.1:8001 \
  --output vllm.json
```

`--model` is optional. When omitted, the harness takes the first ID from
`/v1/models`.

Put credentials in `--api-key` or `QWEN38_BENCH_API_KEY`, never in the base URL.
A base URL with userinfo, a query, or a fragment is rejected. A secret embedded
in the URL path is still recorded, so do not put one there. Selected environment
values that are absolute host paths are recorded as `<absolute-path>`; Nix store
paths are retained. The endpoint's served model name and health payload are
recorded as returned, so keep secrets and host paths out of those fields.

Useful options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--concurrency 4,8` | `8` | Measure one or more batch widths greater than one |
| `--runs 5` | `3` | More measured roots |
| `--warmup-runs 2` | `1` | Prime compile, trace, and cache state |
| `--no-stream` | off | Fallback for an endpoint without SSE; TTFT and decode rate become unavailable |
| `--api-key` | `QWEN38_BENCH_API_KEY` | Bearer token; never written to the result |
| `--header NAME: VALUE` | unset | Additional request header, repeatable |
| `--request-timeout 900` | `600` | Per-request timeout |

## Compare

```sh
nix run .#qwen38-benchmark -- compare onix.json vllm.json \
  --output comparison.json
```

The first result is the baseline unless `--baseline-label` names another. The
comparison emits candidate/baseline ratios for:

- single-stream delivered, decode, and end-to-end throughput;
- single-stream TTFT and p95 wall;
- batch delivered and aggregate decode throughput;
- batch per-user decode throughput, TTFT, and p95 wall;
- exact greedy-text parity per mode and prompt.

A parity mismatch exits non-zero. That is deliberate: if the two endpoints do
not produce the same greedy text, their throughput numbers are not a controlled
comparison. `compare` validates each result first: the stored workload digest
must match its workload, every mode and run must be present, every record must
name the corpus prompt it used, every single-stream run must cover the whole
corpus, and the stored summary must match a summary recomputed from the runs. A
malformed or edited result is rejected instead of producing a ratio.

To cancel linear host drift, run targets in ABBA order:

```sh
nix run .#qwen38-benchmark -- run --label onix --base-url http://127.0.0.1:8000 --output onix-a.json
nix run .#qwen38-benchmark -- run --label vllm --base-url http://127.0.0.1:8001 --output vllm-a.json
nix run .#qwen38-benchmark -- run --label vllm --base-url http://127.0.0.1:8001 --output vllm-b.json
nix run .#qwen38-benchmark -- run --label onix --base-url http://127.0.0.1:8000 --output onix-b.json
nix run .#qwen38-benchmark -- compare onix-a.json vllm-a.json --output compare-a.json
nix run .#qwen38-benchmark -- compare onix-b.json vllm-b.json --output compare-b.json
```

When two serving stacks share the same physical accelerator, run them
sequentially, reset the hardware between owners, and record which package,
runtime, cache mode, and server flags each endpoint used in `--note`.

## Accounting

- **TTFT** is request send to the first streamed content chunk with non-empty
  text. Empty role or usage chunks do not count.
- **Decode rate** is `(completion_tokens - 1) / (last_content_time - TTFT)`.
  The token count must come from `usage.completion_tokens`. A stream or
  non-stream response without that field is rejected; the harness does not
  substitute a content-chunk count for a token count.
- **Delivered throughput** is completion tokens over request or batch wall.
  Batch wall starts at the first request released by the barrier and ends when
  the slowest request finishes.
- **Aggregate decode throughput** sums per-request decode tokens over the span
  from the earliest TTFT to the latest content chunk. It excludes prefill but
  not queueing.
- **Per-user decode throughput** is the median of each request's decode rate.
  For a batch it is the median per-request rate; aggregate throughput is
  reported separately.

Batch mode releases every request from one barrier after its connection is
established. That removes client setup from the window, but it cannot prove the
server actually formed one batch or kept all slots active. Treat an aggregate
number as client-delivered throughput for the measured release pattern, not as a
server-side occupancy measurement.

A non-stream response with empty text or zero completion tokens is rejected.
Without that check two broken endpoints could "agree" on empty text and produce
a meaningless parity pass.

## Agentic mode

`run` answers "how fast is one stream?" and `agentic` answers "how many
concurrent agents meet a response-time target on a fixed multi-call trace?"
It is still a client-side harness: it never reads server cache counters or
starts a server.

```sh
nix run .#qwen38-benchmark -- agentic \
  --label onix \
  --base-url http://127.0.0.1:8000 \
  --agents 1,2,4,8 \
  --ttft-p95 0.5 \
  --itl-p95 0.05 \
  --task-p95 30 \
  --cache-mode disabled \
  --output onix-agentic.json
```

Add a power boundary only when it is measured:

```sh
nix run .#qwen38-benchmark -- agentic ... \
  --power-boundary wall --watts 900 \
  --power-method "wall meter, 1 s samples, five-minute mean"
```

### Frozen trace

The packaged trace is `pkgs/qwen38-benchmark/agentic_trace.json`: one system
prompt, eight coding-agent calls, a 300-token generation bound per call, and
seven one-second tool waits. Each call's message list extends the previous
call's list with a frozen assistant turn, a frozen tool result, and the next
user turn. Assistant turns are frozen rather than fed back from live output so
every target receives the same workload text; the live reply is still measured
and hashed per call.

The history grows, so call *n* reuses the entire prompt of call *n-1* as a
client-known prefix. `--tool-wait-scale` multiplies every wait uniformly; it is
recorded in the workload digest and must match across compared targets.
`--trace` points at a different frozen file.

### SLOs and the sweep

`--ttft-p95`, `--itl-p95`, and `--task-p95` are required and are in seconds.
They are defined before the run and recorded in the result. For each agent
width, every agent runs the whole trace closed-loop from one barrier; a run
repeats the sweep `--runs` times and discards `--warmup-runs` sweeps. The
summary reports, per width, p50/p95 call TTFT, p50/p95 mean ITL, p50/p95/max
task time, tasks/s, output tokens/s, and whether each SLO passed. It reports
the largest width meeting all three SLOs, plus the width with the highest
task throughput as the saturation point. The `curve` array is the knee curve.

Per-call ITL is the mean inter-token interval,
`(last_content_time - TTFT) / (completion_tokens - 1)`, which is the reciprocal
of the per-call decode rate. Task time covers the calls plus the replayed tool
waits for one agent. Every run's summary is retained, so a run whose ITL
deviates from the session spread can be discarded instead of silently averaged
into the curve.

Agentic results use their own schema; `compare` accepts only `run` results.
Two agentic results are directly comparable when `workload_sha256` and the three
SLOs match, and their `curve` arrays can then be read side by side.

### Token accounting

OpenAI-compatible usage reports prompt and completion tokens; it does not
guarantee a server-side prefix-cache report. Every call therefore records:

- `prompt_tokens` and `completion_tokens`, from usage;
- `reused_prefix_characters`, the exact client-known prefix overlap with the
  previous call in the same trace;
- `reused_prefix_fraction`, that overlap over the serialized prompt length;
- `reused_prefix_tokens_estimate` and `new_input_tokens_estimate`, a
  client-side apportionment of the reported `prompt_tokens` by that fraction;
- `server_reported_cached_tokens` only when the endpoint returns
  `usage.prompt_tokens_details.cached_tokens`, with its source named;
- `cache_confirmation: "client-prefix-overlap-only"`.

The estimate is an accounting label, not evidence that the server kept those
tokens in a prefix cache. A result never claims a server cache hit unless the
endpoint itself reported cached tokens.

### Held conditions

The result records the endpoint label, base URL, resolved model and probe; the
trace content, digest, per-call message digests, generation bounds, and scaled
tool-wait schedule; the SLOs; the agent widths, runs, warmup, settle time,
greedy temperature, and streaming flag; the declared cache mode
(`cold`, `warm`, `disabled`, or `unknown`); and the power boundary, watts, and
method. `--power-boundary` and `--watts` must be supplied together; otherwise
`agents_at_slo_per_watt` is `null`.

Cache mode is operator-declared; the client cannot observe or reset a server's
cache. Record exact server-side flags in `--note`.

## Stream-stall and per-token deadline semantics (`tt-onix#17`)

Every streaming request in `run`, `agentic`, and `interference` attaches a
versioned `stream_observation` (`onix.qwen38-stream-deadlines.v1`) alongside
the legacy `onix.qwen38-benchmark.v1` / `onix.qwen38-agentic-benchmark.v1`
mean-ITL fields:

- **Bounded per-stream event capture**: monotonic relative arrival offsets (from
  request dispatch) are recorded only for content-bearing SSE events, separated
  from SSE comment heartbeats (`:...`), role-only deltas, usage-only chunks,
  empty deltas, and `[DONE]` markers (`event_counts`). Capture is bounded by
  `--max-trace-events` (default `4096`). Exceeding the bound sets
  `trace_truncated: true` and `overflow: true`, records `dropped_trace_events`,
  and invalidates gap and deadline compliance (`status: "invalid_truncated"`,
  never passing). Only SHA-256 content hashes and character counts are stored —
  never raw response text or credentials.
- **Chunk-vs-token separation**: client-observed inter-event gaps are labeled
  explicitly as `content_event_gaps` (`event_gap_seconds_p50/p95/p99/max/mean`)
  under `timestamp_domain: "client_sse_delivery_monotonic_seconds"` and
  `generation_time_attribution: "client_delivery_only_not_device_generation_time"`.
  Cumulative per-token deadline checks (`TTFT_SLO + (i - 1) * TBT_SLO`) run
  **only** when the endpoint supplies an explicit, validated token-to-event
  mapping (`endpoint_token_ids` via `choices[0].token_ids` /
  `choices[0].delta.token_ids`, or `endpoint_cumulative_tokens` via
  `choices[0].cumulative_completion_tokens` / per-event `usage.completion_tokens`)
  whose total matches `usage.completion_tokens`. Multi-token chunks are never
  interpolated backward, and SSE chunk counts are never equated with token
  counts. Without an explicit mapping, `token_deadline_compliance.status` is
  `"unavailable"` and `every_token_meets_deadline` is `null`.
- **Distinct max-gap vs cumulative-deadline SLOs & weighting**: summaries report
  both `event_weighted` (pooled inter-event gaps) and `request_weighted`
  (per-request max-gap and p95-gap distributions) with explicit sample counts,
  and keep `max_gap_slo_compliance` (`max_gap_within_budget`) strictly separate
  from `cumulative_token_deadline_compliance` (`every_token_meets_deadline`),
  since early token delivery builds slack under cumulative deadlines.
- **Full failure retention**: mode and run summaries retain every offered
  request in `population_outcomes`: `offered`, `admitted` (`"unknown"` when
  endpoint-only), `completed`, `rejected`, `timed_out`, `incomplete` (with
  explicit `cancelled` count), and `unobservable` (trace overflow or invalid
  timing), with `completed + rejected + timed_out + incomplete + unobservable == offered`.
  Failure and unobservable rates, goodput, `arrival_mode`, duration, offered
  load, warmup count, SLO denominators, and fail-closed `run_status`
  (`passed`, `failed_slo`, `failed_population`, or `incomplete_slo_unavailable`)
  are recorded without dropping failed or truncated streams.

## Prefill-interference mode (`interference`)

The original eight-call `pkgs/qwen38-benchmark/agentic_trace.json` and
`pkgs/qwen38-benchmark/corpus.json` remain byte-identical to donor
`tt-onix@5ed5e90c`. For controlled prefill-vs-decode interference testing,
`pkgs/qwen38-benchmark/prefill_interference.json` provides a frozen, seeded
workload with two bounded long-lived victim streams overlapped by four frozen
long prompts, paired with a matched `short_prompt` (default) or `idle` control
on the same endpoint and settings:

```sh
nix run .#qwen38-benchmark -- interference \
  --label onix \
  --base-url http://127.0.0.1:8000 \
  --control-mode short_prompt \
  --ttft-slo 0.5 \
  --tbt-slo 0.1 \
  --max-gap-slo 0.25 \
  --output onix-interference.json
```

Each run emits `onix.qwen38-prefill-interference.v1`, recording planned and
actual dispatch offsets, TTFT, content-event gaps, max-gap and cumulative
token-deadline compliance, task latency, full population outcomes, and
side-by-side interference-vs-control victim deltas and ratios. Because the
harness is endpoint-only, `server_side_phase_attribution` is always marked
`"unavailable"`.

## Non-claims

- Not an admission receipt and not a substitute for backend-owned hardware
  admission receipts in `tt-onix`.
- Not a profiler, thermal, or device-health measurement.
- Not a sampled-generation or accuracy benchmark. Greedy text parity is a
  validity gate for throughput comparison, not a model-quality claim.
- A `--no-stream` result has no TTFT or decode-rate claim.
- Agentic mode is a client-side closed-loop measurement of a fixed trace. It
  does not prove one server batch, slot occupancy, or a production arrival
  process, and its tool waits are replayed sleeps rather than real tool calls.
- `reused_prefix_tokens_estimate` is a client-side apportionment, not a server
  prefix-cache hit count. Only `server_reported_cached_tokens` restates an
  endpoint's own usage report.
- `agents_at_slo_per_watt` is only reported inside the stated power boundary
  using the supplied watts. It is not a facility-efficiency or cost claim.
- The comparison is only as controlled as the two endpoint configurations. A
  different checkpoint, tokenizer, precision recipe, mesh, or serving flag is a
  different workload even when the prompt digest matches.
