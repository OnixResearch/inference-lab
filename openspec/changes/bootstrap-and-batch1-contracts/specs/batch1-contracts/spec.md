# Specification: Bootstrap and Batch 1 contracts

## Scenario 1: Fresh output directory publication with `COMPLETE` last
- Given any CLI subcommand (`benchmark`, `diffusion`, `trace`, `stage`, `power`, `tokenizer`, `related-work`) and a `--out` path whose parent exists and whose target does not exist:
  - The CLI creates the directory, writes `inputs.json`, `results.json`, and `summary.md`, and creates empty `COMPLETE` last.
  - Running the same subcommand twice into two fresh directories produces byte-identical files across all four artifacts.
  - Pointing `--out` at an existing directory fails non-zero without modifying existing files.

## Scenario 2: Non-collapsing outcomes and explicit execution classes
- Every emitted report explicitly identifies `execution_class` (`bounded_contract_harness`, `unavailable_hardware_runtime`, or `blocked_on_artifacts`).
- Distinct failure classes (`invalid`, `unsupported`, `exhausted`, `timeout`, `oom`, `cancelled`, `internal_failure`, `unavailable_hardware_runtime`, `blocked_on_artifacts`) remain separate and are never collapsed into zero or dropped from denominators.

## Scenario 3: Issue #1 — Diffusion proposal contract and tree interchange
- Capability probes detect unsupported masking/pinning/seeding and record effective sampler capabilities distinctly from requested settings.
- Frozen-region edits, stale base/diagnostic digests, malformed outputs, excessive candidates, and post-deadline arrivals are rejected without altering the incumbent artifact.
- Finite-candidate selection separates `ArgmaxBestOfN` from `WeightedResampling`, quarantines NaN/Inf/missing scores, and enforces deterministic tie-breaking and duplicate policy.
- Hard Noble admission outcomes (`accepted`, `invalid`, `unsupported`, `exhausted`, `internal_failure`) cannot be overridden by informational model confidence or interchange format fields.
- Default-off 3-format interchange (`dependent-weave-v1`, `jsonl-parent-stream-v1`, `socketteer-nested-json-v1`) reports exact bytes and an explicit field-loss list.

## Scenario 4: Issue #40 — Host and GPU trace correlation qualification
- A known positive control with calibrated clock mappings joins host enqueue, device execution/copy, CUDA graph, Linux scheduling, request phase, and client SSE events with explicit units and uncertainty bounds.
- Wrong run/module binding, unmapped cross-clock subtraction, swapped stream/device IDs, unscoped correlation reuse across contexts, buffer/event loss, and late metadata beyond the deferred bound are detected and prevent substantiating a complete causal path.
- Batched kernels serving multiple requests without per-request annotation leave request attribution explicitly `unavailable`.

## Scenario 5: Issue #41 — Versioned transformer stage contract and split reference
- Unsplit (`0..4`), 2-stage (`0..2` + `2..4`), and 3-stage (`0..1` + `1..3` + `3..4`) execution of the frozen 4-layer dense causal transformer match layer-boundary activations and final logits within `1e-12` max absolute error across full prefill, chunked prefill, forced-token decode, and reset/reuse.
- Negative controls reject non-contiguous/overlapping layer intervals, checkpoint/effective-weight/dtype/layout/byte-order mismatch, stale request generation, wrong token position, truncated buffers, multiple sampling owners, and buffer/KV reuse after timeout or cancellation.
- Requesting Apple GPU (Asahi/Metal) or Tenstorrent P150 locally returns `unavailable_hardware_runtime` with zero silent CPU fallback.

## Scenario 6: Issue #43 — Power and throttle cadence qualification
- Host invocation cadence spacing cannot substantiate a device instruction/tile spacing claim.
- `known-equivalent` yields `no_benefit`; `wrong-but-fast` is disqualified (`numerical_mismatch`); `order-drift` yields `inconclusive_order_drift` under counterbalanced AB/BA analysis.
- Dropped, stale, or wrongly-bound telemetry blocks causal attribution; energy metrics are emitted only when backed by actual sensor observations.
- Physical device acceptance remains explicitly `unavailable_hardware_runtime`.

## Scenario 7: Issue #52 — Tokenizer and chat-template conformance
- Reference BPE + special-token encoding/decoding preserves exact UTF-8 round-trips and never splits registered special tokens.
- Chat templates (`chatml-v1`, `qwen-thinking-v1`) defend against role-boundary injection inside user content.
- Any backend mismatch (token ID divergence, split special token, missing EOS, dropped whitespace, role injection) is reported as `disqualifying_parity_mismatch` with exact diff offsets.

## Scenario 8: Issues #56, #58, #60 — Pinned default-off related-work gates
- All three entries have `default_off = true`, `claims_verified = false`, and zero author figures promoted to acceptance gates.
- Empty placeholder repos (`ankitvgupta/fpga-inference`) and HTML-only project pages (`ShivamDuggal4/latent-lattice-language-models@23876839...`) are rejected as runnable implementation pins.
- Undeclared Hugging Face licenses in `#58` remain explicit license gates and are never assumed to inherit Apache-2.0.
