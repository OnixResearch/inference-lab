# Experiment protocol

## Execution classes

Every report and evidence manifest must declare an explicit `execution_class`:

| Class | Meaning | What it can establish | What it cannot establish |
| --- | --- | --- | --- |
| `bounded_contract_harness` | Pure deterministic Rust execution over frozen fixtures, negative controls, and high-precision reference math | Contract schema validity, negative-control rejection, split/unsplit math parity, tokenizer/template parity logic, deterministic repeatability | Physical GPU/P150/M2/TPU/FPGA throughput, real model accuracy, thermal/power behavior, or hardware driver qualification |
| `unavailable_hardware_runtime` | Requested physical backend, accelerator, or model host is absent or unprovisioned in the execution environment | Exact hardware gate, requested-vs-available capability gap, preserved provenance | Any physical performance or quality outcome |
| `blocked_on_artifacts` | Upstream code, checkpoints, reference implementation, or required license terms are not publicly available | Exact artifact/license gate and pinned observation state | Any reproduction or refutation of upstream claims |

## Non-collapsing outcome taxonomy

Outcomes must never be silently collapsed into zero, omitted from denominators, or rewritten by soft scores:

- `accepted`: passed every required independent consumer/numerical/parity check within budget.
- `invalid`: well-formed output rejected by parse, type, effect, contract, numerical, or parity checks.
- `unsupported`: requested capability, collector, backend, or operation is not supported by the target.
- `exhausted`: search, attempt, denoising, or token budget was exhausted before finding an accepted artifact.
- `timeout`: deadline exceeded; in-flight device buffers and KV state enter an explicit unknown/poisoned state until reset.
- `oom`: declared memory or KV-cache bound exceeded.
- `cancelled`: caller cancellation requested (distinct from device release observed).
- `internal_failure`: process crash, malformed transport, or invariant violation.
- `unavailable_hardware_runtime`: physical accelerator or serve host unavailable.
- `blocked_on_artifacts`: required upstream code, weights, reference, or license missing.

## Provenance and digest discipline

- Every external identity (checkpoint, effective weights, base artifact, diagnostic, kernel module, trace artifact, tokenizer, chat template) carries an explicit digest algorithm (`sha256`) alongside its hex digest.
- Scenario labels and package versions are never substitutes for content digests.
- Hardware/device identity, runtime/driver/compiler versions, precision/quantization, batch/sequence dimensions, warm/cold state, and collection settings are retained alongside raw results.
- Committed artifacts and logs must never contain `/home/` or machine-private user paths.

## Budget, timing, and causal-evidence rules

1. **Matched budget views**: Proposal and serving comparisons report both equal end-to-end budget and equal candidate/checker budget. Elapsed critical-path work is kept separate from summed resource work (generation, scoring, checking, retries, restore, loading, I/O) without double-counting overlap.
2. **Clock domains**: Timestamps from distinct clock domains (`cpu_ktime_ns`, `device_global_timer_ns`, `client_monotonic_ns`) may only be compared when an explicit calibrated `ClockMapping` with uncertainty bounds is present.
3. **Completeness before causality**: Dropped events, buffer overflows, excluded collectors, stale bindings, or late unresolved metadata block causal attribution (`incomplete_cannot_substantiate`). A zero count on an unsupported collector is `missing_data`, not zero events.
4. **Correctness before speed**: Candidates failing independent output verification or tokenizer/template parity are disqualified regardless of timing.
5. **No unverified targets**: Author-reported numbers from papers, blog posts, or X threads are logged strictly under `non_gating_unverified_author_claims` and never imported as acceptance thresholds.

## Artifact publication

Every CLI subcommand requires `--out NEW_DIRECTORY` where `NEW_DIRECTORY` does not yet exist and its parent directory already exists:

1. `inputs.json`: complete frozen inputs, bounds, and pinned contract metadata.
2. `results.json`: complete per-item and aggregate outcomes (never dropping failed/rejected/unfinished items).
3. `summary.md`: human-readable summary with explicit scope disclaimers and non-collapsing counts.
4. `COMPLETE`: empty zero-byte file written **last** after all three report files succeed.

Two consecutive runs of any deterministic harness must produce byte-identical `inputs.json`, `results.json`, `summary.md`, and `COMPLETE` files.
