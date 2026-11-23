# Batch 6 Design

## Architectural invariants

Every Batch 6 module follows the pure-core / imperative-shell split established in Batches 1–5:

1. **Zero external dependencies and `#![forbid(unsafe_code)]`**: All numerical references, training loops, compression codecs, parsers, reducers, and validators reside in pure deterministic Rust 1.85.1 with no filesystem, network, clock, process, or accelerator I/O.
2. **Strict NaN & zero-divisor safety**: No float `.max()` or `.min()` calls anywhere in Batch 6 code. Every floating-point input, intermediate, and metric is checked with `!v.is_finite()` -> `Err(...)`, comparisons use explicit branches or finite-checked `total_cmp`, and every denominator is guarded against `0` (`Option<f64>` returning `None` when empty or zero accepted).
3. **Independent reference verification**: Every parity or round-trip claim executes two genuinely independent implementations:
   - `#47`: Independent exact slot-0 target forward rollout vs multi-slot speculative verifier with slot-index-dependent numerical rounding error, plus independent task-outcome verifier.
   - `#48`: Deterministic `nvme get-feature -f 8 -H` parser and event-level SSD-paged Engram/PLE decode latency reducer across `coalescing (off/on) x page_cache (cold/warm)`.
   - `#49`: Independent deterministic task-output verifier and n-gram/digest training-overlap contamination auditor across 4 arms.
   - `#50`: Miniature 2-layer causal sequence learner comparing `LatentFeedbackModel` vs matched-FLOP extra-epoch `StandardTransformerControl`, validated by independent central finite-difference gradient check (`<= 1e-6`).
   - `#51`: Multi-stream (`N = 4`) `DeepSeekMhc` (iterative Sinkhorn-Knopp doubly-stochastic projection) vs `SimpleHc` vs `BlockAttnResRef`, with independent scalar reference forward pass (`<= 1e-12`) and exact FLOP accounting.
   - `#53`: Measured-bandwidth/FLOP ceiling calculator, `Control-Candidate-Control` interleaved median evaluator (`>= 6` samples, `<= 3%` regression bound), immutable failure-retaining ledger, token-accounting/unified-memory/needle/soak/greedy/clean-rebuild gates.
   - `#54`: Pure-Rust fixed-point online-updating byte-level arithmetic/range codec initialized from a serialized quantized offline prior (decoder repeats online weight updates with zero transferred online weights) verified for bit-exact SHA-256 round-trip alongside a pure-Rust LZ77 conventional baseline on a pinned text prefix.
   - `#55`: Parallel slot-autoregressive text/number generator + typed-head extractor vs structured-output baseline checked by independent ground-truth value agreement and UTF-8 character-span grounding.
   - `#57`: Exact 24-stored-layer (middle 12 looped 3× = 48 applications, 8-of-128 MoE routing, NVFP4/MXFP8 config) `forward_unrolled_48_reference` vs `forward_looped_3x_shared` (`<= 1e-12`) plus optional `varloop` (`loop_count = 1..=4`) and admitted host/runtime qualification.
   - `#59`: Independent scalar nibble-unpack FP64 expert GEMV vs packed 32-bit word register-unpack FP16/FP32 expert GEMV (`<= 1e-12` exact parity + dynamically computed FP16 error and full-vocab KL divergence) plus Linux + 24 GB `sm_86` + CUDA 12.8 host gate.
   - `#61`: Independent multi-launch per-operator decode reference vs fused persistent-kernel decode loop enforcing token & logit equivalence (`<= 1e-12`) BEFORE any timing/speedup is permitted, batch scaling (`batch = 1, 4, 8`), latency percentiles, host-sync reduction, and agent kernel-build ledger.
4. **Honest hardware/artifact separation & non-gating upstream pins**:
   - All modules emit `synthetic_contract_fixture: true`, `execution_class: "bounded_contract_harness"`, and explicit `hardware_gate` (`unavailable_hardware_runtime` for `#47`, `#48`, `#50`, `#53`, `#54`, `#55`, `#57`, `#59`, `#61`; `blocked_on_artifacts` for `#49`, `#51`).
   - All default-off campaigns (`#53`, `#54`, `#55`, `#57`, `#59`) emit `default_enabled: false`.
   - Zero upstream benchmark numbers, blog claims, or X thread figures are imported as lab pass/fail criteria.
5. **Deterministic publication**: `src/batch6.rs` aggregates all 11 suites into a fresh directory with per-issue `inputs.json`, `results.json`, `summary.md`, 0-byte `COMPLETE` markers, top-level aggregate files, and SHA-256 `manifest.json`, with zero `/home/` path leaks and two-run byte-identical determinism.
