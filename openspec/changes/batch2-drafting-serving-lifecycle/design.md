# Design: Batch 2 drafting, serving, kernel-opt, KV-routing, recovery, and lifecycle contracts

## Core architecture

- **Pure dependency-free core (`unsafe_code = "forbid"`)**: all ten new modules (`proof_completion`, `drafting`, `kernel_opt`, `kv_routing`, `adaptive_spec`, `serving_bench`, `deepseek_spark`, `learned_compiler`, `structured_read`, `lifecycle`) plus the orchestrator (`batch2`) use only Rust `std` and existing helpers in `src/json.rs` and `src/provenance.rs`. `Cargo.toml`, `Cargo.lock`, `flake.nix`, and `flake.lock` remain untouched.
- **No duplicated infrastructure**:
  - `src/adaptive_spec.rs` (`#6`) directly imports and exercises the speculative draft/verify engine from `src/drafting.rs` (`#3`) and client metric semantics from `src/serving_bench.rs` (`#7`).
  - `src/deepseek_spark.rs` (`#8`), `src/kv_routing.rs` (`#5`), and `src/lifecycle.rs` (`#11`) reuse `WorkloadIdentity` and client serving semantics from `src/serving_bench.rs` (`#7`).
  - `src/proof_completion.rs` (`#2`) and `src/learned_compiler.rs` (`#9`) reuse digest/provenance/outcome structures from `src/provenance.rs` while keeping consumer checking ownership in Compiler Lab.
- **CLI shell (`src/main.rs`)**:
  - Preserves all 7 existing subcommands (`benchmark`, `diffusion`, `trace`, `stage`, `power`, `tokenizer`, `related-work`) unchanged.
  - Adds 10 per-issue subcommands (`proof-completion`, `drafting`, `kernel-opt`, `kv-routing`, `adaptive-spec`, `serving-bench`, `deepseek-spark`, `learned-compiler`, `structured-read`, `lifecycle`), each writing `inputs.json`, `results.json`, `summary.md`, and empty `COMPLETE` last into a fresh `--out NEW_DIRECTORY`.
  - Adds `batch2-suite --out NEW_DIRECTORY`, which creates `NEW_DIRECTORY`, publishes all 10 per-issue subdirectories (each with `inputs.json`, `results.json`, `summary.md`, empty `COMPLETE`), writes top-level `inputs.json`, `results.json`, `summary.md`, and `manifest.json` (recording exact SHA-256 digests of every emitted file across all subdirs and top-level reports), and writes empty `NEW_DIRECTORY/COMPLETE` last.

## Per-issue contract design

1. **Issue `#2` (`src/proof_completion.rs`)**:
   - Models frozen obligations with explicit editable byte spans, immutable prefix/statement/executable/contract/imports/assumptions, and task-family partition tags (`train_adaptation`, `validation`, `held_out`).
   - Enforces single checker family (`lean-4.35.0-pinned`, rejecting mixed Lean+Verus cohorts), frozen-context preservation, no `sorry`/`admit`/new axioms, no unreplaced `leansearch?` placeholders, generator-neutral (`#75`) vs temporary adapter equivalence, producer-confidence non-authority, matched arm budgets (including repairs, retries, retrieval, tactic search, and checker calls; optional Python disabled in common-tool cohort), and offline replay without model calls.
2. **Issue `#3` (`src/drafting.rs`)**:
   - Implements pure-Rust greedy token-by-token conformance checking (retaining exact first mismatch index and margin) and exact speculative stochastic rejection sampling (`min(1, p/q)`, normalized residual `(p - q)_+ / Z`, bonus token on all-accept) with analytical marginal distribution verification (`max_abs_err <= 1e-12`) plus a faulty-sampler negative control.
   - Enforces silent-fallback disqualification, approximate-vs-lossless separation, downstream accepted-task accounting, and pinned non-gating external claim records (JRB_AI DFlash2/TensorFold, DNeo `8449a3ce`).
3. **Issue `#4` (`src/kernel_opt.rs`)**:
   - Enforces strict candidate-independent gate ordering: write-isolation/integrity -> FP64-vs-BF16 relative tolerance + hidden prompts + batch-reorder bit-identity + Cachet schedule budget -> soak stage -> local repeated-baseline noise floor (forbidding B200 roofline reuse).
   - Separates `spec_only`, `reference_assisted`, `reference_plus_profiler`, `compileiq_acf`, `reviewer_guided_continuation` (preserving RSI `#26` NULL confirmation), `persistent_tiered_evaluator` (`#38`), and default-off `fresh_agent_rediscovery` (zero prior notes/traces/incumbent source). Retains all rejected attempts in a `runs/`-style submission log with explicit license provenance.
4. **Issue `#5` (`src/kv_routing.rs`)**:
   - Validates 13-field `SnapshotIdentity` (base digest, effective weights, adapter, perturbation recipe, tokenizer, template, precision, kernel numerics, KV layout, sequence capacity, runtime image, GPU arch, state kind) plus `deployment_generation` fencing.
   - Exercises same-config/different-weights (RSI `#86`), quantization-collapse observation, adapter swap, stale generation, cross-arch (`sm_100a` vs `sm_121` vs `sm_120`), translated vs exact hybrid-state (`tt-onix#46`), and cross-variant fallback attribution.
   - Simulates transfer-free soft-affinity with load escape (admission before affinity; pre-dispatch reconsideration vs post-dispatch uncertainty) and compares sticky routing, host-RDMA migration, and recomputation with ChaosControl fault scenarios.
5. **Issue `#6` (`src/adaptive_spec.rs`)**:
   - Reuses `#3`'s verification core to compare fixed block size vs validation-frozen adaptive confidence-threshold verification, tracking per-position acceptance, draft/verify latency, CUDA-graph capture/replay overhead, and `#22` tiny-MTP rollback controls (first/later rejection, all-accepted, bonus indexing, valid-prefix continuation).
   - Separates single-stream vs concurrent arrivals and enforces a negative control distinguishing CPU preparation overlap (1 active generation/replica) from true continuous batching.
6. **Issue `#7` (`src/serving_bench.rs`)**:
   - Encodes `tt-onix@5ed5e90c` (`LGPL-2.1-only`, `onix.qwen38-benchmark.v1`) provenance and explicit metric-meaning disclaimers, plus an offline artifact reducer & tamper checker (rejecting modified summaries, missing runs/usage, identity mismatches, credential leaks).
   - Validates three opt-in profiles: TensorRT-LLM branch-aware agent trace replay (with 6 negative controls), Eunomia queue/small-write transport controls (preserving negative BPF evidence, `Connection: close` baseline, and non-overlapping per-process vs whole-host CPU), and OASYS RLM recursive-call trace replay (`appended_child_results` vs `externally_retained_results`, root+leaf aggregate charging, 5 negative controls).
7. **Issue `#8` (`src/deepseek_spark.rs`)**:
   - Records `dsv41-1017@7f466a12` migration manifest (MIT notice, private-vs-public selection, retractions preserved, deployment scripts excluded), probed capabilities (`/tokenize`, `ignore_eos`, spec counters, cache telemetry), and repaired metrics parsing (missing -> `unavailable`, multi-series label scope required, counter reset detected).
   - Runs all 9 device-free fixtures, forbids burst-vs-sustained cross-comparison, separates unknown/declared/observed cache states, and proves a stronger semantic code checker rejects an incorrect candidate that passes the legacy substring suite.
8. **Issue `#9` (`src/learned_compiler.rs`)**:
   - Publishes an explicit `blocked_on_artifacts` recovery inventory documenting missing `research/learned-compiler` and `acceptance.json` at `compiler-lab@50dcbdce`, separating publishable contracts from private donor assets.
   - Keeps `historical_claim`, `current_replay_outcome`, and `newly_measured_result` distinct for Qwen `#30` and DiffusionGemma `#31`, enforces the Linux host Darwin/AppleContainer isolation gate without weakening isolation, and verifies frozen ABI/identity, feedback/final separation, attempt budget, and no-teacher-replacement controls.
9. **Issue `#10` (`src/structured_read.rs`)**:
   - Enforces the conditional two-real-backends + second-consumer extraction gate (`blocked_on_artifacts`), rejecting same-adapter URLs, mocks-only, and chat-only models.
   - Preserves semantic separation across `ar_label_read`, `diffusion_canvas_read`, and `clef_joint_head_read` (no silent chat degradation), explicit `permitted_labels_only` vs `full_vocabulary` normalization with retained label mass, calibration invalidation, 3 operation-separated positive fixtures, and 10 adversarial negative fixtures.
10. **Issue `#11` (`src/lifecycle.rs`)**:
    - Implements the versioned disposable Spark NixOS adapter contract (`models.nix@1abf9f9e`, `autoStart=false`, `register=false`), `A -> B -> A` and RSI `#86` `parent -> variant -> parent` sequences (1 active engine at a time), first-checked-completion readiness, Systems `#10` raw overlapping vs exclusive elapsed-time partition, unified-memory counter provenance/overlap guards, all 13 lifecycle/Animus `#17` controls, and residency-policy comparison on frozen agent demand with adapter-evidenced physical release.
