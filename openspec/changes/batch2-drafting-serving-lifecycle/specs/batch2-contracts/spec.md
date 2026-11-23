# Specification: Batch 2 drafting, serving, kernel-opt, KV-routing, recovery, and lifecycle contracts

## Scenario 1: Batch 2 CLI publication, SHA-256 manifest, and backward preservation
- All 7 Batch 1/bootstrap CLI subcommands, `evidence/bootstrap/`, `evidence/batch1/`, `Cargo.toml`, `Cargo.lock`, `flake.nix`, `flake.lock`, and all 22 prior tests remain intact and passing.
- Each of the 10 new per-issue subcommands (`proof-completion`, `drafting`, `kernel-opt`, `kv-routing`, `adaptive-spec`, `serving-bench`, `deepseek-spark`, `learned-compiler`, `structured-read`, `lifecycle`) writes `inputs.json`, `results.json`, `summary.md`, and empty `COMPLETE` last into a fresh `--out NEW_DIRECTORY` and refuses overwrites.
- `batch2-suite --out NEW_DIRECTORY` publishes all 10 per-issue subdirectories (each ending with empty `COMPLETE`), top-level `inputs.json`, `results.json`, `summary.md`, `manifest.json` (with exact SHA-256 digests of every emitted file), and empty `COMPLETE` last.
- Two consecutive runs of `batch2-suite` produce byte-identical outputs across every emitted file matching retained `evidence/batch2/` with zero `/home/` paths.

## Scenario 2: Issue #2 — Masked proof completion through Compiler Lab's frozen gate
- Frozen-context guards reject executable-body edits, weakened statements/contracts, disallowed imports or axioms (`sorry`/`admit`/`axiom`), stale task/profile identities, unreplaced `leansearch?` placeholders, and mixed Lean+Verus cohorts.
- Task-family partition validation detects and rejects related proof-corruption leakage across `train_adaptation`, `validation`, and `held_out` splits.
- Generator-neutral (`compiler-lab#75`) and temporary saved-result adapters agree identically on acceptance/refusal; producer confidence, REPL `Completed`, and exit status 0 cannot override independent fresh recheck refusal.
- Repair attempts, retries, retrieval, tactic search, and checker calls count toward total arm budgets; optional Python is disabled in the common-tool cohort; offline replay reproduces exact verdicts without model calls.

## Scenario 3: Issue #3 — Diffusion drafting conformance and target-model parity
- Greedy conformance retains exact first mismatch offset and token/logit margin without hiding or filtering failures.
- Stochastic speculative rejection sampling verifies exact analytical marginal equivalence (`max_abs_err <= 1e-12`) against target distribution `p(x)` across draft blocks, residual correction `(p - q)_+ / Z`, and bonus tokens; a faulty rejection sampler without residual correction fails parity.
- Silent fallback (`draft_assisted = true` with zero actual draft steps or 100% fallback) is explicitly disqualified; approximate quantization comparisons remain separate from same-target lossless claims; downstream accepted-task checks gate wins; external DFlash2 (6.6x) and DNeo (1.3x / 701 tok/s) claims stay strictly non-gating.

## Scenario 4: Issue #4 — Reference-assisted vs spec-only agent kernel optimization
- Independent evaluator enforces strict gate ordering before timing: integrity/write-isolation -> FP64-vs-BF16 relative tolerance + hidden prompts + batch-reorder bit-identity + Cachet schedule budget -> soak stage -> local repeated-baseline noise floor (no foreign B200 roofline reuse).
- Deliberately wrong-but-fast, batch-reorder-divergent, overbudget-schedule, soak-failing, forged-timing, and within-noise-floor candidates are all rejected and retained in the submission log with exact denominators.
- Experimental factors (`spec_only`, `reference_assisted`, `reference_plus_profiler`, `compileiq_acf`, `reviewer_guided_continuation` preserving RSI `#26` NULL confirmation, `#38` persistent evaluator, and default-off `fresh_agent_rediscovery` with zero prior notes/traces/incumbent source) remain strictly separated; reviewer prose cannot override failed checks.

## Scenario 5: Issue #5 — Session KV routing, bounded soft affinity, and host-RDMA migration
- Snapshot compatibility enforces exact base, effective-weight, adapter, tokenizer, template, precision, kernel-numerics, KV-layout, sequence-capacity, runtime-image, GPU-arch, and state-kind equality plus deployment-generation fencing.
- Same-config/different-weights variants (RSI `#86`), adapter replacements, stale generations, cross-arch (`sm_100a` vs `sm_121` vs `sm_120`), and translated-vs-exact hybrid-state parity failures (`tt-onix#46`) are rejected; quantization-collapse equality is preserved as an explicit observation; cross-variant fallback never credits the intended variant.
- Transfer-free bounded soft affinity checks compatibility, readiness, and admission before affinity, demonstrates wait-bounded load escape, and separates pre-dispatch reconsideration from post-dispatch uncertainty.
- Three-policy comparison (sticky, host-RDMA migration, recompute) exposes full staging/transfer/restore breakdown and ChaosControl fencing/transfer fault outcomes.

## Scenario 6: Issue #6 — Adaptive speculative graph and concurrency characterization
- Reuses `#3`'s verification core and `#7`'s client timing definitions without duplicating benchmark infrastructure.
- Compares fixed vs validation-frozen adaptive verification lengths, records per-position acceptance, draft/verify latency, CUDA-graph capture/replay overhead, and tiny-MTP rollback controls (`#22`).
- Explicitly distinguishes single-active-generation CPU preparation overlap from true continuous batching via a negative control, and rejects B300/GB300 `sm_100a`/`sm_103a` extrapolation to GB10/RTX 5080.

## Scenario 7: Issue #7 — Portable serving benchmark extraction and trace profiles
- Preserves `tt-onix@5ed5e90c` (`LGPL-2.1-only`, `onix.qwen38-benchmark.v1`) provenance and explicit metric-meaning disclaimers (`ttft`, per-call mean `itl`, frozen agentic replay, client prefix overlap != server hit, constant-watt != measured energy).
- Offline reducer verifies retained positive artifacts and rejects tampered summaries, missing runs, missing usage tokens, identity mismatches, and unredacted credentials.
- Branch-aware agent trace replay passes valid serial/two-branch traces and catches all 6 negative controls (randomized shared prefix, retained thinking tokens, skipped tool delay, serialized parallel branches, early join, missing/duplicate events).
- Eunomia controls preserve negative BPF evidence, `Connection: close` baseline, and non-overlapping per-process vs whole-host CPU; OASYS RLM trace profile compares `appended_child_results` vs `externally_retained_results`, charges root+leaf together, and catches all 5 negative controls.

## Scenario 8: Issue #8 — DeepSeek/Spark migration into shared serving runner
- Migration manifest binds `dsv41-1017@7f466a12` (MIT notice, private-vs-public selection, retractions preserved, deployment scripts excluded) and reuses `#7`'s workload identity.
- Repaired counter accounting returns `unavailable` (never 0) for missing counters, requires documented label scope for multi-series metrics, detects counter resets, and refuses token-throughput claims when usage is missing.
- All 9 device-free fixtures pass; burst (`synchronized_stream_wave`) vs sustained workloads cannot compare as identical; cache state distinguishes `unknown`, `requested_unique_prefix`, `operator_declared`, and `server_counter_observed`; semantic code checker rejects an incorrect candidate that passes the legacy substring check.

## Scenario 9: Issue #9 — Learned-compiler asset recovery with preserved baselines
- Reports `execution_class = blocked_on_artifacts` with an explicit recovery inventory documenting unlocated `research/learned-compiler` and `acceptance.json` at `compiler-lab@50dcbdce`, separating publishable contracts from restricted private donor assets.
- Preserves Qwen `#30` and DiffusionGemma `#31` historical claims separately from replay outcomes and new runs; enforces Linux host Darwin/AppleContainer isolation gate (`unavailable_hardware_runtime`) without weakening isolation; verifies ABI/identity, feedback/final separation, attempt budget, and no-teacher-replacement controls.

## Scenario 10: Issue #10 — Conditional shared structured-read extraction gate
- Reports `execution_class = blocked_on_artifacts` until two distinct real backends and a second concrete consumer are qualified; explicitly rejects same-adapter URLs, mocks-only, and chat-only models as gate evidence.
- Preserves `ar_label_read`, `diffusion_canvas_read`, and `clef_joint_head_read` capability separation (no silent chat degradation), explicit `permitted_labels_only` vs `full_vocabulary` normalization with retained label mass, calibration invalidation on identity/budget change, 3 operation-separated positive fixtures, and 10 adversarial negative fixtures.

## Scenario 11: Issue #11 — Centralized model-lifecycle measurements with Spark adapter contract
- Models versioned disposable Spark NixOS adapter (`models.nix@1abf9f9e`, `autoStart=false`, `register=false`), single-active-engine `A -> B -> A` and RSI `#86` `parent -> variant -> parent` transitions, and first-independently-checked-completion readiness.
- Preserves Systems `#10` raw overlapping phase timings separately from exact exclusive elapsed-time partitions (`sum(exclusive) + unknown == wall_ms`) and enforces unified-memory counter provenance/overlap guards (never summing overlapping counters or reporting offload as freed physical capacity).
- Passes all 13 lifecycle/Animus `#17` controls (including delayed readiness, false-ready model listing, stale post-shutdown data, late previous-generation completion, unsupported native sleep not falling back to restart, and foreign-resource cleanup refusal) and compares residency policies requiring adapter-evidenced physical release.
