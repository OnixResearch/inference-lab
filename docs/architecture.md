# Architecture and boundaries

## Pure core and explicit shell

Inference Lab separates pure deterministic evaluation from all external effects:

- **Pure core (`src/lib.rs` and submodules)**: zero external crate dependencies, `unsafe_code = "forbid"`, no filesystem I/O, no system clock reads, no network sockets, no process spawning, and no device probing. Given explicit inputs, every module produces deterministic structured outputs and SHA-256 content digests.
- **Explicit imperative shell (`src/main.rs`)**: parses CLI flags, invokes the selected bounded harness, serializes deterministic JSON and Markdown reports, creates a brand-new output directory (refusing to overwrite existing paths), writes `inputs.json`, `results.json`, `summary.md`, and publishes an empty `COMPLETE` marker last.

## Module structure

| Module | Issue(s) | Role |
| --- | --- | --- |
| `src/json.rs` | Shared | Dependency-free deterministic JSON writer and pure-Rust SHA-256 implementation |
| `src/provenance.rs` | Shared | Execution classes (`bounded_contract_harness`, `unavailable_hardware_runtime`, `blocked_on_artifacts`), algorithm-tagged digests, and non-collapsing outcome enums |
| `src/seed.rs` | Bootstrap | Deterministic serving/KV-budget contract seed (`serving-contract-v1`) |
| `src/diffusion.rs` | `#1` | Versioned Noble proposal/repair contract, sampler capability probes, EBM proposal/scoring/selection/admission separation, offline pool replay, matched budgets, and default-off `DependentWeave`/JSONL/socketteer interchange check |
| `src/trace.rs` | `#40` | Host/GPU trace-correlation schema, clock-domain mapping guard, batched-kernel non-attribution, event-loss completeness checks, and negative controls |
| `src/stage.rs` | `#41` | Versioned transformer stage contract, tensor envelope validation, real pure-Rust 4-layer dense causal transformer split/unsplit numerical reference, poisoned-buffer/stale-KV guards, and explicit hardware-unavailable enforcement |
| `src/power.rs` | `#43` | Power and throttle cadence qualification schema, device-vs-host axis separation, sensor-gated energy reporting, observer perturbation, and paired AB/BA protocol controls |
| `src/tokenizer.rs` | `#52` | Deterministic BPE + special-token reference tokenizer, chat-template renderer (`chatml-v1`, `qwen-thinking-v1`), role-boundary injection checks, and disqualifying backend parity harness |
| `src/related_work.rs` | `#56`, `#58`, `#60` | Pinned default-off related-work gate contracts and validator ensuring unverified external claims never become acceptance targets |
| `src/proof_completion.rs` | `#2` | Masked proof completion contract as second consumer of `#1`, frozen-context guards, partition discipline, intake equivalence, matched budgets, and offline replay |
| `src/drafting.rs` | `#3` | Greedy conformance, exact analytical stochastic rejection-sampling verifier, block verification, silent-fallback guard, and pinned non-gating DFlash2/DNeo claims |
| `src/kernel_opt.rs` | `#4` | Independent kernel-opt evaluator gate ordering, soak stage, local repeat noise floor, separate experimental factors, retained `runs/` log, and license provenance |
| `src/kv_routing.rs` | `#5` | 13-dimension snapshot compatibility & generation fencing, RSI `#86` & TT `#46` guards, soft-affinity load escape, sticky/RDMA/recompute comparison, and ChaosControl transfer faults |
| `src/adaptive_spec.rs` | `#6` | Adaptive speculative graph & concurrency characterization reusing `#3` block verifier and `#7` workload identity, plus CPU-overlap vs continuous-batching guard |
| `src/serving_bench.rs` | `#7` | Portable `tt-onix` serving compatibility contract (`LGPL-2.1-only`), offline reducer & tamper verifier, branch-aware agent trace replay, Eunomia transport controls, and OASYS RLM trace profile |
| `src/deepseek_spark.rs` | `#8` | DeepSeek/Spark migration manifest (`dsv41-1017`), repaired counter accounting, burst/sustained & cache controls, 9 device-free fixtures, and semantic vs legacy quality checks |
| `src/learned_compiler.rs` | `#9` | `blocked_on_artifacts` asset recovery inventory (`compiler-lab@50dcbdce`), preserved `#30`/`#31` baselines, Darwin/AppleContainer isolation gate, and model-free controls |
| `src/structured_read.rs` | `#10` | Conditional `blocked_on_artifacts` extraction gate (2 distinct real backends + 2nd consumer), AR/diffusion/Clef capability separation, normalization scope, calibration invalidation, and adversarial fixtures |
| `src/lifecycle.rs` | `#11` | Disposable Spark NixOS adapter contract, `A -> B -> A` & RSI `#86` variant transitions, Systems `#10` raw vs exclusive elapsed-time partition, unified-memory provenance guards, Animus `#17` controls, and residency policies |
| `src/batch2.rs` | `#2`–`#11` | Deterministic Batch 2 suite orchestrator emitting per-issue bundles, aggregate reports, and SHA-256 `manifest.json` |

## Future real-backend adapter boundary

Physical model servers, GPU profilers, Apple Metal/Asahi stages, Tenstorrent P150 workers, TPU XProf collectors, and FPGA boards belong in explicit opt-in adapters outside the pure core. Any future physical adapter must:

1. Bind exact hardware, driver, runtime, compiler, model/checkpoint, effective-weight, tokenizer, and template revisions before running.
2. Preserve separate `bounded_contract_harness` and physical-run records; never silently fall back to CPU or mock responses when a hardware backend is requested.
3. Never start, stop, or evict resident model services or modify host configuration without an explicit experiment target.
4. Keep hard consumer admission (Noble, Compiler Lab, RSI Lab, numerical tolerance, tokenizer parity) outside soft scoring or format serialization.

## Consumer ownership boundaries

- **Noble** retains structured editor AST semantics, deterministic CEGIS baseline (`cegis.mjs`), proofs, and kernel acceptance.
- **Compiler Lab** retains native candidate checks, source-proof intake, compiler-input reduction, and Verilog/RTL admission (`compiler-lab#270`, `compiler-lab#275`).
- **TT-Onix** retains physical Tenstorrent runtime execution, device telemetry, and host qualification gates (`tt-onix#48`, `#49`, `#50`, `#51`).
- **RSI Lab** retains recurrent-state learning, retention, promotion, and exact-controller gates.
- **Systems Lab** retains cross-system queue composition and userspace/kernel observer-overhead comparisons (`systems-lab#19`).
- **Kiln / Lattice / Animus / Molten** retain global scheduling, workflow bindings, campaign exploration, and compute/storage allocation authority.

Shared library extraction across consumers remains gated on at least two real backends and two concrete consumers demonstrating compatible needs (tracked in `#10`).
