# Implementation tasks: Batch 2 (`#2`–`#11`)

- [x] Write OpenSpec proposal, design, tasks, and requirement spec under `openspec/changes/batch2-drafting-serving-lifecycle/`.
- [x] Implement Issue `#7` portable serving benchmark compatibility contract, offline reducer, branch-aware agent trace replay, Eunomia transport controls, and OASYS RLM trace profile (`src/serving_bench.rs`).
- [x] Implement Issue `#2` masked proof completion contract, frozen-context guards, task-family split discipline, intake equivalence, matched budgets, and offline replay (`src/proof_completion.rs`).
- [x] Implement Issue `#3` diffusion drafting contract with greedy conformance, analytical stochastic rejection-sampling verifier, silent-fallback guard, and pinned non-gating external claim records (`src/drafting.rs`).
- [x] Implement Issue `#6` adaptive speculative graph & concurrency characterization reusing `#3` and `#7` (`src/adaptive_spec.rs`).
- [x] Implement Issue `#4` reference-assisted vs spec-only agent kernel optimization contract with strict gate ordering, soak stage, local noise floor, separate factors, and retained submission log (`src/kernel_opt.rs`).
- [x] Implement Issue `#5` session KV routing, bounded soft affinity with load escape, host-RDMA migration vs recomputation, RSI `#86` variant cache identity, and TT-to-NVIDIA hybrid-state compatibility (`src/kv_routing.rs`).
- [x] Implement Issue `#8` DeepSeek/Spark migration manifest, repaired metrics accounting, burst/sustained & cache controls, 9 device-free fixtures, and semantic vs legacy quality checks (`src/deepseek_spark.rs`).
- [x] Implement Issue `#9` learned-compiler asset recovery inventory (`blocked_on_artifacts`), preserved `#30`/`#31` baselines, Darwin/AppleContainer isolation gate, and model-free integration controls (`src/learned_compiler.rs`).
- [x] Implement Issue `#10` conditional structured-read extraction gate (`blocked_on_artifacts`), capability matrix, explicit normalization semantics, calibration invalidation, 3 operation-separated positive fixtures, and 10 adversarial negative fixtures (`src/structured_read.rs`).
- [x] Implement Issue `#11` centralized model-lifecycle protocol (`A -> B -> A` and RSI `#86` `parent -> variant -> parent`), Systems `#10` raw vs exclusive elapsed-time partition, unified-memory provenance guards, 13 lifecycle/Animus `#17` controls, and residency-policy comparison (`src/lifecycle.rs`).
- [x] Implement `src/batch2.rs` orchestrator and extend `src/lib.rs` and `src/main.rs` with per-issue CLI subcommands and `batch2-suite`.
- [x] Add comprehensive adversarial unit and integration tests covering every Batch 2 contract invariant and negative control while preserving all 22 prior tests.
- [x] Update `README.md`, `docs/architecture.md`, `docs/roadmap.md`, and `docs/source-review.md` to document Batch 2 honestly.
- [x] Generate SHA-256-verified retained `evidence/batch2/` (`manifest.json`, `README.md`, per-issue subdirs, empty `COMPLETE`) and verify 2-run byte-identical output matching `evidence/batch2/`.
- [x] Validate `cargo fmt --all --check`, `cargo clippy --all-targets --locked -- -D warnings`, `cargo test --locked --all-targets`, `cargo test --locked --doc`, and `nix flake check --no-update-lock-file --option builders ''`.
- [x] Complete adversarial audit by `@inf-lab-b2-verifier` before committing and pushing.
