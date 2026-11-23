# Tasks: Batch 4 (`#22`–`#31`)

## OpenSpec & Hygiene
- [x] Add `openspec/changes/batch4-mtp-sparse-quant-placement-qsa/{proposal,design,tasks,specs/batch4-contracts/spec}.md` and register it in `openspec/README.md`.
- [x] Keep `Cargo.toml`, `Cargo.lock`, `flake.nix`, `flake.lock`, `evidence/{bootstrap,batch1,batch2,batch3}/`, and all 45 prior tests completely untouched and passing.

## Core Pure-Rust Modules (`unsafe_code = "forbid"`, zero external dependencies)
- [x] Implement `#22` (`src/mtp_decoder.rs`): tiny frozen target + depth-1 MTP draft-and-verify decoder, separate poisoned target/MTP caches, greedy parity, stochastic rejection sampling (exact small-vocab + empirical), and negative controls.
- [x] Implement `#23` (`src/flash_topk.rs`): independent FlashTopK reference scorer, SM90/SM100 vs SM120/SM121 & 32-head vs 64-head compatibility gates, canonical ties/padding, fused threshold+recovery CPU selector with fault-injected underflow/overflow tests, and negative controls.
- [x] Implement `#24` (`src/macaq_r.rs`): synthetic MoE K-quant GGUF container, real bounded coordinate refinement preserving scales/basis/out-of-scope tensors, artifact inspector, creation/resume receipts, calibration/held-out separation, offline paired rescoring, and negative controls.
- [x] Implement `#25` (`src/hetero_rollout.rs`): frozen-policy teacher-forced logprob/exact-KL qualification, versioned `http-lora` `tensors`-mode activation with probe-verified identity, 10-case fleet-reconciliation fault matrix, and bounded two-backend mixed-pool reducer.
- [x] Implement `#26` (`src/nvfp4_mxfp8.rs`): exact-bit NVFP4/MXFP8 quantizers with deterministic & stochastic rounding, 4 linear-layer surrogate arms with bias/variance/CI/covariance stats, Transformer Engine SM120 vs SM121 refusal matrix, Model Optimizer reconstruction controls, and optional Fabric `#6` Hadamard/S2F6 codec reference.
- [x] Implement `#27` (`src/spark_engines.rs`): Hyperflare & TensorRT Edge-LLM DGX Spark support/capability matrix, hybrid KV+recurrent continuation verifier (append/branch/rollback/cancel/pause vs full recomputation), `#11` lifecycle accounting, and negative controls.
- [x] Implement `#28` (`src/kv_compression.rs`): uncompressed, identity, and 2 lossy KVPress-style prefill compression arms, observed KV + temporary buffer byte accounting, 3-length frozen RULER-style diagnostics, evidence-grounded end-task checker, and negative controls.
- [x] Implement `#29` (`src/expert_placement.rs`): RustingStrata vs `llama.cpp` fixed/adaptive placement & host-page policy simulation, 5-layer warm-state isolation, routing-equivalence observability, counterbalanced paired reducer, custom license record, and negative controls.
- [x] Implement `#30` (`src/vision_infusion.rs`): 4 matched-visual-budget multimodal architectures (`encoder_free`, `separate_encoder`, `early_ffn_infusion`, `early_full_infusion`), single-vs-packed mixed-batch parity, strict modality/attention mask separation, text-only invariance, explicit `quality_unestablished`, and negative controls.
- [x] Implement `#31` (`src/qwen38_qsa.rs`): 4-token block-level QSA vs matched dense (`LLAMA_QSA_DENSE=1`) `2x2` matrix with MTP, complete memory accounting (retained target KV + indexer + GDN + MTP + workspace), 4 independent start/middle/end task checkers, offline rescorer, and negative controls.
- [x] Implement `src/batch4.rs` and wire all 10 per-issue CLI commands + `batch4-suite` in `src/lib.rs` and `src/main.rs`.

## Tests, Docs, Evidence & Validation
- [x] Add adversarial unit and integration tests for `#22`–`#31` and `batch4-suite` CLI/determinism/overwrite/leak guards.
- [x] Update `README.md` and generate retained SHA-256-verified evidence under `evidence/batch4/` (`manifest.json`, `README.md`, empty `COMPLETE` markers last).
- [x] Pass `cargo fmt --all --check`, `cargo clippy --all-targets --locked -- -D warnings`, `cargo test --locked --all-targets`, `cargo test --locked --doc`, `nix flake check --no-update-lock-file --option builders ''`, and 2-run byte-identical checks matching retained `evidence/batch4/` with zero `/home/` leaks.
