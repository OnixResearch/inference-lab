# Implementation tasks: Batch 3 (`#12`–`#21`)

- [x] Write OpenSpec proposal, design, tasks, and requirement spec under `openspec/changes/batch3-rwkv-ggml-mla-moe-quant/`.
- [x] Implement Issue `#12` standalone RSI inference benchmark consolidation manifest, RSI `#82` sweep repair gate, pure-Rust RWKV-7 recurrence invariance & fake-quant controls (`src/rsi_bench.rs`).
- [x] Implement Issue `#13` reusable `djev` protocol client split, canonical wire-byte serializer/parser, closed RSI `#42` cache-identity regressions, capability separation, and deadline/cancellation controls (`src/djev_protocol.rs`).
- [x] Implement Issue `#14` conditional native RWKV numerical leaf gate (`blocked_on_artifacts`), byte-prefix trie tokenizer, structural safetensors admission validator, zero-reallocation reusable-workspace recurrent step, and RSI wrapper controls (`src/rwkv_leaf.rs`).
- [x] Implement Issue `#15` pinned `ggml` `test-backend-ops` qualification harness with 4 independent axes, pure-Rust `F32`/`Q8_0`/`Q4_0` reference kernels, explicit `MXFP4`/`NVFP4` unsupported handling, and silent-fallback disqualification (`src/ggml_ops.rs`).
- [x] Implement Issue `#16` `ggml` RWKV-7 recurrence and state-continuation parity comparing independent FP64 and FP32-ggml-style implementations across token/chunked, 64-step drift, state export/import round-trips, and negative controls (`src/rwkv_ggml.rs`).
- [x] Implement Issue `#17` discrete graphical sampling benchmark with exact enumeration oracle, sequential/serial-color/color-phase CPU Gibbs samplers, deterministic RNG mapping, honest IPS ESS estimator, 8 controls, and non-gating AMARA/CRAM separation (`src/graphical_sampling.rs`).
- [x] Implement Issue `#18` Qwen3.8-27B quantization & serving evaluation profile reusing `#7` and `#3`, `64 KiB/token` KV planning, unified-memory guards, conformance fixtures, MACaQ/QSA/P150x2/Monoglot/ttf/QDA Spearman KL correlation/vegaml modeling (`src/qwen38_quant.rs`).
- [x] Implement Issue `#19` corrected pinned DeepSeek-style reference (`6901be60`, MIT) with pure-Rust TINY MLA + MoE + MTP, all 5 upstream defect reproductions/fixes, parity, gradient, non-finite, and checkpoint regressions (`src/deepseek_ref.rs`).
- [x] Implement Issue `#20` explicit vs absorbed MLA comparison reusing `#19`, lowered-op zero-K/V-expansion evidence, `batch > 1` & sentinel poisoning (`1e6`), exact persistent/workspace/peak byte accounting, gradient parity, and 7 negative controls (`src/mla_comparison.rs`).
- [x] Implement Issue `#21` sparse MoE dispatch vs dense & tiny references reusing `#19`, full analytical backward pass + finite-difference & 3-step update parity, BF16-underflow-immune load counts, capacity/padding/fallback reporting, and 7 negative controls (`src/sparse_moe.rs`).
- [x] Implement `src/batch3.rs` orchestrator and extend `src/lib.rs` and `src/main.rs` with per-issue CLI subcommands and `batch3-suite`.
- [x] Add comprehensive adversarial unit and integration tests covering every Batch 3 contract invariant and negative control while preserving all 34 prior tests.
- [x] Update `README.md`, `docs/architecture.md`, `docs/roadmap.md`, and `docs/source-review.md` to document Batch 3 honestly.
- [x] Generate SHA-256-verified retained `evidence/batch3/` (`manifest.json`, `README.md`, per-issue subdirs, empty `COMPLETE`) and verify 2-run byte-identical output matching `evidence/batch3/`.
- [x] Validate `cargo fmt --all --check`, `cargo clippy --all-targets --locked -- -D warnings`, `cargo test --locked --all-targets`, `cargo test --locked --doc`, and `nix flake check --no-update-lock-file --option builders ''`.
- [x] Complete adversarial audit by `@inf-lab-b3-verifier` before committing and pushing.
