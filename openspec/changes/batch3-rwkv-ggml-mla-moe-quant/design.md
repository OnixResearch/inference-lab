# Design: Batch 3 RWKV, djev, ggml, graphical sampling, Qwen3.8 quant, DeepSeek ref, MLA, and sparse MoE contracts

## Core architecture

- **Pure dependency-free core (`unsafe_code = "forbid"`)**: all ten new modules (`rsi_bench`, `djev_protocol`, `rwkv_leaf`, `ggml_ops`, `rwkv_ggml`, `graphical_sampling`, `qwen38_quant`, `deepseek_ref`, `mla_comparison`, `sparse_moe`) plus the orchestrator (`batch3`) use only Rust `std` and existing helpers in `src/json.rs` and `src/provenance.rs`. `Cargo.toml`, `Cargo.lock`, `flake.nix`, and `flake.lock` remain untouched.
- **Shared composition without duplication**:
  - `src/mla_comparison.rs` (`#20`) and `src/sparse_moe.rs` (`#21`) directly import frozen TINY DeepSeek parameters, RoPE/attention helpers, and corrected MoE reference semantics from `src/deepseek_ref.rs` (`#19`).
  - `src/rwkv_ggml.rs` (`#16`) directly compares its independent ggml-style FP32 `wkv7` implementation against the pure-Rust FP64 RWKV-7 reference recurrence in `src/rsi_bench.rs` (`#12`).
  - `src/qwen38_quant.rs` (`#18`) reuses `WorkloadIdentity` and metric rules from `src/serving_bench.rs` (`#7`) and speculative/MTP gating contracts from `src/drafting.rs` (`#3`).
- **CLI shell (`src/main.rs`)**:
  - Preserves all 18 existing Bootstrap, Batch 1, and Batch 2 subcommands unchanged.
  - Adds 10 per-issue subcommands (`rsi-bench`, `djev-protocol`, `rwkv-leaf`, `ggml-ops`, `rwkv-ggml`, `graphical-sampling`, `qwen38-quant`, `deepseek-ref`, `mla-comparison`, `sparse-moe`) and `batch3-suite`, each writing fresh directories only, SHA-256 `manifest.json` for `batch3-suite`, and empty `COMPLETE` markers last.

## Per-issue contract and numerical design

1. **Issue `#12` (`src/rsi_bench.rs`)**:
   - Models the 5-surface donor manifest at `OnixResearch/rsi-lab@01e47e268d71bc1c53cc36e11c179f2b8f22f5c5`, explicit RSI boundaries (`packing_flip_replay.py`, `gpu_memory_graph_probe.py`, `gpu_packed_read_probe.py`, stateful wrappers, closed `#18` receipts, open `#82` gate), and private-to-public publication allowlist.
   - Implements a pure-Rust FP64 RWKV-7 state-update recurrence (`rwkv7_step_f64`, `rwkv7_forward_f64`) with invariance checks (token-by-token vs chunked output + final-state parity, zero-decay identity), RSI `#82` sweep qualification validator, encoder cold/warm/compile phase separation, and fake-quantization INT8-speedup rejection.
2. **Issue `#13` (`src/djev_protocol.rs`)**:
   - Implements canonical `djev.systemone.v1` request serialization/parsing with deterministic key order, UTF-8 / non-ASCII preservation, ordered candidate labels, and span byte-offset bounds checking.
   - Implements closed RSI `#42` effective cache-identity hashing (every semantic parameter change invalidates reuse; missing immutable model digest emits `non_reusable_run_namespace`), dual-consumer extraction gate, scoring-vs-generative/proof capability separation, and whole-decision deadline/cancellation/stale-result controls.
3. **Issue `#14` (`src/rwkv_leaf.rs`)**:
   - Explicitly reports `execution_class = blocked_on_artifacts` until a second concrete non-mock consumer demonstrates the leaf API (rejecting papers, planned benchmarks, and second mocks).
   - Implements a pure-Rust byte-prefix trie RWKV tokenizer (with negative whitespace-tokenizer divergence check), structural safetensors header admission validator (bounds, overlap, dtype, digest, immutability contract), reusable-workspace FP32/FP64 recurrent step verifying zero per-token reallocation, FP32-CPU vs FLA-BF16-CUDA non-interchangeability guard, and RSI wrapper preservation controls.
4. **Issue `#15` (`src/ggml_ops.rs`)**:
   - Pins `ggml-org/ggml@417a024b4fb6bc3116d41af7c067166a039bdf12` (`test-backend-ops.cpp`, `ggml-backend.h`) with four independent qualification axes (`advertised_support`, `executed_success`, `numerical_pass`, `performance_measured`).
   - Implements pure-Rust bounded `F32`, `Q8_0` (32-element block, `f32` scale + `i8` quants), and `Q4_0` (32-element block, `f32` scale + 4-bit signed quants) reference dot/matmul kernels, explicit `MXFP4`/`NVFP4` unsupported handling, silent CPU fallback & non-finite disqualification, repeated-trial variance accounting, and `grants_placement_authority = false`.
5. **Issue `#16` (`src/rwkv_ggml.rs`)**:
   - Pins `ggml@417a024b` (`ggml.c#L5894-L5939`, `ggml-cpu/ops.cpp`, `ggml-cuda/wkv.cu`) and compares an independent FP32 ggml-style `wkv7` operator against `src/rsi_bench.rs`'s FP64 reference on frozen `[n_heads, head_dim, head_dim]` state layouts.
   - Tests token-at-a-time vs chunked parity, 64-step long-horizon drift checkpoints, binary state export/import continuation with SHA-256 receipt, negative controls (altered state, boundary shift, precision drift, backward/training unsupported, CUDA unavailable without fallback), and `asserts_agent_persistence_correctness = false`.
6. **Issue `#17` (`src/graphical_sampling.rs`)**:
   - Implements exact joint enumeration (`<= 256` states) over a frozen corpus of coupled/irregular positive-factor categorical unary+pairwise models (Compiler Lab `#222` `fd0bb432`).
   - Runs sequential Gibbs, serial color-schedule Gibbs, and color-phase Gibbs with deterministic per-(sweep, node) SplitMix64 RNG mapping (verifying exact trajectory reproduction when schedule + RNG match, and exact-oracle conditional/marginal/energy checks across different valid schedules).
   - Implements an IPS autocorrelation ESS estimator (`ess-ips-v1`) returning `None` on stuck/zero-variance/short chains, 8 semantic/statistical negative controls, and explicit non-gating AMARA & default-off CRAM (`Prefrontal-AI/CRAM@62bbe830`) separation.
7. **Issue `#18` (`src/qwen38_quant.rs`)**:
   - Integrates `#7` metric semantics and `#3` speculative/MTP rules with exact `64 KiB/token` calculated KV planning (`2 GiB` @ 32k, `16 GiB` @ 256k, labeled non-observed), unified-memory overlap guards, matched-target vs changed-quant separation, offline rescoring with exact denominators (`zero accepted -> cost_per_accepted_task = None`), and Animus hidden-reasoning containment.
   - Covers positive/negative conformance fixtures and explicitly models MACaQ (`3810efc`, PR `#4`/`#5`, creation vs serving cost, `max_num_seqs=1`), QSA (`#31`/`#32`), Tenstorrent P150x2 concurrency/contamination (`tt-stack@e55c6527`), default-off Monoglot vocabulary pruning (bit-identical non-vocab check), `blocked_on_artifacts` ternary `ttf`, QDA PTQ checkpoints with pure-Rust Spearman KL-vs-accepted-task rank correlation (aligned + inverted control), and out-of-scope `vegaml` (`rsi-lab#151`).
8. **Issue `#19` (`src/deepseek_ref.rs`)**:
   - Pins `Vaibhaav-Tiwari/deepseek-v3-jax@6901be60941b98dc654ed542465290db9e8bafa6` (MIT notice) and implements a pure-Rust TINY DeepSeek-V3 architectural reference (MLA + routed/shared MoE + MTP) with explicit FP64 master weights and BF16 cast policy.
   - Reproduces and verifies fixes for all 5 upstream defects: Eq. 18 raw-affinity sequence loss vs bias-adjusted dispatch through full `moe()` caller using the exact `logit([0.9, 0.8, 0.2, 0.1])` fixture, `n_mtp = 0` MTP-off, `n_dense == n_layers` all-dense, independent eval/log intervals, and inclusive `get_batch` final window + short-data rejection.
   - Verifies full vs chunked vs cached decode parity, finite-difference gradients, non-finite rejection, negative controls (wrong mask, altered gates, shifted MTP target), and deterministic checkpoint round-trip.
9. **Issue `#20` (`src/mla_comparison.rs`)**:
   - Compares full uncached baseline, compressed-cache explicit expansion, and absorbed MLA under identical `#19` parameters across `batch = 2`, multiple heads, irregular chunks, near-capacity/overflow, and cache reset/reuse.
   - Records lowered-op intermediate counts proving absorbed MLA materializes 0 per-head K/V cache expansion intermediates, verifies unused-slot `1e6` sentinel poisoning causes zero output drift, compares full-allocated vs valid-prefix vs bucketed-prefix recompilation accounting, verifies 144-vs-480 element accounting without claiming speedup, checks finite-difference gradients, and rejects 7 negative controls with first divergent position.
10. **Issue `#21` (`src/sparse_moe.rs`)**:
    - Compares dense reference, independent tiny per-token/per-expert scalar reference, and sparse MoE dispatch under identical `#19` parameters with deterministic tie-breaking, bias-membership-only / raw-affinity-weighting / single `route_scale` / single shared-expert semantics, non-dropping capacity accounting, and assignment-derived load counts immune to BF16 gate underflow.
    - Implements full analytical backward passes (`dX`, `dW_router`, `dW_shared`, `dW_experts`) for dense and sparse paths, verifies dense-vs-sparse and finite-difference gradient parity away from ties, asserts declared tie behavior at discontinuities, compares a 3-step matched update sequence (including bias updates), sweeps `(tokens, experts, top_k, hidden, skew)` reporting actual executed pairs vs dense/padded pairs, and catches all 7 adversarial negative controls.
