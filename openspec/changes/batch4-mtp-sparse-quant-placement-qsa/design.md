# Design: Batch 4 MTP, sparse-indexer, quant, placement, vision in-fusion, and QSA contracts

## Architecture & Purity Boundaries

All ten Batch 4 modules live inside the pure deterministic Rust library crate (`unsafe_code = "forbid"`, zero external dependencies; `Cargo.toml`, `Cargo.lock`, `flake.nix`, and `flake.lock` remain unchanged):

| Issue | Module | CLI subcommand | Subdir under `evidence/batch4/` | Execution class | Physical / artifact gate |
| ---: | --- | --- | --- | --- | --- |
| `#22` | `src/mtp_decoder.rs` | `mtp-decoder` | `issue-22-mtp-decoder` | `bounded_contract_harness` | `unavailable_hardware_runtime` (GPU/JAX serving-scale MTP) |
| `#23` | `src/flash_topk.rs` | `flash-topk` | `issue-23-flash-topk` | `bounded_contract_harness` | `unavailable_hardware_runtime` (SM90/SM100 GPU; SM120/SM121 `unsupported`) |
| `#24` | `src/macaq_r.rs` | `macaq-r` | `issue-24-macaq-r` | `bounded_contract_harness` | `unavailable_hardware_runtime` (22.36 GB Qwen3.6-35B-A3B creation + `llama.cpp` serve) |
| `#25` | `src/hetero_rollout.rs` | `hetero-rollout` | `issue-25-hetero-rollout` | `bounded_contract_harness` | `unavailable_hardware_runtime` (physical two-backend heterogeneous rollout pool) |
| `#26` | `src/nvfp4_mxfp8.rs` | `nvfp4-mxfp8` | `issue-26-nvfp4-mxfp8` | `bounded_contract_harness` | `unavailable_hardware_runtime` (RTX 5080 SM120 / Spark SM121 native CUDA/TE/CUTLASS) |
| `#27` | `src/spark_engines.rs` | `spark-engines` | `issue-27-spark-engines` | `bounded_contract_harness` | `unavailable_hardware_runtime` (DGX Spark GB10 aarch64 SM121 Hyperflare / Edge-LLM) |
| `#28` | `src/kv_compression.rs` | `kv-compression` | `issue-28-kv-compression` | `bounded_contract_harness` | `unavailable_hardware_runtime` (physical GPU Transformers/KVPress long-context run) |
| `#29` | `src/expert_placement.rs` | `expert-placement` | `issue-29-expert-placement` | `bounded_contract_harness` | `unavailable_hardware_runtime` (discrete NVIDIA RTX 5080 RustingStrata vs `llama.cpp`) |
| `#30` | `src/vision_infusion.rs` | `vision-infusion` | `issue-30-vision-infusion` | `bounded_contract_harness` | `unavailable_hardware_runtime` + `blocked_on_artifacts` (trained matched-capacity variants) |
| `#31` | `src/qwen38_qsa.rs` | `qwen38-qsa` | `issue-31-qwen38-qsa` | `bounded_contract_harness` | `unavailable_hardware_runtime` (physical Q8_0 27B QSA vs dense `llama.cpp` serve) |

`src/batch4.rs` aggregates all 10 suites into a single `Batch4Bundle` with exact FIPS-180-4 SHA-256 `manifest.json` digests, and `src/main.rs` remains the sole imperative shell writing to a fresh `--out NEW_DIRECTORY` and emitting empty `COMPLETE` markers last.

## Cross-Cutting Numerical & Contract Invariants

1. **Zero NaN-swallowing comparisons**: Every max-error helper, norm reducer, argmax, softmax, top-k selector, and numerical gate explicitly checks `!v.is_finite()` and returns `Err(...)` or `OutcomeStatus::Invalid` rather than swallowing `NaN` through `f64::max`.
2. **Real executable negative controls**: Every negative control mutates actual inputs, caches, weights, masks, scales, payloads, or state transitions and runs through the real validator/kernel to prove rejection (zero hardcoded `handled_honestly: true` booleans).
3. **Exact non-collapsing denominators**: All planned items (accepted, invalid, unsupported, exhausted, timeout, oom, cancelled, internal_failure) remain in population denominators; `cost_per_accepted_task` is `None` (`null` in JSON) whenever `accepted == 0`.
4. **Honest upstream boundaries**: Every pinned upstream repository, commit SHA, paper, and discovery post is retained as non-gating provenance; hardware refusals (`FlashTopK` SM120/SM121, Transformer Engine `_compute_mxfp8_support()` CC >= 12.0, Edge-LLM TP2 context reuse, Spark ARM64 vs x86-64/sm70 binary) stay explicit `unsupported` results and cannot be bypassed or silently fallen back to BF16/CPU.
5. **Path & publication hygiene**: Zero `/home/` or user-specific paths appear anywhere in code, tests, docs, or emitted JSON/Markdown; every `--out` run refuses existing directories and writes empty `COMPLETE` last.

## Per-Module Key Designs

### `#22` (`src/mtp_decoder.rs`)
- Builds a frozen tiny causal target model (reusing `deepseek_ref::det_weight`, `apply_rope`, `matvec`, `softmax_stable`, and frozen MoE routing biases) plus a depth-1 MTP module taking `(h_t, embed(token_{t+1}))` strictly causally.
- Maintains separate `TargetKvCache` and `MtpKvCache` buffers with explicit `valid_len`; on rollback or unused tail slots, writes `1e6` sentinels into `valid_len..capacity` so any accidental tail read produces immediate logit divergence.
- Implements both greedy acceptance (verifying full token + next-step logit + valid-cache agreement with full recomputation across mixed accept/reject, first-pos reject, later-pos reject, all-accept + bonus, zero-accept, EOS/budget/context bounds, cancel/continue, and multi-request isolation) and stochastic rejection sampling (`min(1, p(x)/q(x))` + normalized `max(p - q, 0)` residual + bonus token) with deterministic uniform draw injection covering all branches, exact small-vocab marginal verification (`< 1e-12`), calibrated finite-sample empirical check, and 6+ executable negative controls.

### `#23` (`src/flash_topk.rs`)
- Implements `score(t, s) = scale(s) * sum_h weights(t, h) * ReLU(dot(q(t, h), key(s)))` independently in FP64 with exact `[ks, ke)` windowing, canonical descending-score / ascending-key-index tie ordering (handling `+0.0` vs `-0.0`), and short-window sentinel padding.
- Implements a fused threshold-and-recovery CPU selector that estimates a threshold from sampled keys, prunes candidates with an optional bound, and recovers rows on candidate underflow (`< k`) or capacity overflow (`> cap`), recording exact recovery counts and distinguishing unmodified runs from test-only fault-injected threshold hooks.
- Enforces `validate_flashtopk_contract` (CUDA CC major 9 or 10 only; shapes `[rows, 32, 128]` / `[keys, 128]`; FP8 E4M3 or BF16; `topk in [1, 4096]`; contiguous keys/windows; explicit refusal of SM120, SM121, 64-head DeepSeek-V3.2, paged KV, block-level QSA).

### `#24` (`src/macaq_r.rs`)
- Implements a synthetic MoE GGUF representation with `Q4_K`, `Q5_K`, `Q6_K` MoE projections (explicit block scales `d`/`dmin`, discrete codes, activation basis) plus `F16`/`Q8_0` out-of-scope tensors and explicit unsupported `IQ4_XS` / split-GGUF markers.
- Runs real bounded coordinate refinement on supported MoE weight codes against a frozen FP64 teacher over a calibration split while holding scales/basis/formats strictly constant and copying out-of-scope tensors unchanged.
- Verifies artifact invariants, creation-vs-serving memory/disk preflight, resume-receipt digest binding, calibration/validation/held-out disjointness, offline paired rescoring (`zero accepted -> None`), and 8 negative controls.

### `#25` (`src/hetero_rollout.rs`)
- Implements frozen-policy teacher-forced log-probability and exact full-vocab KL vs sampled-token residual qualification across Backend A, Backend B, and an independent reference scorer.
- Models versioned `http-lora` `tensors`-mode updates (`parent v0 -> v1 -> v2 -> restore v0`) with deterministic known-answer probes verifying effective adapter identity independently of echoed version strings, episode-boundary switching, and a 10-case fault/reconciliation matrix (corrupt payload, duplicate delivery vs duplicate activation, stale generation, interrupted activation -> uncertain/ineligible, late old-gen response, failed restoration, unsupported multi-LoRA/disaggregation).
- Reduces single-endpoint vs two-endpoint mixed-pool traces keeping protocol-valid rollouts (including valid low-reward episodes), independent task success, and learner acceptance separate.

### `#26` (`src/nvfp4_mxfp8.rs`)
- Implements exact-bit pure-Rust NVFP4 (FP4 E2M1 + block-16 FP8 E4M3 inner scale + FP32 outer scale) and MXFP8 (FP8 E4M3 + block-32 E8M0 power-of-2 scale) quantizers/dequantizers with both round-to-nearest-even and finite-bit counter-based stochastic rounding.
- Evaluates 4 bounded arms on `Y = X_f W_f^T` with surrogate `dX* = G W_f`, `dW* = G^T X_f`, fixed-operand finite-difference verification, finite-sample stochastic bias/variance/CI/operand-covariance statistics, and deterministic orientation-bias ablation.
- Encodes the NVIDIA Transformer Engine (`4f5c08e4`) `_compute_mxfp8_support()` SM120 vs SM121 refusal matrix, CUTLASS SM120 TN-only layout constraints, Model Optimizer fixture reconstruction/mismatch controls, and the optional Fabric `#6` NVFP4-dispatch + Walsh-Hadamard/S2F6-combine codec reference.

### `#27` (`src/spark_engines.rs`)
- Encodes the pinned DGX Spark support matrix across baseline, Hyperflare (`333b11ec`), and TensorRT Edge-LLM (`513dcd18`) for single-Spark and multi-host topologies (preserving `services.kindling.models` nspawn as untested, Spark `#7` NCCL vs host-RDMA boundary, and Edge-LLM TP2 exclusions).
- Implements a pure-Rust hybrid KV+recurrent continuation verifier comparing `append/resume`, `branch`, `rollback`, `cancel`, and `tool pause` against full recomputation, plus `#11` lifecycle phase accounting and 8 negative controls.

### `#28` (`src/kv_compression.rs`)
- Implements a tiny multi-head causal attention model under uncompressed, identity (`ratio = 0.0`, exact logit parity), and two lossy KVPress-style prefill compression settings (`r = 0.25`, `r = 0.50`) with explicit position-ID retention and observed KV + metadata + temporary buffer byte accounting.
- Evaluates a 3-length frozen RULER-style diagnostic matrix (with truncation guards) and an independently checked evidence-grounded end-task family (corrected facts, distractors, boundary facts, required citations) plus 7 negative controls.

### `#29` (`src/expert_placement.rs`)
- Simulates response-derived MoE expert placement (`fixed` vs `adaptive`) and host-page policy (`STRATA_FIXED_EXPERTS`, `STRATA_COLD_KEEP`) one factor at a time across 5 independent warm-state layers, teacher-forced vs free-generation workloads, and >= 3 counterbalanced paired repetitions.
- Tracks routing-equivalence observability, unified-memory non-freeing, Linux `madvise` non-guarantee, custom `RustingStrata` license terms, and 8 negative controls.

### `#30` (`src/vision_infusion.rs`)
- Implements 4 tiny pure-Rust multimodal architectures (`encoder_free`, `separate_encoder`, `early_ffn_infusion`, `early_full_infusion`) with exact matched visual-only parameter count across the 3 principal arms and separate FLOP/active-param accounting.
- Verifies single-request vs packed mixed-batch gather/scatter numerical parity per architecture, text-only invariance under nonzero branch weights, strict modality-vs-attention mask separation, mixed-traffic text/image latency reporting, explicit `quality_unestablished` status, and 7 negative controls.

### `#31` (`src/qwen38_qsa.rs`)
- Implements a pure-Rust 4-token block-level QSA attention selector vs matched dense attention (`LLAMA_QSA_DENSE=1`) across the `2x2` `(dense/QSA) x (MTP off/on)` matrix with full target KV + indexer buffer + GDN + MTP + workspace memory accounting.
- Runs 4 independently checked long-context task families (lookup+decoys, corrected definition, dependency chain, multi-file edit) across start/middle/end evidence placements, preserving dense failures and sparse-induced losses, offline population-completeness checks, ARM64/GB10 vs x86-64/sm70 binary and `#23` FlashTopK non-drop-in guards, and 8 negative controls.
