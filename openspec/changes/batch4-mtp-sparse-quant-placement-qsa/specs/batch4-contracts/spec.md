# Specification: Batch 4 MTP, FlashTopK, MACaQ-R, heterogeneous rollouts, NVFP4/MXFP8, Spark engines, KV compression, expert placement, vision in-fusion, and Qwen3.8 QSA contracts

## Scenario 1: Batch 4 CLI publication, SHA-256 manifest, and backward preservation
- All 29 prior CLI subcommands, `evidence/{bootstrap,batch1,batch2,batch3}/`, `Cargo.toml`, `Cargo.lock`, `flake.nix`, `flake.lock`, and all 45 prior tests remain intact and passing.
- Each of the 10 new per-issue subcommands (`mtp-decoder`, `flash-topk`, `macaq-r`, `hetero-rollout`, `nvfp4-mxfp8`, `spark-engines`, `kv-compression`, `expert-placement`, `vision-infusion`, `qwen38-qsa`) writes `inputs.json`, `results.json`, `summary.md`, and empty `COMPLETE` last into a fresh `--out NEW_DIRECTORY` and refuses overwrites.
- `batch4-suite --out NEW_DIRECTORY` publishes all 10 per-issue subdirectories (each ending with empty `COMPLETE`), top-level `inputs.json`, `results.json`, `summary.md`, `manifest.json` (with exact SHA-256 digests of every emitted file), and empty `COMPLETE` last.
- Two consecutive runs of `batch4-suite` produce byte-identical outputs across every emitted file matching retained `evidence/batch4/` with zero `/home/` paths.

## Scenario 2: Issue #22 — Small MTP draft-and-verify decoder with cache and sampling conformance
- Executes a real frozen tiny target model and depth-1 MTP draft-and-verify decoder with frozen parameters and routing biases (no training-time load-controller updates during decode).
- Tracks target and MTP caches separately with `1e6` sentinel poisoning of rejected/unused tail slots; greedy full output, valid cache prefix, and next-step logits agree with target-only full recomputation across mixed accept/reject, first-pos rejection, later-pos rejection, all-accepted + bonus, zero-accepted, EOS/budget/context bounds, cancellation/continuation, and sequential multi-request isolation.
- Stochastic mode implements exact rejection sampling (`min(1, p(x)/q(x))`, normalized `max(p - q, 0)` residual, all-accepted bonus rule) with deterministic uniform draw injection covering all reachable branches, exact small-vocabulary analytical marginal check (`< 1e-12`), and calibrated empirical check.
- Negative controls (swapped `p`/`q`, bad residual normalization, off-by-one target alignment, missing rollback reading poisoned cache, emitted rejected token, future ground-truth leakage / routing-bias mutation) are detected and fail.

## Scenario 3: Issue #23 — FlashTopK fused sparse-indexer selection
- Pins `Tencent-Hunyuan/FlashTopK@090902abad86e5bb3324519124a680122fff91a1` (`Apache-2.0`) and enforces compatibility gates: CUDA CC major 9 or 10 only (explicitly refusing SM120/SM121), `[rows, 32, 128]` / `[keys, 128]` shapes (explicitly refusing 64-head DeepSeek-V3.2), FP8 E4M3 or BF16, `topk in [1, 4096]`, contiguous keys/windows (refusing paged KV and block-level QSA).
- Independently evaluates `score(t, s) = scale(s) * sum_h weights(t, h) * ReLU(dot(q(t, h), key(s)))` with strict `[ks, ke)` windows, canonical descending-score / ascending-key-index ordering (including signed-zero ties), short-window padding, and non-finite/invalid-window rejection.
- Exercises fused threshold estimation and underflow/overflow recovery (distinguishing unmodified vs fault-injected threshold-hook runs), reports numerical agreement, computed-score selection exactness, and deterministic repetition separately, preserves complete memory accounting, and catches all negative controls.

## Scenario 4: Issue #24 — MACaQ-R native MoE GGUF refinement
- Pins `huawei-csl/MACaQ@3810efca0b9e819c8eb51addd93c3c87225b7476` and isolates `#24` (MoE K-quant refinement) from `#18` (Qwen3.8-27B dense).
- Executes real bounded coordinate refinement on `Q4_K`/`Q5_K`/`Q6_K` MoE projections while strictly preserving block scales, tensor formats, activation basis, and byte-for-byte unchanged out-of-scope tensors; explicitly rejects unsupported `IQ4_XS` / split-GGUF / dense-only silent admission.
- Separates creation-time resources/receipts (with digest-bound resume state) from serving-time residency (independent of HF weights/teacher/calibration), enforces calibration/validation/held-out disjointness, preserves `cost_per_accepted_task = None` at zero accepted tasks, and catches all negative controls.

## Scenario 5: Issue #25 — Heterogeneous rollout serving & versioned HTTP LoRA updates
- Pins Miles `#3690` (`c4d5aa11`), SGLang `#48241` (`f8d0201`), SGLang `#47135`, and Router `#30`, preserving upstream exclusions (multi-LoRA, prefill/decode disaggregation).
- Qualifies frozen policies with teacher-forced per-token logprob residuals and exact full-vocabulary KL (kept distinct from sampled-token probabilities).
- Executes versioned `http-lora` `tensors`-mode transitions (`v0 -> v1 -> v2 -> v0`) with known-answer probe verification independent of echoed version strings, episode-boundary switching, 10 fleet-reconciliation fault controls (keeping uncertain/failed-restore replicas ineligible), and bounded two-backend mixed-pool accounting retaining valid low-reward rollouts.

## Scenario 6: Issue #26 — NVFP4 forward / MXFP8 backward numerics & native layer costs
- Implements exact-bit pure-Rust NVFP4 (E2M1 + block-16 FP8 E4M3 + FP32 outer scale) and MXFP8 (E4M3 + block-32 E8M0) quantizers with round-to-nearest-even and finite-bit stochastic rounding over all required edge cases.
- Compares 4 bounded linear-layer arms against surrogate `dX* = G W_f`, `dW* = G^T X_f` (with fixed-operand FD verification, stochastic bias/variance/CI/operand-covariance accounting, and deterministic orientation-bias ablation).
- Records distinct SM120 vs SM121 rows retaining Transformer Engine (`4f5c08e4`) `_compute_mxfp8_support()` refusal as valid `unsupported` (rejecting monkey-patched gates and BF16-as-MXFP8 fallbacks), verifies Model Optimizer (`ef893e77`) fixture reconstruction controls, and provides the optional Fabric `#6` NVFP4-dispatch + Hadamard/S2F6-combine codec reference.

## Scenario 7: Issue #27 — Hyperflare & TensorRT Edge-LLM on DGX Spark
- Records pinned single-Spark and multi-host support matrices across baseline, Hyperflare (`333b11ec`), and TensorRT Edge-LLM (`513dcd18`), preserving `services.kindling.models` nspawn as untested, Spark `#7` NCCL vs host-RDMA boundaries, and Edge-LLM TP2 exclusions.
- Verifies hybrid KV+recurrent continuation parity (`append/resume`, `branch`, `rollback`, `cancel`, `tool pause`) against full recomputation, accounts for `#11` lifecycle phases separately, and catches stale/missing recurrent state, branch contamination, wrong artifacts, and unsupported fallback controls.

## Scenario 8: Issue #28 — KV compression with frozen long-context diagnostics & end task
- Pins `NVIDIA/kvpress@0cb673ac` and `NVIDIA/RULER@db43f92f`, comparing uncompressed baseline, identity/no-compression control (exact logit parity), and two lossy prefill compression arms with observed KV + metadata + temporary buffer byte accounting.
- Evaluates a 3-length frozen RULER-style subset (with truncation guards) and an independently checked evidence-grounded end-task family (corrected facts, distractors, boundary facts, required citations), separating exact cache conformance from approximate quality thresholds and catching all negative controls.

## Scenario 9: Issue #29 — RustingStrata expert placement & host residency vs llama.cpp
- Pins `RustingStudio/RustingStrata@a39d9e57` (separate from 1.2.1/1.2.2 benchmark notes; custom `LICENSE.md` recorded; RTX 5080 / Strix Halo / Spark unestablished in `SUPPORTED_GPUS.md`).
- Isolates fixed (`STRATA_FIXED_EXPERTS`) vs adaptive placement and host-page (`STRATA_COLD_KEEP`) policies across 5 independent warm-state layers, teacher-forced vs free-generation workloads, routing-equivalence observability, unified-memory non-freeing, `madvise` non-guarantee, >= 3 counterbalanced paired trials, and 8 negative controls.

## Scenario 10: Issue #30 — Separate vision encoders vs early in-fusion at matched visual capacity
- Implements 4 tiny multimodal architectures (`encoder_free`, `separate_encoder`, `early_ffn_infusion`, `early_full_infusion`) with exact matched visual-only parameter budgets across the 3 principal arms and separate FLOP/active-parameter accounting.
- Verifies single-request vs packed mixed-batch gather/scatter numerical parity per architecture, text-only invariance under nonzero branch weights, strict modality-vs-attention mask separation, separate text/image latency distributions in mixed traffic, explicit `quality_unestablished` status, and 7 negative controls.

## Scenario 11: Issue #31 — Qwen3.8-27B QSA speed & long-context task accuracy vs dense control
- Pins `Qwen3.8-27B-QSA-llama.cpp@864934c7` and evaluates 4-token block-level QSA vs matched dense (`LLAMA_QSA_DENSE=1`) across the `2x2` `(dense/QSA) x (MTP off/on)` matrix with full target KV + indexer buffer + GDN + MTP + workspace memory accounting.
- Runs 4 independently checked start/middle/end long-context task families preserving dense-control failures and sparse-induced losses, enforces offline population completeness and `cost_per_accepted_task = None` at zero accepted tasks, rejects x86-64/sm70 binary on Spark and `#23` FlashTopK drop-in claims, and catches all negative controls.
