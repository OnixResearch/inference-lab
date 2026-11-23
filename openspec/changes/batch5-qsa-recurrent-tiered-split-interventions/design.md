# Design: Batch 5 QSA state coherence, Inkling residuals, Phoenix nano, Nemotron Spark, compact hybrids, native recurrent bench, tiered eval, CPU contention, M2+P150 split, Quark eval, intervention replay, and BioLLM protocol

## Architecture & Purity Boundaries

All twelve Batch 5 modules live inside the pure deterministic Rust library crate (`unsafe_code = "forbid"`, zero external dependencies; `Cargo.toml`, `Cargo.lock`, `flake.nix`, and `flake.lock` remain unchanged):

| Issue | Module | CLI subcommand | Subdir under `evidence/batch5/` | Execution class | Physical / artifact gate |
| ---: | --- | --- | --- | --- | --- |
| `#32` | `src/qsa_state_coherence.rs` | `qsa-state-coherence` | `issue-32-qsa-state-coherence` | `bounded_contract_harness` | `unavailable_hardware_runtime` (ARM64/GB10 Qwen3.8-27B QSA live serve) |
| `#33` | `src/inkling_residuals.rs` | `inkling-residuals` | `issue-33-inkling-residuals` | `bounded_contract_harness` | `blocked_on_artifacts` (Inkling-2-Small 594B & matched-budget trained small models) |
| `#34` | `src/phoenix_nano.rs` | `phoenix-nano` | `issue-34-phoenix-nano` | `bounded_contract_harness` | `unavailable_hardware_runtime` (one-node JAX/CUDA + Rust gRPC Phoenix serve) |
| `#35` | `src/nemotron_spark.rs` | `nemotron-spark` | `issue-35-nemotron-spark` | `bounded_contract_harness` | `unavailable_hardware_runtime` (DGX Spark GB10 live Nemotron Lightning & Bolt serve) |
| `#36` | `src/compact_hybrids.rs` | `compact-hybrids` | `issue-36-compact-hybrids` | `bounded_contract_harness` | `unavailable_hardware_runtime` (Spark/RTX 5080/Strix Halo Granite/Jamba2/Falcon H1 serve) |
| `#37` | `src/native_recurrent_bench.rs` | `native-recurrent-bench` | `issue-37-native-recurrent-bench` | `bounded_contract_harness` | `unavailable_hardware_runtime` (official RWKV-7/Mamba-3/xLSTM-7b accelerator runs) |
| `#38` | `src/tiered_eval.rs` | `tiered-eval` | `issue-38-tiered-eval` | `bounded_contract_harness` | `unavailable_hardware_runtime` (physical GPU persistent CUDA/HIP graph worker) |
| `#39` | `src/cpu_contention.rs` | `cpu-contention` | `issue-39-cpu-contention` | `bounded_contract_harness` | `unavailable_hardware_runtime` (physical Linux/Spark CPU contention & GPU serve) |
| `#42` | `src/split_m2_p150.rs` | `split-m2-p150` | `issue-42-split-m2-p150` | `bounded_contract_harness` | `unavailable_hardware_runtime` (Mac Pro M2 Ultra + internal PCIe P150/P150x2) |
| `#44` | `src/quark_eval.rs` | `quark-eval` | `issue-44-quark-eval` | `bounded_contract_harness` | `blocked_on_artifacts` (Quark-150M-128K Base/Zero HF checkpoints & Transformers 5.23+/CUDA) |
| `#45` | `src/intervention_replay.rs` | `intervention-replay` | `issue-45-intervention-replay` | `bounded_contract_harness` | `unavailable_hardware_runtime` (pinned local Qwen transformer serve on admitted host) |
| `#46` | `src/biollm_protocol.rs` | `biollm-protocol` | `issue-46-biollm-protocol` | `bounded_contract_harness` | `blocked_on_artifacts` (paired CL1 biological recordings & BioLLM-8B artifacts) |

`src/batch5.rs` aggregates all 12 suites into a single `Batch5Bundle` with exact FIPS-180-4 SHA-256 `manifest.json` digests, and `src/main.rs` remains the sole imperative shell writing to a fresh `--out NEW_DIRECTORY` and emitting empty `COMPLETE` markers last.

## Cross-Cutting Numerical & Contract Invariants

1. **Zero NaN-swallowing comparisons**: Every max-error helper, norm/moment reducer, argmax, softmax, top-k selector, calibration metric, and numerical gate explicitly checks `!v.is_finite()` and returns `Err(...)` or `OutcomeStatus::Invalid` rather than swallowing `NaN` via `f64::max`.
2. **Real executable negative controls**: Every negative control mutates actual inputs, caches, weights, masks, SIDs, adapters, signals, timings, or state transitions and runs through the real validator/kernel to prove rejection (zero hardcoded booleans).
3. **Synthetic fixture reducer tagging**: Every synthetic timing, cost, latency, or population reducer explicitly emits `"synthetic_contract_fixture":true` so synthetic checks can never be mistaken for live hardware measurements.
4. **Exact non-collapsing denominators**: All planned items (`accepted`, `invalid`, `unsupported`, `exhausted`, `timeout`, `oom`, `cancelled`, `internal_failure`, `unavailable_hardware_runtime`, `blocked_on_artifacts`) remain in population denominators; `cost_per_accepted_task` is `None` (`null` in JSON) whenever `accepted == 0`.
5. **Honest upstream boundaries**: Every pinned upstream repository, commit SHA, paper, blog, and issue comment (Inkling-2-Mini `#57`, Goodfire Mini study, NDIF Looped Lens Challenge, Cloudflare Containers TCP/gRPC, Latent Lattice LMs `#60`, Blink, Eunomia, Together Compiler, BioLLM) is retained strictly as non-gating provenance.
6. **Path & publication hygiene**: Zero `/home/` or user-specific paths appear anywhere in code, tests, docs, or emitted JSON/Markdown; every `--out` run refuses existing directories and writes empty `COMPLETE` last.

## Per-Module Key Designs

### `#32` (`src/qsa_state_coherence.rs`)
- Defines `ContinuationIdentity` binding backbone/indexer hashes, engine/build, tokenizer/template, accepted token history, positions, attention mode/budget, KV precision, recurrent layout, MTP mode, and session/deployment generation, plus explicit `advertised`/`observed`/`verified` cache-reuse telemetry.
- Implements a coupled pure-Rust state engine maintaining Target KV, Raw + 4-token Pooled Indexer Keys, Gated DeltaNet matrix recurrence `S_t`, and MTP draft-2 cache with `1e6` sentinel poisoning on uncommitted tail slots.
- Verifies fresh recomputation vs append/resume, repeated prefix, chunked prefill across 4-token block and sparse-selection boundaries, truncation + changed suffix, checkpoint restore with rebuilt pooled keys, MTP-off vs draft-2 accept/reject/bonus rollback, two-session single-slot isolation, explicit unsupported branching/disk-persistence, and 8 negative controls.

### `#33` (`src/inkling_residuals.rs`)
- Implements independent FP64 reference and candidate paths for a 12-layer (two 6-layer blocks) miniature with RMSNorm, learned depth queries over completed block sums + active block accumulation/reset/final aggregation, input-dependent dynamic sink logits, static sink baseline, and output gating.
- Verifies full prefill vs chunked prefill vs token decode (with conv/cache state and reset) across causal/sliding/full masks and extreme sink/gate values (`-50.0`/`+50.0`), separates all 6 ablation arms, distinguishes weight-disabling sensitivity from matched-budget training claims, pins `Inkling-2-Mini` / Goodfire as non-gating related work, and checks 7 negative controls.

### `#34` (`src/phoenix_nano.rs`)
- Implements `gpu_arch()` qualification matching `xai-org/x-algorithm@30647a85` (`A100/H100/H200/GB200/GB300` only; `GB10` and `RTX 5080` explicitly `unsupported` and separated; spoofed overrides rejected).
- Binds seeded world, provider snapshot, codebook, and training dump by SHA-256, verifies 0-indexed wire <-> 1-indexed model SID encoding (`0` reserved for missing), runs a tiny deterministic retrieval/ranking trainer comparing uninterrupted vs fresh-process checkpoint restore (parameters + optimizer + RNG + data cursor), runs a bounded native gRPC retrieve -> rank check against direct inference plus independent attention-isolation / missing-SID / fixed-score top-k controls, and checks 8 negative controls.

### `#35` (`src/nemotron_spark.rs`)
- Freezes 13-field DGX Spark serving profiles for `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (`OpenMDW-1.1`, `nemotron_v3`, `qwen3_coder`) and `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` with staged arm progression (target-only reasoning-off -> reasoning-on -> DSpark/MTP drafter -> Bolt target-only -> Bolt drafter).
- Executes 10 device-free request/reasoning/tool/stream/incarnation conformance fixtures, independent known-answer & structured-output checking (rejecting malformed JSON and plausible-but-wrong answers regardless of confidence), full lifecycle/cost reduction, and 8 negative controls.

### `#36` (`src/compact_hybrids.rs`)
- Extends the common profile manifest with evidence-bound `ArchitectureFamily`, `StateComponent`, `StateVisibility`, and `StateOperation` declarations across Granite 4.0 H-Tiny, Jamba2 3B, Falcon H1 3B, and qualified controls, refusing unverified operations under `opaque_endpoint`.
- Implements request/dialect conformance, 3 frozen workload classes (short structured agent tasks, context/correction tasks across 3 per-tokenizer length buckets, and separately labeled serving-mechanics trace replay), concurrency-1 vs bounded-concurrent session-density and unified-memory provenance accounting, complete-profile vs architecture attribution guards, and 8 negative controls.

### `#37` (`src/native_recurrent_bench.rs`)
- Defines the versioned native recurrent-state contract (`fresh_state`, `step`, `chunk`, `snapshot`, `serialize`, `restore`, `reset`, optional `fork`, `rollback`, `batch_step`) with 6 explicit dispositions (`supported`, `unsupported`, `invalid`, `resource_stop`, `numerical_failure`, `infrastructure_failure`).
- Implements independent tiny scalar references and chunked/stateful engines for RWKV-7, Mamba-3 SISO, and Milestone-B-gated xLSTM (log-stabilized mLSTM), verifying Phases 1–5 (reference parity, step/chunk/serialize/restore/poisoned-slot/capacity bounds, fork/rollback/cancel/stale-bytes, 2-session interleaved isolation + batched parity, state bytes + recompute cost), pinning Latent Lattice LMs (`#60`) as non-gating related work, and checking 9 negative controls.

### `#38` (`src/tiered_eval.rs`)
- Evaluates a frozen candidate set across `(A) restart-per-candidate`, `(B) full-fidelity persistent`, and `(C) fast screening + full persistent` (plus explicitly scoped reduced-layer tier) under equal total budgets.
- Enforces immutable weight digests, KV/recurrent/workspace/RNG/global resets, verified worker termination on failed cleanup, `A->B->A` parity vs fresh-process A, preregistered fast-tier false-rejection and rank-disagreement audits, cold final held-out replay before performance scoring, and 8 negative controls.

### `#39` (`src/cpu_contention.rs`)
- Models host topology (physical/logical cores, asymmetric classes, NUMA, NIC/GPU locality), requested vs effective masks, and single-factor arms (`default_placement`, `static_topology_affinity`, `host_light_decode`) across idle/moderate/heavy background CPU load while keeping priority and IRQ affinity as separate labeled arms.
- Tracks completed background work (detecting starvation-based recovery), computes Blink-comparable colocated/isolated throughput ratios and P99 TTFT/TPOT per load level (recording Blink's non-reproducible interference workload gap), enforces `#40` trace attribution & ITL-naming rules, retains regression/no-benefit regimes and conditional `sched_ext` gating, and checks 8 negative controls.

### `#42` (`src/split_m2_p150.rs`)
- Freezes Mac Pro M2 Ultra + P150 / P150x2 profiles separating Linux+Asahi (`tt-onix#48`) vs macOS+Metal (`tt-onix#49`), internal PCIe vs conventional Linux/network prototypes, distributed unified-memory vs GDDR inventories, and `P150x1` vs `P150x2` vs `MoE` acceptance.
- Reuses `crate::stage` for real dense split-model numerical qualification (FP64 ref, Apple-only, TT-only, TT split-vs-unsplit, Apple+TT split across full/chunked prefill, forced-token decode, reset/reuse, and >= 2 placements) plus independent disjoint-expert MoE split verification, derived logical payload math (`16 KiB/token`, `64 MiB` / 4096 rows), and 8 negative controls.

### `#44` (`src/quark_eval.rs`)
- Pins `Quark-150M-128K-Base` and `Quark-150M-128K-Zero` separately (Transformers 5.23+ check, ~165.3M params, bounded local attention/conv + 4 global KV layers).
- Implements a pure-Rust miniature local-window/conv + global-KV hybrid verifying full-prefix recomputation vs cached decode across irregular chunks, local-window rollover, exact positions, and changed tool results; enforces turn/delimiter/tool-execution vs hallucinated-text checks, explicit state capability declarations, progressive `1K/8K/32K/128K` context accounting, held-out diagnostics + RSI task rescoring, and 8 negative controls.

### `#45` (`src/intervention_replay.rs`)
- Implements a model-agnostic immutable experiment manifest and deterministic replay verifier with a tiny pure-Rust Qwen-style transformer backend supporting paired baseline, targeted activation interventions (`Scale`, `Ablate`, `Inject`), and matched-norm random-direction controls with KL/L1/top-1/task-outcome comparison.
- Defines recurrent/stateful hook boundaries and forward-compatible looped-block readout descriptors (recording Goodfire/NDIF Mini notes strictly as non-gating related work), strictly enforcing non-portability of neuron IDs and layer indices across architectures, and checks 8 negative controls.

### `#46` (`src/biollm_protocol.rs`)
- Defines a reproducible 5-arm protocol (`base_model`, `text_only_lora`, `bio_paired_adapter`, `bio_shuffled_control`, `matched_synthetic_control`) at equal training budgets with strict held-out task AND held-out culture/session split enforcement and `MeasuredTiming` vs `ReconstructedTiming` separation.
- Implements a pure-Rust signal moment/autocorrelation matcher and miniature 5-arm adapter evaluator computing accuracy, retention, calibration (`ECE` + Brier score), compute, and latency with falsifiable biological-attribution criteria (requiring paired bio to beat text LoRA, shuffled, AND synthetic controls on held-out tasks & cultures), and checks 8 negative controls.
