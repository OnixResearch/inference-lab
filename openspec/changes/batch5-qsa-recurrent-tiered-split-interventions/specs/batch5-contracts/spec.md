# Specification: Batch 5 QSA state coherence, Inkling residuals, Phoenix nano, Nemotron Spark, compact hybrids, native recurrent bench, tiered eval, CPU contention, M2+P150 split, Quark eval, intervention replay, and BioLLM protocol

## Scenario 1: Batch 5 CLI publication, SHA-256 manifest, and backward preservation
- All 40 prior CLI subcommands, `evidence/{bootstrap,batch1,batch2,batch3,batch4}/`, `Cargo.toml`, `Cargo.lock`, `flake.nix`, `flake.lock`, and all 56 prior tests remain intact and passing.
- Each of the 12 new per-issue subcommands (`qsa-state-coherence`, `inkling-residuals`, `phoenix-nano`, `nemotron-spark`, `compact-hybrids`, `native-recurrent-bench`, `tiered-eval`, `cpu-contention`, `split-m2-p150`, `quark-eval`, `intervention-replay`, `biollm-protocol`) writes `inputs.json`, `results.json`, `summary.md`, and empty `COMPLETE` last into a fresh `--out NEW_DIRECTORY` and refuses overwrites.
- `batch5-suite --out NEW_DIRECTORY` publishes all 12 per-issue subdirectories (each ending with empty `COMPLETE`), top-level `inputs.json`, `results.json`, `summary.md`, `manifest.json` (with exact SHA-256 digests of every emitted file), and empty `COMPLETE` last.
- Two consecutive runs of `batch5-suite` produce byte-identical outputs across every emitted file matching retained `evidence/batch5/` with zero `/home/` paths.

## Scenario 2: Issue #32 — QSA cache reuse and target/indexer/GDN/MTP state coherence
- Binds complete continuation identity across backbone/indexer hashes, engine/build, tokenizer/template, accepted token history, positions, attention mode/budget, KV precision, recurrent layout, MTP mode, and session/deployment generation, explicitly separating `advertised`, `observed`, and `verified` cache reuse.
- Verifies coupled Target KV, raw+pooled 4-token Indexer keys, GDN recurrent matrix state, and MTP draft-2 cache against fresh recomputation across append/resume, chunked prefill (block & sparse-selection boundaries), truncation + changed suffix, checkpoint restore with rebuilt pooled keys, MTP-off vs draft-2 rejection/rollback, and two-session single-slot isolation.
- Rejects stale index keys, wrong positions, omitted GDN rollback, retained rejected drafts, mismatched artifact identity, cross-session prefix reuse, telemetry-free reuse claims, and stale deployment generations.

## Scenario 3: Issue #33 — Inkling block attention residuals and dynamic sinks
- Implements independent FP64 reference and candidate paths for a 12-layer (two 6-layer blocks) miniature with RMSNorm, learned depth queries, block accumulation/reset/final aggregation, input-dependent dynamic sinks, static sink baseline, and output gating.
- Verifies full prefill, chunked prefill, and token decode parity across causal/sliding/full masks and extreme sink/gate values (`-50.0`, `+50.0`), separates all 6 ablation arms, distinguishes pretrained disabling sensitivity from matched-budget training superiority, keeps `Inkling-2-Mini` and Goodfire notes strictly non-gating, and catches all 7 negative controls.

## Scenario 4: Issue #34 — Phoenix nano training, checkpoint resume, and gRPC serving
- Enforces `xai-org/x-algorithm@30647a85` `gpu_arch()` qualification (`A100/H100/H200/GB200/GB300` only; `GB10` and `RTX 5080` explicitly `unsupported` and separate; device-name spoof overrides rejected).
- Verifies SHA-256 snapshot/codebook/dump binding, 0-indexed wire <-> 1-indexed model SID encoding (`0` reserved for missing), fresh-process checkpoint restore parity (parameters, optimizer, RNG, data cursor), bounded native gRPC retrieve -> rank vs direct inference, independent attention-isolation/missing-SID/top-k controls, and all 8 negative controls.

## Scenario 5: Issue #35 — Nemotron 3.5 Lightning & Bolt DGX Spark serving profiles
- Freezes 13-field serving profiles for Lightning 30B-A3B NVFP4 (`OpenMDW-1.1`, `nemotron_v3`, `qwen3_coder`) and Bolt 74B-A9B 24GB, rejecting floating `/v1/models` aliases and separating staged target-only/reasoning/drafter arms.
- Passes 10 device-free request/reasoning/tool/SSE/incarnation conformance fixtures, independently rejects malformed JSON and plausible-but-wrong answers regardless of confidence, preserves lifecycle/cost accounting and non-gating upstream tables, and catches all 8 negative controls.

## Scenario 6: Issue #36 — Compact hybrid recurrent language model comparison profile
- Extends the common profile manifest with evidence-bound `ArchitectureFamily`, `StateComponent`, `StateVisibility`, and `StateOperation` declarations for Granite 4.0 H-Tiny, Jamba2 3B, Falcon H1 3B, and qualified controls, refusing unverified operations under `opaque_endpoint`.
- Covers request/dialect conformance, 3 frozen workload classes across 3 per-tokenizer token-length buckets (keeping serving-mechanics trace replay separate from task checkers), concurrency-1 vs bounded-concurrent session-density and unified-memory provenance accounting, `cost_per_accepted_task = None` at zero accepted tasks, and all 8 negative controls.

## Scenario 7: Issue #37 — Native recurrent-state benchmark across RWKV-7, Mamba-3, and xLSTM
- Implements the versioned native recurrent-state contract with 6 explicit dispositions (`supported`, `unsupported`, `invalid`, `resource_stop`, `numerical_failure`, `infrastructure_failure`), forbidding mislabeling full-prefix recomputation as native state support.
- Exercises independent tiny scalar references and chunked/stateful engines for RWKV-7, Mamba-3 SISO, and Milestone-B xLSTM (mLSTM) across Phases 1–5 (equation parity, step/chunk/serialize/restore/poisoned slots, fork/rollback/cancel, 2-session interleaved isolation & batched parity, state bytes & recompute economics), pins Latent Lattice LMs (`#60`) as non-gating, and catches all 9 negative controls.

## Scenario 8: Issue #38 — Tiered persistent-engine evaluation with cold final replay
- Compares restart-per-candidate, full-fidelity persistent, and fast-screen + full-persistent arms under equal budgets across a frozen candidate matrix with immutable-weight verification, full state/buffer/RNG/global resets, verified worker termination, and `A->B->A` restoration parity against fresh-process A.
- Audits fast-tier false rejection and rank disagreement, requires frozen-candidate cold final held-out replay (numerical + state continuation before performance scoring), retains all planned candidates, and catches all 8 negative controls.

## Scenario 9: Issue #39 — Host CPU contention and topology-aware placement
- Discovers host topology (cores, asymmetric classes, NUMA, NIC/GPU locality) and records requested vs effective masks across single-factor arms (`default_placement`, `static_topology_affinity`, `host_light_decode`) under idle/moderate/heavy background CPU load while keeping priority and IRQ affinity as separate labeled arms.
- Tracks completed background work, reports Blink-comparable colocated/isolated throughput ratios and P99 TTFT/TPOT (recording Blink's workload reproducibility gap), enforces `#40` trace attribution & ITL-naming rules, retains regression regimes and conditional `sched_ext` gating, and catches all 8 negative controls.

## Scenario 10: Issue #42 — M2 Ultra + P150 / P150x2 split-model inference
- Separates Linux+Asahi (`tt-onix#48`) vs macOS+Metal (`tt-onix#49`), internal PCIe vs conventional Linux/network prototypes, distributed unified-memory vs GDDR inventories, and independent `P150x1` / `P150x2` / `MoE` acceptance.
- Reuses `crate::stage` for dense split parity (FP64 ref, Apple-only, TT-only, TT split-vs-unsplit, Apple+TT split across full/chunked prefill, forced-token decode, reset/reuse, and >= 2 placements), verifies disjoint-expert MoE splitting, records exact derived logical payload (`16 KiB/token`, `64 MiB` / 4096 rows) separately from measured throughput, and catches all 8 negative controls.

## Scenario 11: Issue #44 — Quark Base/Zero cached-decode parity & context costs
- Pins `Quark-150M-128K-Base` and `Quark-150M-128K-Zero` separately (Transformers 5.23+, bounded local window/conv + 4 global KV layers), verifying full-prefix recomputation vs cached decoding within each checkpoint across irregular chunks, local-window rollover, exact positions, and changed tool results.
- Rejects hallucinated tool-result text as executed effect, separates native state capabilities from recomputation recovery, accounts for progressive `1K/8K/32K/128K` context costs, scores held-out diagnostics and RSI tasks with `cost_per_verified_success = None` at zero successes, and catches all 8 negative controls.

## Scenario 12: Issue #45 — Reproducible model intervention and replay harness
- Captures immutable experiment manifests and deterministic replay receipts on a tiny Qwen-style transformer backend with paired baseline, targeted activation interventions (`Scale`, `Ablate`, `Inject`), and matched-norm random-direction controls comparing token distributions (KL/L1/top-1) and task outcomes.
- Defines recurrent/stateful hook boundaries and non-gating looped-model readout notes (Goodfire/NDIF), strictly enforces non-portability of neuron IDs and layer indices across architectures, and catches all 8 negative controls.

## Scenario 13: Issue #46 — Biological-recording adapter protocol & matched controls
- Enforces equal-budget 5-arm comparison (`base_model`, `text_only_lora`, `bio_paired_adapter`, `bio_shuffled_control`, `matched_synthetic_control`), disjoint held-out tasks AND held-out cultures/sessions, `MeasuredTiming` vs `ReconstructedTiming` separation, and verified synthetic moment/autocorrelation matching.
- Evaluates accuracy, retention, calibration (`ECE` + Brier score), compute, and latency with falsifiable biological-attribution criteria requiring paired biological recordings to outperform all three controls without breaking retention/calibration bounds, and catches all 8 negative controls.
