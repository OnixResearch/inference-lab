# Batch 5: QSA state coherence, Inkling residuals, Phoenix nano, Nemotron Spark, compact hybrids, native recurrent bench, tiered eval, CPU contention, M2+P150 split, Quark eval, intervention replay, and BioLLM protocol

## Why

Batches 1–4 established `OnixResearch/inference-lab` with a deterministic, dependency-free pure-Rust core (`unsafe_code = "forbid"`), offline Nix tooling, and bounded contract harnesses for `#1`–`#31`, `#40`, `#41`, `#43`, `#52`, `#56`, `#58`, and `#60`. Batch 5 extends this pure-Rust numerical reference, contract verification, and provenance foundation across the next twelve tracked issues (`#32`–`#39`, `#42`, `#44`–`#46`), covering:

- `#32`: Qwen3.8-27B QSA cache reuse and coupled target KV / raw+pooled indexer / Gated DeltaNet recurrent / MTP state coherence for agent sessions;
- `#33`: Inkling 2 Small 6-layer block attention residuals, input-dependent dynamic sinks, static-sink isolation, and output-gating numerical conformance and matched ablations;
- `#34`: Phoenix nano retrieval/ranking one-node `gpu_arch()` qualification, 0/1-indexed SID wire encoding, checkpoint/optimizer/RNG/cursor resume parity, and bounded native gRPC retrieve-to-rank controls;
- `#35`: Nemotron 3.5 Lightning (30B-A3B NVFP4) and Bolt (74B-A9B 24GB) pinned DGX Spark serving profiles, staged arm progression, device-free request/reasoning/tool conformance, and independent structured-output checking;
- `#36`: Compact hybrid recurrent language model comparison profile (Granite 4.0 H-Tiny, Jamba2 3B, Falcon H1 3B) with evidence-bound architecture/state manifest extensions, 3 frozen workload classes, session-density accounting, and complete-profile decision gates;
- `#37`: Native recurrent-state benchmark contract across RWKV-7, Mamba-3 SISO, and xLSTM (mLSTM) with independent tiny references, step/chunk/snapshot/serialize/restore/fork/rollback/multi-session isolation, and milestone gating;
- `#38`: Tiered persistent-engine evaluation for inference optimization (`restart` vs `full persistent` vs `fast screen + full persistent`) with immutable-weight isolation, `A->B->A` restoration, fast-tier false-rejection auditing, and cold final held-out replay;
- `#39`: Host CPU contention and topology-aware placement sensitivity (default vs static affinity vs host-light decode), comparable background-work accounting, Blink-comparable colocated/isolated ratios, `#40` attribution guards, and conditional `sched_ext` gating;
- `#42`: Mac Pro M2 Ultra + Tenstorrent P150 / P150x2 split-model inference qualification (Linux/Asahi vs macOS/Metal separation, dense & MoE split parity reusing `#41`, distributed memory inventory, logical payload derivation, and independent P150x1/P150x2/MoE acceptance);
- `#44`: Quark-150M-128K Base/Zero hybrid local-window/conv + 4 global KV cached-decode parity, tool-result execution vs text-hallucination guards, progressive 1K–128K context accounting, and held-out RSI task rescoring;
- `#45`: Model-agnostic reproducible activation intervention (`Scale`, `Ablate`, `Inject`) and deterministic replay harness on a tiny Qwen-style transformer backend with matched-norm random-direction controls and non-portable recurrent/looped hook boundaries;
- `#46`: Reproducible 5-arm biological-recording adapter protocol (base, text-only LoRA, paired biological, shuffled, and statistics-matched synthetic controls) at equal training budgets with held-out tasks and cultures, measured-vs-reconstructed timing separation, calibration/retention metrics, and falsifiable attribution criteria.

## Change

1. Add twelve pure-Rust, dependency-free contract and numerical-reference modules (`unsafe_code = "forbid"`) without touching `Cargo.toml`, `Cargo.lock`, `flake.nix`, `flake.lock`, `evidence/{bootstrap,batch1,batch2,batch3,batch4}/`, or any of the 56 existing tests:
   - `#32` (`src/qsa_state_coherence.rs`)
   - `#33` (`src/inkling_residuals.rs`)
   - `#34` (`src/phoenix_nano.rs`)
   - `#35` (`src/nemotron_spark.rs`)
   - `#36` (`src/compact_hybrids.rs`)
   - `#37` (`src/native_recurrent_bench.rs`)
   - `#38` (`src/tiered_eval.rs`)
   - `#39` (`src/cpu_contention.rs`)
   - `#42` (`src/split_m2_p150.rs`)
   - `#44` (`src/quark_eval.rs`)
   - `#45` (`src/intervention_replay.rs`)
   - `#46` (`src/biollm_protocol.rs`)
2. Add `src/batch5.rs` and extend `src/lib.rs` and `src/main.rs` with 12 per-issue CLI subcommands plus `batch5-suite`, emitting deterministic artifacts, SHA-256 `manifest.json`, and empty `COMPLETE` markers last.
3. Add comprehensive adversarial unit and integration tests covering every Batch 5 invariant and negative control while preserving all 56 prior tests, update `README.md`, and generate two-run byte-identical retained evidence under `evidence/batch5/`.

## Scope and limitations

All executable harnesses in Batch 5 run deterministically on CPU as `bounded_contract_harness` checks. Physical DGX Spark GB10, RTX 5080, AMD Strix Halo, Mac Pro M2 Ultra + internal PCIe Tenstorrent P150 / P150x2, A100/H100/GB300 Phoenix nodes, full Qwen3.8-27B QSA / Nemotron Lightning & Bolt / Granite H-Tiny / Jamba2 / Falcon H1 / RWKV-7 / Mamba-3 / xLSTM-7b / Quark-150M / Inkling-2-Small 594B / BioLLM-8B checkpoints, and biological culture recordings are explicitly recorded as `unavailable_hardware_runtime` and/or `blocked_on_artifacts`. All external upstream claims (Together Compiler §4.2, Eunomia CPU-pressure/SchedCP, Blink 8.47x/3.40x, Goodfire Mini study / NDIF Looped Lens Challenge, Latent Lattice LMs, Cloudflare Containers TCP/gRPC, NVIDIA Nemotron tables, BioLLM-8B claims) remain strictly unverified non-gating references.
