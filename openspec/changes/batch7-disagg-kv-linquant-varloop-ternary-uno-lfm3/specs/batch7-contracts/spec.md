# Batch 7 Observable Requirement Specifications

## R1: Pure core, finite-checked math, and non-gating external pins
- All Batch 7 modules MUST compile under `#![forbid(unsafe_code)]` with zero external crate dependencies on Rust 1.85.1 without modifying `Cargo.toml`, `Cargo.lock`, `flake.nix`, `flake.lock`, or prior `evidence/{bootstrap,batch1,batch2,batch3,batch4,batch5,batch6}/`.
- No Batch 7 module MAY invoke `.max(...)` or `.min(...)` on `f64` or `f32`. Non-finite floats MUST return `Err(...)` or non-`Accepted` outcomes. Zero denominators MUST produce `None` or `Err(...)`.
- Every Batch 7 report MUST emit `default_enabled: false`, `synthetic_contract_fixture: true`, `execution_class: "bounded_contract_harness"`, and an explicit `hardware_gate` (`unavailable_hardware_runtime` or `blocked_on_artifacts`), and MUST NOT import external X/blog/model-card/paper claims as pass/fail criteria.
- Every Batch 7 module MUST retain `>= 8` adversarial negative controls, each producing a non-`Accepted` `OutcomeStatus`.

## R2: Issue #62 (`disagg-kv-transfer`)
- MUST define a bounded default-off wire format for prefix-cache transfer from a CUDA prefill host to a Metal decode host over 10GbE (header, model/weight/tokenizer digests, layer/head/dtype metadata, chunk framing, SHA-256 verification).
- MUST verify exact token and logit parity (`<= 1e-12`) between disaggregated wire-transferred KV prefill+decode and single-host prefill+decode on frozen prompts.
- MUST report cold and warm TTFT, wire bytes/token, and end-to-end latency against single-host baselines while marking physical Spark+Mac Studio runs `unavailable_hardware_runtime`.

## R3: Issue #63 (`spark-rdma-tp`)
- MUST log RLX as related work with honest offline pin status (`post_only_unlocated_paper_repo`) and `hardware_gate: blocked_on_artifacts`.
- MUST verify 2-rank RDMA tensor-parallel parity (`<= 1e-12`) and 4-bit quantized attention numerical/task parity BEFORE permitting throughput reporting.
- MUST compare 2-node Spark RDMA TP with 4-bit attention against replicated and layer-split baselines at matched context lengths (`8K`, `128K`, `1M`).

## R4: Issue #64 (`spark-kv-checkpoint`)
- MUST pin `MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104@6c55c20da31a02d6d381a268fa42f8a9ebe0c5d2` (Apache-2.0) as a default-off experimental profile.
- MUST verify exact token/logit parity (`<= 1e-12`) between 2-node sharded disk-resumed KV checkpoints and fresh full recomputation on frozen prompts.
- MUST report KV capacity headroom, save/load I/O latency, and cold-restart TTFT against recomputation.

## R5: Issue #65 (`spark-copydrafter`)
- MUST pin `christopherowen/spark-ds41f@75fb3e72c657534cb6cf0f7a989e0edb4db64a6f` (Apache-2.0) and isolate `CopyDrafter` behind a default-off toggle.
- MUST verify lossless output equivalence of `CopyDrafter` against non-drafted target decoding (exact greedy token identity and exact stochastic rejection sampling) on frozen prompts.
- MUST report single-stream and multi-stream (`16` streams) decode tok/s, acceptance rate, and TTFT across `131K`–`1M` context against baseline serving profiles.

## R6: Issue #66 (`hypertune-kv-adapters`)
- MUST pin `Hyperstitioncc/Hypertune@7aaaf8c2fc88bfb7dfd91019c40428d4528ae121` (MIT) and compare soft-prefill paged KV adapters against heterogeneous batched LoRA.
- MUST verify exact mathematical equivalence (`<= 1e-12`) between prepended soft-prefill paged KV blocks and full prepended forward passes, report KV-cache overhead, per-matrix bandwidth, and multi-adapter batching scaling under matched concurrency, and independently check co-resident multi-adapter task accuracy.

## R7: Issue #67 (`qwen38-longbook-gdn`)
- MUST pin `summerAIO/Qwen3.8-27B-Longbook27-v2@46319430c16af66860b20f14986ad1032e991796` (alias `summerMC/Qwen3.8-27B-Longbook27-v2`, Apache-2.0) and base Qwen3.8-27B under matched serving and quantization settings.
- MUST measure constant-size GDN state vs linear-growth full-attention KV memory footprint, TTFT, and decode throughput from `8K` to `262K` context, and compare independently checked long-context recall and frozen agent-task pass rates against base Qwen3.8-27B.

## R8: Issue #68 (`linquant-recurrent`)
- MUST log LinQuant as related work (`post_only_unlocated_paper_repo`, `hardware_gate: blocked_on_artifacts`) and explicitly link Tenstorrent Blackhole kernel verification to `OnixResearch/tt-onix#34` (`grants_blackhole_acceptance: false`).
- MUST implement real pure-Rust Gated DeltaNet (GDN) and Mamba-2 recurrent prefill engines verifying exact FP64 algebraic invariance (`<= 1e-12`) between direct recurrence and the LinQuant rewrite BEFORE evaluating low-precision (BF16/FP8-E4M3/INT8) state/logit numerical error and prefill throughput.

## R9: Issue #69 (`varloop-budget-protocol`)
- MUST pin `thinkingmachines/TML-d24-varloop-research@657b53ba5da7baefe9ff15dca5cd25f5bd3dbaa2`, `SqueezeAILab/2DSD@962874f303c3142406ab3eecfae2bb51567815c9` (`feat/inkling-dspark`, Apache-2.0; noting `main@431aade2` lacks Inkling), and `minseo25/2dsd-inkling-2-mini-varloop-dspark-drafter@57f09bbe7bc67a1643f541a7e568de34f65b5762` (Apache-2.0).
- MUST enforce the Linux x86_64 + CUDA 13.0 + BF16 GPU host gate (refusing ARM64 DGX Spark or 24 GB substitution with `unavailable_hardware_runtime`).
- MUST verify greedy 2DSD token-identical losslessness at matched `R` BEFORE timing, evaluate `R = 1` AR, `R = 3` AR, `R = 3` MTP, `R = 3` 2DSD, and `R = 4, 5` transfer with the `R = 3` drafter, record per-loop acceptance length and first-loop vs final-loop argmax agreement, feed measured 2DSD cost into loop-count vs thinking-token comparisons with repetition-filtered and unfiltered columns side-by-side on long-context reasoning vs closed-book knowledge, and ablate shared-vs-separate per-loop KV caches (keeping `#89` separate) and quantization-vs-looping tradeoffs.

## R10: Issue #70 (`ttf-ternary-qwen`)
- MUST pin `oguzeldereli/ttf-home@2f470627050cbb6e163494c2a97eda48c7248d80` (license `NOASSERTION`) against 4-bit GGUF and FP16/BF16 Qwen3.8-27B controls.
- MUST implement a real pure-Rust ternary `{-1, 0, +1}` quantizer with 5-trits-per-byte packing (`<= 1.6` bpw) and exact dequantized matvec parity (`<= 1e-12` vs scalar FP64 reference) compared against `Q4_0` and FP64 baselines.
- MUST report VRAM footprint, prefill/decode tok/s, independently checked math/code/agent pass rates, `cost_per_accepted_task` (`None` at zero accepted), and explicit code/reasoning degradation relative to 4-bit and FP16/BF16 baselines.

## R11: Issue #71 (`drafter-opd-replay`)
- MUST define a bounded default-off open-loop replay harness with explicit noise-floor gating (gains inside repeat noise floor MUST NOT be accepted) and strict separation of lossless vs lossy verdicts.
- MUST compare frozen baseline drafter vs on-policy `Draft-OPD` and `D-PACE` rejection-point weighted adaptation on acceptance length and end-to-end throughput, verifying zero target-distribution drift on lossless arms.

## R12: Issue #72 (`k2-uno-mova`)
- MUST record K2 Horizon (`arXiv:2611.21003`) with honest offline pin status (`arxiv_id_only_artifacts_unverified_offline`, `hardware_gate: blocked_on_artifacts`).
- MUST implement real pure-Rust MoVA + overflow-safe softplus attention gating compared against standard softmax attention, and a draftless `Uno` adapter verified for exact lossless token equivalence (`0` mismatches) against standard autoregressive decoding, reporting speedup, memory overhead, and `batch = 1, 4, 8` scaling against AR and MTP/speculative controls.

## R13: Issue #73 (`lfm3-d1-decision`)
- MUST define default-off adapters for `LFM3-1.7B` (tri-hybrid 256k) and `d1-2B` single-pass typed decision model (`hardware_gate: blocked_on_artifacts`) while explicitly excluding `LFM3-Audio-2B` (`audio-lab#2`).
- MUST measure schema/typed-output conformance, exact ECE and Brier probability calibration with confidence-independent consumer ground-truth grading, single-pass vs autoregressive latency on frozen fixtures, and `LFM3-1.7B` tri-hybrid cached-vs-full parity (`<= 1e-12`) and `8K`–`256K` throughput/retention scaling.
