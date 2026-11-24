# Batch 8 Observable Requirement Specifications

## R1: Pure core, finite-checked math, and non-gating external pins
- All Batch 8 modules MUST compile under `#![forbid(unsafe_code)]` with zero external crate dependencies on Rust 1.85.1 without modifying `Cargo.toml`, `Cargo.lock`, `flake.nix`, `flake.lock`, or prior `evidence/{bootstrap,batch1,batch2,batch3,batch4,batch5,batch6,batch7}/`.
- No Batch 8 module MAY invoke `.max(...)` or `.min(...)` on `f64` or `f32`. Non-finite floats MUST return `Err(...)` or non-`Accepted` outcomes. Zero denominators MUST produce `None` or `Err(...)`.
- Every Batch 8 report MUST emit `default_enabled: false`, `synthetic_contract_fixture: true`, `execution_class: "bounded_contract_harness"`, and an explicit `hardware_gate` (`unavailable_hardware_runtime` or `blocked_on_artifacts`), and MUST NOT import external X/article/video/paper claims as pass/fail criteria.
- Every Batch 8 module MUST retain `>= 8` adversarial negative controls, each producing a non-`Accepted` `OutcomeStatus`.

## R2: Issue #74 (`pdl-breakable-graphs`)
- MUST log Baseten's PDL and breakable prefill CUDA-graph write-up as unverified related work (`@thealexker` X post `2117989442090598798`).
- MUST isolate `pdl_overlap` and `breakable_prefill_graph` behind explicit default-off toggles on a pinned flash-MoE fixture.
- MUST verify exact numerical parity (`<= 1e-12`) against standard launches BEFORE reporting correlated host/GPU kernel-gap bubble and TTFT reductions (`#39`, `#40`).

## R3: Issue #75 (`tensorlean-kernel-verify`)
- MUST log TensorLean (CertiK + Columbia, `@RonghuiGu` X post `2123786837797773807`) as related work with explicit verification scope, TCB assumptions, and `hardware_gate: blocked_on_artifacts`.
- MUST define a default-off Lean formal-verification check boundary on one pinned small reference GPU kernel.
- MUST compare symbolic/formal invariant coverage against finite-input differential numerical testing across seeded kernel bugs (`>= 5` distinct bug classes).

## R4: Issue #76 (`blink-cpu-free-serving`)
- MUST log Blink's CPU-free SmartNIC GPUDirect RDMA + persistent GPU kernel serving architecture (`@thesupermannx` X post `2123339605311111649`) as unverified related work (`hardware_gate: blocked_on_artifacts`).
- MUST compare P50/P99 TTFT and decode jitter of host-scheduled serving (vLLM/SGLang-style) vs GPU-resident persistent scheduling across controlled host-CPU contention levels (`idle`, `moderate`, `heavy`).
- MUST enforce a default-off RDMA-to-GPU qualification gate requiring public Blink artifacts and compatible GPUDirect RDMA NIC hardware.

## R5: Issue #77 (`qwen35-neuron-steering`)
- MUST encode the exact Qwen3.5-4B two-neuron `down_proj` scaling formula (`L14 N8949 +1`, `L13 N7956 -1`, `dose * median(||resid||) / ||W_down[:, n]||` at `down_proj`) as a default-off fixture extending `#45` (`intervention-replay`).
- MUST verify deterministic bit-identical replay across runs and record dose-response curves against both an unsteered control and a matched-L2-norm random-direction control.

## R6: Issue #78 (`hnet-byte-modeling`)
- MUST log H-Net v2 byte-level hierarchical modeling (`@_albertgu` X post `2115502110578290900`) as unverified related work with `hardware_gate: blocked_on_artifacts` noting public weights/code are required before live execution.
- MUST implement a pure-Rust byte-level UTF-8 dynamic chunking and hierarchical reconstruction parity check (`<= 1e-12`) comparing tokenizer-free edge cases against a BPE baseline (`#52`) alongside decode-step overhead accounting.

## R7: Issue #79 (`tkcc-hillclimb-profile`)
- MUST log Together Compiler (TKCC, `@simran_s_arora` X post `2118032475498983540`) as unverified related work beside `#53` and `#61`.
- MUST profile local hill-climb experiment-cycle bottlenecks (`compile`, `warm_up`, `parity_check`, `measurement`, `teardown_overhead`), report Amdahl speedup ceilings, preserve failed/rejected candidates in denominators, and reject parity-skipping shortcuts.

## R8: Issue #80 (`exl3-longctx-quant`)
- MUST pin `pepuscz/strix-halo-llm@529935497a97bc61a1d71aea9a8864f81ebc82bb` (`MIT`) and reference `mw00/project-maya#908` as a default-off EXL3 2.25-bpw quantization arm against 4-bit/FP4 and FP64 baselines.
- MUST implement a real pure-Rust exact `2.25`-bpw (36 bits per 16-weight tile) trellis/block quantizer with bit-exact pack/unpack round-trip and dequantized matvec parity (`<= 1e-12` vs scalar FP64 reference).
- MUST evaluate independently checked long-context reasoning and coding tasks (explicitly separated from needle-in-a-haystack retrieval) across `128K`–`1M` context alongside VRAM, throughput, and `cost_per_accepted_task` (`None` at zero accepted).

## R9: Issue #81 (`saber-dflash2-power`)
- MUST pin `GGZ14/saber-engine@1da27661b3bf8a702a7532c8d17651de8c635e44` (`Apache-2.0`) as a default-off backend arm with DFlash2 speculative decoding on Qwen3.8-27B.
- MUST verify numerical and token parity (greedy exact match and stochastic rejection sampling) against vLLM/llama.cpp references BEFORE admitting acceptance length, TTFT, and sensor-gated power-capped `tok/s/W` and `J/token` (`#43`). Missing power telemetry MUST remain `unknown`, never zero.

## R10: Issue #82 (`mimo-hysparse2-kv`)
- MUST log Xiaomi MiMo HySparse2 (`@SemiAnalysis_` X post `2119142144749293863`) as unverified related work (`hardware_gate: blocked_on_artifacts`, 5x prefill / 4.5x KV claims unverified).
- MUST implement a pure-Rust causal encoder-decoder KV projection engine verifying exact cached-vs-full parity (`<= 1e-12`), strict causal mask non-leakage, prefill-compute and KV-byte ablation across `8K`–`1M` context, and independent long-context accuracy checks (`#28`).

## R11: Issue #83 (`apple-nvidia-sidecar`)
- MUST pin `joesinvestments/apple-nvidia-sidecar@c59a276a1ce5fc3c2e12e05f070a0180550cb619` (`MIT`) as related work beside `#41` and `#42`.
- MUST verify exact numerical parity (`<= 1e-12`) between single-device execution and a 2-stage heterogeneous Apple-host + discrete-NVIDIA-eGPU split transformer, measure link bandwidth, stage handoff latency, and concurrency scaling, and refuse silent CPU fallback.

## R12: Issue #84 (`silicon-codesign-audit`)
- MUST log `arXiv:2611.18859` (`@vithursant19` X post `2122697450196730308`) as unverified pre-silicon related work (`hardware_gate: blocked_on_artifacts`).
- MUST record workload, precision, memory-bandwidth, and energy assumptions in a structured audit compared via deterministic roofline and energy-per-token calculations against lab hardware baseline envelopes (`#43`), explicitly flagging every item requiring physical silicon validation.
