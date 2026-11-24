# Batch 9 Observable Requirement Specifications

## R1: Pure core, finite-checked math, and non-gating external pins
- All Batch 9 modules MUST compile under `#![forbid(unsafe_code)]` with zero external crate dependencies on Rust 1.85.1 without modifying `Cargo.toml`, `Cargo.lock`, `flake.nix`, `flake.lock`, or prior `evidence/{bootstrap,batch1,batch2,batch3,batch4,batch5,batch6,batch7,batch8}/`.
- No Batch 9 module MAY invoke `.max(...)` or `.min(...)` on `f64` or `f32`. Non-finite floats MUST return `Err(...)` or non-`Accepted` outcomes. Zero denominators MUST produce `None` or `Err(...)`.
- Every Batch 9 report MUST emit `default_enabled: false`, `synthetic_contract_fixture: true`, `execution_class: "bounded_contract_harness"`, and an explicit `hardware_gate` (`unavailable_hardware_runtime` or `blocked_on_artifacts`), and MUST NOT import external X/blog/video/model-card/paper claims as pass/fail criteria.
- Every Batch 9 module MUST retain `>= 8` adversarial negative controls, each producing a non-`Accepted` `OutcomeStatus`.

## R2: Issue #85 (`link-sparse-attention`)
- MUST log Constructor AGI's `Link` (`@michaeltgao`, X post `2123204373773983992`) as unverified related work (`hardware_gate: blocked_on_artifacts`, +53% retrieval at 4M and 9.9x GPU efficiency unverified).
- MUST verify pure-Rust learned block-router full-read parity (`<= 1e-12`) against dense causal attention and bound a default-off long-context retrieval AND end-task check (`#23`, `#31`) disqualifying faster-but-wrong sparse candidates.

## R3: Issue #86 (`mit-qda-quant`)
- MUST log MIT QDA (`@songhan_mit` `2123621155756036191`, `@xyli_bruce` `2123613817129116038`) as unverified related work (`hardware_gate: blocked_on_artifacts`).
- MUST compare a pure-Rust QDA-proposed mixed-precision quantization recipe against pinned uniform GGUF-INT4, NVFP4, and FP64 baselines on bit budget, Spearman KL rank correlation (`#18`), independently checked task accuracy, and `cost_per_accepted_task` (`None` at zero accepted).

## R4: Issue #87 (`knowledgator-sys1-extract`)
- MUST log Knowledgator System 1 (`@knowledgator` `2123851865821700260`, `knowledgator.com/blog/system-one-model`) as a default-off arm (`hardware_gate: blocked_on_artifacts`, ~100x claim unverified) alongside pinned Clairic (`#55`) and a structured-output LLM baseline.
- MUST compare exact-field F1, strict schema conformance, span grounding, and throughput on a frozen hierarchical-extraction fixture while treating model confidence as informational only.

## R5: Issue #88 (`unlimited-ocr-longdoc`)
- MUST log Unlimited-OCR 3B (`@byteonai` `2124163291157008857`, MIT, 32K context, vLLM/SGLang compatible, 93% OmniDocBench v1.5 and `<0.11` edit distance unverified) behind a default-off adapter (`hardware_gate: blocked_on_artifacts`).
- MUST compute exact pure-Rust normalized Levenshtein edit distance, cross-page table/heading preservation, memory & 32K context overflow bounds, and local pages/sec against page-chunked OCR on a frozen multi-page PDF structure fixture.

## R6: Issue #89 (`loop-kv-compression`)
- MUST log `ResidualQuant` (Oct 2026), `LoopVQ` (Nov 2026), and *Looping Deeper Without Growing the KV Cache* (Nov 2026) as unverified related work beside `#28` and `#69`.
- MUST implement pure-Rust cross-loop KV cosine similarity, bit-exact INT2 cross-loop residual quantization, and sub-1-bit (`0.5` and `0.25` bpw) LoopVQ codebook quantization with zero-residual exact parity (`<= 1e-12`), reconstruction/attention error, and `O(L*K*N)` vs compressed memory accounting.

## R7: Issue #90 (`alora-prefix-cache`)
- MUST log `aLoRA` (`arXiv:2512.17910`) and `FastLibra`/`ELORA` (`arXiv:2505.03756`) alongside Punica/S-LoRA, `#25`, `#49`, and `#66` (`Hypertune`).
- MUST prove exact pure-Rust prefix-KV invariance (`<= 1e-12`) and cached-prefix-vs-full equivalence (`<= 1e-12`) for delayed-suffix `aLoRA` and `Hypertune` soft-prefill KV vs standard LoRA prefix perturbation, alongside shared-prefix TTFT, HBM usage, and independent task accuracy.

## R8: Issue #91 (`perseus-mpk-rdma`)
- MUST log `Perseus`, `MPK` (Mirage Persistent Kernel), and `ForgeMegakernel` as unverified related work beside `#61`, `#63`, and `#76`.
- MUST verify deterministic multi-node collective value/ordering parity (`<= 1e-12`) while modeling NIC fence/signaling stalls (`naive_proxy_put_signal`) vs decoupled per-destination signaling (`perseus_decoupled_signaling`) and SM-level persistent graphs (`mpk_sm_persistent_graph`) for 2-node DGX Spark RDMA decode.

## R9: Issue #92 (`moe-spec-roofline`)
- MUST log `arXiv:2610.38981`, `QuantSpec` (`arXiv:2502.10424`), and `MoE-SpeQ` beside `#6`, `#21`, and `#65`.
- MUST measure exact pure-Rust expert-routing union and temporal overlap across consecutive draft tokens, verify batched MoE verification parity (`<= 1e-12`), and evaluate an analytical memory-bandwidth & KV-cache roofline governor against naive dense roofline predictions.

## R10: Issue #93 (`anchor-label-drafter`)
- MUST log `arXiv:2609.38795` (*Anchor-Label Relabelling* & *In-Rollout Anchors*) beside `#71` (`Draft-OPD` & `D-PACE`).
- MUST implement pure-Rust cascade-invalidation recovery via Anchor-Label Relabelling vs naive off-policy, Draft-OPD, and D-PACE under exact matched drafter training budgets, proving exact analytical lossless target-distribution invariance (`<= 1e-12`) and enforcing 2-sigma noise-floor gating.

## R11: Issue #94 (`parasite-feature-control`)
- MUST log *Parasite Features* (`arXiv:2611.26711`), `dounick/parasite-features@1df61727bb2f502b882d334eb7f2ca1a38e05712` (missing license; reference only, zero code copied), and both X posts (`@SashaBoguraev`, `@tpimentelms`) as non-gating related work.
- MUST add typed rank-k `SubspaceInterchange` and `ProjectionAblation` manifest kinds with SHA-256 digests covering basis, rank, layer, and token positions, strict orthonormality/rank validation, necessity ablation against `>= 10` (16) matched-rank random orthonormal subspaces, separate `sufficiency_passed` and `necessity_passed` fields, a planted-parasite backend variant classified as `sufficient_not_necessary` (NEVER `found_mechanism`), a positive mechanism control passing both, bit-identical replay receipts, and `#77`/`rsi-lab#172` necessity-reporting notes.

## R12: Issue #95 (`ling-fp8kv-alloc-cap`)
- MUST log `tonyd2wild/Ling-3.2-air-NVFP4-DGX-Spark@d2548801e02c3bfb337a35a9fd9b0d16d71df3f2` (`MIT`), both `@Tech2Wild` posts, pinned vLLM image digest, NVFP4 weights `e7461aad4dfaedf0e4064d3db0e08d53ce641711`, and base weights `1613b1a4dcc6aa0d8f346606628cd9a92061a525` as non-gating related work.
- MUST freeze one profile per arm via the `#35` schema whose digest changes on KV dtype, allocator cap (`FP8FIX_TORCH_GROWTH_GIB`), MTP draft count, `--gpu-memory-utilization`, max context (`262144`), and each patch file digest, rejecting floating tags/unpinned revisions and marking missing-patch arms `Unsupported`.
- MUST record `host_mem_available`, `reported_kv_pool`, `post_warmup_torch_reservation`, and `peak_torch_reservation` as separate `#11` `MemoryCounterObservation` rows under a hard `MemAvailable` brake without silent retry, evaluate multi-depth needle recall (`32K`/`128K`/`215K+`, flagging position-0 probe as memory-pressure check only) and full-pool stress, measure pure-Rust BF16-vs-FP8 KV & isolated 1-stage MLA decode parity, and report `1K`/`32K` MTP off/on throughput with local noise floor.
