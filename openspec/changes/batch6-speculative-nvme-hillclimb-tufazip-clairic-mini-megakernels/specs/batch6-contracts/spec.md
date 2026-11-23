# Batch 6 Observable Requirement Specifications

## R1: Pure core, finite-checked math, and non-gating external pins
- All Batch 6 modules MUST compile under `#![forbid(unsafe_code)]` with zero external crate dependencies on Rust 1.85.1 without modifying `Cargo.toml`, `Cargo.lock`, `flake.nix`, `flake.lock`, or prior `evidence/{bootstrap,batch1,batch2,batch3,batch4,batch5}/`.
- No Batch 6 module MAY invoke `.max(...)` or `.min(...)` on `f64` or `f32`. Non-finite floats MUST return `Err(...)` or non-`Accepted` outcomes. Zero denominators MUST produce `None` or `Err(...)`.
- Every Batch 6 report MUST emit `synthetic_contract_fixture: true`, `execution_class: "bounded_contract_harness"`, and an explicit `hardware_gate` (`unavailable_hardware_runtime` or `blocked_on_artifacts`), and MUST NOT import external X/blog/paper claims as pass/fail criteria.

## R2: Issue #47 (`adv-spec-reasoning`)
- MUST extend `#3`/`#6`/`#22` drafting rather than claiming a competing benchmark.
- MUST measure per-slot `max_abs_err` and `l2_err` comparing independent exact slot-0 target reference against multi-slot speculative verifier slots `k = 0..K-1`.
- MUST compare unbiased draft vs decision-point-rejecting adversarial draft under matched seeds on accepted-task rate (verified by independent task checker), wall-time proxy, and slot-0 exactness.

## R3: Issue #48 (`nvme-coalescing`)
- MUST record NVMe interrupt coalescing state (`on`/`off`/`unknown`), drive identity/model, systemd unit status (`nvidia-nvme-interrupt-coalescing.service`), and page-cache state (`cold`/`warm`/`unknown`) as explicit controlled factors.
- MUST parse `nvme get-feature /dev/nvme0 -f 8 -H` output deterministically without mutating host state.
- MUST keep 4K QD1 storage latency strictly separate from SSD-paged Engram/PLE decode latency and tok/s claims.

## R4: Issue #49 (`harness-lora-compare`)
- MUST compare `unadapted_base`, `prompt_only_change`, `harness_only_change`, and `harness_specific_lora` (`harness-hub/Qwen3.8-27B-pi-adapted`) on a frozen agent task set using an independent task checker.
- MUST report accepted tasks, tool calls, total cost including tool calls, and `cost_per_accepted_task = None` when `accepted_tasks == 0`.
- MUST perform a benchmark-overlap / contamination audit rejecting tasks overlapping the adapter training manifest from clean held-out scoring.

## R5: Issue #50 (`latent-feedback`)
- MUST execute a real miniature pure-Rust sequence learner comparing token-to-token top-layer latent feedback against a standard Transformer control given matched exact FLOPs via extra epochs on the same data-limited training set and disjoint held-out set.
- MUST validate analytical gradients against an independent central finite-difference checker (`<= 1e-6`) and keep large-scale stability explicitly `open_untested`.

## R6: Issue #51 (`simplehc-mhc`)
- MUST record `primary_source_pin_status: "post_only_unlocated_paper_repo"` and `hardware_gate: blocked_on_artifacts`.
- MUST compare multi-stream DeepSeek mHC (real iterative Sinkhorn-Knopp doubly-stochastic manifold projection verified to `<= 1e-6`), SimpleHC, and `#33` block attention residuals at matched parameter count and tokens with independent scalar reference parity (`<= 1e-12`) and exact op/FLOP accounting.

## R7: Issue #53 (`serving-hill-climb`)
- MUST be default-off (`default_enabled: false`) and pin `sybil-solutions/inference-optimisation-skill@fda968e7fc1996aa23c67139c755fe541c9d4be0` (MIT), reusing `#7` contract conventions.
- MUST compute practical ceilings from measured bandwidth & FLOPs BEFORE lever search and refuse targets above ceiling.
- MUST evaluate bounded one-lever candidates under `Control-Candidate-Control` interleaving with `>= 6` samples per run, accepting ONLY if candidate median beats BOTH neighboring controls and regresses no declared secondary metric by `> 3%`, while retaining ALL failures in the ledger.
- MUST enforce server-usage/tokenizer token accounting (rejecting streamed chunks), unified-memory residency, needle retrieval, concurrent full-KV soak, greedy lossless equivalence (flagging lossy levers as quality trades), and clean-checkout rebuild recheck.

## R8: Issue #54 (`tufazip-online-codec`)
- MUST be default-off (`default_enabled: false`), pin `Tufalabs/tufazip@b9d678db3a0f726e117c8cab51927975830395ac` (MIT), refuse full `enwik9` as first arm, and record the 2×B200 CUDA 13.0 hardware gate.
- MUST execute a real pure-Rust online-learning fixed-point range codec (initialized from serialized quantized offline prior, decoder repeating online weight updates with zero transferred online weights) verified for bit-exact SHA-256 round-trip, reporting `payload + prior` total bytes alongside a conventional LZ77 baseline on the same pinned prefix.
- MUST verify as a negative control that skipping decoder-side online weight updates fails lossless round-trip.

## R9: Issue #55 (`clairic-extraction`)
- MUST be default-off (`default_enabled: false`), pin `Cloudflare/clairic@2c161be5cc71bacd6605b99a8c07547e58cfa434` (27B, Apache-2.0) and `Cloudflare/clairic-flash@36ad68b999560f56b57ef16a22180f8cb83c6d77` (9B preferred), and NOT start `#10` shared-library extraction.
- MUST reject the outdated "single forward pass" preview claim and model the 18 Nov parallel slot-autoregressive text/number + trained typed-head architecture.
- MUST compare against a documented structured-output baseline on a frozen independently labeled fixture verifying schema validity, independent ground-truth value agreement (well-typed wrong value = FAIL), exact character-span grounding, and non-gating `p_null`/confidence scores.

## R10: Issue #57 (`inkling-mini-serve`)
- MUST be default-off (`default_enabled: false`), pin `thinkingmachines/Inkling-2-Mini@6e7c8248f75265bdd4fbab5273304243b5d8c2cb`, `thinkingmachines/TML-d24-varloop-research@767c9d0250aa2938e289e50ca6db5f1b39ebae74`, and `thinking-machines-lab/tinker-cookbook@37261cae3d3ff11f08573b2a84a4d44c809c38f3` (Apache-2.0 + Model AUP), and stay distinct from `#33`, `#26`, `#35`, `#18`, `rsi-lab#88`, and `#59`.
- MUST enforce the admitted host floor (`DGX Spark` / `>=32 GB` Apple Silicon / `>=32 GB` VRAM) and official runtime allowlist, returning clean `unsupported` when unmet.
- MUST verify miniature 24-stored-layer (middle 12 looped 3× = 48 applications, 8-of-128 MoE, NVFP4/MXFP8) unrolled-vs-looped numerical equivalence (`<= 1e-12`) and optional varloop contract.

## R11: Issue #59 (`maya-ik2-serve`)
- MUST be default-off (`default_enabled: false`), pin `alesha-pro/maya-inkling-mini@c520fd14de3b21d67ddc6b7e7348b761aad26d52` (MIT), `ggml-org/llama.cpp@34ff2e047ce520b1d83bd274bf3ff0bfbc72a69b` (`b12590`), and `unsloth/Inkling-2-Mini-GGUF@ae9327c5fbc7333116ba32eee78bb5d2bbefae4f` (`Inkling-2-Mini-UD-Q4_K_XL.gguf`, Apache-2.0), stay distinct from `#57`, and record X post claims separately from later README claims.
- MUST enforce Linux + 24 GB `sm_86` + CUDA 12.8 + `UD-Q4_K_XL` GGUF host gate, record `gpu_unlock` status without installing it, and verify packed 4-bit expert register-unpack parity against an independent scalar reference.

## R12: Issue #61 (`decode-megakernels`)
- MUST pin `dangxingyu/megakernel-zoo` with honest offline pin status and complement `#4` and `#39`.
- MUST compare an independent multi-launch reference decode loop against a fused persistent-kernel decode loop on a miniature transformer, enforcing token & logit equivalence (`<= 1e-12`) BEFORE any timing/speedup is reported.
- MUST report batch=1 vs batch>1 (`batch = 1, 4, 8`) scaling survival, latency percentiles (`p50/p95/p99`), host-sync reduction, and an agent kernel-build ledger retaining failure modes.
