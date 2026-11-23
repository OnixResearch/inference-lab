# Bootstrap and Batch 1 bounded contract harnesses

## Why

`OnixResearch/inference-lab` starts as an empty repository with 61 tracked issues spanning diffusion proposal benchmarking, host/GPU trace correlation, split-model transformer stages, power/throttle cadence qualification, tokenizer/chat-template conformance, and default-off related-work gates. Establishing a deterministic, dependency-free Rust core with offline-reproducible Nix tooling and strict provenance/non-collapsing outcome contracts gives every subsequent hardware adapter an honest verification boundary.

## Change

1. Bootstrap `inference-lab` with Rust 1.85.1 pinned via offline-compatible Nix flake (`nixpkgs` + `rust-overlay` pins matching `systems-lab`), zero external crate dependencies, `unsafe_code = "forbid"`, CI workflow, documentation, and seed `benchmark` CLI + `evidence/bootstrap/`.
2. Implement pure-Rust `bounded_contract_harness` modules, CLI subcommands, adversarial test suites, and SHA-256-verified retained evidence for Batch 1 issues:
   - `#1`: Bounded diffusion proposal benchmark contract, Noble consumer admission, EBM proposal/scoring/selection/admission fixture, offline pool replay, matched budgets, and default-off `DependentWeave`/JSONL/socketteer candidate/repair tree interchange check.
   - `#40`: Host/GPU trace correlation qualification contract, clock-domain guards, batched-kernel non-attribution, completeness/event-loss tracking, and negative controls.
   - `#41`: Versioned transformer stage & tensor envelope contract, real pure-Rust 4-layer dense causal transformer split/unsplit numerical reference (full/chunked prefill, forced-token decode, reset/reuse), poisoned-buffer/stale-KV guards, and explicit zero-silent-CPU-fallback enforcement for Apple GPU / Tenstorrent P150.
   - `#43`: Power and throttle cadence qualification contract, device-vs-host axis separation, sensor-gated energy reporting, observer perturbation, and known-equivalent / wrong-but-fast / order-drift controls.
   - `#52`: Pure-Rust deterministic BPE + special-token reference tokenizer, `chatml-v1` / `qwen-thinking-v1` chat-template fixtures, role-injection defense, and disqualifying backend parity checker.
3. Log pinned, default-off related-work gate documentation and machine-readable gate contracts for `#56` (Maple FPGA), `#58` (compositional-harness RLM-27B), and `#60` (Latent Lattice Language Models).

## Scope and limitations

All executable harnesses in this change run as deterministic CPU `bounded_contract_harness` checks. Physical GPU, Apple M2 Ultra, Tenstorrent P150, TPU v7x, AWS F2 FPGA, and 27B serving measurements are explicitly reported as `unavailable_hardware_runtime` or `blocked_on_artifacts`. No external claims are treated as lab targets or verified results.
