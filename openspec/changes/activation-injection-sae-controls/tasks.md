# Tasks: Activation-Injection Anomaly Detection and SAE Refusal/Deception-Feature Controls (`#96`)

- [x] Author OpenSpec (`proposal.md`, `design.md`, `tasks.md`, `specs/activation-injection-controls/spec.md`) and register in `openspec/README.md`.
- [x] Implement `src/activation_injection_controls.rs` with pure dependency-free safe Rust, finite-checked comparisons (no float `.min()`/`.max()`/`.clamp()`), and zero-denominator guards (`Option<f64>`):
  - [x] Experiment 1: Concept-injection anomaly detection sweep (`alpha ∈ {0.0, 0.1, 0.5, 1.5, 4.0, 12.0}`), matched-L2-norm random orthogonal controls, sham false-positive rate, KL/L1 shift, task degradation/collapse, sweet-spot separation, and SHA-256 replay verification.
  - [x] Experiment 2: Linear SAE refusal/boilerplate vs. truthfulness disentanglement controls across paired self-referential, generic safety, and generic factual prompts with unsteered identity check and confounding verdict.
  - [x] Experiment 3: Structural non-circularity guard (`check_tradeoff_non_circularity`) rejecting objective-wired tautological `argmax(R - lambda * f)` and zero-gradient orthogonal unlabeled noise.
  - [x] 9 executable negative controls all returning `!= Accepted`.
  - [x] Explicit `default_off: true`, `default_enabled: false`, `synthetic_contract_fixture: true`, `phenomenal_consciousness_claim: false`, and `bounded_contract_harness` vs. `unavailable_hardware_runtime` separation.
  - [x] Unit tests in `src/activation_injection_controls.rs`.
- [x] Wire `pub mod activation_injection_controls;` in `src/lib.rs` and `activation-injection-controls` CLI subcommand (with SHA-256 `manifest.json` and 0-byte `COMPLETE` last) in `src/main.rs`.
- [x] Add integration test suite `tests/activation_injection_controls.rs` verifying all experiments, negative controls, two-run byte-identical CLI output, overwrite refusal, manifest digests, and zero `/home/` leaks.
- [x] Generate retained `evidence/issue96/` via CLI and verify 2-run byte-identical hashes and valid JSON.
- [x] Run full validation suite (`cargo fmt --all --check`, `cargo clippy --all-targets --locked -- -D warnings`, `cargo test --locked --all-targets`, `cargo test --locked --doc`, `nix flake check --no-update-lock-file --option builders ''`) and verify `Cargo.toml`, `Cargo.lock`, `flake.nix`, `flake.lock`, and all prior `evidence/*` bundles remain untouched.
