# Activation-Injection Anomaly Detection and SAE Refusal/Deception-Feature Controls (`#96`)

## Why

Two white-box mechanistic interpretability lines probe model responses to direct internal activation perturbations rather than surface prompts:

1. **Jack Lindsey et al. (Anthropic), *"Emergent Introspective Awareness in Large Language Models"* (`transformer-circuits.pub`):** injects isolated concept vectors into mid-to-late residual streams and tests whether the model detects the perturbation prior to generating telltale text, finding fragile, injection-strength-sensitive anomaly detection that the authors frame as functional anomaly detection rather than consciousness.
2. **Cameron Berg, Diogo de Lucena, & Judd Rosenblatt (`arXiv:2510.24797`), *"Large Language Models Report Subjective Experience Under Self-Referential Processing"*:** steers Sparse Autoencoder (SAE) features labeled for deception/roleplay on Llama 3.3 70B under recursive self-reference, observing higher first-person reports when suppressing those features and boilerplate denials when amplifying them—a pattern that can reflect suppressing generic alignment/refusal boilerplate or perturbing factual calibration rather than unmasking an authentic inner state.

`inference-lab` already owns the deterministic intervention replay harness (`#45`), steering regression fixtures (`#77`), and subspace interchange / necessity ablation / planted-parasite controls (`#94`). Tracking `OnixResearch/inference-lab#96` extends this family with a strictly **default-off** deterministic contract harness covering:
- an activation-injection strength sweep with `alpha = 0.0` sham and matched-L2-norm random orthogonal controls separating raw norm-shift reflexes from concept-specific identification and downstream task collapse;
- a linear SAE feature-disentanglement control pairing self-referential prompts with generic non-self-referential safety and factual prompts; and
- a structural non-circularity guard for internal perturbation vs. reward trade-off claims.

## Change

1. **Pure-Rust contract module (`src/activation_injection_controls.rs`)**:
   - Strictly `#![forbid(unsafe_code)]`, zero external dependencies, finite-checked comparisons (no float `.min()` / `.max()` / `.clamp()`), and zero-denominator guards (`Option<f64> = None`).
   - Records `default_off: true`, `default_enabled: false`, `synthetic_contract_fixture: true`, `phenomenal_consciousness_claim: false`, `execution_class: "bounded_contract_harness"`, and an explicit `unavailable_hardware_runtime` gate for large-model hardware execution.
   - **Experiment 1 (Concept-Injection Anomaly Detection Sweep)**: deterministic miniature residual-stream transformer/MLP with independent unperturbed baseline vs. perturbed forward passes across `alpha ∈ {0.0, 0.1, 0.5, 1.5, 4.0, 12.0}`, comparing target concept vectors against matched-L2-norm random orthogonal directions with SHA-256 replay verification.
   - **Experiment 2 (SAE Refusal/Boilerplate vs. Truthfulness Disentanglement)**: deterministic linear SAE over residual activations with entangled `refusal_boilerplate`, `truthfulness`, `deception_roleplay`, and `self_ref_cluster` features evaluated across paired self-referential, generic safety-refusal, and generic factual-calibration prompts.
   - **Experiment 3 (Non-Circularity Guard for Internal Perturbation Trade-Off Tests)**: structural validator `check_tradeoff_non_circularity` rejecting objective-wired circular claims (`U = R - lambda * f`, tautological `argmax`) and zero-gradient orthogonal unlabeled noise.
   - **Executable Negative Controls (`>= 8`)**: 9 executable adversarial checks all returning non-`Accepted` outcomes.
2. **CLI & Evidence (`src/main.rs`, `src/lib.rs`, `evidence/issue96/`)**:
   - Add `inference-lab activation-injection-controls --out NEW_DIRECTORY` publishing `inputs.json`, `results.json`, `summary.md`, SHA-256 `manifest.json`, and 0-byte `COMPLETE` written last.
   - Retain byte-identical `evidence/issue96/` verified across two independent runs without modifying any prior `evidence/*` bundles, `Cargo.toml`, `Cargo.lock`, `flake.nix`, or `flake.lock`.
3. **Tests (`tests/activation_injection_controls.rs` + unit tests)**:
   - Verify all three experiments, all 9 negative controls, replay determinism, two-run byte-identical CLI output, overwrite refusal, and zero `/home/` leaks.

## Scope and Limitations

- This change implements a bounded deterministic CPU contract harness (`bounded_contract_harness`) and does NOT claim live Llama 3.3 70B, frontier-model residual-stream injection, or GPU execution (`unavailable_hardware_runtime`).
- Strictly functional mechanistic evaluation (`phenomenal_consciousness_claim: false`).
- Default-off (`default_off: true`); does not alter default serving profiles or break `#45`, `#77`, or `#94` contracts.
