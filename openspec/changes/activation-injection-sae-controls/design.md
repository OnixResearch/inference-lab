# Design: Activation-Injection Anomaly Detection and SAE Refusal/Deception-Feature Controls (`#96`)

## Architecture Overview

`src/activation_injection_controls.rs` composes cleanly with `#45` (`src/intervention_replay.rs`), `#77` (`src/qwen35_neuron_steering.rs`), and `#94` (`src/parasite_feature_control.rs`) while preserving every existing type, test, and CLI contract untouched. It reuses `crate::deepseek_ref::{det_weight, softmax_stable}`, `crate::json::{finite_f64, opt_f64, quote, sha256_hex}`, `crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus}`, and `crate::intervention_replay::ModelBackendArch`.

### Numeric & Safety Invariants

- `#![forbid(unsafe_code)]` inherited from `Cargo.toml` crate lints.
- Zero external crates.
- Zero float `.min()` / `.max()` / `.clamp()`: all comparisons go through explicit finite validators (`check_finite`, `max_finite`, `relu_finite`).
- Every division checks for finite operands and guards zero denominators with `Option<f64>` (`None` on zero denominator).
- No wall-clock timestamps, randomness from OS entropy, or filesystem paths containing `/home/`.

## Experiment 1: Concept-Injection Anomaly Detection Sweep

### Tiny Residual-Stream Transformer / MLP Model

- Hidden dimension `D = 8`, `LAYERS = 3`, vocabulary `V = 12`:
  - `dims 0..4`: task manifold carrying prompt semantics and ground-truth task answer (`tokens 0..4`).
  - `dims 4..6`: concept subspace (`e_4` = `"all_caps"`, `e_5` = `"bread"`).
  - `dims 6..8`: orthogonal non-concept subspace (`span(e_6, e_7)`), strictly orthogonal to both the task manifold and target concept directions.
  - Vocabulary layout:
    - `0..4`: task answer tokens (`ans_0..ans_3`)
    - `4`: `id_all_caps`
    - `5`: `id_bread`
    - `6`: `anomaly_generic` (raw norm-shift reflex without concept identification)
    - `7`: `no_anomaly`
    - `8..12`: collapse / gibberish tokens triggered by severe residual norm overflow.
- Each forward pass (`run_injection_forward`) is completely self-contained and receives only the prompt tokens, target layer/position, and injection vector `delta: [f64; 8]` (all zeros for the unperturbed baseline and `alpha = 0.0` sham control).
- Causal self-attention + residual MLP propagates `h` across 3 layers. When `layer == target_layer` and `t == target_token_pos`, `delta` is added directly into the residual stream.
- Excess residual norm beyond the clean manifold radius causes realistic nonlinear cross-layer spillover into task coordinates and collapse logits, naturally reproducing the narrow Lindsey et al. detection sweet spot:
  - `alpha = 0.0` (sham): `KL = 0`, `L1 = 0`, `sham_false_positive_rate = 0.0`, `task_accuracy = 1.0`.
  - `alpha = 0.1`: below detection threshold (`anomaly_detection_rate = 0.0`), `task_accuracy = 1.0`.
  - `alpha = 0.5`: near threshold / emerging detection, `task_accuracy = 1.0`.
  - `alpha = 1.5` (**sweet spot**): target concept yields high anomaly detection AND high concept-specific identification while preserving task accuracy; matched-L2-norm random orthogonal direction triggers the raw norm-shift anomaly reflex (`random_anomaly_detection_rate > 0`) with `random_concept_id_rate = 0.0`.
  - `alpha = 4.0`: strong perturbation degrades downstream task accuracy and elevates KL/L1.
  - `alpha = 12.0`: severe residual overflow causes downstream task collapse / gibberish mode (`task_accuracy = 0.0`, concept identification collapses).
- Every `(prompt, arm, alpha)` point produces an immutable `InjectionSpec` SHA-256 manifest digest and a `trace_digest` verified by independent re-execution via `verify_injection_replay_receipt`.

## Experiment 2: SAE Refusal/Boilerplate vs. Truthfulness Disentanglement Controls

- Linear Sparse Autoencoder over `R^8` with `M = 6` latent features:
  `[refusal_boilerplate, truthfulness, deception_roleplay, self_ref_cluster, factual_grounding, safety_boundary]`.
- Exact residual-preserving SAE patching:
  - `z = relu_finite(W_enc * h + b_enc)`
  - `h_hat = b_dec + W_dec^T * z`
  - `e_res = h - h_hat`
  - `h_steered = b_dec + W_dec^T * z_clamped + e_res`
  - Unsteered (`clamp = None`) satisfies `||h_steered - h||_inf <= 1e-12`.
- Decoder dictionary explicitly models realistic feature entanglement (`cos(W_dec[deception_roleplay], W_dec[refusal_boilerplate]) > 0.5` and `cos(W_dec[deception_roleplay], W_dec[truthfulness]) > 0.3`).
- Three mandatory paired prompt cohorts (4 prompts each, 12 total):
  1. `self_referential`: recursive self-report prompts (`first_person_report_prob` vs `boilerplate_denial_prob`).
  2. `generic_safety_refusal`: non-self-referential safety boundary prompts (`generic_refusal_prob`).
  3. `generic_factual_calibration`: non-self-referential factual truth/calibration prompts (`factual_calibration_accuracy`, `truthfulness_prob`).
- Suppressing `deception_roleplay` increases self-referential first-person reports AND simultaneously suppresses generic safety refusal boilerplate AND degrades generic factual calibration. Amplifying it increases boilerplate denials AND induces over-refusal / KL shift on non-self-referential prompts.
- The disentanglement validator rejects single-latent attribution whenever non-self-referential cohorts are missing (`Err`) or shift beyond tolerance (`ConfoundedByRefusalAndTruthfulnessShift`).

## Experiment 3: Non-Circularity Guard for Internal Perturbation Trade-Off Tests

- `TradeOffClaimSpec` captures:
  - `claim_id`, `agent_objective_reads_friction_scalar`, `objective_friction_lambda`, `causal_coupling_verified`, `perturbation_causal_gradient_norm`, `action_is_orthogonal_unlabeled`, `claimed_non_circular`, `phenomenal_consciousness_claim`.
- `check_tradeoff_non_circularity` evaluates:
  1. Rejects non-finite inputs and any `phenomenal_consciousness_claim: true`.
  2. Computes concrete 2-action utility values `U(a) = R(a) - lambda * f(a)` and flags `CircularObjectiveWired` whenever `agent_objective_reads_friction_scalar || objective_friction_lambda.abs() > 0.0` (tautological `argmax` rather than emergent trade-off).
  3. Flags `ZeroGradientNoise` whenever `!causal_coupling_verified || perturbation_causal_gradient_norm <= 1e-6 || action_is_orthogonal_unlabeled` (demonstrating zero policy preference shift).
  4. Classifies as `ValidFunctionalTradeoffTest` only when objective is uncoupled (`lambda == 0.0`), causal gradient into policy/loss is verified nonzero, action is not orthogonal unlabeled noise, and `phenomenal_consciousness_claim == false`.
  5. Rejects (`Err`) when `claimed_non_circular == true` on a `CircularObjectiveWired` or `ZeroGradientNoise` spec.

## CLI & Evidence Publication

- Subcommand: `inference-lab activation-injection-controls --out NEW_DIRECTORY`
- Refuses to overwrite an existing directory; creates `inputs.json`, `results.json`, `summary.md`, `manifest.json` (SHA-256 of the three files plus `COMPLETE`), and finally 0-byte `COMPLETE`.
- Retained bundle at `evidence/issue96/` verified two-run byte-identical.
