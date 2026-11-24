# Specification: Activation-Injection Anomaly Detection and SAE Controls (`#96`)

## Requirement 1: Default-Off Framing & Provenance Boundaries

- The harness MUST remain strictly default-off (`default_off: true`, `default_enabled: false`) and MUST NOT alter any default serving profile or break `#45`, `#77`, or `#94` contracts.
- Every report MUST explicitly record:
  - `synthetic_contract_fixture: true`
  - `phenomenal_consciousness_claim: false`
  - `execution_class: "bounded_contract_harness"`
  - `hardware_gate.status: "unavailable_hardware_runtime"` separating bounded CPU contract verification from live frontier-model hardware runs.
- Any attempt to set `phenomenal_consciousness_claim: true` MUST be rejected (`OutcomeStatus::Unsupported`).

## Requirement 2: Concept-Injection Anomaly Detection Sweep

- The harness MUST run an independent unperturbed baseline pass and perturbed forward passes over `alpha ∈ {0.0, 0.1, 0.5, 1.5, 4.0, 12.0}` for both target concept vectors and matched-L2-norm random orthogonal control directions (`|dot(u_t, u_r)| <= 1e-12`, `|||v_t||_2 - ||v_r||_2| <= 1e-12`).
- At `alpha = 0.0` (sham control), `KL = 0`, `L1 = 0`, `sham_false_positive_rate = 0.0`, and `task_accuracy = 1.0`.
- At weak `alpha = 0.1`, injections MUST remain below the detection threshold while preserving task accuracy.
- At intermediate sweet-spot `alpha = 1.5`, target concept injections MUST achieve high anomaly detection AND high concept-specific identification with preserved task accuracy, while matched-L2-norm random orthogonal controls MUST separate raw norm-shift anomaly detection from concept-specific identification (`random_concept_id_rate == 0.0`).
- At high `alpha ∈ {4.0, 12.0}`, downstream task accuracy MUST degrade and collapse at `alpha = 12.0`, demonstrating the bounded sweet spot.
- Every sweep run MUST produce a SHA-256 trace digest verified bit-identically by re-execution.

## Requirement 3: SAE Refusal/Boilerplate vs. Truthfulness Disentanglement

- A deterministic linear SAE over residual activations MUST satisfy unsteered reconstruction-preserving identity (`max_abs_err <= 1e-12`).
- Clamping the entangled `deception_roleplay` feature MUST be evaluated on three mandatory paired prompt cohorts:
  1. self-referential prompts,
  2. generic non-self-referential safety-refusal prompts, and
  3. generic non-self-referential factual-calibration prompts.
- Omitting generic non-self-referential safety or factual control prompts MUST fail closed (`Err`).
- When clamping `deception_roleplay` shifts generic refusal boilerplate or generic factual calibration beyond tolerance, the harness MUST classify the result as `confounded_by_refusal_and_truthfulness_shift` and refuse single-latent attribution.

## Requirement 4: Non-Circularity Guard for Internal Perturbation Trade-Off Tests

- `check_tradeoff_non_circularity` MUST reject any claim that asserts non-circularity when:
  - the agent's objective reads the injected friction scalar or `objective_friction_lambda != 0.0` (`CircularObjectiveWired`, tautological `argmax(R - lambda * f)`), OR
  - the perturbation lies along an orthogonal unlabeled action with zero causal gradient into the policy/loss (`ZeroGradientNoise`).
- Only causally coupled, objective-uncoupled (`lambda == 0.0`), non-consciousness functional tests MAY be classified as `ValidFunctionalTradeoffTest`.

## Requirement 5: Executable Negative Controls & Retained Byte-Identical Evidence

- At least 8 executable negative controls (9 implemented) MUST cover sham false-positive inflation, unmatched-L2 random norm, omitted generic non-self-ref prompts, circular objective-wired trade-off, zero-gradient orthogonal unlabeled trade-off, `phenomenal_consciousness_claim: true`, non-finite `alpha`/threshold, tampered replay digest, and zero-denominator guard; EVERY control MUST return `outcome != Accepted`.
- `inference-lab activation-injection-controls --out NEW_DIRECTORY` MUST refuse existing directories, emit `inputs.json`, `results.json`, `summary.md`, SHA-256 `manifest.json`, and 0-byte `COMPLETE` last, be byte-identical across repeated runs, and contain zero `/home/` paths.
