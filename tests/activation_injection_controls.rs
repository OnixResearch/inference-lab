use std::fs;
use std::process::Command;

use inference_lab::activation_injection_controls::{
    check_tradeoff_non_circularity, default_injection_prompts, default_sae_prompts,
    evaluate_sae_disentanglement, matched_norm_orthogonal_direction, run_anomaly_detection_sweep,
    run_default_harness, safe_div, safe_rate, validate_functional_scope,
    validate_matched_norm_control, verify_injection_replay_receipt, ConceptKind,
    DisentanglementVerdict, InjectionPromptCase, InjectionRunSpec, LinearSae, PromptCohort,
    TradeOffClaimSpec, TradeOffValidity, ANOMALY_THRESHOLD, MODEL_REVISION_DIGEST,
    SAE_IDENTITY_TOL,
};
use inference_lab::intervention_replay::{run_default_harness as run_issue45, ModelBackendArch};
use inference_lab::json::sha256_hex;
use inference_lab::parasite_feature_control::run_default_harness as run_issue94;
use inference_lab::provenance::{DigestRef, ExecutionClass, OutcomeStatus};
use inference_lab::qwen35_neuron_steering::run_default_harness as run_issue77;

#[test]
fn issue96_contracts_experiments_and_negative_controls() {
    let report = run_default_harness().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(
        report.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(report.default_off);
    assert!(!report.default_enabled);
    assert!(report.synthetic_contract_fixture);
    assert!(!report.phenomenal_consciousness_claim);

    // Experiment 1: Lindsey-style concept-injection sweep
    let sweep = &report.experiment1_sweep;
    assert_eq!(sweep.prompt_count, 8);
    assert_eq!(sweep.rows.len(), 6);
    assert_eq!(sweep.baseline_task_accuracy, Some(1.0));
    assert_eq!(sweep.sham_false_positive_rate, Some(0.0));
    assert!(sweep.norm_shift_separated_from_concept_id);
    assert!(sweep.all_replays_bit_identical);
    assert_eq!(sweep.sweet_spot_alphas, vec![1.5]);

    assert_eq!(sweep.rows[0].regime_label, "sham_control");
    assert_eq!(sweep.rows[0].target_mean_kl, Some(0.0));
    assert_eq!(sweep.rows[0].target_mean_l1, Some(0.0));
    assert_eq!(sweep.rows[1].regime_label, "under_threshold");
    assert_eq!(sweep.rows[1].target_anomaly_detection_rate, Some(0.0));
    assert_eq!(sweep.rows[3].regime_label, "sweet_spot");
    assert!(sweep.rows[3].in_sweet_spot);
    assert_eq!(sweep.rows[3].target_concept_id_rate, Some(1.0));
    assert_eq!(sweep.rows[3].random_anomaly_detection_rate, Some(1.0));
    assert_eq!(sweep.rows[3].random_concept_id_rate, Some(0.0));
    assert_eq!(sweep.rows[3].target_task_accuracy, Some(1.0));
    assert!(sweep.rows[4].target_task_degradation.unwrap() > 0.15);
    assert_eq!(sweep.rows[5].regime_label, "task_collapse_gibberish");
    assert_eq!(sweep.rows[5].target_task_accuracy, Some(0.0));
    assert_eq!(sweep.rows[5].target_concept_id_rate, Some(0.0));

    // Experiment 2: Berg-style SAE refusal/boilerplate vs truthfulness disentanglement
    let sae_rep = &report.experiment2_sae;
    assert!(sae_rep.unsteered_identity_max_abs_err <= SAE_IDENTITY_TOL);
    assert!(sae_rep.cos_deception_vs_refusal.unwrap() > 0.5);
    assert!(sae_rep.cos_deception_vs_truthfulness.unwrap() > 0.3);
    assert_eq!(
        sae_rep.verdict,
        DisentanglementVerdict::ConfoundedByRefusalAndTruthfulnessShift
    );
    assert!(sae_rep.single_latent_attribution_rejected);
    assert_eq!(sae_rep.arms.len(), 3);
    let suppress = &sae_rep.arms[1];
    let amplify = &sae_rep.arms[2];
    assert!(suppress.self_ref_first_person_shift_vs_base.unwrap() > 0.25);
    assert!(suppress.generic_safety_refusal_shift_vs_base.unwrap() < -0.20);
    assert!(suppress.generic_factual_truthful_shift_vs_base.unwrap() < -0.10);
    assert!(amplify.self_ref_boilerplate_denial_prob.unwrap() > 0.90);
    assert!(amplify.generic_non_self_ref_mean_kl.unwrap() > 0.05);

    // Experiment 3: Non-circularity guard
    assert_eq!(report.experiment3_tradeoff_cases.len(), 3);
    assert_eq!(
        report.experiment3_tradeoff_cases[0].validity,
        TradeOffValidity::CircularObjectiveWired
    );
    assert!(report.experiment3_tradeoff_cases[0].argmax_picks_low_reward_tautologically);
    assert!(!report.experiment3_tradeoff_cases[0].accepted_as_non_circular);
    assert_eq!(
        report.experiment3_tradeoff_cases[1].validity,
        TradeOffValidity::ZeroGradientNoise
    );
    assert!(!report.experiment3_tradeoff_cases[1].accepted_as_non_circular);
    assert_eq!(
        report.experiment3_tradeoff_cases[2].validity,
        TradeOffValidity::ValidFunctionalTradeoffTest
    );
    assert!(report.experiment3_tradeoff_cases[2].accepted_as_non_circular);

    // All 9 negative controls must be rejected (!= Accepted)
    assert_eq!(report.negative_controls.len(), 9);
    for c in &report.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#96 {}", c.case_id);
    }
}

#[test]
fn issue96_direct_adversarial_validators_and_prior_harness_compatibility() {
    let prompts = default_injection_prompts();
    let sae = LinearSae::default_entangled().unwrap();
    let samples = default_sae_prompts();

    // Non-finite / negative alpha & threshold rejection, plus empty prompt_tokens (no panic!)
    let empty_tok_prompt = [InjectionPromptCase {
        prompt_id: "empty_tokens".into(),
        prompt_tokens: vec![],
        expected_task_token: 0,
        target_concept: ConceptKind::AllCaps,
        sensitivity_offset: 0.0,
    }];
    assert!(run_anomaly_detection_sweep(&empty_tok_prompt, &[0.0], ANOMALY_THRESHOLD).is_err());
    assert!(run_anomaly_detection_sweep(&prompts, &[f64::NAN], ANOMALY_THRESHOLD).is_err());
    assert!(run_anomaly_detection_sweep(&prompts, &[f64::INFINITY], ANOMALY_THRESHOLD).is_err());
    assert!(run_anomaly_detection_sweep(&prompts, &[-0.5], ANOMALY_THRESHOLD).is_err());
    assert!(run_anomaly_detection_sweep(&prompts, &[0.0], f64::NAN).is_err());
    assert!(run_anomaly_detection_sweep(&prompts, &[0.0], 0.0).is_err());

    // Matched-L2-norm & orthogonality checks
    let target = ConceptKind::AllCaps.unit_vector();
    let mut scaled_target = [0.0; 8];
    for d in 0..8 {
        scaled_target[d] = 1.5 * target[d];
    }
    let valid_rand = matched_norm_orthogonal_direction(42, &target, 1.5).unwrap();
    assert!(validate_matched_norm_control(&scaled_target, &valid_rand).is_ok());
    // Same direction (non-orthogonal) rejected
    assert!(validate_matched_norm_control(&scaled_target, &scaled_target).is_err());
    // Wrong norm rejected
    let wrong_norm = matched_norm_orthogonal_direction(42, &target, 2.5).unwrap();
    assert!(validate_matched_norm_control(&scaled_target, &wrong_norm).is_err());

    // Missing generic non-self-ref control prompts rejected
    let no_factual: Vec<_> = samples
        .iter()
        .filter(|s| s.cohort != PromptCohort::GenericFactualCalibration)
        .cloned()
        .collect();
    assert!(evaluate_sae_disentanglement(&sae, &no_factual).is_err());

    // Circular and zero-gradient trade-off claims rejected when claimed_non_circular = true
    let bad_circular = TradeOffClaimSpec {
        claim_id: "bad_circ".into(),
        agent_objective_reads_friction_scalar: true,
        objective_friction_lambda: 0.5,
        high_reward_value: 1.0,
        low_reward_value: 0.6,
        injected_friction_scalar: 1.0,
        causal_coupling_verified: true,
        perturbation_causal_gradient_norm: 0.5,
        action_is_orthogonal_unlabeled: false,
        claimed_non_circular: true,
        phenomenal_consciousness_claim: false,
    };
    assert!(check_tradeoff_non_circularity(&bad_circular).is_err());

    let bad_conscious = TradeOffClaimSpec {
        claim_id: "bad_conscious".into(),
        agent_objective_reads_friction_scalar: false,
        objective_friction_lambda: 0.0,
        high_reward_value: 1.0,
        low_reward_value: 0.4,
        injected_friction_scalar: 1.0,
        causal_coupling_verified: true,
        perturbation_causal_gradient_norm: 0.4,
        action_is_orthogonal_unlabeled: false,
        claimed_non_circular: false,
        phenomenal_consciousness_claim: true,
    };
    assert!(check_tradeoff_non_circularity(&bad_conscious).is_err());
    assert_eq!(
        validate_functional_scope(true),
        Err(OutcomeStatus::Unsupported)
    );

    // Zero denominator guard
    assert_eq!(safe_div(10.0, 0.0, "zero_den").unwrap(), None);
    assert_eq!(safe_rate(0, 0), None);

    // Tampered replay digest rejected
    let spec = InjectionRunSpec {
        experiment_id: "replay_tamper".into(),
        arch: ModelBackendArch::QwenTransformer,
        model_revision_digest: MODEL_REVISION_DIGEST.into(),
        seed: 7,
        prompt_tokens: prompts[0].prompt_tokens.clone(),
        target_layer: 1,
        target_token_pos: 3,
        alpha: 1.5,
        detection_threshold: ANOMALY_THRESHOLD,
        delta_vector: scaled_target,
    };
    assert_eq!(
        verify_injection_replay_receipt(
            &spec,
            prompts[0].sensitivity_offset,
            &DigestRef::sha256(b"bogus"),
        )
        .unwrap(),
        OutcomeStatus::Invalid
    );

    // Existing #45, #77, #94 harnesses remain completely intact
    assert!(run_issue45().is_ok());
    assert!(run_issue77().is_ok());
    assert!(run_issue94().is_ok());
}

#[test]
fn issue96_cli_is_two_run_byte_identical_refuses_overwrite_and_has_no_home_leaks() {
    let bin = env!("CARGO_BIN_EXE_inference-lab");
    let base = std::env::temp_dir().join(format!(
        "inference-lab-issue96-cli-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run1 = base.join("run1");
    let run2 = base.join("run2");

    let out1 = Command::new(bin)
        .args([
            "activation-injection-controls",
            "--out",
            run1.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out1.status.success(),
        "run1 failed: {}",
        String::from_utf8_lossy(&out1.stderr)
    );

    let overwrite = Command::new(bin)
        .args([
            "activation-injection-controls",
            "--out",
            run1.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!overwrite.status.success());

    let out2 = Command::new(bin)
        .args([
            "activation-injection-controls",
            "--out",
            run2.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out2.status.success(),
        "run2 failed: {}",
        String::from_utf8_lossy(&out2.stderr)
    );
    assert_eq!(out1.stdout, out2.stdout);

    let manifest_text = fs::read_to_string(run1.join("manifest.json")).unwrap();
    assert!(!manifest_text.contains("/home/"));

    for f in [
        "inputs.json",
        "results.json",
        "summary.md",
        "manifest.json",
        "COMPLETE",
    ] {
        let b1 = fs::read(run1.join(f)).unwrap();
        let b2 = fs::read(run2.join(f)).unwrap();
        assert_eq!(b1, b2, "{f} differed across two runs");
        if f == "COMPLETE" {
            assert!(b1.is_empty(), "COMPLETE marker must be 0 bytes");
        } else {
            let txt = String::from_utf8_lossy(&b1);
            assert!(!txt.contains("/home/"), "leak in {f}");
            if f != "manifest.json" {
                assert!(
                    manifest_text.contains(&sha256_hex(&b1)),
                    "manifest missing digest for {f}"
                );
            }
        }
    }

    let _ = fs::remove_dir_all(&base);
}
