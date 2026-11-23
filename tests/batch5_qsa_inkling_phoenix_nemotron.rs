use inference_lab::inkling_residuals::{
    forward_full_reference, run_default_harness as run_inkling, validate_behavioral_claim,
    AblationConfig, BehavioralClaimCheck, GateMode, InklingFaults, ResidualMode, SinkMode,
};
use inference_lab::nemotron_spark::{
    check_structured_known_answer, reduce_attempt_costs, run_default_harness as run_nemotron,
    AttemptCostRecord,
};
use inference_lab::phoenix_nano::{
    build_seeded_world, model_to_wire_sid, qualify_phoenix_gpu_arch,
    run_default_harness as run_phoenix, validate_serving_completion, validate_sid_mapping,
    wire_to_model_sid, DeviceQualificationInput, PhoenixTrainerState,
};
use inference_lab::provenance::{ExecutionClass, OutcomeStatus};
use inference_lab::qsa_state_coherence::{
    classify_cache_reuse, reduce_session_reuse_costs, run_default_harness as run_qsa_coherence,
    CacheReuseEvidence, ReuseTelemetryClaim, SessionCostRecord,
};

#[test]
fn qsa_state_coherence_verifies_coupled_paths_mtp_rollback_and_all_negative_controls() {
    let report = run_qsa_coherence().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(
        report.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(report.synthetic_contract_fixture);
    assert!(report.zero_accepted_session_cost_undefined);
    assert_eq!(report.service_verdict, "INCONCLUSIVE");
    assert_eq!(report.tested_slot_limit, 1);
    assert_eq!(report.path_checks.len(), 7);
    for p in &report.path_checks {
        assert_eq!(
            p.outcome,
            OutcomeStatus::Accepted,
            "path check {} failed",
            p.case_id
        );
        assert!(p.max_state_and_logit_err <= 1.0e-10);
    }

    let zero_red = reduce_session_reuse_costs(&[SessionCostRecord {
        session_id: "s0".into(),
        outcome: OutcomeStatus::Invalid,
        cold_ttft_ms: 10.0,
        reused_ttft_ms: 0.0,
        rollback_recovery_ms: 0.0,
        total_session_ms: 10.0,
    }])
    .unwrap();
    assert!(zero_red.cost_per_accepted_session_ms.is_none());

    let (ev, st) = classify_cache_reuse(&ReuseTelemetryClaim {
        client_prefix_overlap_tokens: 512,
        server_observed_reused_tokens: Some(0),
        verified_against_fresh: false,
        deployment_gen_request: 1,
        deployment_gen_active: 1,
    });
    assert_eq!(ev, CacheReuseEvidence::AdvertisedOnly);
    assert_eq!(st, OutcomeStatus::Invalid);

    assert_eq!(report.negative_controls.len(), 8);
    for c in &report.negative_controls {
        assert_ne!(
            c.outcome,
            OutcomeStatus::Accepted,
            "QSA coherence negative control {} must reject",
            c.case_id
        );
    }
    assert!(inference_lab::qsa_state_coherence::results_json(&report)
        .contains("\"synthetic_contract_fixture\": true"));
}

#[test]
fn inkling_residuals_verifies_six_layer_block_ablations_extreme_sinks_gates_and_controls() {
    let report = run_inkling().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(
        report.hardware_gate.status,
        ExecutionClass::BlockedOnArtifacts
    );
    assert!(report.extreme_sink_neg50_finite);
    assert!(report.extreme_sink_pos50_finite);
    assert!(report.extreme_gate_neg50_finite);
    assert!(report.extreme_gate_pos50_finite);
    assert!(!report.training_superiority_established);
    assert!(!report.full_594b_claim_supported);
    assert_eq!(report.ablation_arms.len(), 6);
    for a in &report.ablation_arms {
        assert_eq!(a.outcome, OutcomeStatus::Accepted);
        assert!(a.full_vs_chunked_err <= 1.0e-12);
        assert!(a.full_vs_decode_err <= 1.0e-12);
    }

    let cfg = AblationConfig {
        arm_id: "test",
        residual: ResidualMode::BlockAttentionResidual,
        sink: SinkMode::DynamicInputSink,
        gate: GateMode::FixedEnabled,
    };
    assert!(forward_full_reference(
        &[1, 2],
        cfg,
        InklingFaults {
            force_extreme_sink_override: Some(f64::INFINITY),
            ..InklingFaults::default()
        }
    )
    .is_err());
    assert!(forward_full_reference(
        &[1, 2],
        cfg,
        InklingFaults {
            force_extreme_gate_override: Some(f64::NAN),
            ..InklingFaults::default()
        }
    )
    .is_err());
    assert_eq!(
        validate_behavioral_claim(&BehavioralClaimCheck {
            matched_small_models_trained: false,
            equal_data_token_seed_budgets: true,
            pretrained_component_disabling_only: true,
        }),
        OutcomeStatus::Invalid
    );

    assert_eq!(report.negative_controls.len(), 7);
    for c in &report.negative_controls {
        assert_ne!(
            c.outcome,
            OutcomeStatus::Accepted,
            "Inkling control {} must reject",
            c.case_id
        );
    }
}

#[test]
fn phoenix_nano_verifies_gpu_cohort_sid_codec_resume_parity_and_controls() {
    let report = run_phoenix().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert!(!report.synthetic_loss_proves_rec_quality);
    assert!(report.checkpoint_resume_max_err <= 1.0e-12);
    assert!(report.direct_vs_grpc_served_max_err <= 1.0e-12);
    assert_eq!(report.retrieved_and_ranked_count, 4);

    // Empty dump_rows rejected cleanly instead of `% 0` panic
    let mut empty_world = build_seeded_world();
    empty_world.dump_rows.clear();
    let mut tr = PhoenixTrainerState::initial(&empty_world);
    assert!(tr.train_steps(&empty_world, 1).is_err());

    assert_eq!(
        qualify_phoenix_gpu_arch(&DeviceQualificationInput {
            detected_gpu_name: "GB10".into(),
            override_gpu_name: None,
            jax_cuda_verified: false,
        })
        .status,
        OutcomeStatus::Unsupported
    );
    assert_eq!(
        qualify_phoenix_gpu_arch(&DeviceQualificationInput {
            detected_gpu_name: "RTX 5080".into(),
            override_gpu_name: Some("H100".into()),
            jax_cuda_verified: true,
        })
        .status,
        OutcomeStatus::Invalid
    );
    assert_eq!(wire_to_model_sid(None, 16).unwrap(), 0);
    assert_eq!(wire_to_model_sid(Some(0), 16).unwrap(), 1);
    assert_eq!(model_to_wire_sid(1, 16).unwrap(), Some(0));
    assert_eq!(validate_sid_mapping(Some(0), 0, 16), OutcomeStatus::Invalid);
    assert_eq!(
        validate_serving_completion(true, 0, true),
        OutcomeStatus::Invalid
    );

    assert_eq!(report.negative_controls.len(), 9);
    for c in &report.negative_controls {
        assert_ne!(
            c.outcome,
            OutcomeStatus::Accepted,
            "Phoenix control {} must reject",
            c.case_id
        );
    }
}

#[test]
fn nemotron_spark_verifies_strict_json_distinct_profiles_conformance_and_controls() {
    let report = run_nemotron().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(report.staged_arms.len(), 4);
    assert_ne!(
        report.staged_arms[0].profile_digest,
        report.staged_arms[1].profile_digest
    );
    assert_ne!(
        report.staged_arms[0].profile_digest,
        report.staged_arms[2].profile_digest
    );
    assert_eq!(report.positive_conformance_passed, 4);
    assert_eq!(report.positive_conformance_total, 4);
    assert_eq!(report.lifecycle_reduction.planned_attempts, 5);
    assert_eq!(report.lifecycle_reduction.accepted_attempts, 3);
    assert_eq!(report.lifecycle_reduction.failed_or_incomplete_attempts, 2);
    assert!(report.cost_per_accepted_zero_case.is_none());

    // Strict JSON checker rejects unquoted, unbalanced, trailing-comma, and wrong-value inputs
    assert_eq!(
        check_structured_known_answer("{\"answer\":\"42\"}", "answer", "42", 0.01),
        OutcomeStatus::Accepted
    );
    for bad_json in [
        "{answer: 42}",
        "{\"answer\":\"42}",
        "{\"answer\":\"42\",}",
        "{\"answer\":\"99\"}",
        "{}",
    ] {
        assert_eq!(
            check_structured_known_answer(bad_json, "answer", "42", 0.9999),
            OutcomeStatus::Invalid,
            "strict JSON checker must reject `{bad_json}`"
        );
    }

    assert!(reduce_attempt_costs(&[AttemptCostRecord {
        attempt_id: "nan".into(),
        is_warm: true,
        outcome: OutcomeStatus::Accepted,
        ttft_ms: f64::NAN,
        total_wall_ms: 10.0,
    }])
    .is_err());

    assert_eq!(report.negative_controls.len(), 12);
    for c in &report.negative_controls {
        assert_ne!(
            c.outcome,
            OutcomeStatus::Accepted,
            "Nemotron control {} must reject",
            c.case_id
        );
    }
}
