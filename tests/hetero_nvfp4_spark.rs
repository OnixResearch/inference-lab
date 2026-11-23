use inference_lab::hetero_rollout::{
    qualify_frozen_policy, run_default_harness as run_hetero_rollout, HttpLoraPayload,
    ReplicaHealth, RolloutReplica,
};
use inference_lab::nvfp4_mxfp8::{
    encode_nvfp4, quantize_fp4_e2m1, run_default_harness as run_nvfp4_mxfp8, walsh_hadamard_norm,
};
use inference_lab::provenance::{ExecutionClass, OutcomeStatus};
use inference_lab::spark_engines::{
    run_default_harness as run_spark_engines, validate_spark_topology_claim, SparkTopologyClaim,
};

#[test]
fn hetero_rollout_qualifies_frozen_policy_http_lora_probes_reconciliation_and_mixed_pool() {
    let report = run_hetero_rollout().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(report.frozen_policy_checks.len(), 2);
    for q in &report.frozen_policy_checks {
        assert_eq!(q.outcome, OutcomeStatus::Accepted);
        assert!(q.tokenizer_parity_ok);
        assert!(q.sampled_is_not_full_kl_verified);
    }
    assert_eq!(report.activation_transitions.len(), 3);
    assert!(report
        .activation_transitions
        .iter()
        .all(|t| t.probe_verified));

    // Mixed pool keeps valid low-reward episodes retained AND counts failed episodes.
    assert_eq!(report.pool_summaries.len(), 3);
    let mixed = &report.pool_summaries[2];
    assert_eq!(mixed.node_count, 2);
    assert_eq!(mixed.planned_episodes, 6);
    assert_eq!(mixed.protocol_valid_rollouts, 4);
    assert_eq!(mixed.valid_low_reward_retained, 2);
    assert_eq!(mixed.independent_task_successes, 2);
    assert_eq!(mixed.failed_or_rejected_episodes, 2);

    // Verify all 10 fault controls executed honestly.
    assert_eq!(report.fault_controls.len(), 10);
    // Duplicate delivery is Accepted idempotently without re-activation; all others fail/reject/unsupported.
    assert_eq!(report.fault_controls[2].outcome, OutcomeStatus::Accepted);
    assert!(report.fault_controls[2].replica_eligible_after);
    // Interrupted activation & failed parent restoration leave replica INELIGIBLE!
    assert!(!report.fault_controls[5].replica_eligible_after);
    assert!(!report.fault_controls[7].replica_eligible_after);
    assert!(!report.fault_controls[9].replica_eligible_after);

    // Direct API test: mid-episode switch rejected; uncertain replica stays ineligible on duplicate delivery!
    let mut rep = RolloutReplica::new("r", "base", "v0", "sig0");
    rep.in_flight_episode = true;
    let p = HttpLoraPayload::valid("base", "v1", 1, "d1", "sig1");
    assert!(rep.apply_http_lora(&p, false, false, false, false).is_err());
    assert_eq!(rep.health, ReplicaHealth::Ready);

    rep.in_flight_episode = false;
    assert_eq!(
        rep.apply_http_lora(&p, false, true, false, false).unwrap(),
        OutcomeStatus::InternalFailure
    );
    assert_eq!(rep.health, ReplicaHealth::UncertainIneligible);
    // Duplicate delivery on an UncertainIneligible replica must NOT return Accepted!
    assert_eq!(
        rep.apply_http_lora(&p, false, false, false, false).unwrap(),
        OutcomeStatus::Invalid
    );
    assert!(!rep.health.is_eligible());

    // NaN logits or NaN tolerances rejected cleanly.
    assert!(qualify_frozen_policy(
        "nan",
        &[vec![0.1, 0.2]],
        &[vec![f64::NAN, 0.2]],
        &[0],
        &[1],
        &[1],
        1e-2,
        1e-4,
    )
    .is_err());
    assert!(qualify_frozen_policy(
        "nan-tol",
        &[vec![0.1, 0.2]],
        &[vec![0.1, 0.2]],
        &[0],
        &[1],
        &[1],
        f64::NAN,
        1e-4,
    )
    .is_err());
    assert!(inference_lab::hetero_rollout::results_json(&report)
        .contains("\"synthetic_contract_fixture\":true"));
}

#[test]
fn nvfp4_mxfp8_verifies_exact_bit_surrogate_stochastic_stats_te_sm120_sm121_and_fabric_codec() {
    let report = run_nvfp4_mxfp8().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(report.known_answer_edge_cases_passed, 5);
    assert!(report.fixed_operand_fd_grad_max_err <= 1e-7);
    assert!(report.arm3_stochastic_stats.within_practical_bias_margin);
    // Deterministic un-reconstructed Arm 4 exhibits larger systematic error than Arm 3 stochastic mean!
    assert!(
        report.arm4_deterministic_ablation_dx_max_abs_bias
            > report.arm3_stochastic_stats.dx_max_abs_mean_err
    );
    assert!(
        report.arm4_deterministic_ablation_dw_max_abs_bias
            > report.arm3_stochastic_stats.dw_max_abs_mean_err
    );

    // TE matrix keeps SM120 and SM121 strictly distinct and retains both MXFP8 refusals as Unsupported!
    assert_eq!(report.te_cutlass_matrix.len(), 5);
    assert_eq!(report.te_cutlass_matrix[2].compute_capability, "sm_120");
    assert_eq!(
        report.te_cutlass_matrix[2].outcome,
        OutcomeStatus::Unsupported
    );
    assert_eq!(report.te_cutlass_matrix[3].compute_capability, "sm_121");
    assert_eq!(
        report.te_cutlass_matrix[3].outcome,
        OutcomeStatus::Unsupported
    );

    // Fabric #6 codec checks pass.
    assert!(report.fabric_codec.hadamard_involution_max_err <= 1e-12);
    assert!(report.fabric_codec.incompatible_transform_rejected);
    assert!(report.fabric_codec.duplicate_or_missing_contrib_rejected);

    // Direct API NaN, E8M0 guard, near-underflow consistency & Fabric JSON flags!
    assert!(quantize_fp4_e2m1(f64::NAN, None).is_err());
    assert!(encode_nvfp4(&[f64::INFINITY, 1.0], 1, 2).is_err());
    assert!(walsh_hadamard_norm(&[1.0, 2.0, 3.0]).is_err());
    assert!(inference_lab::nvfp4_mxfp8::quantize_e8m0_scale(1.0, 0.0).is_err());
    assert!(inference_lab::nvfp4_mxfp8::quantize_e8m0_scale(f64::NAN, 16.0).is_err());
    let nv_json = inference_lab::nvfp4_mxfp8::results_json(&report);
    assert!(nv_json.contains("\"native_s2f6_instruction_available\":false"));
    assert!(nv_json.contains("\"emulated_reconstruction\":true"));

    assert_eq!(report.negative_controls.len(), 8);
    for c in &report.negative_controls {
        assert_ne!(
            c.outcome,
            OutcomeStatus::Accepted,
            "NVFP4/MXFP8 control {} must reject",
            c.case_id
        );
    }
}

#[test]
fn spark_engines_verifies_support_matrix_kv_recurrent_continuations_and_controls() {
    let report = run_spark_engines().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(report.support_matrix.len(), 6);
    assert_eq!(report.support_matrix[1].status, "untested"); // services.kindling.models nspawn!
    assert_eq!(report.support_matrix[3].outcome, OutcomeStatus::Unsupported);
    assert_eq!(report.support_matrix[5].outcome, OutcomeStatus::Unsupported);

    assert_eq!(report.continuation_checks.len(), 4);
    for c in &report.continuation_checks {
        assert!(c.passed, "continuation check {} failed", c.op_name);
        assert!(c.logit_max_err <= 1e-12);
        assert!(c.recurrent_max_err <= 1e-12);
    }
    assert!(report.end_task_passed);

    // Direct validator test: valid Edge-LLM TP2 with Spark #7 receipt & no context reuse passes.
    let (ok_status, _) = validate_spark_topology_claim(&SparkTopologyClaim {
        engine: "tensorrt_edge_llm".into(),
        claimed_topology: "tp2".into(),
        observed_ranks: 2,
        adapter_requested: false,
        context_reuse_requested: false,
        has_spark_7_nccl_receipt: true,
        host_rdma_only: false,
        finish_reason: Some("stop".into()),
    });
    assert_eq!(ok_status, OutcomeStatus::Accepted);

    assert_eq!(report.negative_controls.len(), 9);
    for c in &report.negative_controls {
        assert_ne!(
            c.outcome,
            OutcomeStatus::Accepted,
            "Spark engines control {} must reject",
            c.case_id
        );
    }
}
