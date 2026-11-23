use inference_lab::compact_hybrids::{
    reduce_profile_tasks, run_default_harness as run_compact_hybrids, validate_attribution_scope,
    validate_memory_and_concurrency, AttributionClaim, ConcurrencyMemoryRecord, DecisionThresholds,
};
use inference_lab::cpu_contention::{
    compute_colocated_ratio, run_default_harness as run_cpu_contention,
    validate_background_fairness,
};
use inference_lab::native_recurrent_bench::{
    compare_output_sequences, run_default_harness as run_native_recurrent, scalar_full_reference,
    validate_native_operation_claim, NativeRecurrentState, OperationDisposition, RecFaults,
    RecurrentArch, RecurrentProfileIdentity, REC_CAPACITY,
};
use inference_lab::provenance::{DigestRef, ExecutionClass, OutcomeStatus};
use inference_lab::tiered_eval::{
    admit_final_candidate, run_default_harness as run_tiered_eval, verify_worker_cleanup,
};

#[test]
fn compact_hybrids_verifies_manifest_extensions_decision_rule_and_controls() {
    let report = run_compact_hybrids().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert!(report.positive_concurrency_memory_valid);
    assert_eq!(report.candidate_profiles.len(), 3);
    assert!(report.zero_accepted_cost_undefined);
    assert_eq!(report.reductions[0].planned_tasks, 4);
    assert_eq!(report.reductions[0].accepted_tasks, 3);
    assert_eq!(
        report.reductions[0].trace_replay_items_excluded_from_quality,
        1
    );
    assert_eq!(report.reductions[0].family_breakdown.len(), 4);
    assert_eq!(report.reductions[0].token_length_buckets_observed.len(), 4);
    assert_eq!(report.reductions[1].cost_per_accepted_task_ms, None);

    // queued + admitted > submitted rejected
    assert!(validate_memory_and_concurrency(&ConcurrencyMemoryRecord {
        submitted: 4,
        queued: 3,
        admitted: 2,
        resident_slots: 1,
        weight_file_bytes: 1000,
        observed_resident_rss_bytes: 2000,
        unified_memory_host: false,
        summed_overlapping_cpu_gpu_bytes: false,
        reported_weight_file_as_resident: false,
    })
    .is_err());

    // NaN decision threshold rejected
    assert!(reduce_profile_tasks(
        "nan-thresh",
        &[],
        &DecisionThresholds {
            control_accepted_rate: f64::NAN,
            noninferiority_margin: 0.05,
            control_cost_per_accepted_ms: 100.0,
            required_cost_improvement_ratio: 0.1,
            within_latency_memory_limits: true,
            hardware_executed: true,
        }
    )
    .is_err());

    assert_eq!(
        validate_attribution_scope(&AttributionClaim {
            unmatched_quant_or_engine: true,
            claims_isolated_architecture_speedup: true,
        }),
        OutcomeStatus::Invalid
    );

    assert_eq!(report.negative_controls.len(), 9);
    for c in &report.negative_controls {
        assert_ne!(
            c.outcome,
            OutcomeStatus::Accepted,
            "Compact hybrids control {} must reject",
            c.case_id
        );
    }
}

#[test]
fn native_recurrent_bench_verifies_independent_scalar_refs_nan_guards_and_controls() {
    let report = run_native_recurrent().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert!(report.milestone_a_passed);
    assert!(report.milestone_b_xlstm_gated_passed);
    assert_eq!(report.arch_rows.len(), 3);
    for row in &report.arch_rows {
        assert!(row.scalar_ref_vs_irregular_chunk_err <= 1.0e-12);
        assert!(row.uninterrupted_vs_restored_err <= 1.0e-12);
        assert!(row.fork_branch_max_err <= 1.0e-12);
        assert!(row.batch_vs_isolated_err <= 1.0e-12);
        assert!(row.empty_and_single_token_ok);
        assert_eq!(row.first_divergent_token_on_clean_run, None);
        assert_eq!(
            row.native_rollback_disposition,
            OperationDisposition::Unsupported
        );
    }

    // Direct independent scalar reference check + NaN guard in xLSTM
    let seq = [2u32, 4, 1];
    let ref_out = scalar_full_reference(RecurrentArch::XlstmMlstm, &seq).unwrap();
    let id = RecurrentProfileIdentity {
        arch: RecurrentArch::XlstmMlstm,
        model_digest: DigestRef::sha256(b"test"),
        tokenizer_rev: "t".into(),
        state_dtype: "f64".into(),
        serialization_version: 1,
        worker_generation: 1,
        session_id: "s".into(),
    };
    let mut st = NativeRecurrentState::fresh(id.clone(), REC_CAPACITY);
    let cand_out = st.chunk(&seq, RecFaults::default()).unwrap();
    let (e, div) = compare_output_sequences(&ref_out, &cand_out, 1.0e-12).unwrap();
    assert!(e <= 1.0e-12 && div.is_none());

    let mut nan_st = NativeRecurrentState::fresh(id, REC_CAPACITY);
    nan_st.log_stab_m = f64::NAN;
    assert!(nan_st.step_one(1, RecFaults::default()).is_err());

    assert_eq!(
        validate_native_operation_claim(true, OperationDisposition::Supported),
        OutcomeStatus::Invalid
    );

    assert_eq!(report.negative_controls.len(), 10);
    for c in &report.negative_controls {
        assert_ne!(
            c.outcome,
            OutcomeStatus::Accepted,
            "Native recurrent control {} must reject",
            c.case_id
        );
    }
}

#[test]
fn tiered_eval_verifies_arm_restarts_preregistered_audit_cold_replay_and_controls() {
    let report = run_tiered_eval().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert!(report.aba_restored_vs_fresh_max_err <= 1.0e-12);
    assert!(!report.reduced_layer_tier_is_whole_model_measure);
    assert_eq!(report.arms.len(), 3);
    for arm in &report.arms {
        assert_eq!(arm.planned_candidates, 8);
        assert_eq!(
            arm.worker_restarts, 1,
            "arm {} must retain mandatory weight-mutation restart",
            arm.arm_id
        );
        assert_eq!(
            arm.winning_candidate_id.as_deref(),
            Some("cand_1_valid_faster")
        );
        assert_eq!(arm.cold_final_replay_outcome, OutcomeStatus::Accepted);
    }
    // Arm C audited fast-tier false rejection AND recorded rank disagreement vs slower survivor
    assert_eq!(report.arms[2].fast_tier_false_rejects_audited, 1);
    assert_eq!(report.arms[2].fast_tier_rank_disagreements, 1);

    assert_eq!(
        admit_final_candidate(true, true, false, 0.0, 1.0e-12),
        OutcomeStatus::Invalid
    );
    assert_eq!(
        admit_final_candidate(true, true, true, f64::NAN, 1.0e-12),
        OutcomeStatus::Invalid
    );
    assert_eq!(
        verify_worker_cleanup(true, false, true),
        OutcomeStatus::Invalid
    );

    assert_eq!(report.negative_controls.len(), 8);
    for c in &report.negative_controls {
        assert_ne!(
            c.outcome,
            OutcomeStatus::Accepted,
            "Tiered eval control {} must reject",
            c.case_id
        );
    }
}

#[test]
fn cpu_contention_verifies_blink_ratios_topology_guards_and_controls() {
    let report = run_cpu_contention().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert!(!report.blink_interference_workload_reproducible);
    assert!(!report.sched_ext_followon_triggered);
    assert_eq!(report.trials.len(), 9);
    assert!(report.trials[3].throughput_req_s < report.trials[0].throughput_req_s);

    assert!(compute_colocated_ratio(f64::NAN, 4.0).is_err());
    assert_eq!(compute_colocated_ratio(2.0, 0.0).unwrap(), None);
    assert_eq!(
        validate_background_fairness(100.0, 20.0, 0.70).unwrap(),
        OutcomeStatus::Invalid
    );

    assert_eq!(report.negative_controls.len(), 8);
    for c in &report.negative_controls {
        assert_ne!(
            c.outcome,
            OutcomeStatus::Accepted,
            "CPU contention control {} must reject",
            c.case_id
        );
    }
}
