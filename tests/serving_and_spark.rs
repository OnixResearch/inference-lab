use inference_lab::deepseek_spark::{
    check_workload_comparability, compute_repaired_counter_delta, run_default_harness as run_spark,
    CounterDeltaOutcome, MetricSample, SparkWorkloadClass,
};
use inference_lab::provenance::{ExecutionClass, OutcomeStatus};
use inference_lab::serving_bench::{
    reduce_and_verify_artifact, run_default_harness as run_serving, validate_rlm_trace,
    ClaimedSummary, RetainedCallRecord, RlmTraceCall,
};

#[test]
fn serving_bench_verifies_offline_reducer_branch_trace_eunomia_and_rlm_controls() {
    let report = run_serving();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(report.donor.license, "LGPL-2.1-only");
    assert_eq!(report.donor.legacy_schema, "onix.qwen38-benchmark.v1");

    // 1 positive + 4 negative offline reduction controls.
    assert!(report.offline_reductions[0].valid);
    assert!(report.offline_reductions[0].throughput_claim_eligible);
    assert_eq!(report.offline_reductions[0].failed_or_unfinished, 1);
    for neg in &report.offline_reductions[1..] {
        assert!(!neg.valid, "{} must fail validation", neg.fixture_id);
        assert!(!neg.throughput_claim_eligible);
    }

    // Tampered p50 or None/NaN p95 must be rejected!
    let rec = vec![RetainedCallRecord {
        request_id: "r1".into(),
        endpoint_url: "http://127.0.0.1:8000/v1/chat/completions".into(),
        model_id: "m".into(),
        status: OutcomeStatus::Accepted,
        ttft_ms: Some(100.0),
        total_latency_ms: Some(600.0),
        prompt_tokens: Some(10),
        completion_tokens: Some(51),
        sse_chunk_count: 10,
        greedy_text_matches_baseline: Some(true),
        task_quality_passed: Some(true),
        authorization_header_redacted: true,
    }];
    let bad_p50 = reduce_and_verify_artifact(
        "bad-p50",
        "m",
        &rec,
        &ClaimedSummary {
            planned_requests: 1,
            accepted_requests: 1,
            total_completion_tokens: 51,
            p50_call_mean_itl_ms: Some(1.0),
            p95_call_mean_itl_ms: Some(10.0),
            claimed_greedy_parity_pass: true,
        },
    );
    assert!(!bad_p50.valid);

    let nan_p95 = reduce_and_verify_artifact(
        "nan-p95",
        "m",
        &rec,
        &ClaimedSummary {
            planned_requests: 1,
            accepted_requests: 1,
            total_completion_tokens: 51,
            p50_call_mean_itl_ms: Some(10.0),
            p95_call_mean_itl_ms: Some(f64::NAN),
            claimed_greedy_parity_pass: true,
        },
    );
    assert!(!nan_p95.valid);

    let none_p95 = reduce_and_verify_artifact(
        "none-p95",
        "m",
        &rec,
        &ClaimedSummary {
            planned_requests: 1,
            accepted_requests: 1,
            total_completion_tokens: 51,
            p50_call_mean_itl_ms: Some(10.0),
            p95_call_mean_itl_ms: None,
            claimed_greedy_parity_pass: true,
        },
    );
    assert!(!none_p95.valid);

    // 1 positive + 6 negative branch-trace controls; missing event case cannot report completed_jobs=1!
    assert!(report.branch_trace_controls[0].valid);
    assert_eq!(report.branch_trace_controls[0].completed_jobs, 1);
    for neg in &report.branch_trace_controls[1..] {
        assert!(!neg.valid, "{} must catch violation", neg.control_id);
        assert!(!neg.violations.is_empty());
    }
    let missing_ev = report
        .branch_trace_controls
        .iter()
        .find(|b| b.control_id == "negative-missing-and-duplicate-events")
        .unwrap();
    assert_eq!(missing_ev.completed_jobs, 0);

    // Eunomia negative BPF evidence preserved; client-only cannot assign server phase.
    assert!(!report.eunomia_controls[0].bpf_kernel_loop_justified);
    assert!(!report.eunomia_controls[0].overlapping_cpu_summed);
    assert!(!report.eunomia_controls[1].causal_phase_attribution_allowed);

    // OASYS RLM: 2 valid shapes + 5 negative controls + explicit missing-parent & child-before-parent check.
    assert!(report.rlm_trace_controls[0].valid);
    assert!(report.rlm_trace_controls[1].valid);
    for neg in &report.rlm_trace_controls[2..] {
        assert!(!neg.valid, "{} must fail RLM validation", neg.control_id);
    }
    let bad_rlm = validate_rlm_trace(
        "child-before-parent-and-missing-parent",
        "externally_retained_results",
        &["root", "c1", "c2"],
        &[
            RlmTraceCall {
                call_id: "root".into(),
                parent_call_id: None,
                expected_parent_id: None,
                is_recursive_subcall: false,
                root_context_tokens: 100,
                call_total_tokens: Some(100),
                start_ms: 50,
                end_ms: 200,
                replay_output_matches_recorded: true,
                status: OutcomeStatus::Accepted,
            },
            RlmTraceCall {
                call_id: "c1".into(),
                parent_call_id: Some("root".into()),
                expected_parent_id: Some("root".into()),
                is_recursive_subcall: false,
                root_context_tokens: 0,
                call_total_tokens: Some(50),
                start_ms: 10,
                end_ms: 100,
                replay_output_matches_recorded: true,
                status: OutcomeStatus::Accepted,
            },
            RlmTraceCall {
                call_id: "c2".into(),
                parent_call_id: Some("ghost".into()),
                expected_parent_id: Some("ghost".into()),
                is_recursive_subcall: true,
                root_context_tokens: 0,
                call_total_tokens: Some(50),
                start_ms: 60,
                end_ms: 120,
                replay_output_matches_recorded: true,
                status: OutcomeStatus::Accepted,
            },
        ],
    );
    assert!(!bad_rlm.valid);
    assert!(bad_rlm
        .violations
        .iter()
        .any(|v| v.starts_with("child_started_before_parent:")));
    assert!(bad_rlm
        .violations
        .iter()
        .any(|v| v.starts_with("missing_parent_reference:")));
}

#[test]
fn deepseek_spark_repairs_metrics_runs_9_fixtures_and_catches_substring_bypass() {
    let report = run_spark();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert!(report.workload_cross_compare_rejected);
    assert_eq!(report.device_free_fixtures.len(), 9);
    for fix in &report.device_free_fixtures {
        assert!(fix.handled_honestly, "fixture {} must pass", fix.fixture_id);
    }

    // Correct alternate syntax passes executable semantic check; wrong candidate passes substring but fails concrete inputs [-5, -1].
    assert!(report.quality_comparisons[0].semantic_executable_passed);
    let bypass = &report.quality_comparisons[1];
    assert!(bypass.legacy_substring_passed);
    assert!(!bypass.semantic_executable_passed);
    assert_eq!(bypass.failed_inputs, vec![-5, -1]);

    // Direct counter math verification including NaN & negative rejection!
    let s = vec![MetricSample {
        metric_name: "m".into(),
        label_scope: "a=1".into(),
        value: 50.0,
    }];
    let e_reset = vec![MetricSample {
        metric_name: "m".into(),
        label_scope: "a=1".into(),
        value: 10.0,
    }];
    let e_nan = vec![MetricSample {
        metric_name: "m".into(),
        label_scope: "a=1".into(),
        value: f64::NAN,
    }];
    let s_neg = vec![MetricSample {
        metric_name: "m".into(),
        label_scope: "a=1".into(),
        value: -1.0,
    }];
    assert_eq!(
        compute_repaired_counter_delta(&[], &s, "m", "a=1"),
        CounterDeltaOutcome::UnavailableMissingCounter
    );
    assert_eq!(
        compute_repaired_counter_delta(&s, &e_reset, "m", "a=1"),
        CounterDeltaOutcome::CounterResetDetected {
            start: 50.0,
            end: 10.0
        }
    );
    assert_eq!(
        compute_repaired_counter_delta(&s, &e_nan, "m", "a=1"),
        CounterDeltaOutcome::InvalidNonFiniteOrNegativeSample
    );
    assert_eq!(
        compute_repaired_counter_delta(&s_neg, &s, "m", "a=1"),
        CounterDeltaOutcome::InvalidNonFiniteOrNegativeSample
    );
    assert!(check_workload_comparability(
        SparkWorkloadClass::SynchronizedStreamWaveBurst,
        SparkWorkloadClass::SustainedArrival
    )
    .is_err());
    assert_eq!(
        report.device_free_fixtures[6].outcome,
        OutcomeStatus::InternalFailure
    );
}
