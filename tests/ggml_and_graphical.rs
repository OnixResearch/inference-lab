use inference_lab::ggml_ops::{
    qualify_backend_op, run_default_harness as run_ggml_ops, BackendOpQualificationCase, GgmlDtype,
    Q4_0Block, Q8_0Block, GGML_BLOCK_SIZE,
};
use inference_lab::graphical_sampling::{
    estimate_ess_ips, exact_enumerate, frozen_coupled_model,
    run_default_harness as run_graphical_sampling, run_sampler_variant,
    validate_chain_population_denominator, SamplerVariant,
};
use inference_lab::provenance::{ExecutionClass, OutcomeStatus};
use inference_lab::rsi_bench::synthetic_rwkv7_sequence;
use inference_lab::rwkv_ggml::{
    ggml_wkv7_step_f32, run_default_harness as run_rwkv_ggml, validate_ggml_rwkv7_request,
    Rwkv7GgmlStateF32,
};

#[test]
fn ggml_ops_enforces_4_independent_axes_quant_reference_and_no_silent_cpu_fallback() {
    let report = run_ggml_ops().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert!(!report.grants_placement_authority);
    assert!(report.q8_0_abs_err < 0.15);
    assert!(report.q4_0_abs_err < 1.5);

    let by_id = |id: &str| report.verdicts.iter().find(|v| v.case_id == id).unwrap();
    assert_eq!(
        by_id("01-positive-cpu-ref-mul-mat-f32").outcome,
        OutcomeStatus::Accepted
    );
    assert_eq!(
        by_id("02-positive-cpu-ref-mul-mat-q8_0").outcome,
        OutcomeStatus::Accepted
    );
    assert_eq!(
        by_id("03-positive-cpu-ref-mul-mat-q4_0").outcome,
        OutcomeStatus::Accepted
    );
    assert_eq!(
        by_id("04-control-unsupported-mxfp4-explicit").outcome,
        OutcomeStatus::Unsupported
    );
    assert_eq!(
        by_id("05-control-unsupported-nvfp4-explicit").outcome,
        OutcomeStatus::Unsupported
    );

    let fallback = by_id("06-negative-silent-cpu-fallback-disqualified");
    assert!(fallback.silent_cpu_fallback_detected);
    assert_eq!(fallback.outcome, OutcomeStatus::Invalid);
    assert!(!fallback.performance_measured);

    let bad_num = by_id("07-negative-numerical-mismatch-disqualified");
    assert!(bad_num.executed_success);
    assert!(!bad_num.numerical_pass);
    assert_eq!(bad_num.outcome, OutcomeStatus::Invalid);

    assert_eq!(
        by_id("08-negative-nonfinite-nan-output-disqualified").outcome,
        OutcomeStatus::Invalid
    );
    assert_eq!(
        by_id("09-negative-advertised-supported-but-exec-failed").outcome,
        OutcomeStatus::InternalFailure
    );
    assert_eq!(
        by_id("10-control-explicit-skip-distinct-from-pass-or-fail").outcome,
        OutcomeStatus::Unsupported
    );

    assert_eq!(
        by_id("04-control-unsupported-mxfp4-explicit").qualification_state,
        "unsupported"
    );
    assert_eq!(
        by_id("10-control-explicit-skip-distinct-from-pass-or-fail").qualification_state,
        "skipped"
    );
    assert!(by_id("10-control-explicit-skip-distinct-from-pass-or-fail").skipped);
    assert!(!by_id("04-control-unsupported-mxfp4-explicit").skipped);
    assert_eq!(
        by_id("01-positive-cpu-ref-mul-mat-f32").qualification_state,
        "passed"
    );
    assert_eq!(
        by_id("07-negative-numerical-mismatch-disqualified").qualification_state,
        "failed"
    );

    // Negative error, negative tolerance, and non-CPU backend mismatch (CUDA0 -> Vulkan0) must fail!
    let neg_err = qualify_backend_op(&BackendOpQualificationCase {
        case_id: "neg-err".into(),
        mode: "test".into(),
        op_name: "MUL_MAT".into(),
        dtype: GgmlDtype::F32,
        shape: [32, 16, 1],
        requested_backend: "CPU".into(),
        actual_backend: "CPU".into(),
        advertised_support: true,
        executed_success: true,
        skipped: false,
        max_abs_err: Some(-0.01),
        tolerance: 1e-4,
        has_nonfinite_output: false,
        repeated_latencies_us: vec![],
        warmup_runs: 1,
        graph_reallocations: 0,
        memory_growth_bytes: 0,
    });
    assert_eq!(neg_err.outcome, OutcomeStatus::Invalid);
    assert!(!neg_err.numerical_pass);

    let cross_mismatch = qualify_backend_op(&BackendOpQualificationCase {
        case_id: "cuda-to-vulkan".into(),
        mode: "perf".into(),
        op_name: "MUL_MAT".into(),
        dtype: GgmlDtype::F32,
        shape: [32, 16, 1],
        requested_backend: "CUDA0".into(),
        actual_backend: "Vulkan0".into(),
        advertised_support: true,
        executed_success: true,
        skipped: false,
        max_abs_err: Some(0.0),
        tolerance: 1e-4,
        has_nonfinite_output: false,
        repeated_latencies_us: vec![50.0, 51.0],
        warmup_runs: 1,
        graph_reallocations: 0,
        memory_growth_bytes: 0,
    });
    assert!(cross_mismatch.backend_mismatch_detected);
    assert_eq!(cross_mismatch.outcome, OutcomeStatus::Invalid);
    assert!(!cross_mismatch.performance_measured);

    // Quantizers reject NaN cleanly.
    let mut bad = [0.0_f32; GGML_BLOCK_SIZE];
    bad[5] = f32::NAN;
    assert!(Q8_0Block::quantize(&bad).is_err());
    assert!(Q4_0Block::quantize(&bad).is_err());

    // Negative latency rejected cleanly.
    let bad_lat = qualify_backend_op(&BackendOpQualificationCase {
        case_id: "neg-lat".into(),
        mode: "perf".into(),
        op_name: "MUL_MAT".into(),
        dtype: GgmlDtype::F32,
        shape: [32, 16, 1],
        requested_backend: "CPU".into(),
        actual_backend: "CPU".into(),
        advertised_support: true,
        executed_success: true,
        skipped: false,
        max_abs_err: Some(0.0),
        tolerance: 1e-4,
        has_nonfinite_output: false,
        repeated_latencies_us: vec![10.0, -1.0],
        warmup_runs: 1,
        graph_reallocations: 0,
        memory_growth_bytes: 0,
    });
    assert_eq!(bad_lat.outcome, OutcomeStatus::Invalid);
}

#[test]
fn rwkv_ggml_verifies_64step_drift_export_import_continuation_and_negative_controls() {
    let report = run_rwkv_ggml().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert!(!report.asserts_agent_persistence_correctness);
    assert_eq!(report.drift_checkpoints.len(), 5);
    for cp in &report.drift_checkpoints {
        assert!(cp.max_output_abs_err <= report.parity_tolerance);
        assert!(cp.max_state_abs_err <= report.parity_tolerance);
    }

    let by_id = |id: &str| report.controls.iter().find(|c| c.case_id == id).unwrap();
    assert_eq!(
        by_id("01-positive-token-and-full-parity-f32-vs-f64-ref").outcome,
        OutcomeStatus::Accepted
    );
    assert_eq!(
        by_id("02-positive-export-import-state-continuation-24-plus-40").outcome,
        OutcomeStatus::Accepted
    );
    assert_eq!(
        by_id("03-negative-altered-exported-state-detected").outcome,
        OutcomeStatus::Invalid
    );
    assert_eq!(
        by_id("04-negative-sequence-boundary-duplicate-token-detected").outcome,
        OutcomeStatus::Invalid
    );
    assert_eq!(
        by_id("05-negative-bf16-precision-drift-exceeds-f32-tolerance").outcome,
        OutcomeStatus::Invalid
    );
    assert_eq!(
        by_id("06-negative-backward-training-claim-unsupported").outcome,
        OutcomeStatus::Unsupported
    );
    assert_eq!(
        by_id("07-control-ggml-cuda-wkv-unavailable-no-cpu-fallback").outcome,
        OutcomeStatus::UnavailableHardwareRuntime
    );

    // Import rejects truncated bytes, NaN payloads, and dimension overflow; step rejects shape/NaN cleanly.
    assert!(Rwkv7GgmlStateF32::import_le_bytes(2, 4, &[0u8; 4]).is_err());
    assert!(Rwkv7GgmlStateF32::import_le_bytes(usize::MAX, 2, &[]).is_err());
    let mut nan_bytes = vec![0u8; 2 * 4 * 4 * 4];
    nan_bytes[0..4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(Rwkv7GgmlStateF32::import_le_bytes(2, 4, &nan_bytes).is_err());

    let mut st = Rwkv7GgmlStateF32::zeros(2, 4).unwrap();
    let bad_shape_seq = synthetic_rwkv7_sequence(1, 4, 1);
    assert!(ggml_wkv7_step_f32(&mut st, &bad_shape_seq[0], false).is_err());
    let mut bad_nan_seq = synthetic_rwkv7_sequence(2, 4, 1);
    bad_nan_seq[0].key_k[0][0] = f64::NAN;
    assert!(ggml_wkv7_step_f32(&mut st, &bad_nan_seq[0], false).is_err());
    assert_eq!(
        validate_ggml_rwkv7_request("forward_inference", "ggml-cpu", false),
        OutcomeStatus::Accepted
    );
}

#[test]
fn graphical_sampling_matches_exact_oracle_reproduces_color_schedule_and_catches_all_faults() {
    let report = run_graphical_sampling().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert!(!report.cram_default_off_enabled);
    assert!(!report.amara_speedup_used_as_gate);
    assert!(report.conditional_oracle_max_err <= 1e-12);
    assert!(report.serial_and_parallel_color_trajectories_bit_identical);

    let by_var = |v: SamplerVariant| report.sampler_runs.iter().find(|r| r.variant == v).unwrap();

    // All three valid CPU samplers pass exact oracle and have finite positive ESS.
    for v in [
        SamplerVariant::SequentialReference,
        SamplerVariant::SerialColorSchedule,
        SamplerVariant::ColorPhaseParallelModel,
    ] {
        let run = by_var(v);
        assert_eq!(
            run.outcome,
            OutcomeStatus::Accepted,
            "variant {:?} failed",
            v
        );
        assert!(run.schedule_valid);
        assert!(run.ess_energy.unwrap() > 100.0);
    }

    // All five faulty samplers fail semantic/statistical/schedule checks!
    for v in [
        SamplerVariant::FaultyStaleNeighborCache,
        SamplerVariant::FaultyInvalidOneColoring,
        SamplerVariant::FaultyOmittedCommit,
        SamplerVariant::FaultySwappedPairwiseIndex,
        SamplerVariant::FaultyReusedRngStream,
    ] {
        let run = by_var(v);
        assert_eq!(
            run.outcome,
            OutcomeStatus::Invalid,
            "faulty variant {:?} must fail",
            v
        );
    }

    // Omitted commit produces stuck chain -> ESS MUST be None (never invented).
    assert!(by_var(SamplerVariant::FaultyOmittedCommit)
        .ess_energy
        .is_none());
    assert!(estimate_ess_ips(&vec![1.0; 100]).is_none());
    assert!(estimate_ess_ips(&[1.0, 2.0, 3.0]).is_none());

    // Exact enumeration validates positive-factor & <= 256 state profile, and guards zero chains/sweeps & chain denominator.
    let (mut graph, phases) = frozen_coupled_model();
    let oracle = exact_enumerate(&graph).unwrap();
    assert!(run_sampler_variant(
        &graph,
        &oracle,
        &phases,
        SamplerVariant::SequentialReference,
        0,
        10,
        100,
        1,
        0.05
    )
    .is_err());
    assert!(run_sampler_variant(
        &graph,
        &oracle,
        &phases,
        SamplerVariant::SequentialReference,
        2,
        10,
        0,
        1,
        0.05
    )
    .is_err());
    assert_eq!(
        validate_chain_population_denominator(4, 3, 1, 0),
        OutcomeStatus::Accepted
    );
    assert_eq!(
        validate_chain_population_denominator(4, 3, 0, 0),
        OutcomeStatus::Invalid
    );
    graph.cardinalities = vec![4; 5]; // 1024 > 256
    assert!(exact_enumerate(&graph).is_err());
}
