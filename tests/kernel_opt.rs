use inference_lab::kernel_opt::{
    check_rediscovery_isolation, compute_local_noise_floor, run_default_harness,
};
use inference_lab::provenance::{ExecutionClass, OutcomeStatus};

#[test]
fn kernel_opt_enforces_strict_gate_ordering_soak_and_local_noise_floor() {
    let report = run_default_harness().expect("kernel_opt harness runs");
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert!(!report.noise_floor.foreign_roofline_reused);
    assert!(report.noise_floor.noise_floor_abs_ms > 0.0);

    let by_id = |id: &str| {
        report
            .evaluations
            .iter()
            .find(|e| e.submission_id == id)
            .unwrap()
    };

    // Valid candidate clears every gate + soak + local noise floor and preserves retained runs/ fields.
    let valid = by_id("sub-01-valid-clears-soak-and-noise");
    assert_eq!(valid.outcome, OutcomeStatus::Accepted);
    assert!(valid.timing_eligible);
    assert!(valid.cleared_noise_floor);
    assert!(valid.measured_speedup_vs_baseline.unwrap() > 1.1);
    assert_eq!(valid.parent_id.as_deref(), Some("sub-00-baseline"));
    assert!(!valid.hypothesis.is_empty());
    assert!(valid.total_agent_and_eval_cost_units > 0);

    // Identity no-op is NOT counted as an optimization.
    let noop = by_id("sub-02-identity-noop-baseline");
    assert_eq!(noop.outcome, OutcomeStatus::Invalid);

    // Wrong-but-fast, batch-reorder drift, overbudget schedule, soak fail, forged timing,
    // all-NaN logits, and zero latency are ALL rejected BEFORE timing eligibility!
    for id in [
        "sub-03-wrong-but-fast-logit-drift",
        "sub-04-fast-batch-reorder-drift",
        "sub-05-correct-but-overbudget-schedule",
        "sub-06-passes-dev-fails-soak",
        "sub-08-forged-timing-tampered-eval",
        "sub-09-nan-logits-must-fail-gate2a",
        "sub-10-zero-latency-invalid-timing",
    ] {
        let e = by_id(id);
        assert_eq!(e.outcome, OutcomeStatus::Invalid, "{id} must be rejected");
        assert!(!e.timing_eligible, "{id} must not be timing-eligible");
        assert_eq!(e.measured_speedup_vs_baseline, None);
    }

    // All-NaN logits specifically failed Gate 2a (not bypassed by f64::max!).
    let nan_sub = by_id("sub-09-nan-logits-must-fail-gate2a");
    assert!(!nan_sub.gate_output_tolerance_passed);
    assert_eq!(nan_sub.max_abs_err_vs_fp64, None);

    // Within local noise floor is timing-eligible (passed correctness/soak) but NOT a win!
    let noise = by_id("sub-07-within-local-noise-floor");
    assert!(noise.timing_eligible);
    assert!(!noise.cleared_noise_floor);
    assert_eq!(noise.outcome, OutcomeStatus::Invalid);
    assert_eq!(
        noise.rejection_stage.as_deref(),
        Some("gate_4_within_local_noise_floor_no_improvement")
    );

    // Rediscovery isolation check catches leaked notes/incumbent source.
    assert!(report.rediscovery_checks[0].isolation_valid);
    assert!(report.rediscovery_checks[0].default_off);
    assert!(!report.rediscovery_checks[1].isolation_valid);
    assert!(!check_rediscovery_isolation("leak", true, false, true, false, false).isolation_valid);

    // Noise floor rejects < 3 repeats or non-finite values.
    assert!(compute_local_noise_floor("x", &[1.0, 1.1]).is_err());
    assert!(compute_local_noise_floor("x", &[1.0, f64::NAN, 1.1]).is_err());
}
