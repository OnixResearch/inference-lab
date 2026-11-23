use inference_lab::power::{
    results_json, run_default_harness, CausalMechanism, ComparisonVerdict, SchedulingAxis,
};
use inference_lab::provenance::ExecutionClass;

#[test]
fn power_protocol_controls_enforce_correctness_drift_axis_and_telemetry_gates() {
    let report = run_default_harness();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(
        report.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(
        !report.physical_acceptance_complete,
        "physical acceptance must remain false without physical device execution"
    );

    let find = |name: &str| {
        report
            .evaluations
            .iter()
            .find(|e| e.arm_name == name)
            .unwrap()
    };

    // 1. Known-equivalent -> NoBenefit, energy unavailable without sensor observation,
    // absent throttle counter does NOT prove absence of throttling (UnresolvedMechanism).
    let eq = find("control-known-equivalent");
    assert_eq!(eq.verdict, ComparisonVerdict::NoBenefit);
    assert_eq!(eq.causal_mechanism, CausalMechanism::UnresolvedMechanism);
    assert_eq!(eq.energy_joules, None);
    assert_eq!(eq.energy_status, "unavailable_no_sensor_observation");

    // 2. Wrong-but-fast -> disqualified regardless of speed!
    let wrong = find("control-wrong-but-fast");
    assert_eq!(
        wrong.verdict,
        ComparisonVerdict::DisqualifiedNumericalMismatch
    );
    assert_eq!(wrong.mean_short_run_speedup_pct, None);
    assert_eq!(wrong.mean_sustained_speedup_pct, None);

    // 3. Order/drift control -> counterbalanced AB/BA catches drift and reports InconclusiveOrderDrift,
    // while sensor observation populates energy with scope & units.
    let drift = find("control-order-drift");
    assert_eq!(drift.verdict, ComparisonVerdict::InconclusiveOrderDrift);
    assert!(drift.order_drift_gap_pct.unwrap() > 10.0);
    assert_eq!(drift.energy_joules, Some(12.5));
    assert!(drift.energy_status.starts_with("measured_sensor:"));

    // 4. Host invocation cadence cannot substantiate device instruction spacing.
    let axis = find("control-host-cadence-cannot-claim-instruction-spacing");
    assert_eq!(axis.verdict, ComparisonVerdict::InvalidAxisClaim);
    assert_eq!(
        axis.claimed_axis,
        SchedulingAxis::DeviceInstructionTileSpacing
    );
    assert_eq!(axis.observed_axis, SchedulingAxis::HostInvocationCadence);
    assert!(!axis.causal_claim_allowed);

    // 5. Corrupted/dropped/stale telemetry blocks causal claims.
    let bad_tel = find("control-corrupted-telemetry-blocks-causal-claim");
    assert_eq!(
        bad_tel.verdict,
        ComparisonVerdict::CausalClaimBlockedByTelemetry
    );
    assert!(!bad_tel.causal_claim_allowed);
    assert_eq!(
        bad_tel.causal_mechanism,
        CausalMechanism::UnresolvedMechanism
    );

    // Timing breakdown, cadence axes, and input distribution are preserved in results JSON.
    let json = results_json(&report);
    for key in [
        "\"claimed_axis\":",
        "\"observed_axis\":",
        "\"input_distribution\":",
        "\"timing_breakdown\":",
        "\"compilation_us\":",
        "\"completed_device_work_us\":",
        "\"total_invocation_us\":",
    ] {
        assert!(json.contains(key), "power results JSON missing {key}");
    }
}
