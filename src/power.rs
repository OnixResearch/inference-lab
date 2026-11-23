//! Issue #43: Power and throttle measurement qualification for kernel cadence experiments.
//!
//! Explicitly separates device instruction/tile scheduling from host invocation cadence,
//! separates short-run latency from sustained throughput, enforces independent correctness
//! verification before scoring, gates energy reporting on actual sensor observations,
//! reuses Issue #40's clock/completeness/event-loss conventions, and executes paired AB/BA
//! known-equivalent, wrong-but-fast, and order-drift controls.

use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_f64, quote, str_array};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate};
use crate::trace::{ClockDomain, CollectorCountStatus};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingAxis {
    DeviceInstructionTileSpacing,
    HostInvocationCadence,
}

impl SchedulingAxis {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DeviceInstructionTileSpacing => "device_instruction_tile_spacing",
            Self::HostInvocationCadence => "host_invocation_cadence",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalMechanism {
    ChipThermalOrPowerThrottle,
    HbmBandwidthOrPowerThrottle,
    InternalOperandStall,
    UnresolvedMechanism,
}

impl CausalMechanism {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ChipThermalOrPowerThrottle => "chip_thermal_or_power_throttle",
            Self::HbmBandwidthOrPowerThrottle => "hbm_bandwidth_or_power_throttle",
            Self::InternalOperandStall => "internal_operand_stall",
            Self::UnresolvedMechanism => "unresolved_mechanism",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonVerdict {
    Improvement,
    NoBenefit,
    InconclusiveOrderDrift,
    DisqualifiedNumericalMismatch,
    InvalidAxisClaim,
    CausalClaimBlockedByTelemetry,
}

impl ComparisonVerdict {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Improvement => "improvement",
            Self::NoBenefit => "no_benefit",
            Self::InconclusiveOrderDrift => "inconclusive_order_drift",
            Self::DisqualifiedNumericalMismatch => "disqualified_numerical_mismatch",
            Self::InvalidAxisClaim => "invalid_axis_claim",
            Self::CausalClaimBlockedByTelemetry => "causal_claim_blocked_by_telemetry",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhaseTimingBreakdown {
    pub compilation_us: f64,
    pub preparation_us: f64,
    pub transfer_us: f64,
    pub launch_us: f64,
    pub completed_device_work_us: f64,
    pub validation_us: f64,
    pub total_invocation_us: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SensorObservation {
    pub sensor_scope: String,
    pub units: String,
    pub energy_joules: f64,
    pub sample_interval_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TelemetryCompleteness {
    pub clock_domain: ClockDomain,
    pub clock_mapped: bool,
    pub dropped_events: CollectorCountStatus,
    pub binding_valid: bool,
    pub stale_samples: u64,
    pub throttle_counter_supported: bool,
    pub throttle_events_observed: Option<u64>,
    pub sensor_observation: Option<SensorObservation>,
}

/// One paired, counterbalanced AB/BA trial block on fixed representative/diagnostic inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedCadenceTrial {
    pub trial_id: String,
    pub order: String, // "AB" or "BA"
    pub baseline_short_run_us: f64,
    pub candidate_short_run_us: f64,
    pub baseline_sustained_ops_per_s: f64,
    pub candidate_sustained_ops_per_s: f64,
    pub candidate_output_verified: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CadenceExperimentArm {
    pub arm_name: String,
    pub claimed_axis: SchedulingAxis,
    pub observed_axis: SchedulingAxis,
    pub input_distribution: String,
    pub practical_improvement_threshold_pct: f64,
    pub untraced_baseline_us: f64,
    pub instrumented_baseline_us: f64,
    pub timing_breakdown: PhaseTimingBreakdown,
    pub telemetry: TelemetryCompleteness,
    pub trials: Vec<PairedCadenceTrial>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CadenceArmEvaluation {
    pub arm_name: String,
    pub claimed_axis: SchedulingAxis,
    pub observed_axis: SchedulingAxis,
    pub input_distribution: String,
    pub timing_breakdown: PhaseTimingBreakdown,
    pub verdict: ComparisonVerdict,
    pub causal_mechanism: CausalMechanism,
    pub causal_claim_allowed: bool,
    pub mean_short_run_speedup_pct: Option<f64>,
    pub mean_sustained_speedup_pct: Option<f64>,
    pub ab_mean_speedup_pct: Option<f64>,
    pub ba_mean_speedup_pct: Option<f64>,
    pub order_drift_gap_pct: Option<f64>,
    pub observer_perturbation_pct: f64,
    pub energy_joules: Option<f64>,
    pub energy_status: String,
    pub diagnostics: Vec<String>,
}

/// Evaluate one power/cadence comparison arm with strict correctness, axis, order-drift,
/// telemetry-integrity, and sensor-gated energy checks.
pub fn evaluate_cadence_arm(arm: &CadenceExperimentArm) -> CadenceArmEvaluation {
    let mut diagnostics = Vec::new();
    let observer_perturbation_pct = if arm.untraced_baseline_us > 0.0 {
        ((arm.instrumented_baseline_us - arm.untraced_baseline_us) / arm.untraced_baseline_us)
            * 100.0
    } else {
        0.0
    };

    // Energy reported ONLY when backed by actual sensor observations with declared scope and units.
    let (energy_joules, energy_status) = match &arm.telemetry.sensor_observation {
        Some(obs) if obs.energy_joules.is_finite() && obs.energy_joules >= 0.0 => (
            Some(obs.energy_joules),
            format!("measured_sensor:{}:{}", obs.sensor_scope, obs.units),
        ),
        _ => (None, "unavailable_no_sensor_observation".into()),
    };

    // Check telemetry integrity for causal attribution (reusing #40 rules).
    let mut telemetry_ok = true;
    if !arm.telemetry.clock_mapped {
        telemetry_ok = false;
        diagnostics.push("unmapped_clock_domain_blocks_causal_attribution".into());
    }
    if !arm.telemetry.binding_valid {
        telemetry_ok = false;
        diagnostics.push("wrongly_bound_telemetry_blocks_causal_attribution".into());
    }
    if arm.telemetry.stale_samples > 0 {
        telemetry_ok = false;
        diagnostics.push(format!(
            "stale_telemetry_samples:{}",
            arm.telemetry.stale_samples
        ));
    }
    match arm.telemetry.dropped_events {
        CollectorCountStatus::SupportedZero => {}
        CollectorCountStatus::SupportedNonZero(n) => {
            telemetry_ok = false;
            diagnostics.push(format!("dropped_telemetry_events:{n}"));
        }
        CollectorCountStatus::MissingData | CollectorCountStatus::UnsupportedCollector => {
            telemetry_ok = false;
            diagnostics.push(format!(
                "incomplete_drop_counter:{}",
                arm.telemetry.dropped_events.status_str()
            ));
        }
    }

    // Rule 1: Spacing host invocations cannot substantiate a device instruction-spacing claim.
    if arm.claimed_axis == SchedulingAxis::DeviceInstructionTileSpacing
        && arm.observed_axis == SchedulingAxis::HostInvocationCadence
    {
        diagnostics
            .push("host_invocation_cadence_cannot_substantiate_device_instruction_spacing".into());
        return CadenceArmEvaluation {
            arm_name: arm.arm_name.clone(),
            claimed_axis: arm.claimed_axis,
            observed_axis: arm.observed_axis,
            input_distribution: arm.input_distribution.clone(),
            timing_breakdown: arm.timing_breakdown.clone(),
            verdict: ComparisonVerdict::InvalidAxisClaim,
            causal_mechanism: CausalMechanism::UnresolvedMechanism,
            causal_claim_allowed: false,
            mean_short_run_speedup_pct: None,
            mean_sustained_speedup_pct: None,
            ab_mean_speedup_pct: None,
            ba_mean_speedup_pct: None,
            order_drift_gap_pct: None,
            observer_perturbation_pct,
            energy_joules,
            energy_status,
            diagnostics,
        };
    }

    // Rule 2: Independent output check before ANY candidate can receive a performance score.
    if arm.trials.iter().any(|t| !t.candidate_output_verified) {
        diagnostics.push("candidate_failed_independent_output_verification".into());
        return CadenceArmEvaluation {
            arm_name: arm.arm_name.clone(),
            claimed_axis: arm.claimed_axis,
            observed_axis: arm.observed_axis,
            input_distribution: arm.input_distribution.clone(),
            timing_breakdown: arm.timing_breakdown.clone(),
            verdict: ComparisonVerdict::DisqualifiedNumericalMismatch,
            causal_mechanism: CausalMechanism::UnresolvedMechanism,
            causal_claim_allowed: false,
            mean_short_run_speedup_pct: None,
            mean_sustained_speedup_pct: None,
            ab_mean_speedup_pct: None,
            ba_mean_speedup_pct: None,
            order_drift_gap_pct: None,
            observer_perturbation_pct,
            energy_joules,
            energy_status,
            diagnostics,
        };
    }

    // Compute paired short-run and sustained speedups, separately for AB and BA orders.
    let mut short_gains = Vec::new();
    let mut sustained_gains = Vec::new();
    let mut ab_gains = Vec::new();
    let mut ba_gains = Vec::new();

    for t in &arm.trials {
        let short_gain = ((t.baseline_short_run_us - t.candidate_short_run_us)
            / t.baseline_short_run_us)
            * 100.0;
        let sust_gain = ((t.candidate_sustained_ops_per_s - t.baseline_sustained_ops_per_s)
            / t.baseline_sustained_ops_per_s)
            * 100.0;
        short_gains.push(short_gain);
        sustained_gains.push(sust_gain);
        if t.order == "AB" {
            ab_gains.push(sust_gain);
        } else if t.order == "BA" {
            ba_gains.push(sust_gain);
        }
    }

    let mean = |xs: &[f64]| -> Option<f64> {
        if xs.is_empty() {
            None
        } else {
            Some(xs.iter().sum::<f64>() / (xs.len() as f64))
        }
    };

    let mean_short = mean(&short_gains);
    let mean_sust = mean(&sustained_gains);
    let ab_mean = mean(&ab_gains);
    let ba_mean = mean(&ba_gains);
    let order_drift_gap = match (ab_mean, ba_mean) {
        (Some(a), Some(b)) => Some((a - b).abs()),
        _ => None,
    };

    // Detect severe order/drift confounding when AB and BA disagree in opposite directions or exceed threshold.
    let order_confounded = match (ab_mean, ba_mean, order_drift_gap) {
        (Some(a), Some(b), Some(gap)) => {
            (a * b < 0.0) && gap > arm.practical_improvement_threshold_pct
        }
        _ => false,
    };

    let verdict = if !telemetry_ok {
        ComparisonVerdict::CausalClaimBlockedByTelemetry
    } else if order_confounded {
        diagnostics.push(
            "ab_vs_ba_sign_reversal_exceeds_practical_threshold_indicating_run_history_drift"
                .into(),
        );
        ComparisonVerdict::InconclusiveOrderDrift
    } else if mean_sust.unwrap_or(0.0) >= arm.practical_improvement_threshold_pct {
        ComparisonVerdict::Improvement
    } else {
        ComparisonVerdict::NoBenefit
    };

    // Absent throttle events NEVER prove absence of throttling; require explicit supported counter > 0.
    let causal_mechanism = if telemetry_ok
        && arm.telemetry.throttle_counter_supported
        && arm.telemetry.throttle_events_observed.unwrap_or(0) > 0
    {
        CausalMechanism::ChipThermalOrPowerThrottle
    } else {
        if !arm.telemetry.throttle_counter_supported {
            diagnostics.push("absent_throttle_counter_does_not_prove_no_throttling".into());
        }
        CausalMechanism::UnresolvedMechanism
    };

    CadenceArmEvaluation {
        arm_name: arm.arm_name.clone(),
        claimed_axis: arm.claimed_axis,
        observed_axis: arm.observed_axis,
        input_distribution: arm.input_distribution.clone(),
        timing_breakdown: arm.timing_breakdown.clone(),
        verdict,
        causal_mechanism,
        causal_claim_allowed: telemetry_ok,
        mean_short_run_speedup_pct: mean_short,
        mean_sustained_speedup_pct: mean_sust,
        ab_mean_speedup_pct: ab_mean,
        ba_mean_speedup_pct: ba_mean,
        order_drift_gap_pct: order_drift_gap,
        observer_perturbation_pct,
        energy_joules,
        energy_status,
        diagnostics,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PowerContractReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub physical_acceptance_complete: bool,
    pub kernel_module_digest: DigestRef,
    pub unverified_external_reference_note: String,
    pub evaluations: Vec<CadenceArmEvaluation>,
}

fn default_breakdown() -> PhaseTimingBreakdown {
    PhaseTimingBreakdown {
        compilation_us: 1200.0,
        preparation_us: 45.0,
        transfer_us: 30.0,
        launch_us: 12.0,
        completed_device_work_us: 410.0,
        validation_us: 25.0,
        total_invocation_us: 522.0,
    }
}

/// Run the Issue #43 protocol qualification controls:
/// known-equivalent, wrong-but-fast, order-drift, invalid-axis, and broken-telemetry controls.
pub fn run_default_harness() -> PowerContractReport {
    let arms = vec![
        // 1. Known-equivalent candidate -> no_benefit, no sensor -> energy unavailable
        CadenceExperimentArm {
            arm_name: "control-known-equivalent".into(),
            claimed_axis: SchedulingAxis::DeviceInstructionTileSpacing,
            observed_axis: SchedulingAxis::DeviceInstructionTileSpacing,
            input_distribution: "representative-gemm-v1".into(),
            practical_improvement_threshold_pct: 2.0,
            untraced_baseline_us: 410.0,
            instrumented_baseline_us: 428.0,
            timing_breakdown: default_breakdown(),
            telemetry: TelemetryCompleteness {
                clock_domain: ClockDomain::DeviceGlobalTimerNs,
                clock_mapped: true,
                dropped_events: CollectorCountStatus::SupportedZero,
                binding_valid: true,
                stale_samples: 0,
                throttle_counter_supported: false,
                throttle_events_observed: None,
                sensor_observation: None,
            },
            trials: vec![
                PairedCadenceTrial {
                    trial_id: "t1".into(),
                    order: "AB".into(),
                    baseline_short_run_us: 410.0,
                    candidate_short_run_us: 409.5,
                    baseline_sustained_ops_per_s: 2400.0,
                    candidate_sustained_ops_per_s: 2402.0,
                    candidate_output_verified: true,
                },
                PairedCadenceTrial {
                    trial_id: "t2".into(),
                    order: "BA".into(),
                    baseline_short_run_us: 410.5,
                    candidate_short_run_us: 410.2,
                    baseline_sustained_ops_per_s: 2398.0,
                    candidate_sustained_ops_per_s: 2400.0,
                    candidate_output_verified: true,
                },
            ],
        },
        // 2. Wrong-but-fast candidate -> disqualified (numerical_mismatch), never scored!
        CadenceExperimentArm {
            arm_name: "control-wrong-but-fast".into(),
            claimed_axis: SchedulingAxis::DeviceInstructionTileSpacing,
            observed_axis: SchedulingAxis::DeviceInstructionTileSpacing,
            input_distribution: "diagnostic-edge-values-v1".into(),
            practical_improvement_threshold_pct: 2.0,
            untraced_baseline_us: 410.0,
            instrumented_baseline_us: 428.0,
            timing_breakdown: default_breakdown(),
            telemetry: TelemetryCompleteness {
                clock_domain: ClockDomain::DeviceGlobalTimerNs,
                clock_mapped: true,
                dropped_events: CollectorCountStatus::SupportedZero,
                binding_valid: true,
                stale_samples: 0,
                throttle_counter_supported: true,
                throttle_events_observed: Some(0),
                sensor_observation: None,
            },
            trials: vec![PairedCadenceTrial {
                trial_id: "t1".into(),
                order: "AB".into(),
                baseline_short_run_us: 410.0,
                candidate_short_run_us: 280.0,
                baseline_sustained_ops_per_s: 2400.0,
                candidate_sustained_ops_per_s: 3400.0,
                candidate_output_verified: false,
            }],
        },
        // 3. Order/drift control: AB looks +6%, BA looks -6% -> counterbalanced detects drift!
        CadenceExperimentArm {
            arm_name: "control-order-drift".into(),
            claimed_axis: SchedulingAxis::DeviceInstructionTileSpacing,
            observed_axis: SchedulingAxis::DeviceInstructionTileSpacing,
            input_distribution: "representative-gemm-v1".into(),
            practical_improvement_threshold_pct: 2.0,
            untraced_baseline_us: 410.0,
            instrumented_baseline_us: 425.0,
            timing_breakdown: default_breakdown(),
            telemetry: TelemetryCompleteness {
                clock_domain: ClockDomain::DeviceGlobalTimerNs,
                clock_mapped: true,
                dropped_events: CollectorCountStatus::SupportedZero,
                binding_valid: true,
                stale_samples: 0,
                throttle_counter_supported: true,
                throttle_events_observed: Some(4),
                sensor_observation: Some(SensorObservation {
                    sensor_scope: "synthetic_contract_fixture_board_rail".into(),
                    units: "joules".into(),
                    energy_joules: 12.5,
                    sample_interval_ms: 10.0,
                }),
            },
            trials: vec![
                PairedCadenceTrial {
                    trial_id: "t1-ab".into(),
                    order: "AB".into(),
                    baseline_short_run_us: 410.0,
                    candidate_short_run_us: 390.0,
                    baseline_sustained_ops_per_s: 2400.0,
                    candidate_sustained_ops_per_s: 2544.0,
                    candidate_output_verified: true,
                },
                PairedCadenceTrial {
                    trial_id: "t2-ba".into(),
                    order: "BA".into(),
                    baseline_short_run_us: 390.0,
                    candidate_short_run_us: 410.0,
                    baseline_sustained_ops_per_s: 2544.0,
                    candidate_sustained_ops_per_s: 2400.0,
                    candidate_output_verified: true,
                },
            ],
        },
        // 4. Invalid axis claim: host invocation spacing cannot substantiate device instruction spacing
        CadenceExperimentArm {
            arm_name: "control-host-cadence-cannot-claim-instruction-spacing".into(),
            claimed_axis: SchedulingAxis::DeviceInstructionTileSpacing,
            observed_axis: SchedulingAxis::HostInvocationCadence,
            input_distribution: "representative-gemm-v1".into(),
            practical_improvement_threshold_pct: 2.0,
            untraced_baseline_us: 410.0,
            instrumented_baseline_us: 425.0,
            timing_breakdown: default_breakdown(),
            telemetry: TelemetryCompleteness {
                clock_domain: ClockDomain::DeviceGlobalTimerNs,
                clock_mapped: true,
                dropped_events: CollectorCountStatus::SupportedZero,
                binding_valid: true,
                stale_samples: 0,
                throttle_counter_supported: false,
                throttle_events_observed: None,
                sensor_observation: None,
            },
            trials: vec![PairedCadenceTrial {
                trial_id: "t1".into(),
                order: "AB".into(),
                baseline_short_run_us: 410.0,
                candidate_short_run_us: 350.0,
                baseline_sustained_ops_per_s: 2400.0,
                candidate_sustained_ops_per_s: 2700.0,
                candidate_output_verified: true,
            }],
        },
        // 5. Dropped/stale/wrongly-bound telemetry blocks causal attribution
        CadenceExperimentArm {
            arm_name: "control-corrupted-telemetry-blocks-causal-claim".into(),
            claimed_axis: SchedulingAxis::DeviceInstructionTileSpacing,
            observed_axis: SchedulingAxis::DeviceInstructionTileSpacing,
            input_distribution: "representative-gemm-v1".into(),
            practical_improvement_threshold_pct: 2.0,
            untraced_baseline_us: 410.0,
            instrumented_baseline_us: 430.0,
            timing_breakdown: default_breakdown(),
            telemetry: TelemetryCompleteness {
                clock_domain: ClockDomain::DeviceGlobalTimerNs,
                clock_mapped: false,
                dropped_events: CollectorCountStatus::SupportedNonZero(12),
                binding_valid: false,
                stale_samples: 3,
                throttle_counter_supported: true,
                throttle_events_observed: Some(9),
                sensor_observation: None,
            },
            trials: vec![PairedCadenceTrial {
                trial_id: "t1".into(),
                order: "AB".into(),
                baseline_short_run_us: 410.0,
                candidate_short_run_us: 380.0,
                baseline_sustained_ops_per_s: 2400.0,
                candidate_sustained_ops_per_s: 2600.0,
                candidate_output_verified: true,
            }],
        },
    ];

    let evaluations = arms.iter().map(evaluate_cadence_arm).collect();

    PowerContractReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "physical-tpu-v7x-or-nvidia-or-tenstorrent-accelerator",
            "No physical accelerator or XProf/NVML telemetry collector executed locally; physical-measurement acceptance item remains explicitly incomplete.",
        ),
        physical_acceptance_complete: false,
        kernel_module_digest: DigestRef::sha256(b"inference-lab:cadence-contract-kernel:v1"),
        unverified_external_reference_note: "Ayaka TPU v7x 25.9%/26.4% claims (x.com/ayaka14732/status/2122033206719639812) are unverified author reports and NOT lab targets.".into(),
        evaluations,
    }
}

pub fn inputs_json(report: &PowerContractReport) -> String {
    format!(
        "{{\"schema\":\"inference-lab.power-inputs.v1\",\"issue\":43,\"kernel_module_digest\":{},\"practical_improvement_threshold_pct\":2.0,\"controls\":5,\"unverified_external_reference_note\":{}}}\n",
        report.kernel_module_digest.to_json(),
        quote(&report.unverified_external_reference_note)
    )
}

pub fn results_json(report: &PowerContractReport) -> String {
    let evals: Vec<String> = report
        .evaluations
        .iter()
        .map(|e| {
            let tb = &e.timing_breakdown;
            let tb_json = format!(
                "{{\"compilation_us\":{},\"preparation_us\":{},\"transfer_us\":{},\"launch_us\":{},\"completed_device_work_us\":{},\"validation_us\":{},\"total_invocation_us\":{}}}",
                finite_f64(tb.compilation_us),
                finite_f64(tb.preparation_us),
                finite_f64(tb.transfer_us),
                finite_f64(tb.launch_us),
                finite_f64(tb.completed_device_work_us),
                finite_f64(tb.validation_us),
                finite_f64(tb.total_invocation_us)
            );
            format!(
                "{{\"arm_name\":{},\"claimed_axis\":{},\"observed_axis\":{},\"input_distribution\":{},\"timing_breakdown\":{},\"verdict\":{},\"causal_mechanism\":{},\"causal_claim_allowed\":{},\"mean_short_run_speedup_pct\":{},\"mean_sustained_speedup_pct\":{},\"ab_mean_speedup_pct\":{},\"ba_mean_speedup_pct\":{},\"order_drift_gap_pct\":{},\"observer_perturbation_pct\":{},\"energy_joules\":{},\"energy_status\":{},\"diagnostics\":{}}}",
                quote(&e.arm_name),
                quote(e.claimed_axis.as_str()),
                quote(e.observed_axis.as_str()),
                quote(&e.input_distribution),
                tb_json,
                quote(e.verdict.as_str()),
                quote(e.causal_mechanism.as_str()),
                e.causal_claim_allowed,
                opt_f64(e.mean_short_run_speedup_pct),
                opt_f64(e.mean_sustained_speedup_pct),
                opt_f64(e.ab_mean_speedup_pct),
                opt_f64(e.ba_mean_speedup_pct),
                opt_f64(e.order_drift_gap_pct),
                finite_f64(e.observer_perturbation_pct),
                opt_f64(e.energy_joules),
                quote(&e.energy_status),
                str_array(&e.diagnostics)
            )
        })
        .collect();
    format!(
        "{{\"schema\":\"inference-lab.power-report.v1\",\"issue\":43,\"package_version\":{},\"execution_class\":{},\"hardware_gate\":{},\"physical_acceptance_complete\":{},\"kernel_module_digest\":{},\"evaluations\":[{}]}}\n",
        quote(env!("CARGO_PKG_VERSION")),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.physical_acceptance_complete,
        report.kernel_module_digest.to_json(),
        evals.join(",\n")
    )
}

pub fn summary_markdown(report: &PowerContractReport) -> String {
    let mut out = format!(
        "# Issue #43 — Power and throttle cadence qualification report\n\n\
         - **Execution class**: `{}`\n\
         - **Physical hardware gate**: `{}` (`physical_acceptance_complete = false`)\n\
         - **External reference rule**: {}\n\n\
         ## Protocol control outcomes\n\n\
         | Control arm | Verdict | Causal mechanism | Causal allowed | Observer perturbation (%) | Energy status |\n\
         | --- | --- | --- | --- | --- | --- |\n",
        report.execution_class.as_str(),
        report.hardware_gate.status.as_str(),
        report.unverified_external_reference_note
    );
    for e in &report.evaluations {
        writeln!(
            out,
            "| `{}` | `{}` | `{}` | `{}` | {} | `{}` |",
            e.arm_name,
            e.verdict.as_str(),
            e.causal_mechanism.as_str(),
            e.causal_claim_allowed,
            finite_f64(e.observer_perturbation_pct),
            e.energy_status
        )
        .expect("writing to String cannot fail");
    }
    out.push_str(
        "\nPhysical-device acceptance stays explicitly incomplete until a real accelerator run \
         with qualified raw timing and sensor/throttle telemetry is executed.\n",
    );
    out
}
