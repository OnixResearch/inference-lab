//! Issue #11: Centralized model-lifecycle measurements with a Spark NixOS execution adapter
//! (`spark-nixos@1abf9f9e:nix/modules/kindling/models.nix`, `autoStart=false`, `register=false`),
//! `A -> B -> A` and RSI `#86` `parent -> variant -> parent` single-active-engine sequences,
//! Systems `#10` raw overlapping vs exclusive elapsed-time accounting, whole-host unified-memory
//! provenance/overlap guards, Animus `#17` readiness/capacity observations, and residency-policy
//! comparison on frozen agent demand.

use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_str, opt_u64, quote, str_array};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};
use crate::serving_bench::{default_workload_identity, WorkloadIdentity};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleOperation {
    ProcessRestart,
    NativeSleepWake,
    ResidentIdle,
}

impl LifecycleOperation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ProcessRestart => "process_restart",
            Self::NativeSleepWake => "native_sleep_wake",
            Self::ResidentIdle => "resident_idle",
        }
    }
}

/// Whole-host/device memory counter with explicit provenance and overlap group (GB10 unified memory).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryCounterObservation {
    pub counter_name: String,
    pub provenance_source: String,
    pub overlap_group: String,
    pub observed_mib: Option<u64>,
    pub represents_freed_physical_capacity: bool,
}

/// Raw overlapping phase timings kept strictly separate from exclusive elapsed-time partition (`systems-lab#10`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifecyclePhaseAccounting {
    pub raw_drain_ms: u64,
    pub raw_stop_or_sleep_ms: u64,
    pub raw_preparation_materialization_ms: u64,
    pub raw_load_or_wake_ms: u64,
    pub raw_warmup_ms: u64,
    pub raw_first_valid_completion_ms: u64,
    pub raw_steady_serving_ms: u64,
    pub raw_cleanup_ms: u64,
    pub exclusive_segments_ms: Vec<(String, u64)>,
    pub unknown_coverage_ms: u64,
    pub total_wall_ms: u64,
}

impl LifecyclePhaseAccounting {
    pub fn exclusive_partition_exact(&self) -> bool {
        let sum: u64 = self.exclusive_segments_ms.iter().map(|(_, ms)| *ms).sum();
        sum.saturating_add(self.unknown_coverage_ms) == self.total_wall_ms
    }

    pub fn raw_overlapping_sum_ms(&self) -> u64 {
        self.raw_drain_ms
            + self.raw_stop_or_sleep_ms
            + self.raw_preparation_materialization_ms
            + self.raw_load_or_wake_ms
            + self.raw_warmup_ms
            + self.raw_first_valid_completion_ms
            + self.raw_steady_serving_ms
            + self.raw_cleanup_ms
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifecycleControlCase {
    pub case_id: String,
    pub requested_op: LifecycleOperation,
    pub adapter_supports_native_sleep: bool,
    pub intended_model_digest: DigestRef,
    pub observed_model_digest: DigestRef,
    pub intended_generation: u64,
    pub response_generation: u64,
    pub model_listed_in_v1_models: bool,
    pub first_checked_completion_ok: bool,
    pub shutdown_occurred_since_probe: bool,
    pub wake_succeeded: bool,
    pub timed_out_or_interrupted: bool,
    pub cleanup_target_owned_by_experiment: bool,
    pub cleanup_succeeded: bool,
    pub concurrent_clients_for_one_slot: usize,
    pub telemetry_available: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifecycleControlVerdict {
    pub case_id: String,
    pub classified_ready: bool,
    pub false_ready_detected: bool,
    pub silent_restart_fallback_blocked: bool,
    pub foreign_resource_untouched: bool,
    pub telemetry_state: String,
    pub outcome: OutcomeStatus,
    pub reason: Option<String>,
}

/// Enforce lifecycle readiness, generation fencing, native-sleep non-fallback, and cleanup ownership.
pub fn evaluate_lifecycle_control(case: &LifecycleControlCase) -> LifecycleControlVerdict {
    let telemetry_state = if case.telemetry_available {
        "observed".to_string()
    } else {
        "unknown_telemetry_unavailable".to_string()
    };

    if !case.cleanup_target_owned_by_experiment {
        return LifecycleControlVerdict {
            case_id: case.case_id.clone(),
            classified_ready: false,
            false_ready_detected: false,
            silent_restart_fallback_blocked: true,
            foreign_resource_untouched: true,
            telemetry_state,
            outcome: OutcomeStatus::Invalid,
            reason: Some("refused_cleanup_of_non_experiment_owned_resource".into()),
        };
    }
    if case.requested_op == LifecycleOperation::NativeSleepWake
        && !case.adapter_supports_native_sleep
    {
        return LifecycleControlVerdict {
            case_id: case.case_id.clone(),
            classified_ready: false,
            false_ready_detected: false,
            silent_restart_fallback_blocked: true,
            foreign_resource_untouched: true,
            telemetry_state,
            outcome: OutcomeStatus::Unsupported,
            reason: Some("native_sleep_unsupported_no_silent_process_restart".into()),
        };
    }
    if case.timed_out_or_interrupted {
        return LifecycleControlVerdict {
            case_id: case.case_id.clone(),
            classified_ready: false,
            false_ready_detected: false,
            silent_restart_fallback_blocked: true,
            foreign_resource_untouched: true,
            telemetry_state,
            outcome: OutcomeStatus::Timeout,
            reason: Some("interrupted_or_timed_out_transition_unresolved".into()),
        };
    }
    if !case.wake_succeeded {
        return LifecycleControlVerdict {
            case_id: case.case_id.clone(),
            classified_ready: false,
            false_ready_detected: false,
            silent_restart_fallback_blocked: true,
            foreign_resource_untouched: true,
            telemetry_state,
            outcome: OutcomeStatus::InternalFailure,
            reason: Some("failed_wake_or_parent_restoration".into()),
        };
    }
    if !case.cleanup_succeeded {
        return LifecycleControlVerdict {
            case_id: case.case_id.clone(),
            classified_ready: false,
            false_ready_detected: false,
            silent_restart_fallback_blocked: true,
            foreign_resource_untouched: true,
            telemetry_state,
            outcome: OutcomeStatus::InternalFailure,
            reason: Some("cleanup_failure_retained_in_denominator".into()),
        };
    }
    if case.shutdown_occurred_since_probe {
        return LifecycleControlVerdict {
            case_id: case.case_id.clone(),
            classified_ready: false,
            false_ready_detected: true,
            silent_restart_fallback_blocked: true,
            foreign_resource_untouched: true,
            telemetry_state,
            outcome: OutcomeStatus::Invalid,
            reason: Some("stale_healthy_observation_after_shutdown".into()),
        };
    }
    if case.response_generation != case.intended_generation {
        return LifecycleControlVerdict {
            case_id: case.case_id.clone(),
            classified_ready: false,
            false_ready_detected: true,
            silent_restart_fallback_blocked: true,
            foreign_resource_untouched: true,
            telemetry_state,
            outcome: OutcomeStatus::Invalid,
            reason: Some("late_completion_from_previous_deployment_generation".into()),
        };
    }
    if case.observed_model_digest != case.intended_model_digest {
        return LifecycleControlVerdict {
            case_id: case.case_id.clone(),
            classified_ready: false,
            false_ready_detected: true,
            silent_restart_fallback_blocked: true,
            foreign_resource_untouched: true,
            telemetry_state,
            outcome: OutcomeStatus::Invalid,
            reason: Some("wrong_model_or_variant_loaded".into()),
        };
    }
    if !case.first_checked_completion_ok {
        let (false_ready, reason) = if case.model_listed_in_v1_models {
            (
                true,
                "false_ready_model_listed_before_first_checked_completion",
            )
        } else {
            (
                false,
                "delayed_readiness_first_checked_completion_not_observed",
            )
        };
        return LifecycleControlVerdict {
            case_id: case.case_id.clone(),
            classified_ready: false,
            false_ready_detected: false_ready,
            silent_restart_fallback_blocked: true,
            foreign_resource_untouched: true,
            telemetry_state,
            outcome: if false_ready {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Timeout
            },
            reason: Some(reason.into()),
        };
    }
    if case.concurrent_clients_for_one_slot > 1 {
        return LifecycleControlVerdict {
            case_id: case.case_id.clone(),
            classified_ready: true,
            false_ready_detected: false,
            silent_restart_fallback_blocked: true,
            foreign_resource_untouched: true,
            telemetry_state,
            outcome: OutcomeStatus::Exhausted,
            reason: Some("second_client_admission_rejected_for_single_slot".into()),
        };
    }

    LifecycleControlVerdict {
        case_id: case.case_id.clone(),
        classified_ready: true,
        false_ready_detected: false,
        silent_restart_fallback_blocked: true,
        foreign_resource_untouched: true,
        telemetry_state,
        outcome: OutcomeStatus::Accepted,
        reason: None,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResidencyPolicyOutcome {
    pub policy_id: String,
    pub max_active_engines: usize,
    pub planned_requests: usize,
    pub completed_requests: usize,
    pub rejected_or_expired: usize,
    pub failed_requests: usize,
    pub failed_transitions: usize,
    pub adapter_evidenced_physical_releases: usize,
    pub heartbeat_loss_counted_as_release: bool,
    pub total_transition_overhead_ms: u64,
    pub p95_task_completion_ms: f64,
}

#[derive(Debug, Clone)]
pub struct LifecycleReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub shared_workload_identity: WorkloadIdentity,
    pub spark_adapter_pin: String,
    pub aba_phase_accounting: LifecyclePhaseAccounting,
    pub rsi86_variant_phase_accounting: LifecyclePhaseAccounting,
    pub unified_memory_observations: Vec<MemoryCounterObservation>,
    pub overlapping_memory_counters_summed: bool,
    pub lifecycle_controls: Vec<LifecycleControlVerdict>,
    pub residency_policies: Vec<ResidencyPolicyOutcome>,
    pub decision: String,
}

pub fn run_default_harness() -> LifecycleReport {
    let shared_workload_identity = default_workload_identity();
    let model_a = DigestRef::sha256(b"model-a-parent-frozen-weights");
    let model_b = DigestRef::sha256(b"model-b-perturbed-variant-weights");

    // Systems #10: raw overlapping phases (sum=1880ms) kept separate from exact exclusive wall partition (1500ms).
    let aba_phase_accounting = LifecyclePhaseAccounting {
        raw_drain_ms: 120,
        raw_stop_or_sleep_ms: 210,
        raw_preparation_materialization_ms: 350, // Overlaps with stop/load
        raw_load_or_wake_ms: 680,
        raw_warmup_ms: 180,
        raw_first_valid_completion_ms: 90,
        raw_steady_serving_ms: 200,
        raw_cleanup_ms: 50,
        exclusive_segments_ms: vec![
            ("drain".into(), 120),
            ("stop_and_prep_overlap_wall".into(), 260),
            ("load_and_warmup_wall".into(), 780),
            ("first_valid_completion".into(), 90),
            ("steady_serving".into(), 200),
            ("cleanup".into(), 30),
        ],
        unknown_coverage_ms: 20,
        total_wall_ms: 1500,
    };

    let rsi86_variant_phase_accounting = LifecyclePhaseAccounting {
        raw_drain_ms: 100,
        raw_stop_or_sleep_ms: 190,
        raw_preparation_materialization_ms: 420,
        raw_load_or_wake_ms: 710,
        raw_warmup_ms: 160,
        raw_first_valid_completion_ms: 95,
        raw_steady_serving_ms: 240,
        raw_cleanup_ms: 45,
        exclusive_segments_ms: vec![
            ("drain_parent".into(), 100),
            ("stop_and_materialize_variant_wall".into(), 390),
            ("load_variant_and_warmup".into(), 810),
            ("first_checked_variant_completion".into(), 95),
            ("steady_episode_fresh_state".into(), 240),
            ("restore_parent_and_cleanup".into(), 45),
        ],
        unknown_coverage_ms: 20,
        total_wall_ms: 1700,
    };

    let unified_memory_observations = vec![
        MemoryCounterObservation {
            counter_name: "whole_host_mem_available_delta".into(),
            provenance_source: "/proc/meminfo (whole-host unified pool)".into(),
            overlap_group: "gb10_unified_dram_pool".into(),
            observed_mib: Some(61440),
            represents_freed_physical_capacity: false,
        },
        MemoryCounterObservation {
            counter_name: "device_cuda_alloc_mib".into(),
            provenance_source: "nvml_or_driver_counter (overlaps unified pool)".into(),
            overlap_group: "gb10_unified_dram_pool".into(),
            observed_mib: Some(49152),
            represents_freed_physical_capacity: false,
        },
        MemoryCounterObservation {
            counter_name: "rsi86_candidate_preparation_buffer_mib".into(),
            provenance_source: "host_staging_rss (overlaps unified pool)".into(),
            overlap_group: "gb10_unified_dram_pool".into(),
            observed_mib: Some(8192),
            represents_freed_physical_capacity: false,
        },
        MemoryCounterObservation {
            counter_name: "page_cache_weight_residency_mib".into(),
            provenance_source: "unavailable_without_host_telemetry".into(),
            overlap_group: "gb10_unified_dram_pool".into(),
            observed_mib: None,
            represents_freed_physical_capacity: false,
        },
    ];

    let base_case = |id: &str| LifecycleControlCase {
        case_id: id.into(),
        requested_op: LifecycleOperation::ProcessRestart,
        adapter_supports_native_sleep: false,
        intended_model_digest: model_a.clone(),
        observed_model_digest: model_a.clone(),
        intended_generation: 3,
        response_generation: 3,
        model_listed_in_v1_models: true,
        first_checked_completion_ok: true,
        shutdown_occurred_since_probe: false,
        wake_succeeded: true,
        timed_out_or_interrupted: false,
        cleanup_target_owned_by_experiment: true,
        cleanup_succeeded: true,
        concurrent_clients_for_one_slot: 1,
        telemetry_available: true,
    };

    let mut c_wrong = base_case("02-negative-wrong-model-loaded");
    c_wrong.observed_model_digest = model_b;

    let mut c_listed_not_ready = base_case("03-negative-animus17-model-listed-before-ready");
    c_listed_not_ready.first_checked_completion_ok = false;

    let mut c_stale_after_stop = base_case("04-negative-animus17-stale-healthy-after-shutdown");
    c_stale_after_stop.shutdown_occurred_since_probe = true;

    let mut c_late_gen = base_case("05-negative-late-completion-previous-generation");
    c_late_gen.response_generation = 2;

    let mut c_unsup_sleep = base_case("06-negative-unsupported-native-sleep-no-restart-fallback");
    c_unsup_sleep.requested_op = LifecycleOperation::NativeSleepWake;
    c_unsup_sleep.adapter_supports_native_sleep = false;

    let mut c_fail_wake = base_case("07-negative-failed-wake-or-parent-restore");
    c_fail_wake.wake_succeeded = false;

    let mut c_timeout = base_case("08-negative-interrupted-transition-timeout");
    c_timeout.timed_out_or_interrupted = true;

    let mut c_cleanup_fail = base_case("09-negative-cleanup-failure-retained");
    c_cleanup_fail.cleanup_succeeded = false;

    let mut c_foreign = base_case("10-negative-refuse-foreign-resource-cleanup");
    c_foreign.cleanup_target_owned_by_experiment = false;

    let mut c_two_clients = base_case("11-negative-animus17-two-clients-one-slot-rejected");
    c_two_clients.concurrent_clients_for_one_slot = 2;

    let mut c_no_telem = base_case("12-control-unavailable-telemetry-stays-unknown");
    c_no_telem.telemetry_available = false;

    let mut c_delayed = base_case("13-negative-delayed-readiness-unlisted-not-ready");
    c_delayed.model_listed_in_v1_models = false;
    c_delayed.first_checked_completion_ok = false;

    let lifecycle_controls = vec![
        evaluate_lifecycle_control(&base_case("01-positive-aba-first-checked-completion")),
        evaluate_lifecycle_control(&c_wrong),
        evaluate_lifecycle_control(&c_listed_not_ready),
        evaluate_lifecycle_control(&c_stale_after_stop),
        evaluate_lifecycle_control(&c_late_gen),
        evaluate_lifecycle_control(&c_unsup_sleep),
        evaluate_lifecycle_control(&c_fail_wake),
        evaluate_lifecycle_control(&c_timeout),
        evaluate_lifecycle_control(&c_cleanup_fail),
        evaluate_lifecycle_control(&c_foreign),
        evaluate_lifecycle_control(&c_two_clients),
        evaluate_lifecycle_control(&c_no_telem),
        evaluate_lifecycle_control(&c_delayed),
    ];

    let residency_policies = vec![
        ResidencyPolicyOutcome {
            policy_id: "keep_resident_across_idle".into(),
            max_active_engines: 1,
            planned_requests: 12,
            completed_requests: 11,
            rejected_or_expired: 1,
            failed_requests: 0,
            failed_transitions: 0,
            adapter_evidenced_physical_releases: 0,
            heartbeat_loss_counted_as_release: false,
            total_transition_overhead_ms: 0,
            p95_task_completion_ms: 640.0,
        },
        ResidencyPolicyOutcome {
            policy_id: "unload_at_eligible_idle".into(),
            max_active_engines: 1,
            planned_requests: 12,
            completed_requests: 9,
            rejected_or_expired: 2,
            failed_requests: 1,
            failed_transitions: 1,
            adapter_evidenced_physical_releases: 3,
            heartbeat_loss_counted_as_release: false,
            total_transition_overhead_ms: 4200,
            p95_task_completion_ms: 2150.0,
        },
        ResidencyPolicyOutcome {
            policy_id: "delayed_unload_min_residency".into(),
            max_active_engines: 1,
            planned_requests: 12,
            completed_requests: 10,
            rejected_or_expired: 1,
            failed_requests: 1,
            failed_transitions: 1,
            adapter_evidenced_physical_releases: 1,
            heartbeat_loss_counted_as_release: false,
            total_transition_overhead_ms: 1500,
            p95_task_completion_ms: 1120.0,
        },
    ];

    LifecycleReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "spark_nixos_gb10_disposable_adapter_and_cuda_completion",
            "Device-free A/B/A, RSI #86 parent->variant->parent, Animus #17 readiness, Systems #10 exclusive partition, and unified-memory overlap controls verified locally; physical Spark NixOS #4 GB10 execution remains an external hardware gate.",
        ),
        shared_workload_identity,
        spark_adapter_pin: "OnixResearch/spark-nixos@1abf9f9efa56ba097d3728a2ecb2362d4746675f:nix/modules/kindling/models.nix (autoStart=false, register=false, opt-in disposable only)".into(),
        aba_phase_accounting,
        rsi86_variant_phase_accounting,
        unified_memory_observations,
        overlapping_memory_counters_summed: false,
        lifecycle_controls,
        residency_policies,
        decision: "INCONCLUSIVE_UNAVAILABLE_HARDWARE_RUNTIME".into(),
    }
}

fn phase_to_json(p: &LifecyclePhaseAccounting) -> String {
    let segs: Vec<String> = p
        .exclusive_segments_ms
        .iter()
        .map(|(k, v)| format!("{{\"segment\":{},\"ms\":{v}}}", quote(k)))
        .collect();
    format!(
        concat!(
            "{{\"raw_overlapping_phases_ms\":{{\"drain\":{},\"stop_or_sleep\":{},",
            "\"preparation_materialization\":{},\"load_or_wake\":{},\"warmup\":{},",
            "\"first_valid_completion\":{},\"steady_serving\":{},\"cleanup\":{},\"raw_sum\":{}}},",
            "\"exclusive_elapsed_partition\":{{\"segments\":[{}],\"unknown_coverage_ms\":{},",
            "\"total_wall_ms\":{},\"partition_exact\":{}}}}}"
        ),
        p.raw_drain_ms,
        p.raw_stop_or_sleep_ms,
        p.raw_preparation_materialization_ms,
        p.raw_load_or_wake_ms,
        p.raw_warmup_ms,
        p.raw_first_valid_completion_ms,
        p.raw_steady_serving_ms,
        p.raw_cleanup_ms,
        p.raw_overlapping_sum_ms(),
        segs.join(","),
        p.unknown_coverage_ms,
        p.total_wall_ms,
        p.exclusive_partition_exact(),
    )
}

pub fn inputs_json(report: &LifecycleReport) -> String {
    let notes = vec![
        "One active engine at a time; A -> B -> A and RSI #86 parent -> variant -> parent with fresh episode state.".into(),
        "Harness never invokes host-wide cache dropping, device reset, or production router registration.".into(),
        "Spark NixOS #4 owns disposable GB10 adapter; Systems #1/#4/#10, Animus #17, Kiln #49, Aspen #42, Lattice #24, and RSI #86 retain their consumer authorities.".into(),
    ];
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.issue-11-lifecycle.inputs.v1\",\n",
            "  \"issue\": 11,\n",
            "  \"reuses_serving_runner_issue\": 7,\n",
            "  \"execution_class\": {},\n",
            "  \"spark_adapter_pin\": {},\n",
            "  \"shared_workload_identity\": {},\n",
            "  \"protocol_invariants\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(&report.spark_adapter_pin),
        report.shared_workload_identity.to_json(),
        str_array(&notes),
    )
}

pub fn results_json(report: &LifecycleReport) -> String {
    let mut out = String::new();
    writeln!(out, "{{").unwrap();
    writeln!(
        out,
        "  \"schema\": \"inference-lab.issue-11-lifecycle.results.v1\","
    )
    .unwrap();
    writeln!(out, "  \"issue\": 11,").unwrap();
    writeln!(
        out,
        "  \"execution_class\": {},",
        quote(report.execution_class.as_str())
    )
    .unwrap();
    writeln!(
        out,
        "  \"hardware_gate\": {},",
        report.hardware_gate.to_json()
    )
    .unwrap();
    writeln!(out, "  \"decision\": {},", quote(&report.decision)).unwrap();
    writeln!(
        out,
        "  \"aba_phase_accounting\": {},",
        phase_to_json(&report.aba_phase_accounting)
    )
    .unwrap();
    writeln!(
        out,
        "  \"rsi86_variant_phase_accounting\": {},",
        phase_to_json(&report.rsi86_variant_phase_accounting)
    )
    .unwrap();
    writeln!(
        out,
        "  \"overlapping_memory_counters_summed\": {},",
        report.overlapping_memory_counters_summed
    )
    .unwrap();

    let mem: Vec<String> = report
        .unified_memory_observations
        .iter()
        .map(|m| {
            format!(
                concat!(
                    "{{\"counter_name\":{},\"provenance_source\":{},\"overlap_group\":{},",
                    "\"observed_mib\":{},\"represents_freed_physical_capacity\":{}}}"
                ),
                quote(&m.counter_name),
                quote(&m.provenance_source),
                quote(&m.overlap_group),
                opt_u64(m.observed_mib),
                m.represents_freed_physical_capacity,
            )
        })
        .collect();
    writeln!(
        out,
        "  \"unified_memory_observations\": [{}],",
        mem.join(",")
    )
    .unwrap();

    let ctrls: Vec<String> = report
        .lifecycle_controls
        .iter()
        .map(|c| {
            format!(
                concat!(
                    "{{\"case_id\":{},\"classified_ready\":{},\"false_ready_detected\":{},",
                    "\"silent_restart_fallback_blocked\":{},\"foreign_resource_untouched\":{},",
                    "\"telemetry_state\":{},\"outcome\":{},\"reason\":{}}}"
                ),
                quote(&c.case_id),
                c.classified_ready,
                c.false_ready_detected,
                c.silent_restart_fallback_blocked,
                c.foreign_resource_untouched,
                quote(&c.telemetry_state),
                quote(c.outcome.as_str()),
                opt_str(c.reason.as_deref()),
            )
        })
        .collect();
    writeln!(out, "  \"lifecycle_controls\": [{}],", ctrls.join(",")).unwrap();

    let pols: Vec<String> = report
        .residency_policies
        .iter()
        .map(|p| {
            let accounted = p.completed_requests + p.rejected_or_expired + p.failed_requests;
            format!(
                concat!(
                    "{{\"policy_id\":{},\"max_active_engines\":{},\"planned_requests\":{},",
                    "\"accounted_requests\":{},\"completed_requests\":{},\"rejected_or_expired\":{},",
                    "\"failed_requests\":{},\"failed_transitions\":{},\"adapter_evidenced_physical_releases\":{},",
                    "\"heartbeat_loss_counted_as_release\":{},\"total_transition_overhead_ms\":{},",
                    "\"p95_task_completion_ms\":{}}}"
                ),
                quote(&p.policy_id),
                p.max_active_engines,
                p.planned_requests,
                accounted,
                p.completed_requests,
                p.rejected_or_expired,
                p.failed_requests,
                p.failed_transitions,
                p.adapter_evidenced_physical_releases,
                p.heartbeat_loss_counted_as_release,
                p.total_transition_overhead_ms,
                finite_f64(p.p95_task_completion_ms),
            )
        })
        .collect();
    writeln!(out, "  \"residency_policies\": [{}]", pols.join(",")).unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn summary_markdown(report: &LifecycleReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #11: Centralized Model-Lifecycle Measurements (Spark Adapter Contract)"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Execution class: `{}`",
        report.execution_class.as_str()
    )
    .unwrap();
    writeln!(
        out,
        "- Hardware gate: `{}` (`{}`)",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(
        out,
        "- A/B/A exclusive elapsed-time partition exact: `{}` (wall=`{}ms`, raw overlapping sum=`{}ms`)",
        report.aba_phase_accounting.exclusive_partition_exact(),
        report.aba_phase_accounting.total_wall_ms,
        report.aba_phase_accounting.raw_overlapping_sum_ms()
    )
    .unwrap();
    writeln!(
        out,
        "- RSI #86 parent->variant->parent exclusive partition exact: `{}` (wall=`{}ms`, raw overlapping sum=`{}ms`)",
        report.rsi86_variant_phase_accounting.exclusive_partition_exact(),
        report.rsi86_variant_phase_accounting.total_wall_ms,
        report.rsi86_variant_phase_accounting.raw_overlapping_sum_ms()
    )
    .unwrap();
    writeln!(out, "- Decision: `{}`", report.decision).unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Lifecycle & Animus #17 Controls (13/13)").unwrap();
    writeln!(out, "| Control | Ready | False-Ready | Outcome | Reason |").unwrap();
    writeln!(out, "| --- | --- | --- | --- | --- |").unwrap();
    for c in &report.lifecycle_controls {
        writeln!(
            out,
            "| `{}` | `{}` | `{}` | `{}` | {} |",
            c.case_id,
            c.classified_ready,
            c.false_ready_detected,
            c.outcome.as_str(),
            c.reason.as_deref().unwrap_or("none")
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "> Device-free contract harness only. Physical Spark NixOS #4 GB10 execution and CUDA completion remain required external gates."
    )
    .unwrap();
    out
}
