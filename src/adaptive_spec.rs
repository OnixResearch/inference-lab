//! Issue #6: Adaptive speculative graph and concurrency characterization.
//!
//! Direct extension to Issue #3 (`src/drafting.rs`) reusing its exact block-verification
//! engine and Issue #7 (`src/serving_bench.rs`) client metric identities with zero duplicated
//! serving harness infrastructure.

use std::fmt::Write as FmtWrite;

use crate::drafting::{verify_draft_block, BlockVerifyOutcome};
use crate::json::{finite_f64, opt_str, quote, str_array};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};
use crate::serving_bench::{default_workload_identity, WorkloadIdentity};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecVerifyPolicy {
    FixedBlock(usize),
    AdaptiveFrozenThreshold {
        min_len: usize,
        max_len: usize,
        confidence_threshold_milli: u16,
    },
}

impl SpecVerifyPolicy {
    pub fn label(self) -> String {
        match self {
            Self::FixedBlock(k) => format!("fixed_block_{k}"),
            Self::AdaptiveFrozenThreshold {
                min_len,
                max_len,
                confidence_threshold_milli,
            } => format!(
                "adaptive_{min_len}..{max_len}_thr_{:.3}",
                f64::from(confidence_threshold_milli) / 1000.0
            ),
        }
    }
}

/// Choose verification length using a validation-frozen confidence threshold.
pub fn choose_verify_length(policy: SpecVerifyPolicy, step_confidences_milli: &[u16]) -> usize {
    match policy {
        SpecVerifyPolicy::FixedBlock(k) => k.min(step_confidences_milli.len()).max(1),
        SpecVerifyPolicy::AdaptiveFrozenThreshold {
            min_len,
            max_len,
            confidence_threshold_milli,
        } => {
            let cap = max_len.min(step_confidences_milli.len());
            let mut chosen = 0usize;
            for &c in step_confidences_milli.iter().take(cap) {
                if c >= confidence_threshold_milli {
                    chosen += 1;
                } else {
                    break;
                }
            }
            chosen.max(min_len).min(cap).max(1)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveGraphStepSummary {
    pub arm_id: String,
    pub policy_label: String,
    pub validation_frozen_before_held_out: bool,
    pub mean_verify_len: f64,
    pub mean_accepted_tokens_per_iter: f64,
    pub acceptance_by_position: Vec<f64>,
    pub draft_step_latency_ms: f64,
    pub verify_step_latency_ms: f64,
    pub graph_capture_overhead_ms: f64,
    pub graph_replay_overhead_ms: f64,
    pub memory_footprint_mib: u64,
    pub concurrency: usize,
    pub active_generations_per_replica: usize,
    pub claims_continuous_batching: bool,
    pub p50_ttft_ms: f64,
    pub p95_ttft_ms: f64,
    pub p95_completion_ms: f64,
    pub queue_wait_p95_ms: f64,
    pub accepted_tasks_per_hour: f64,
    pub target_arch: String,
    pub outcome: OutcomeStatus,
    pub violation_reason: Option<String>,
}

/// Evaluate an adaptive-graph serving condition, rejecting mislabeled CPU overlap and
/// invalid cross-architecture extrapolation (`sm_100a`/`sm_103a` -> GB10 `sm_121` / RTX 5080 `sm_120`).
pub fn evaluate_adaptive_condition(
    mut cond: AdaptiveGraphStepSummary,
    allowed_arch: &str,
) -> AdaptiveGraphStepSummary {
    if !cond.validation_frozen_before_held_out {
        cond.outcome = OutcomeStatus::Invalid;
        cond.violation_reason = Some("threshold_not_frozen_on_validation".into());
        return cond;
    }
    if cond.claims_continuous_batching && cond.active_generations_per_replica <= 1 {
        cond.outcome = OutcomeStatus::Invalid;
        cond.violation_reason =
            Some("cpu_preparation_overlap_mislabeled_as_continuous_batching".into());
        return cond;
    }
    if cond.target_arch != allowed_arch {
        cond.outcome = OutcomeStatus::Unsupported;
        cond.violation_reason = Some(format!(
            "unsupported_arch_extrapolation:requested={} allowed={allowed_arch}",
            cond.target_arch
        ));
        return cond;
    }
    cond.outcome = OutcomeStatus::Accepted;
    cond.violation_reason = None;
    cond
}

#[derive(Debug, Clone)]
pub struct AdaptiveSpecReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub shared_workload_identity: WorkloadIdentity,
    pub reused_drafting_controls: Vec<BlockVerifyOutcome>,
    pub adaptive_policy_lengths: Vec<(String, usize)>,
    pub conditions: Vec<AdaptiveGraphStepSummary>,
    pub license_gate_notes: Vec<String>,
    pub decision: String,
}

pub fn run_default_harness() -> Result<AdaptiveSpecReport, String> {
    let shared_workload_identity = default_workload_identity();

    // Reuse #3's exact block-verification engine for #22 tiny-MTP rollback controls!
    let reused_drafting_controls = vec![
        verify_draft_block("mtp22-first-rejection", &[99, 2, 3, 4], &[1, 2, 3, 4], 5)?,
        verify_draft_block("mtp22-later-rejection", &[1, 2, 99, 4], &[1, 2, 3, 4], 5)?,
        verify_draft_block("mtp22-all-accepted-bonus", &[1, 2, 3, 4], &[1, 2, 3, 4], 5)?,
    ];

    let conf_high = [950u16, 920, 890, 860, 830, 810];
    let conf_drop = [940u16, 880, 420, 310, 200, 150];
    let fixed = SpecVerifyPolicy::FixedBlock(4);
    let adaptive = SpecVerifyPolicy::AdaptiveFrozenThreshold {
        min_len: 1,
        max_len: 6,
        confidence_threshold_milli: 800,
    };
    let adaptive_policy_lengths = vec![
        (
            "fixed-high-conf".into(),
            choose_verify_length(fixed, &conf_high),
        ),
        (
            "fixed-low-tail".into(),
            choose_verify_length(fixed, &conf_drop),
        ),
        (
            "adaptive-high-conf".into(),
            choose_verify_length(adaptive, &conf_high),
        ),
        (
            "adaptive-low-tail".into(),
            choose_verify_length(adaptive, &conf_drop),
        ),
    ];

    let conditions = vec![
        evaluate_adaptive_condition(
            AdaptiveGraphStepSummary {
                arm_id: "fixed-block-4-single-stream".into(),
                policy_label: fixed.label(),
                validation_frozen_before_held_out: true,
                mean_verify_len: 4.0,
                mean_accepted_tokens_per_iter: 3.1,
                acceptance_by_position: vec![0.92, 0.84, 0.72, 0.62],
                draft_step_latency_ms: 1.8,
                verify_step_latency_ms: 8.2,
                graph_capture_overhead_ms: 42.0,
                graph_replay_overhead_ms: 0.15,
                memory_footprint_mib: 18432,
                concurrency: 1,
                active_generations_per_replica: 1,
                claims_continuous_batching: false,
                p50_ttft_ms: 95.0,
                p95_ttft_ms: 120.0,
                p95_completion_ms: 780.0,
                queue_wait_p95_ms: 0.5,
                accepted_tasks_per_hour: 410.0,
                target_arch: "gb10_sm121_contract".into(),
                outcome: OutcomeStatus::Accepted,
                violation_reason: None,
            },
            "gb10_sm121_contract",
        ),
        evaluate_adaptive_condition(
            AdaptiveGraphStepSummary {
                arm_id: "adaptive-frozen-c4-cpu-overlap-labeled".into(),
                policy_label: adaptive.label(),
                validation_frozen_before_held_out: true,
                mean_verify_len: 3.4,
                mean_accepted_tokens_per_iter: 3.3,
                acceptance_by_position: vec![0.94, 0.87, 0.78, 0.66, 0.55, 0.48],
                draft_step_latency_ms: 1.6,
                verify_step_latency_ms: 7.4,
                graph_capture_overhead_ms: 118.0,
                graph_replay_overhead_ms: 0.18,
                memory_footprint_mib: 19200,
                concurrency: 4,
                active_generations_per_replica: 1,
                claims_continuous_batching: false,
                p50_ttft_ms: 210.0,
                p95_ttft_ms: 480.0,
                p95_completion_ms: 1620.0,
                queue_wait_p95_ms: 360.0,
                accepted_tasks_per_hour: 445.0,
                target_arch: "gb10_sm121_contract".into(),
                outcome: OutcomeStatus::Accepted,
                violation_reason: None,
            },
            "gb10_sm121_contract",
        ),
        evaluate_adaptive_condition(
            AdaptiveGraphStepSummary {
                arm_id: "negative-cpu-overlap-mislabeled-continuous-batching".into(),
                policy_label: adaptive.label(),
                validation_frozen_before_held_out: true,
                mean_verify_len: 3.4,
                mean_accepted_tokens_per_iter: 3.3,
                acceptance_by_position: vec![0.94, 0.87, 0.78, 0.66],
                draft_step_latency_ms: 1.6,
                verify_step_latency_ms: 7.4,
                graph_capture_overhead_ms: 118.0,
                graph_replay_overhead_ms: 0.18,
                memory_footprint_mib: 19200,
                concurrency: 4,
                active_generations_per_replica: 1,
                claims_continuous_batching: true,
                p50_ttft_ms: 210.0,
                p95_ttft_ms: 480.0,
                p95_completion_ms: 1620.0,
                queue_wait_p95_ms: 360.0,
                accepted_tasks_per_hour: 445.0,
                target_arch: "gb10_sm121_contract".into(),
                outcome: OutcomeStatus::Accepted,
                violation_reason: None,
            },
            "gb10_sm121_contract",
        ),
        evaluate_adaptive_condition(
            AdaptiveGraphStepSummary {
                arm_id: "negative-sm100a-b300-extrapolation".into(),
                policy_label: adaptive.label(),
                validation_frozen_before_held_out: true,
                mean_verify_len: 4.5,
                mean_accepted_tokens_per_iter: 4.1,
                acceptance_by_position: vec![0.95, 0.90, 0.85, 0.80],
                draft_step_latency_ms: 0.9,
                verify_step_latency_ms: 3.2,
                graph_capture_overhead_ms: 90.0,
                graph_replay_overhead_ms: 0.10,
                memory_footprint_mib: 40960,
                concurrency: 1,
                active_generations_per_replica: 1,
                claims_continuous_batching: false,
                p50_ttft_ms: 45.0,
                p95_ttft_ms: 60.0,
                p95_completion_ms: 320.0,
                queue_wait_p95_ms: 0.5,
                accepted_tasks_per_hour: 800.0,
                target_arch: "b300_sm100a_sm103a".into(),
                outcome: OutcomeStatus::Accepted,
                violation_reason: None,
            },
            "gb10_sm121_contract",
        ),
    ];

    Ok(AdaptiveSpecReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "cuda_graph_capture_replay_and_concurrent_gpu_serving",
            "Adaptive threshold selection, #3/#22 rollback controls, and CPU-overlap vs continuous-batching guards verified locally; physical CUDA-graph replay and GPU concurrency runs remain unavailable.",
        ),
        shared_workload_identity,
        reused_drafting_controls,
        adaptive_policy_lengths,
        conditions,
        license_gate_notes: vec![
            "basetenlabs/deepseek-ultrafast@3aad78f1f6a3ae87ddb6b7344edbb40433b519aa is Elastic License 2.0; zero source copied.".into(),
            "Upstream targets sm_100a/sm_103a only; B300/GB300 results cannot extrapolate to GB10 sm_121 or RTX 5080 sm_120.".into(),
            "Reuses #3 block verifier and #7 WorkloadIdentity without duplicating serving infrastructure.".into(),
        ],
        decision: "INCONCLUSIVE_UNAVAILABLE_HARDWARE_RUNTIME".into(),
    })
}

pub fn inputs_json(report: &AdaptiveSpecReport) -> String {
    let lens: Vec<String> = report
        .adaptive_policy_lengths
        .iter()
        .map(|(k, v)| format!("{{\"case\":{},\"chosen_verify_len\":{v}}}", quote(k)))
        .collect();
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.issue-06-adaptive-spec.inputs.v1\",\n",
            "  \"issue\": 6,\n",
            "  \"extends_issue\": 3,\n",
            "  \"reuses_serving_runner_issue\": 7,\n",
            "  \"execution_class\": {},\n",
            "  \"shared_workload_identity\": {},\n",
            "  \"adaptive_policy_lengths\": [{}],\n",
            "  \"license_and_scope_notes\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.shared_workload_identity.to_json(),
        lens.join(","),
        str_array(&report.license_gate_notes),
    )
}

pub fn results_json(report: &AdaptiveSpecReport) -> String {
    let mut out = String::new();
    writeln!(out, "{{").unwrap();
    writeln!(
        out,
        "  \"schema\": \"inference-lab.issue-06-adaptive-spec.results.v1\","
    )
    .unwrap();
    writeln!(out, "  \"issue\": 6,").unwrap();
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

    let ctrls: Vec<String> = report
        .reused_drafting_controls
        .iter()
        .map(|b| {
            format!(
                concat!(
                    "{{\"case_id\":{},\"accepted_draft_tokens\":{},\"rejected_draft_tokens\":{},",
                    "\"bonus_token_emitted\":{},\"correction_token_emitted\":{},\"kv_rollback_len\":{}}}"
                ),
                quote(&b.case_id),
                b.accepted_draft_tokens,
                b.rejected_draft_tokens,
                b.bonus_token_emitted,
                b.correction_token_emitted,
                b.kv_rollback_len,
            )
        })
        .collect();
    writeln!(
        out,
        "  \"reused_drafting_controls\": [{}],",
        ctrls.join(",")
    )
    .unwrap();

    let conds: Vec<String> = report
        .conditions
        .iter()
        .map(|c| {
            let acc: Vec<String> = c
                .acceptance_by_position
                .iter()
                .map(|v| finite_f64(*v))
                .collect();
            format!(
                concat!(
                    "{{\"arm_id\":{},\"policy_label\":{},\"mean_verify_len\":{},",
                    "\"mean_accepted_tokens_per_iter\":{},\"acceptance_by_position\":[{}],",
                    "\"draft_step_latency_ms\":{},\"verify_step_latency_ms\":{},",
                    "\"graph_capture_overhead_ms\":{},\"graph_replay_overhead_ms\":{},",
                    "\"memory_footprint_mib\":{},\"concurrency\":{},\"active_generations_per_replica\":{},",
                    "\"claims_continuous_batching\":{},\"p50_ttft_ms\":{},\"p95_ttft_ms\":{},",
                    "\"p95_completion_ms\":{},\"queue_wait_p95_ms\":{},\"accepted_tasks_per_hour\":{},",
                    "\"target_arch\":{},\"outcome\":{},\"violation_reason\":{}}}"
                ),
                quote(&c.arm_id),
                quote(&c.policy_label),
                finite_f64(c.mean_verify_len),
                finite_f64(c.mean_accepted_tokens_per_iter),
                acc.join(","),
                finite_f64(c.draft_step_latency_ms),
                finite_f64(c.verify_step_latency_ms),
                finite_f64(c.graph_capture_overhead_ms),
                finite_f64(c.graph_replay_overhead_ms),
                c.memory_footprint_mib,
                c.concurrency,
                c.active_generations_per_replica,
                c.claims_continuous_batching,
                finite_f64(c.p50_ttft_ms),
                finite_f64(c.p95_ttft_ms),
                finite_f64(c.p95_completion_ms),
                finite_f64(c.queue_wait_p95_ms),
                finite_f64(c.accepted_tasks_per_hour),
                quote(&c.target_arch),
                quote(c.outcome.as_str()),
                opt_str(c.violation_reason.as_deref()),
            )
        })
        .collect();
    writeln!(out, "  \"conditions\": [{}]", conds.join(",")).unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn summary_markdown(report: &AdaptiveSpecReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #6: Adaptive Speculative Graph & Concurrency Characterization"
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
    writeln!(out, "- Decision: `{}`", report.decision).unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "| Condition | Policy | Concurrency | Active/Replica | Outcome | Violation |"
    )
    .unwrap();
    writeln!(out, "| --- | --- | ---: | ---: | --- | --- |").unwrap();
    for c in &report.conditions {
        writeln!(
            out,
            "| `{}` | `{}` | {} | {} | `{}` | {} |",
            c.arm_id,
            c.policy_label,
            c.concurrency,
            c.active_generations_per_replica,
            c.outcome.as_str(),
            c.violation_reason.as_deref().unwrap_or("none")
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "> Reuses Issue #3 verification primitives and Issue #7 workload identity. No CUDA graphs or GPU serving executed locally."
    )
    .unwrap();
    out
}
