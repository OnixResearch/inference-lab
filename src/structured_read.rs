//! Issue #10: Conditional extraction of shared structured-read contracts after two real
//! distinct backends and a second concrete consumer are qualified.
//!
//! Remains explicitly `blocked_on_artifacts` until two genuine distinct backend implementations
//! and two concrete consumers satisfy the dependency gate. Rejects same-adapter URLs, mocks-only,
//! and chat-only models as gate evidence; validates minimal serializable DTO/readout semantics,
//! AR-label vs diffusion-canvas vs Clef joint-head separation, explicit normalization scope,
//! calibration invalidation, 3 operation-separated positive fixtures, and 10 adversarial negative fixtures.

use std::collections::BTreeSet;
use std::fmt::Write as FmtWrite;

use crate::json::{opt_f64, opt_str, quote, str_array};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendGateCandidate {
    pub backend_id: String,
    pub adapter_implementation_id: String,
    pub is_live_real_backend: bool,
    pub exposes_structured_read_not_chat_only: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractionGateVerdict {
    pub gate_case_id: String,
    pub gate_satisfied: bool,
    pub distinct_real_adapters: usize,
    pub qualified_consumers: usize,
    pub rejection_reason: Option<String>,
}

/// Strictly enforce Issue #10's dependency gate: >= 2 live real distinct backend implementations
/// exposing non-chat structured reads AND >= 2 concrete consumers.
pub fn evaluate_extraction_gate(
    gate_case_id: &str,
    backends: &[BackendGateCandidate],
    concrete_consumers: &[&str],
) -> ExtractionGateVerdict {
    let real_structured: Vec<&BackendGateCandidate> = backends
        .iter()
        .filter(|b| b.is_live_real_backend && b.exposes_structured_read_not_chat_only)
        .collect();
    let distinct_adapters: BTreeSet<&str> = real_structured
        .iter()
        .map(|b| b.adapter_implementation_id.as_str())
        .collect();
    let distinct_consumers: BTreeSet<&str> = concrete_consumers.iter().copied().collect();

    let reason = if backends.iter().all(|b| !b.is_live_real_backend) {
        Some("mocks_only_cannot_satisfy_extraction_gate".into())
    } else if backends
        .iter()
        .all(|b| !b.exposes_structured_read_not_chat_only)
    {
        Some("chat_only_models_cannot_satisfy_extraction_gate".into())
    } else if distinct_adapters.len() < 2 {
        Some("fewer_than_two_distinct_real_backend_implementations".into())
    } else if distinct_consumers.len() < 2 {
        Some("second_concrete_consumer_not_qualified".into())
    } else {
        None
    };

    ExtractionGateVerdict {
        gate_case_id: gate_case_id.into(),
        gate_satisfied: reason.is_none(),
        distinct_real_adapters: distinct_adapters.len(),
        qualified_consumers: distinct_consumers.len(),
        rejection_reason: reason,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredReadOp {
    ArLabelRead,
    DiffusionCanvasRead,
    ClefJointHeadRead,
}

impl StructuredReadOp {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ArLabelRead => "ar_label_read",
            Self::DiffusionCanvasRead => "diffusion_canvas_read",
            Self::ClefJointHeadRead => "clef_joint_head_read",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationScope {
    PermittedLabelsOnly,
    FullVocabulary,
}

impl NormalizationScope {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PermittedLabelsOnly => "permitted_labels_only",
            Self::FullVocabulary => "full_vocabulary",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructuredReadFixture {
    pub fixture_id: String,
    pub requested_op: StructuredReadOp,
    pub backend_supported_ops: Vec<StructuredReadOp>,
    pub expected_ordered_labels: Vec<String>,
    pub observed_ordered_labels: Vec<String>,
    pub label_token_counts: Vec<usize>,
    pub probabilities: Vec<f64>,
    pub normalization_scope: NormalizationScope,
    pub retained_label_mass: Option<f64>,
    pub claimed_full_vocab_distribution: bool,
    pub reads_used: usize,
    pub read_budget: usize,
    pub expected_calibration_digest: DigestRef,
    pub observed_calibration_digest: DigestRef,
    pub cancelled_or_late: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructuredReadVerdict {
    pub fixture_id: String,
    pub op: StructuredReadOp,
    pub outcome: OutcomeStatus,
    pub calibration_valid: bool,
    pub entropy_nats: Option<f64>,
    pub winning_label: Option<String>,
    pub action_authorized_by_confidence: bool,
    pub rejection_reason: Option<String>,
}

/// Validate a bounded serializable structured-read result (no Torch/TT tensor dependency).
/// Confidence NEVER becomes a checker verdict or action authorization (`action_authorized_by_confidence = false`).
pub fn validate_structured_read(fix: &StructuredReadFixture) -> StructuredReadVerdict {
    let cal_valid = fix.expected_calibration_digest == fix.observed_calibration_digest;
    let reject = |outcome: OutcomeStatus, reason: &str| StructuredReadVerdict {
        fixture_id: fix.fixture_id.clone(),
        op: fix.requested_op,
        outcome,
        calibration_valid: cal_valid,
        entropy_nats: None,
        winning_label: None,
        action_authorized_by_confidence: false,
        rejection_reason: Some(reason.into()),
    };

    if !fix.backend_supported_ops.contains(&fix.requested_op) {
        return reject(
            OutcomeStatus::Unsupported,
            "missing_required_capability_no_silent_chat_degradation",
        );
    }
    if fix.cancelled_or_late {
        return reject(
            OutcomeStatus::Cancelled,
            "cancelled_or_late_result_discarded",
        );
    }
    if fix.reads_used > fix.read_budget {
        return reject(OutcomeStatus::Exhausted, "read_or_token_budget_exceeded");
    }
    if !cal_valid {
        return reject(
            OutcomeStatus::Invalid,
            "stale_identity_calibration_invalidated",
        );
    }
    if fix.expected_ordered_labels.is_empty() {
        return reject(OutcomeStatus::Invalid, "empty_expected_ordered_labels");
    }
    if fix.observed_ordered_labels.len() < fix.expected_ordered_labels.len() {
        return reject(OutcomeStatus::Invalid, "missing_permitted_label");
    }
    if fix.observed_ordered_labels != fix.expected_ordered_labels {
        return reject(OutcomeStatus::Invalid, "label_order_drift_detected");
    }
    if fix.label_token_counts.len() != fix.observed_ordered_labels.len()
        || fix.label_token_counts.iter().any(|&c| c != 1)
    {
        return reject(
            OutcomeStatus::Invalid,
            "multi_token_or_empty_label_unsupported_in_single_slot_read",
        );
    }
    if fix.probabilities.len() != fix.observed_ordered_labels.len()
        || fix.probabilities.iter().any(|p| !p.is_finite() || *p < 0.0)
    {
        return reject(
            OutcomeStatus::Invalid,
            "malformed_or_nonfinite_probability_distribution",
        );
    }
    let sum: f64 = fix.probabilities.iter().sum();
    if sum <= 0.0 || sum > 1.0 + 1e-6 {
        return reject(OutcomeStatus::Invalid, "invalid_probability_mass");
    }
    if fix.normalization_scope == NormalizationScope::PermittedLabelsOnly
        && (sum - 1.0).abs() > 1e-6
    {
        return reject(
            OutcomeStatus::Invalid,
            "permitted_labels_distribution_not_normalized_to_one",
        );
    }
    if fix.normalization_scope == NormalizationScope::PermittedLabelsOnly
        && fix.claimed_full_vocab_distribution
    {
        return reject(
            OutcomeStatus::Invalid,
            "restricted_label_renormalization_presented_as_full_vocab",
        );
    }
    if let Some(mass) = fix.retained_label_mass {
        if !mass.is_finite() || !(0.0..=1.0 + 1e-6).contains(&mass) {
            return reject(OutcomeStatus::Invalid, "invalid_retained_label_mass");
        }
    }

    let mut entropy = 0.0;
    let mut best_idx = 0usize;
    for (i, &p) in fix.probabilities.iter().enumerate() {
        if p > 0.0 {
            entropy -= p * p.ln();
        }
        if p > fix.probabilities[best_idx] {
            best_idx = i;
        }
    }

    StructuredReadVerdict {
        fixture_id: fix.fixture_id.clone(),
        op: fix.requested_op,
        outcome: OutcomeStatus::Accepted,
        calibration_valid: true,
        entropy_nats: Some(entropy),
        winning_label: Some(fix.observed_ordered_labels[best_idx].clone()),
        action_authorized_by_confidence: false,
        rejection_reason: None,
    }
}

#[derive(Debug, Clone)]
pub struct StructuredReadReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub donor_pin: String,
    pub donor_license: String,
    pub gate_checks: Vec<ExtractionGateVerdict>,
    pub fixture_verdicts: Vec<StructuredReadVerdict>,
    pub boundary_notes: Vec<String>,
}

pub fn run_default_harness() -> StructuredReadReport {
    let gate_checks = vec![
        evaluate_extraction_gate(
            "current-lab-state-blocked-on-live-backends",
            &[BackendGateCandidate {
                backend_id: "contract-fixture-only".into(),
                adapter_implementation_id: "synthetic-fixture".into(),
                is_live_real_backend: false,
                exposes_structured_read_not_chat_only: true,
            }],
            &["noble-1"],
        ),
        evaluate_extraction_gate(
            "negative-two-urls-same-adapter",
            &[
                BackendGateCandidate {
                    backend_id: "qwen-url-1".into(),
                    adapter_implementation_id: "tt-qwen38-adapter".into(),
                    is_live_real_backend: true,
                    exposes_structured_read_not_chat_only: true,
                },
                BackendGateCandidate {
                    backend_id: "qwen-url-2".into(),
                    adapter_implementation_id: "tt-qwen38-adapter".into(),
                    is_live_real_backend: true,
                    exposes_structured_read_not_chat_only: true,
                },
            ],
            &["noble-1", "proof-2"],
        ),
        evaluate_extraction_gate(
            "negative-two-chat-only-models",
            &[
                BackendGateCandidate {
                    backend_id: "chat-a".into(),
                    adapter_implementation_id: "openai-chat-a".into(),
                    is_live_real_backend: true,
                    exposes_structured_read_not_chat_only: false,
                },
                BackendGateCandidate {
                    backend_id: "chat-b".into(),
                    adapter_implementation_id: "openai-chat-b".into(),
                    is_live_real_backend: true,
                    exposes_structured_read_not_chat_only: false,
                },
            ],
            &["noble-1", "proof-2"],
        ),
    ];

    let cal_digest = DigestRef::sha256(b"qwen38-p150x2:template-v1:labels-ABC:bf16:budget-16");
    let base_fix = |id: &str| StructuredReadFixture {
        fixture_id: id.into(),
        requested_op: StructuredReadOp::ArLabelRead,
        backend_supported_ops: vec![StructuredReadOp::ArLabelRead],
        expected_ordered_labels: vec!["A".into(), "B".into(), "C".into()],
        observed_ordered_labels: vec!["A".into(), "B".into(), "C".into()],
        label_token_counts: vec![1, 1, 1],
        probabilities: vec![0.70, 0.20, 0.10],
        normalization_scope: NormalizationScope::PermittedLabelsOnly,
        retained_label_mass: Some(0.94),
        claimed_full_vocab_distribution: false,
        reads_used: 2,
        read_budget: 8,
        expected_calibration_digest: cal_digest.clone(),
        observed_calibration_digest: cal_digest.clone(),
        cancelled_or_late: false,
    };

    let mut p_diff = base_fix("00b-positive-diffusion-canvas-read");
    p_diff.requested_op = StructuredReadOp::DiffusionCanvasRead;
    p_diff.backend_supported_ops = vec![StructuredReadOp::DiffusionCanvasRead];
    p_diff.probabilities = vec![0.50, 0.35, 0.15];

    let mut p_clef = base_fix("00c-positive-clef-joint-head-read");
    p_clef.requested_op = StructuredReadOp::ClefJointHeadRead;
    p_clef.backend_supported_ops = vec![StructuredReadOp::ClefJointHeadRead];
    p_clef.normalization_scope = NormalizationScope::FullVocabulary;
    p_clef.probabilities = vec![0.45, 0.30, 0.15];

    let mut f_missing = base_fix("01-negative-missing-label");
    f_missing.observed_ordered_labels = vec!["A".into(), "B".into()];

    let mut f_multitok = base_fix("02-negative-multi-token-label");
    f_multitok.label_token_counts = vec![1, 2, 1];

    let mut f_order = base_fix("03-negative-label-order-drift");
    f_order.observed_ordered_labels = vec!["B".into(), "A".into(), "C".into()];

    let mut f_nan = base_fix("04-negative-nonfinite-distribution");
    f_nan.probabilities = vec![f64::NAN, 0.5, 0.5];

    let mut f_mass = base_fix("05-negative-invalid-probability-mass");
    f_mass.probabilities = vec![0.10, 0.10, 0.10];

    let mut f_budget = base_fix("06-negative-exceeded-budget");
    f_budget.reads_used = 9;

    let mut f_cap = base_fix("07-negative-missing-diffusion-capability");
    f_cap.requested_op = StructuredReadOp::DiffusionCanvasRead;

    let mut f_stale_cal = base_fix("08-negative-stale-calibration-identity");
    f_stale_cal.observed_calibration_digest = DigestRef::sha256(b"different-template-or-budget");

    let mut f_cancel = base_fix("09-negative-cancelled-or-late-result");
    f_cancel.cancelled_or_late = true;

    let mut f_renorm_lie = base_fix("10-negative-renormalized-claimed-as-full-vocab");
    f_renorm_lie.claimed_full_vocab_distribution = true;

    let fixture_verdicts = vec![
        validate_structured_read(&base_fix("00a-positive-ar-label-read-explicit-scope")),
        validate_structured_read(&p_diff),
        validate_structured_read(&p_clef),
        validate_structured_read(&f_missing),
        validate_structured_read(&f_multitok),
        validate_structured_read(&f_order),
        validate_structured_read(&f_nan),
        validate_structured_read(&f_mass),
        validate_structured_read(&f_budget),
        validate_structured_read(&f_cap),
        validate_structured_read(&f_stale_cal),
        validate_structured_read(&f_cancel),
        validate_structured_read(&f_renorm_lie),
    ];

    StructuredReadReport {
        execution_class: ExecutionClass::BlockedOnArtifacts,
        hardware_gate: HardwareGate::blocked_on_artifacts(
            "two_distinct_real_structured_read_backends_and_second_consumer",
            "Conditional extraction gate remains BLOCKED until two distinct live backend implementations (not two URLs to one adapter, mocks, or chat-only models) and a second concrete consumer are qualified.",
        ),
        donor_pin: "OnixResearch/tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a (structured_read_core.py, check.py)".into(),
        donor_license: "LGPL-2.1-only".into(),
        gate_checks,
        fixture_verdicts,
        boundary_notes: vec![
            "TT/Torch-specific details (ENGINE_NAME=qwen38-p150x2, MAXIMUM_SLOTS_PER_READ=31, GDN/KV/conv state) stay private behind the TT adapter.".into(),
            "Per-field normalized distribution is not proof of jointly valid multi-field action; confidence never authorizes execution.".into(),
            "Offline re-prefill is not online early stopping; cropping KV cannot be assumed to roll back hybrid GDN/recurrent state.".into(),
            "tt-onix#16 owns Qwen escalation/stopping (AutoThink/DeLLMa); Animus owns routing; RSI owns policy training.".into(),
        ],
    }
}

pub fn inputs_json(report: &StructuredReadReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.issue-10-structured-read.inputs.v1\",\n",
            "  \"issue\": 10,\n",
            "  \"execution_class\": {},\n",
            "  \"donor_pin\": {},\n",
            "  \"donor_license\": {},\n",
            "  \"distinct_operations\": [\"ar_label_read\",\"diffusion_canvas_read\",\"clef_joint_head_read\"],\n",
            "  \"boundary_notes\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(&report.donor_pin),
        quote(&report.donor_license),
        str_array(&report.boundary_notes),
    )
}

pub fn results_json(report: &StructuredReadReport) -> String {
    let mut out = String::new();
    writeln!(out, "{{").unwrap();
    writeln!(
        out,
        "  \"schema\": \"inference-lab.issue-10-structured-read.results.v1\","
    )
    .unwrap();
    writeln!(out, "  \"issue\": 10,").unwrap();
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

    let gates: Vec<String> = report
        .gate_checks
        .iter()
        .map(|g| {
            format!(
                concat!(
                    "{{\"gate_case_id\":{},\"gate_satisfied\":{},\"distinct_real_adapters\":{},",
                    "\"qualified_consumers\":{},\"rejection_reason\":{}}}"
                ),
                quote(&g.gate_case_id),
                g.gate_satisfied,
                g.distinct_real_adapters,
                g.qualified_consumers,
                opt_str(g.rejection_reason.as_deref()),
            )
        })
        .collect();
    writeln!(out, "  \"extraction_gate_checks\": [{}],", gates.join(",")).unwrap();

    let fixs: Vec<String> = report
        .fixture_verdicts
        .iter()
        .map(|v| {
            format!(
                concat!(
                    "{{\"fixture_id\":{},\"op\":{},\"outcome\":{},\"calibration_valid\":{},",
                    "\"entropy_nats\":{},\"winning_label\":{},",
                    "\"action_authorized_by_confidence\":{},\"rejection_reason\":{}}}"
                ),
                quote(&v.fixture_id),
                quote(v.op.as_str()),
                quote(v.outcome.as_str()),
                v.calibration_valid,
                opt_f64(v.entropy_nats),
                opt_str(v.winning_label.as_deref()),
                v.action_authorized_by_confidence,
                opt_str(v.rejection_reason.as_deref()),
            )
        })
        .collect();
    writeln!(out, "  \"fixture_verdicts\": [{}]", fixs.join(",")).unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn summary_markdown(report: &StructuredReadReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #10: Conditional Shared Structured-Read Contracts"
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
        "- Conditional gate: `{}` (`{}`)",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(
        out,
        "- Donor pin: `{}` (`{}`)",
        report.donor_pin, report.donor_license
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Dependency Gate Checks").unwrap();
    for g in &report.gate_checks {
        writeln!(
            out,
            "- `{}`: satisfied=`{}` ({})",
            g.gate_case_id,
            g.gate_satisfied,
            g.rejection_reason.as_deref().unwrap_or("passed")
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "## Adversarial Readout Fixtures").unwrap();
    writeln!(out, "| Fixture | Op | Outcome | Reason |").unwrap();
    writeln!(out, "| --- | --- | --- | --- |").unwrap();
    for v in &report.fixture_verdicts {
        writeln!(
            out,
            "| `{}` | `{}` | `{}` | {} |",
            v.fixture_id,
            v.op.as_str(),
            v.outcome.as_str(),
            v.rejection_reason.as_deref().unwrap_or("none")
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "> Explicitly `blocked_on_artifacts`. No shared runtime library was manufactured ahead of two live distinct backends and a second concrete consumer."
    )
    .unwrap();
    out
}
