//! Issue #4: Reference-assisted agent kernel optimization vs specification-only search.
//!
//! Implements candidate-independent evaluator gate ordering (integrity/write-isolation ->
//! FP64-vs-BF16 relative tolerance + hidden prompts + batch-reorder bit-identity + Cachet
//! schedule resource bounds -> soak stage -> local run-to-run noise floor), retained `runs/`
//! submission logs, separate experimental factors (CompileIQ ACF, reviewer-guided continuation
//! preserving RSI #26 NULL confirmation, #38 persistent evaluator, default-off fresh-agent
//! rediscovery), and explicit license provenance gates.

use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_f64, opt_str, quote, str_array};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationArm {
    SpecOnly,
    ReferenceAssisted,
    ReferencePlusProfiler,
    CompileIqAcf,
    ReviewerGuidedContinuation,
    PersistentTieredEvaluator,
    FreshAgentRediscovery,
}

impl OptimizationArm {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SpecOnly => "spec_only",
            Self::ReferenceAssisted => "reference_assisted",
            Self::ReferencePlusProfiler => "reference_plus_profiler",
            Self::CompileIqAcf => "compileiq_acf_separate",
            Self::ReviewerGuidedContinuation => "reviewer_guided_continuation",
            Self::PersistentTieredEvaluator => "persistent_tiered_evaluator_38",
            Self::FreshAgentRediscovery => "fresh_agent_rediscovery_default_off",
        }
    }
}

/// Local run-to-run noise floor computed strictly from frozen local baseline repeats (never a foreign B200 roofline).
#[derive(Debug, Clone, PartialEq)]
pub struct LocalNoiseFloor {
    pub hardware_label: String,
    pub baseline_repeats_ms: Vec<f64>,
    pub baseline_mean_ms: f64,
    pub noise_floor_abs_ms: f64,
    pub noise_floor_rel_fraction: f64,
    pub foreign_roofline_reused: bool,
}

pub fn compute_local_noise_floor(
    hardware_label: &str,
    baseline_repeats_ms: &[f64],
) -> Result<LocalNoiseFloor, String> {
    if baseline_repeats_ms.len() < 3
        || baseline_repeats_ms
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err("at_least_3_positive_finite_baseline_repeats_required".into());
    }
    let n = baseline_repeats_ms.len() as f64;
    let mean = baseline_repeats_ms.iter().sum::<f64>() / n;
    let min_v = baseline_repeats_ms
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_v = baseline_repeats_ms
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let var = baseline_repeats_ms
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    let std = var.sqrt();
    let floor_abs = ((max_v - min_v) * 0.5).max(2.0 * std);
    Ok(LocalNoiseFloor {
        hardware_label: hardware_label.into(),
        baseline_repeats_ms: baseline_repeats_ms.to_vec(),
        baseline_mean_ms: mean,
        noise_floor_abs_ms: floor_abs,
        noise_floor_rel_fraction: floor_abs / mean,
        foreign_roofline_reused: false,
    })
}

/// Candidate submission evaluated strictly in megakernel-zoo + Cachet + Snarkify gate order.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelSubmission {
    pub submission_id: String,
    pub parent_id: Option<String>,
    pub arm: OptimizationArm,
    pub hypothesis: String,
    pub source_digest: DigestRef,
    pub is_identity_noop: bool,
    pub evaluator_hash_intact: bool,
    pub timing_forged_or_incomplete: bool,
    pub fp64_ref_logits: Vec<f64>,
    pub bf16_ref_logits: Vec<f64>,
    pub candidate_logits: Vec<f64>,
    pub hidden_prompt_max_err: f64,
    pub batch_order_abc_bits: Vec<u64>,
    pub batch_order_cab_realigned_bits: Vec<u64>,
    pub modeled_scratch_bytes: u64,
    pub scratch_budget_bytes: u64,
    pub soak_passed: bool,
    pub reviewer_claimed_goal_met: bool,
    pub candidate_median_ms: f64,
    pub total_agent_and_eval_cost_units: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubmissionEvaluation {
    pub submission_id: String,
    pub parent_id: Option<String>,
    pub arm: OptimizationArm,
    pub hypothesis: String,
    pub source_digest: DigestRef,
    pub reviewer_claimed_goal_met: bool,
    pub total_agent_and_eval_cost_units: u64,
    pub gate_integrity_passed: bool,
    pub gate_output_tolerance_passed: bool,
    pub gate_hidden_prompts_passed: bool,
    pub gate_batch_reorder_bit_identical: bool,
    pub gate_schedule_budget_passed: bool,
    pub gate_soak_passed: bool,
    pub timing_eligible: bool,
    pub cleared_noise_floor: bool,
    pub measured_speedup_vs_baseline: Option<f64>,
    pub max_abs_err_vs_fp64: Option<f64>,
    pub allowed_err_2x_bf16: Option<f64>,
    pub outcome: OutcomeStatus,
    pub rejection_stage: Option<String>,
}

fn checked_max_abs_err(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.is_empty() || a.len() != b.len() {
        return None;
    }
    let mut max_err = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let d = (x - y).abs();
        if !d.is_finite() {
            return None;
        }
        if d > max_err {
            max_err = d;
        }
    }
    Some(max_err)
}

/// Evaluate a candidate strictly in order:
/// 1. Integrity/write-isolation
/// 2. FP64 vs 2x BF16 error (strict finite non-empty logits) + hidden prompts + batch-reorder bit identity + schedule budget
/// 3. Soak stage
/// 4. Local noise floor with strict positive finite candidate latency (timing is NEVER eligible before stages 1-3 pass).
pub fn evaluate_submission(
    sub: &KernelSubmission,
    noise_floor: &LocalNoiseFloor,
) -> SubmissionEvaluation {
    let allowed_err =
        checked_max_abs_err(&sub.fp64_ref_logits, &sub.bf16_ref_logits).map(|e| 2.0 * e);
    let cand_err = checked_max_abs_err(&sub.fp64_ref_logits, &sub.candidate_logits);

    let integrity_ok = sub.evaluator_hash_intact && !sub.timing_forged_or_incomplete;
    let tol_ok = match (cand_err, allowed_err) {
        (Some(ce), Some(ae)) => ce <= ae,
        _ => false,
    };
    let hidden_ok = match allowed_err {
        Some(ae) => {
            sub.hidden_prompt_max_err.is_finite()
                && sub.hidden_prompt_max_err >= 0.0
                && sub.hidden_prompt_max_err <= ae
        }
        None => false,
    };
    let reorder_ok = !sub.batch_order_abc_bits.is_empty()
        && sub.batch_order_abc_bits == sub.batch_order_cab_realigned_bits;
    let sched_ok = sub.modeled_scratch_bytes <= sub.scratch_budget_bytes;

    let mut stage: Option<String> = None;
    let mut outcome = OutcomeStatus::Invalid;
    let mut timing_eligible = false;
    let mut cleared_noise_floor = false;
    let mut speedup = None;

    if !integrity_ok {
        stage = Some("gate_1_integrity_or_forged_timing_rejected".into());
    } else if !tol_ok {
        stage = Some("gate_2a_fp64_relative_tolerance_exceeded_or_nonfinite".into());
    } else if !hidden_ok {
        stage = Some("gate_2b_hidden_held_out_prompt_drift".into());
    } else if !reorder_ok {
        stage = Some("gate_2c_batch_reorder_not_bit_identical".into());
    } else if !sched_ok {
        stage = Some("gate_2d_cachet_schedule_scratch_overbudget".into());
    } else if !sub.soak_passed {
        stage = Some("gate_3_soak_stage_failed".into());
    } else if !sub.candidate_median_ms.is_finite() || sub.candidate_median_ms <= 0.0 {
        stage = Some("gate_4_invalid_nonpositive_or_nonfinite_latency".into());
    } else {
        timing_eligible = true;
        if sub.is_identity_noop {
            outcome = OutcomeStatus::Invalid;
            stage = Some("identity_noop_baseline_not_an_optimization".into());
        } else {
            let improvement_ms = noise_floor.baseline_mean_ms - sub.candidate_median_ms;
            if improvement_ms > noise_floor.noise_floor_abs_ms {
                cleared_noise_floor = true;
                speedup = Some(noise_floor.baseline_mean_ms / sub.candidate_median_ms);
                outcome = OutcomeStatus::Accepted;
            } else {
                outcome = OutcomeStatus::Invalid;
                stage = Some("gate_4_within_local_noise_floor_no_improvement".into());
            }
        }
    }

    SubmissionEvaluation {
        submission_id: sub.submission_id.clone(),
        parent_id: sub.parent_id.clone(),
        arm: sub.arm,
        hypothesis: sub.hypothesis.clone(),
        source_digest: sub.source_digest.clone(),
        reviewer_claimed_goal_met: sub.reviewer_claimed_goal_met,
        total_agent_and_eval_cost_units: sub.total_agent_and_eval_cost_units,
        gate_integrity_passed: integrity_ok,
        gate_output_tolerance_passed: tol_ok,
        gate_hidden_prompts_passed: hidden_ok,
        gate_batch_reorder_bit_identical: reorder_ok,
        gate_schedule_budget_passed: sched_ok,
        gate_soak_passed: sub.soak_passed,
        timing_eligible,
        cleared_noise_floor,
        measured_speedup_vs_baseline: speedup,
        max_abs_err_vs_fp64: cand_err,
        allowed_err_2x_bf16: allowed_err,
        outcome,
        rejection_stage: stage,
    }
}

/// Fresh-agent rediscovery contamination guard (drouyang/Snarkify ask).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RediscoveryIsolationCheck {
    pub arm_id: String,
    pub default_off: bool,
    pub sees_prior_notes: bool,
    pub sees_reviewer_prose: bool,
    pub sees_incumbent_kernel_source: bool,
    pub sees_other_agent_traces: bool,
    pub isolation_valid: bool,
}

pub fn check_rediscovery_isolation(
    arm_id: &str,
    default_off: bool,
    sees_prior_notes: bool,
    sees_reviewer_prose: bool,
    sees_incumbent_kernel_source: bool,
    sees_other_agent_traces: bool,
) -> RediscoveryIsolationCheck {
    let isolation_valid = !sees_prior_notes
        && !sees_reviewer_prose
        && !sees_incumbent_kernel_source
        && !sees_other_agent_traces;
    RediscoveryIsolationCheck {
        arm_id: arm_id.into(),
        default_off,
        sees_prior_notes,
        sees_reviewer_prose,
        sees_incumbent_kernel_source,
        sees_other_agent_traces,
        isolation_valid,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewerContinuationControl {
    pub case_id: String,
    pub stop_reason: String,
    pub budget_exceeded: bool,
    pub restarted_finished_work: bool,
    pub prompt_cache_telemetry_status: String,
    pub reviewer_overrode_failed_check: bool,
    pub control_passed: bool,
}

#[derive(Debug, Clone)]
pub struct KernelOptReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub noise_floor: LocalNoiseFloor,
    pub evaluations: Vec<SubmissionEvaluation>,
    pub rediscovery_checks: Vec<RediscoveryIsolationCheck>,
    pub reviewer_controls: Vec<ReviewerContinuationControl>,
    pub license_provenance: Vec<String>,
    pub decision: String,
}

pub fn run_default_harness() -> Result<KernelOptReport, String> {
    let noise_floor = compute_local_noise_floor(
        "local_frozen_contract_baseline",
        &[10.00, 10.12, 9.91, 10.05, 9.97],
    )?;

    let fp64 = vec![1.000000, -0.500000, 2.250000, 0.125000];
    let bf16 = vec![1.002000, -0.499000, 2.252000, 0.125500];
    let good_cand = vec![1.001500, -0.499200, 2.251800, 0.125400];
    let drift_cand = vec![1.015000, -0.499000, 2.252000, 0.125500];
    let bits_ok = vec![
        0x3ff0_0000_0000_0000u64,
        0xbfe0_0000_0000_0000,
        0x4002_0000_0000_0000,
    ];
    let bits_bad = vec![
        0x3ff0_0000_0000_0001u64,
        0xbfe0_0000_0000_0000,
        0x4002_0000_0000_0000,
    ];

    let base_sub = |id: &str, arm: OptimizationArm, hyp: &str| KernelSubmission {
        submission_id: id.into(),
        parent_id: Some("sub-00-baseline".into()),
        arm,
        hypothesis: hyp.into(),
        source_digest: DigestRef::sha256(id.as_bytes()),
        is_identity_noop: false,
        evaluator_hash_intact: true,
        timing_forged_or_incomplete: false,
        fp64_ref_logits: fp64.clone(),
        bf16_ref_logits: bf16.clone(),
        candidate_logits: good_cand.clone(),
        hidden_prompt_max_err: 0.0018,
        batch_order_abc_bits: bits_ok.clone(),
        batch_order_cab_realigned_bits: bits_ok.clone(),
        modeled_scratch_bytes: 32_768,
        scratch_budget_bytes: 65_536,
        soak_passed: true,
        reviewer_claimed_goal_met: false,
        candidate_median_ms: 8.40,
        total_agent_and_eval_cost_units: 120,
    };

    let s_valid = base_sub(
        "sub-01-valid-clears-soak-and-noise",
        OptimizationArm::ReferenceAssisted,
        "tile-packed loop within scratchpad clearing local noise floor",
    );

    let mut s_noop = base_sub(
        "sub-02-identity-noop-baseline",
        OptimizationArm::SpecOnly,
        "unchanged baseline identity",
    );
    s_noop.is_identity_noop = true;

    let mut s_wrong_fast = base_sub(
        "sub-03-wrong-but-fast-logit-drift",
        OptimizationArm::SpecOnly,
        "aggressive approx skipping tail reduction",
    );
    s_wrong_fast.candidate_logits = drift_cand;
    s_wrong_fast.candidate_median_ms = 3.10;
    s_wrong_fast.reviewer_claimed_goal_met = true;

    let mut s_reorder = base_sub(
        "sub-04-fast-batch-reorder-drift",
        OptimizationArm::ReferenceAssisted,
        "non-deterministic atomic reduction across batch",
    );
    s_reorder.batch_order_cab_realigned_bits = bits_bad;
    s_reorder.candidate_median_ms = 5.00;

    let mut s_overbudget = base_sub(
        "sub-05-correct-but-overbudget-schedule",
        OptimizationArm::ReferencePlusProfiler,
        "correct matmul exceeding Cachet scratchpad limit",
    );
    s_overbudget.modeled_scratch_bytes = 131_072;

    let mut s_soak_fail = base_sub(
        "sub-06-passes-dev-fails-soak",
        OptimizationArm::ReviewerGuidedContinuation,
        "passes short check then drifts during soak",
    );
    s_soak_fail.soak_passed = false;
    s_soak_fail.reviewer_claimed_goal_met = true;

    let mut s_noise = base_sub(
        "sub-07-within-local-noise-floor",
        OptimizationArm::CompileIqAcf,
        "1% apparent gain inside local repeat spread",
    );
    s_noise.candidate_median_ms = 9.92;

    let mut s_forged = base_sub(
        "sub-08-forged-timing-tampered-eval",
        OptimizationArm::PersistentTieredEvaluator,
        "tampered timing harness",
    );
    s_forged.timing_forged_or_incomplete = true;
    s_forged.candidate_median_ms = 0.50;

    let mut s_nan_logits = base_sub(
        "sub-09-nan-logits-must-fail-gate2a",
        OptimizationArm::SpecOnly,
        "all-NaN candidate output must never bypass max-abs-error check",
    );
    s_nan_logits.candidate_logits = vec![f64::NAN; 4];

    let mut s_zero_latency = base_sub(
        "sub-10-zero-latency-invalid-timing",
        OptimizationArm::SpecOnly,
        "zero or non-positive reported latency cannot claim infinite speedup",
    );
    s_zero_latency.candidate_median_ms = 0.0;

    let submissions = vec![
        s_valid,
        s_noop,
        s_wrong_fast,
        s_reorder,
        s_overbudget,
        s_soak_fail,
        s_noise,
        s_forged,
        s_nan_logits,
        s_zero_latency,
    ];
    let evaluations = submissions
        .iter()
        .map(|s| evaluate_submission(s, &noise_floor))
        .collect();

    let rediscovery_checks = vec![
        check_rediscovery_isolation(
            "fresh-agent-harness-only-default-off",
            true,
            false,
            false,
            false,
            false,
        ),
        check_rediscovery_isolation(
            "negative-leaked-prior-notes-and-incumbent",
            true,
            true,
            false,
            true,
            false,
        ),
    ];

    let reviewer_controls = vec![
        ReviewerContinuationControl {
            case_id: "goal-already-met-stops-cleanly".into(),
            stop_reason: "goal_already_met".into(),
            budget_exceeded: false,
            restarted_finished_work: false,
            prompt_cache_telemetry_status: "unknown_provider_telemetry_unavailable".into(),
            reviewer_overrode_failed_check: false,
            control_passed: true,
        },
        ReviewerContinuationControl {
            case_id: "budget-exhausted-enforced-by-runtime".into(),
            stop_reason: "budget_exhausted".into(),
            budget_exceeded: false,
            restarted_finished_work: false,
            prompt_cache_telemetry_status: "observed_prefix_cache_miss".into(),
            reviewer_overrode_failed_check: false,
            control_passed: true,
        },
        ReviewerContinuationControl {
            case_id: "premature-stop-distinguished-from-blocked".into(),
            stop_reason: "premature_stop_diagnosed".into(),
            budget_exceeded: false,
            restarted_finished_work: false,
            prompt_cache_telemetry_status: "unknown_provider_telemetry_unavailable".into(),
            reviewer_overrode_failed_check: false,
            control_passed: true,
        },
    ];

    Ok(KernelOptReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "sm120_rtx5080_sm121_gb10_or_b200_gpu_kernel_execution",
            "Evaluator gate ordering, soak/noise-floor rejection, and rediscovery isolation verified locally; physical GPU kernel builds, RWKV/MLA/MoE/ComputeEval timing, and CompileIQ ACF runs remain external hardware gates.",
        ),
        noise_floor,
        evaluations,
        rediscovery_checks,
        reviewer_controls,
        license_provenance: vec![
            "basetenlabs/deepseek-ultrafast@3aad78f1f6a3ae87ddb6b7344edbb40433b519aa: Elastic License 2.0 (inspect only, no incompatible copy)".into(),
            "NVIDIA/compute-eval@00fbad91: Apache-2.0 harness, NVIDIA Evaluation Dataset License for data/".into(),
            "NVIDIA/CompileIQ@ed52623f: NVIDIA Software License Agreement (distinct from Apache-2.0)".into(),
            "dangxingyu/megakernel-zoo@e3cef3b1cca3dc9164c062899e3e3215380640ba: Apache-2.0 (H100/B200 claims 1.13-2.05x unverified & non-gating)".into(),
            "RSI #26 confirmation preserved: 156/264 plain vs 153/264 forced across 42 paired trials -> frozen verdict NULL (off by default)".into(),
        ],
        decision: "INCONCLUSIVE_UNAVAILABLE_HARDWARE_RUNTIME".into(),
    })
}

pub fn inputs_json(report: &KernelOptReport) -> String {
    let reps: Vec<String> = report
        .noise_floor
        .baseline_repeats_ms
        .iter()
        .map(|v| finite_f64(*v))
        .collect();
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.issue-04-kernel-opt.inputs.v1\",\n",
            "  \"issue\": 4,\n",
            "  \"execution_class\": {},\n",
            "  \"local_noise_floor\": {{\"hardware_label\":{},\"baseline_repeats_ms\":[{}],\"baseline_mean_ms\":{},\"noise_floor_abs_ms\":{},\"noise_floor_rel_fraction\":{},\"foreign_roofline_reused\":{}}},\n",
            "  \"license_and_prior_verdict_provenance\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(&report.noise_floor.hardware_label),
        reps.join(","),
        finite_f64(report.noise_floor.baseline_mean_ms),
        finite_f64(report.noise_floor.noise_floor_abs_ms),
        finite_f64(report.noise_floor.noise_floor_rel_fraction),
        report.noise_floor.foreign_roofline_reused,
        str_array(&report.license_provenance),
    )
}

pub fn results_json(report: &KernelOptReport) -> String {
    let mut out = String::new();
    writeln!(out, "{{").unwrap();
    writeln!(
        out,
        "  \"schema\": \"inference-lab.issue-04-kernel-opt.results.v1\","
    )
    .unwrap();
    writeln!(out, "  \"issue\": 4,").unwrap();
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

    let evals: Vec<String> = report
        .evaluations
        .iter()
        .map(|e| {
            format!(
                concat!(
                    "{{\"submission_id\":{},\"parent_id\":{},\"arm\":{},\"hypothesis\":{},",
                    "\"source_digest\":{},\"reviewer_claimed_goal_met\":{},",
                    "\"total_agent_and_eval_cost_units\":{},",
                    "\"gate_integrity_passed\":{},\"gate_output_tolerance_passed\":{},",
                    "\"gate_hidden_prompts_passed\":{},\"gate_batch_reorder_bit_identical\":{},",
                    "\"gate_schedule_budget_passed\":{},\"gate_soak_passed\":{},",
                    "\"timing_eligible\":{},\"cleared_noise_floor\":{},",
                    "\"measured_speedup_vs_baseline\":{},\"max_abs_err_vs_fp64\":{},",
                    "\"allowed_err_2x_bf16\":{},\"outcome\":{},\"rejection_stage\":{}}}"
                ),
                quote(&e.submission_id),
                opt_str(e.parent_id.as_deref()),
                quote(e.arm.as_str()),
                quote(&e.hypothesis),
                e.source_digest.to_json(),
                e.reviewer_claimed_goal_met,
                e.total_agent_and_eval_cost_units,
                e.gate_integrity_passed,
                e.gate_output_tolerance_passed,
                e.gate_hidden_prompts_passed,
                e.gate_batch_reorder_bit_identical,
                e.gate_schedule_budget_passed,
                e.gate_soak_passed,
                e.timing_eligible,
                e.cleared_noise_floor,
                opt_f64(e.measured_speedup_vs_baseline),
                opt_f64(e.max_abs_err_vs_fp64),
                opt_f64(e.allowed_err_2x_bf16),
                quote(e.outcome.as_str()),
                opt_str(e.rejection_stage.as_deref()),
            )
        })
        .collect();
    writeln!(out, "  \"retained_runs_log\": [{}],", evals.join(",")).unwrap();

    let red: Vec<String> = report
        .rediscovery_checks
        .iter()
        .map(|r| {
            format!(
                concat!(
                    "{{\"arm_id\":{},\"default_off\":{},\"sees_prior_notes\":{},",
                    "\"sees_reviewer_prose\":{},\"sees_incumbent_kernel_source\":{},",
                    "\"sees_other_agent_traces\":{},\"isolation_valid\":{}}}"
                ),
                quote(&r.arm_id),
                r.default_off,
                r.sees_prior_notes,
                r.sees_reviewer_prose,
                r.sees_incumbent_kernel_source,
                r.sees_other_agent_traces,
                r.isolation_valid,
            )
        })
        .collect();
    writeln!(
        out,
        "  \"rediscovery_isolation_checks\": [{}],",
        red.join(",")
    )
    .unwrap();

    let rev: Vec<String> = report
        .reviewer_controls
        .iter()
        .map(|r| {
            format!(
                concat!(
                    "{{\"case_id\":{},\"stop_reason\":{},\"budget_exceeded\":{},",
                    "\"restarted_finished_work\":{},\"prompt_cache_telemetry_status\":{},",
                    "\"reviewer_overrode_failed_check\":{},\"control_passed\":{}}}"
                ),
                quote(&r.case_id),
                quote(&r.stop_reason),
                r.budget_exceeded,
                r.restarted_finished_work,
                quote(&r.prompt_cache_telemetry_status),
                r.reviewer_overrode_failed_check,
                r.control_passed,
            )
        })
        .collect();
    writeln!(
        out,
        "  \"reviewer_continuation_controls\": [{}]",
        rev.join(",")
    )
    .unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn summary_markdown(report: &KernelOptReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #4: Reference-Assisted vs Spec-Only Agent Kernel Optimization"
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
        "- Local noise floor: `{:.4} ms` ({:.2}% of `{:.4} ms` baseline; foreign roofline reused: `{}`)",
        report.noise_floor.noise_floor_abs_ms,
        report.noise_floor.noise_floor_rel_fraction * 100.0,
        report.noise_floor.baseline_mean_ms,
        report.noise_floor.foreign_roofline_reused
    )
    .unwrap();
    writeln!(out, "- Decision: `{}`", report.decision).unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "| Submission | Arm | Timing Eligible | Cleared Noise | Outcome | Rejection Stage |"
    )
    .unwrap();
    writeln!(out, "| --- | --- | --- | --- | --- | --- |").unwrap();
    for e in &report.evaluations {
        writeln!(
            out,
            "| `{}` | `{}` | `{}` | `{}` | `{}` | {} |",
            e.submission_id,
            e.arm.as_str(),
            e.timing_eligible,
            e.cleared_noise_floor,
            e.outcome.as_str(),
            e.rejection_stage.as_deref().unwrap_or("none")
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "> Bounded contract harness only. Upstream megakernel-zoo/Snarkify/Han Xiao numbers remain unverified non-gating references."
    )
    .unwrap();
    out
}
