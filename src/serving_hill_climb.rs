//! Issue #53: Measure a ceiling-first, interleaved-control serving hill-climb on one pinned lab target.
//!
//! Pinned external skill & non-gating motivation:
//! - `sybil-solutions/inference-optimisation-skill@fda968e7fc1996aa23c67139c755fe541c9d4be0` (MIT).
//! - Default-off (`default_enabled: false`): selected only by explicit request; does not alter
//!   default serving, Spark NixOS modules, or tt-onix defaults.
//! - Reuses `#7` (`src/serving_bench.rs`) `WorkloadIdentity` and token-accounting conventions.
//! - Published B390, Arc Pro B70, RTX 3090, H200 NVL, and "2.1x" prefill figures from the skill
//!   README are authors' unverified external measurements and are NOT local pass/fail gates.
//!
//! Provides:
//! - Physics-based practical ceiling calculator (`compute_practical_ceilings`) from measured memory
//!   bandwidth and measured matmul FLOPs computed BEFORE lever search, explicitly refusing targets
//!   above practical ceiling.
//! - Bounded one-lever candidate evaluator with `Control-Candidate-Control` interleaving (medians
//!   of `>= 6` finite-checked samples, accepting ONLY when candidate beats BOTH neighboring controls
//!   and regresses no declared secondary metric by `> 3.0%`), retaining ALL accepted and rejected
//!   runs in a permanent ledger.
//! - Protocol enforcement for: server-usage/tokenizer token accounting (rejecting streamed chunks),
//!   real-tokenizer prompt lengths, unified-memory residency checks on unified-memory hosts,
//!   long-context exact-match needle gate, concurrent full-KV soak gate (rejecting looping, empty,
//!   or failed fast runs), greedy lossless equivalence + lossy quality-trade labeling, clean-checkout
//!   rebuild recheck with labeled unmixed speed tables, and 8 executable negative controls.

use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_f64, quote, str_array};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};
use crate::serving_bench::{default_workload_identity, WorkloadIdentity};

pub const SKILL_REPO: &str = "sybil-solutions/inference-optimisation-skill";
pub const SKILL_COMMIT: &str = "fda968e7fc1996aa23c67139c755fe541c9d4be0";
pub const SKILL_LICENSE: &str = "MIT";
pub const DEFAULT_ENABLED: bool = false;
pub const MIN_DECODE_SAMPLES: usize = 6;
pub const MAX_SECONDARY_REGRESSION_PCT: f64 = 3.0;
pub const MAX_LOOPING_NGRAM_RATIO: f64 = 0.35;

#[derive(Debug, Clone)]
pub struct HardwareCeilingInputs {
    pub measured_mem_bw_gb_s: f64,
    pub measured_matmul_tflops: f64,
    pub bandwidth_from_measurement: bool,
    pub flops_from_measurement: bool,
    pub active_weight_bytes_per_token: f64,
    pub kv_bytes_per_token: f64,
    pub active_flops_per_token: f64,
    pub practical_efficiency: f64,
    pub requested_target_decode_tok_s: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct PracticalCeilings {
    pub theoretical_decode_bw_ceiling_tok_s: f64,
    pub practical_decode_ceiling_tok_s: f64,
    pub theoretical_prefill_compute_ceiling_tok_s: f64,
    pub practical_prefill_ceiling_tok_s: f64,
    pub requested_target_decode_tok_s: Option<f64>,
    pub target_exceeds_practical_ceiling: bool,
    pub target_feasibility_outcome: OutcomeStatus,
}

fn require_pos_finite(v: f64, name: &str) -> Result<f64, String> {
    if !v.is_finite() || v <= 0.0 {
        Err(format!("{name} must be finite and strictly positive"))
    } else {
        Ok(v)
    }
}

/// Computes physics-based decode (bandwidth) and prefill (compute) ceilings from measured host
/// bandwidth and matmul throughput BEFORE any lever search, refusing targets above practical ceiling.
pub fn compute_practical_ceilings(
    inputs: &HardwareCeilingInputs,
) -> Result<PracticalCeilings, String> {
    if !inputs.bandwidth_from_measurement || !inputs.flops_from_measurement {
        return Err("ceilings must come from measured bandwidth and FLOPs, not spec sheets".into());
    }
    let bw = require_pos_finite(inputs.measured_mem_bw_gb_s, "measured_mem_bw_gb_s")?;
    let tflops = require_pos_finite(inputs.measured_matmul_tflops, "measured_matmul_tflops")?;
    let w_bytes = require_pos_finite(
        inputs.active_weight_bytes_per_token,
        "active_weight_bytes_per_token",
    )?;
    if !inputs.kv_bytes_per_token.is_finite() || inputs.kv_bytes_per_token < 0.0 {
        return Err("kv_bytes_per_token must be finite and non-negative".into());
    }
    let flops_tok = require_pos_finite(inputs.active_flops_per_token, "active_flops_per_token")?;
    let eff = require_pos_finite(inputs.practical_efficiency, "practical_efficiency")?;
    if eff > 1.0 {
        return Err("practical_efficiency cannot exceed 1.0".into());
    }

    let total_bytes = w_bytes + inputs.kv_bytes_per_token;
    if !total_bytes.is_finite() || total_bytes <= 0.0 {
        return Err("total decode bytes per token must be positive and finite".into());
    }
    let theoretical_decode_bw_ceiling_tok_s = (bw * 1.0e9) / total_bytes;
    let practical_decode_ceiling_tok_s = theoretical_decode_bw_ceiling_tok_s * eff;
    let theoretical_prefill_compute_ceiling_tok_s = (tflops * 1.0e12) / flops_tok;
    let practical_prefill_ceiling_tok_s = theoretical_prefill_compute_ceiling_tok_s * eff;

    if !practical_decode_ceiling_tok_s.is_finite() || !practical_prefill_ceiling_tok_s.is_finite() {
        return Err("computed ceiling is non-finite".into());
    }

    let (target_exceeds, target_outcome) = match inputs.requested_target_decode_tok_s {
        None => (false, OutcomeStatus::Accepted),
        Some(target) => {
            let t = require_pos_finite(target, "requested_target_decode_tok_s")?;
            if t > practical_decode_ceiling_tok_s {
                (true, OutcomeStatus::Unsupported)
            } else {
                (false, OutcomeStatus::Accepted)
            }
        }
    };

    Ok(PracticalCeilings {
        theoretical_decode_bw_ceiling_tok_s,
        practical_decode_ceiling_tok_s,
        theoretical_prefill_compute_ceiling_tok_s,
        practical_prefill_ceiling_tok_s,
        requested_target_decode_tok_s: inputs.requested_target_decode_tok_s,
        target_exceeds_practical_ceiling: target_exceeds,
        target_feasibility_outcome: target_outcome,
    })
}

/// Finite-checked exact median (never calls float `.max`/`.min`, verifies all finite before sorting).
pub fn checked_median(samples: &[f64]) -> Result<f64, String> {
    if samples.is_empty() {
        return Err("cannot compute median of empty sample slice".into());
    }
    for &v in samples {
        if !v.is_finite() {
            return Err("non-finite sample in checked_median".into());
        }
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        Ok(sorted[n / 2])
    } else {
        Ok((sorted[n / 2 - 1] + sorted[n / 2]) * 0.5)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenAccountingSource {
    ServerUsage,
    RealTokenizer,
    StreamedChunks,
}

impl TokenAccountingSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ServerUsage => "server_usage",
            Self::RealTokenizer => "real_tokenizer",
            Self::StreamedChunks => "streamed_chunks",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricDirection {
    HigherIsBetter,
    LowerIsBetter,
}

#[derive(Debug, Clone)]
pub struct SecondaryMetricObservation {
    pub name: String,
    pub direction: MetricDirection,
    pub control_before: f64,
    pub candidate: f64,
    pub control_after: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeverQualityClass {
    ClaimedLossless,
    DeclaredLossy,
}

impl LeverQualityClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ClaimedLossless => "claimed_lossless",
            Self::DeclaredLossy => "declared_lossy",
        }
    }
}

#[derive(Debug, Clone)]
pub struct NeedleGateInput {
    pub context_tokens: u64,
    pub expected_needle: String,
    pub observed_needle: String,
}

#[derive(Debug, Clone)]
pub struct SoakSample {
    pub request_id: String,
    pub completed_ok: bool,
    pub token_count: u64,
    pub output_text: String,
    pub repeated_ngram_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct SoakGateInput {
    pub concurrent_users: usize,
    pub kv_context_tokens: u64,
    pub runs: Vec<SoakSample>,
}

#[derive(Debug, Clone)]
pub struct QualityGateInput {
    pub quality_class: LeverQualityClass,
    pub greedy_exact_match: bool,
    pub greedy_mismatched_tokens: usize,
    pub independent_task_check_passed: bool,
}

#[derive(Debug, Clone)]
pub struct HostPolicy {
    pub is_unified_memory_host: bool,
    pub memory_cap_mib: f64,
    pub required_context_tokens: u64,
    pub required_soak_concurrency: usize,
    pub max_secondary_regression_pct: f64,
}

#[derive(Debug, Clone)]
pub struct InterleavedCandidateRun {
    pub run_id: String,
    pub harness_id: String,
    pub thinking_enabled: bool,
    pub warmup_enabled: bool,
    pub levers_changed: Vec<String>,
    pub token_accounting: TokenAccountingSource,
    pub prompt_length_measured_with_real_tokenizer: bool,
    pub unified_memory_residency_mib: Option<f64>,
    pub control_before_decode_samples: Vec<f64>,
    pub candidate_decode_samples: Vec<f64>,
    pub control_after_decode_samples: Vec<f64>,
    pub secondary_metrics: Vec<SecondaryMetricObservation>,
    pub needle: NeedleGateInput,
    pub soak: SoakGateInput,
    pub quality: QualityGateInput,
}

#[derive(Debug, Clone)]
pub struct LedgerEntry {
    pub run_id: String,
    pub harness_id: String,
    pub thinking_enabled: bool,
    pub warmup_enabled: bool,
    pub levers_changed: Vec<String>,
    pub control_before_median: Option<f64>,
    pub candidate_median: Option<f64>,
    pub control_after_median: Option<f64>,
    pub worst_secondary_regression_pct: Option<f64>,
    pub quality_class: LeverQualityClass,
    pub reported_as_quality_trade: bool,
    pub headline_speed_win_eligible: bool,
    pub needle_passed: bool,
    pub soak_passed: bool,
    pub greedy_passed: bool,
    pub outcome: OutcomeStatus,
    pub rejection_reasons: Vec<String>,
}

fn secondary_regression_pct(
    control_val: f64,
    cand_val: f64,
    dir: MetricDirection,
) -> Result<Option<f64>, String> {
    if !control_val.is_finite() || !cand_val.is_finite() {
        return Err("non-finite secondary metric observation".into());
    }
    if control_val <= 0.0 {
        return Ok(None);
    }
    let reg = match dir {
        MetricDirection::HigherIsBetter => {
            if cand_val < control_val {
                ((control_val - cand_val) / control_val) * 100.0
            } else {
                0.0
            }
        }
        MetricDirection::LowerIsBetter => {
            if cand_val > control_val {
                ((cand_val - control_val) / control_val) * 100.0
            } else {
                0.0
            }
        }
    };
    if reg.is_finite() {
        Ok(Some(reg))
    } else {
        Ok(None)
    }
}

pub fn evaluate_needle_gate(needle: &NeedleGateInput, required_context_tokens: u64) -> bool {
    needle.context_tokens >= required_context_tokens
        && !needle.expected_needle.is_empty()
        && needle.observed_needle == needle.expected_needle
}

pub fn evaluate_soak_gate(
    soak: &SoakGateInput,
    required_concurrency: usize,
    required_context_tokens: u64,
) -> Result<(bool, Vec<String>), String> {
    let mut reasons = Vec::new();
    if soak.concurrent_users < required_concurrency {
        reasons.push(format!(
            "soak_insufficient_concurrency:{}/{}",
            soak.concurrent_users, required_concurrency
        ));
    }
    if soak.kv_context_tokens < required_context_tokens {
        reasons.push(format!(
            "soak_short_kv_context:{}/{}",
            soak.kv_context_tokens, required_context_tokens
        ));
    }
    if soak.runs.is_empty() {
        reasons.push("soak_empty_run_list".into());
    }
    for r in &soak.runs {
        if !r.repeated_ngram_ratio.is_finite() || r.repeated_ngram_ratio < 0.0 {
            return Err(format!(
                "non-finite or negative repeated_ngram_ratio in soak run {}",
                r.request_id
            ));
        }
        if !r.completed_ok {
            reasons.push(format!("soak_failed_run:{}", r.request_id));
        }
        if r.token_count == 0 || r.output_text.trim().is_empty() {
            reasons.push(format!("soak_empty_output:{}", r.request_id));
        }
        if r.repeated_ngram_ratio > MAX_LOOPING_NGRAM_RATIO {
            reasons.push(format!(
                "soak_looping_output:{}(ratio={:.3})",
                r.request_id, r.repeated_ngram_ratio
            ));
        }
    }
    Ok((reasons.is_empty(), reasons))
}

/// Evaluates a single `Control-Candidate-Control` interleaved hill-climb run against all policy,
/// token-accounting, residency, median, regression, needle, soak, and greedy/lossy gates.
pub fn evaluate_interleaved_candidate(
    run: &InterleavedCandidateRun,
    policy: &HostPolicy,
    ceilings: &PracticalCeilings,
) -> Result<LedgerEntry, String> {
    require_pos_finite(policy.memory_cap_mib, "policy.memory_cap_mib")?;
    if !policy.max_secondary_regression_pct.is_finite() || policy.max_secondary_regression_pct < 0.0
    {
        return Err("policy.max_secondary_regression_pct must be finite and non-negative".into());
    }
    require_pos_finite(
        ceilings.theoretical_decode_bw_ceiling_tok_s,
        "ceilings.theoretical_decode_bw_ceiling_tok_s",
    )?;
    require_pos_finite(
        ceilings.practical_decode_ceiling_tok_s,
        "ceilings.practical_decode_ceiling_tok_s",
    )?;

    let mut reasons = Vec::new();

    if run.levers_changed.len() != 1 {
        reasons.push(format!(
            "must_change_exactly_one_lever:got_{}",
            run.levers_changed.len()
        ));
    }

    match run.token_accounting {
        TokenAccountingSource::ServerUsage | TokenAccountingSource::RealTokenizer => {}
        TokenAccountingSource::StreamedChunks => {
            reasons.push("streamed_chunk_token_accounting_rejected".into());
        }
    }
    if !run.prompt_length_measured_with_real_tokenizer {
        reasons.push("prompt_length_not_measured_with_real_tokenizer".into());
    }

    if policy.is_unified_memory_host {
        match run.unified_memory_residency_mib {
            Some(mib) if mib.is_finite() && mib > 0.0 => {
                if mib > policy.memory_cap_mib {
                    reasons.push(format!(
                        "unified_memory_residency_exceeds_cap:{mib:.1}>{:.1}",
                        policy.memory_cap_mib
                    ));
                }
            }
            _ => {
                reasons.push("missing_or_invalid_unified_memory_residency".into());
            }
        }
    }

    if run.control_before_decode_samples.len() < MIN_DECODE_SAMPLES
        || run.candidate_decode_samples.len() < MIN_DECODE_SAMPLES
        || run.control_after_decode_samples.len() < MIN_DECODE_SAMPLES
    {
        reasons.push(format!(
            "insufficient_decode_samples:require_at_least_{MIN_DECODE_SAMPLES}"
        ));
    }

    let ctrl_before = if run.control_before_decode_samples.is_empty() {
        None
    } else {
        Some(checked_median(&run.control_before_decode_samples)?)
    };
    let cand_med = if run.candidate_decode_samples.is_empty() {
        None
    } else {
        Some(checked_median(&run.candidate_decode_samples)?)
    };
    let ctrl_after = if run.control_after_decode_samples.is_empty() {
        None
    } else {
        Some(checked_median(&run.control_after_decode_samples)?)
    };

    if let (Some(cb), Some(cm), Some(ca)) = (ctrl_before, cand_med, ctrl_after) {
        if !(cm > cb && cm > ca) {
            reasons.push(format!(
                "did_not_beat_both_neighboring_controls:cand={cm:.3}_vs_before={cb:.3}_after={ca:.3}"
            ));
        }
        if cm > ceilings.theoretical_decode_bw_ceiling_tok_s {
            reasons.push(format!(
                "suspicious_speed_exceeds_theoretical_bw_ceiling:{cm:.2}>{:.2}",
                ceilings.theoretical_decode_bw_ceiling_tok_s
            ));
        }
    }

    let mut worst_reg: Option<f64> = Some(0.0);
    for sm in &run.secondary_metrics {
        let r_before = secondary_regression_pct(sm.control_before, sm.candidate, sm.direction)?;
        let r_after = secondary_regression_pct(sm.control_after, sm.candidate, sm.direction)?;
        match (r_before, r_after) {
            (Some(rb), Some(ra)) => {
                let worst_pair = if rb > ra { rb } else { ra };
                if let Some(cur) = worst_reg {
                    if worst_pair > cur {
                        worst_reg = Some(worst_pair);
                    }
                }
                if worst_pair > policy.max_secondary_regression_pct {
                    reasons.push(format!(
                        "secondary_metric_regressed:{}(reg={worst_pair:.2}%>{:.2}%)",
                        sm.name, policy.max_secondary_regression_pct
                    ));
                }
            }
            _ => {
                worst_reg = None;
                reasons.push(format!(
                    "zero_or_invalid_control_secondary_metric:{}",
                    sm.name
                ));
            }
        }
    }

    let needle_passed = evaluate_needle_gate(&run.needle, policy.required_context_tokens);
    if !needle_passed {
        reasons.push("long_context_exact_match_needle_gate_failed".into());
    }

    let (soak_passed, soak_reasons) = evaluate_soak_gate(
        &run.soak,
        policy.required_soak_concurrency,
        policy.required_context_tokens,
    )?;
    if !soak_passed {
        reasons.extend(soak_reasons);
    }

    let greedy_passed = run.quality.greedy_exact_match && run.quality.greedy_mismatched_tokens == 0;
    let (reported_as_quality_trade, headline_speed_win_eligible) = match run.quality.quality_class {
        LeverQualityClass::ClaimedLossless => {
            if !greedy_passed {
                reasons.push(format!(
                    "claimed_lossless_failed_greedy_equivalence:mismatches={}",
                    run.quality.greedy_mismatched_tokens
                ));
            }
            (false, greedy_passed)
        }
        LeverQualityClass::DeclaredLossy => {
            let eligible = run.quality.independent_task_check_passed || greedy_passed;
            (true, eligible)
        }
    };

    let outcome = if reasons.is_empty() {
        OutcomeStatus::Accepted
    } else {
        OutcomeStatus::Invalid
    };

    Ok(LedgerEntry {
        run_id: run.run_id.clone(),
        harness_id: run.harness_id.clone(),
        thinking_enabled: run.thinking_enabled,
        warmup_enabled: run.warmup_enabled,
        levers_changed: run.levers_changed.clone(),
        control_before_median: ctrl_before,
        candidate_median: cand_med,
        control_after_median: ctrl_after,
        worst_secondary_regression_pct: worst_reg,
        quality_class: run.quality.quality_class,
        reported_as_quality_trade,
        headline_speed_win_eligible,
        needle_passed,
        soak_passed,
        greedy_passed,
        outcome,
        rejection_reasons: reasons,
    })
}

#[derive(Debug, Clone)]
pub struct CleanRebuildRecheck {
    pub candidate_run_id: String,
    pub clean_checkout_verified: bool,
    pub pinned_engine_digest_matches: bool,
    pub pinned_weights_digest_matches: bool,
    pub harness_id: String,
    pub thinking_enabled: bool,
    pub warmup_enabled: bool,
    pub tables_labeled_and_not_mixed: bool,
    pub recheck_entry: LedgerEntry,
    pub final_acceptance_outcome: OutcomeStatus,
}

pub fn evaluate_clean_rebuild_recheck(
    initial: &LedgerEntry,
    clean_run: &InterleavedCandidateRun,
    clean_checkout_verified: bool,
    pinned_engine_digest_matches: bool,
    pinned_weights_digest_matches: bool,
    policy: &HostPolicy,
    ceilings: &PracticalCeilings,
) -> Result<CleanRebuildRecheck, String> {
    let recheck_entry = evaluate_interleaved_candidate(clean_run, policy, ceilings)?;
    let tables_labeled_and_not_mixed = !initial.harness_id.is_empty()
        && !recheck_entry.harness_id.is_empty()
        && initial.thinking_enabled == recheck_entry.thinking_enabled
        && initial.warmup_enabled == recheck_entry.warmup_enabled;
    let accepted = initial.outcome == OutcomeStatus::Accepted
        && recheck_entry.outcome == OutcomeStatus::Accepted
        && clean_checkout_verified
        && pinned_engine_digest_matches
        && pinned_weights_digest_matches
        && tables_labeled_and_not_mixed;
    Ok(CleanRebuildRecheck {
        candidate_run_id: initial.run_id.clone(),
        clean_checkout_verified,
        pinned_engine_digest_matches,
        pinned_weights_digest_matches,
        harness_id: recheck_entry.harness_id.clone(),
        thinking_enabled: recheck_entry.thinking_enabled,
        warmup_enabled: recheck_entry.warmup_enabled,
        tables_labeled_and_not_mixed,
        recheck_entry,
        final_acceptance_outcome: if accepted {
            OutcomeStatus::Accepted
        } else {
            OutcomeStatus::Invalid
        },
    })
}

pub fn validate_external_benchmark_gate_policy(
    import_b390_b70_3090_h200_or_2_1x_as_gate: bool,
) -> OutcomeStatus {
    if import_b390_b70_3090_h200_or_2_1x_as_gate {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    }
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct ServingHillClimbReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub synthetic_contract_fixture: bool,
    pub default_enabled: bool,
    pub skill_repo: String,
    pub skill_commit: String,
    pub skill_license: String,
    pub workload_identity: WorkloadIdentity,
    pub protocol_digest: DigestRef,
    pub ceilings: PracticalCeilings,
    pub ledger: Vec<LedgerEntry>,
    pub clean_rebuild_recheck: CleanRebuildRecheck,
    pub negative_controls: Vec<NegativeControl>,
}

fn clean_soak_samples() -> Vec<SoakSample> {
    vec![
        SoakSample {
            request_id: "soak-u1".into(),
            completed_ok: true,
            token_count: 128,
            output_text: "Verified pinned KV residency across full context window without drift."
                .into(),
            repeated_ngram_ratio: 0.04,
        },
        SoakSample {
            request_id: "soak-u2".into(),
            completed_ok: true,
            token_count: 132,
            output_text: "Second concurrent user completed exact-match continuation cleanly."
                .into(),
            repeated_ngram_ratio: 0.05,
        },
        SoakSample {
            request_id: "soak-u3".into(),
            completed_ok: true,
            token_count: 124,
            output_text: "Third concurrent user validated deterministic decode within memory cap."
                .into(),
            repeated_ngram_ratio: 0.03,
        },
        SoakSample {
            request_id: "soak-u4".into(),
            completed_ok: true,
            token_count: 130,
            output_text: "Fourth concurrent user passed full-KV soak with zero looping.".into(),
            repeated_ngram_ratio: 0.04,
        },
    ]
}

fn base_valid_candidate_run() -> InterleavedCandidateRun {
    InterleavedCandidateRun {
        run_id: "cand-01-fused-kv-page-layout".into(),
        harness_id: "inference-lab-issue7-serving-runner".into(),
        thinking_enabled: false,
        warmup_enabled: true,
        levers_changed: vec!["kv_page_block_size=32".into()],
        token_accounting: TokenAccountingSource::ServerUsage,
        prompt_length_measured_with_real_tokenizer: true,
        unified_memory_residency_mib: Some(21_400.0),
        control_before_decode_samples: vec![44.1, 44.3, 44.0, 44.2, 44.4, 44.1],
        candidate_decode_samples: vec![47.8, 48.0, 47.9, 48.1, 47.7, 48.0],
        control_after_decode_samples: vec![44.2, 44.0, 44.3, 44.1, 44.2, 44.0],
        secondary_metrics: vec![
            SecondaryMetricObservation {
                name: "prefill_tok_s".into(),
                direction: MetricDirection::HigherIsBetter,
                control_before: 1820.0,
                candidate: 1812.0, // -0.44% (within 3.0%)
                control_after: 1818.0,
            },
            SecondaryMetricObservation {
                name: "p50_ttft_ms".into(),
                direction: MetricDirection::LowerIsBetter,
                control_before: 118.0,
                candidate: 119.2, // +1.02% (within 3.0%)
                control_after: 118.5,
            },
        ],
        needle: NeedleGateInput {
            context_tokens: 32_768,
            expected_needle: "ONIX-NEEDLE-7491".into(),
            observed_needle: "ONIX-NEEDLE-7491".into(),
        },
        soak: SoakGateInput {
            concurrent_users: 4,
            kv_context_tokens: 32_768,
            runs: clean_soak_samples(),
        },
        quality: QualityGateInput {
            quality_class: LeverQualityClass::ClaimedLossless,
            greedy_exact_match: true,
            greedy_mismatched_tokens: 0,
            independent_task_check_passed: true,
        },
    }
}

pub fn run_default_harness() -> Result<ServingHillClimbReport, String> {
    let workload_identity = default_workload_identity();
    let policy = HostPolicy {
        is_unified_memory_host: true,
        memory_cap_mib: 28_672.0,
        required_context_tokens: 32_768,
        required_soak_concurrency: 4,
        max_secondary_regression_pct: MAX_SECONDARY_REGRESSION_PCT,
    };

    // 1. Physics-based ceilings computed BEFORE lever search from measured host bandwidth & FLOPs.
    let ceiling_inputs = HardwareCeilingInputs {
        measured_mem_bw_gb_s: 240.0,
        measured_matmul_tflops: 72.0,
        bandwidth_from_measurement: true,
        flops_from_measurement: true,
        active_weight_bytes_per_token: 3.2e9,
        kv_bytes_per_token: 0.4e9,
        active_flops_per_token: 6.8e9,
        practical_efficiency: 0.85,
        requested_target_decode_tok_s: Some(50.0),
    };
    let ceilings = compute_practical_ceilings(&ceiling_inputs)?;

    // 2. Bounded sequence of one-lever interleaved candidates (both accepted & rejected kept in ledger!).
    let cand1 = base_valid_candidate_run();
    let entry1 = evaluate_interleaved_candidate(&cand1, &policy, &ceilings)?;

    // Rejected candidate retained in ledger: regresses TTFT by > 3.0%
    let mut cand2 = base_valid_candidate_run();
    cand2.run_id = "cand-02-aggressive-batch-coalesce".into();
    cand2.levers_changed = vec!["max_batch_wait_ms=12".into()];
    cand2.candidate_decode_samples = vec![49.2, 49.0, 49.4, 49.1, 49.3, 49.2];
    cand2.secondary_metrics[1].candidate = 132.0; // +11.86% TTFT regression -> rejected!
    let entry2 = evaluate_interleaved_candidate(&cand2, &policy, &ceilings)?;

    // Quality-trade lossy lever retained in ledger (not a headline speed win without task check!)
    let mut cand3 = base_valid_candidate_run();
    cand3.run_id = "cand-03-lossy-prefill-quant".into();
    cand3.levers_changed = vec!["prefill_fp8_approx=true".into()];
    cand3.quality = QualityGateInput {
        quality_class: LeverQualityClass::DeclaredLossy,
        greedy_exact_match: false,
        greedy_mismatched_tokens: 7,
        independent_task_check_passed: false,
    };
    let entry3 = evaluate_interleaved_candidate(&cand3, &policy, &ceilings)?;

    let ledger = vec![entry1.clone(), entry2, entry3];

    // 3. Clean-checkout rebuild recheck of the accepted winner (entry1), labeled separately.
    let mut clean_run = cand1.clone();
    clean_run.run_id = "cand-01-clean-rebuild-recheck".into();
    clean_run.harness_id = "inference-lab-issue7-clean-rebuild-runner".into();
    clean_run.control_before_decode_samples = vec![44.0, 44.2, 44.1, 44.3, 44.1, 44.2];
    clean_run.candidate_decode_samples = vec![47.6, 47.9, 47.7, 47.8, 47.7, 47.9];
    clean_run.control_after_decode_samples = vec![44.1, 44.2, 44.0, 44.1, 44.3, 44.2];
    let clean_rebuild_recheck =
        evaluate_clean_rebuild_recheck(&entry1, &clean_run, true, true, true, &policy, &ceilings)?;

    let protocol_digest = DigestRef::sha256(
        format!(
            "{SKILL_REPO}@{SKILL_COMMIT}|{}|cap={}|ctx={}",
            workload_identity.model_id, policy.memory_cap_mib, policy.required_context_tokens
        )
        .as_bytes(),
    );

    // 8 executable negative controls
    let mut negative_controls = Vec::new();

    // NC1: target above practical ceiling refused before chasing
    {
        let mut high_target = ceiling_inputs.clone();
        high_target.requested_target_decode_tok_s =
            Some(ceilings.practical_decode_ceiling_tok_s + 15.0);
        let c = compute_practical_ceilings(&high_target)?;
        negative_controls.push(NegativeControl {
            case_id: "nc1_target_above_practical_ceiling_refused".into(),
            outcome: c.target_feasibility_outcome,
            detail: format!(
                "requested target {:.1} tok/s exceeds practical ceiling {:.1} tok/s",
                high_target.requested_target_decode_tok_s.unwrap_or(0.0),
                c.practical_decode_ceiling_tok_s
            ),
        });
    }

    // NC2: streamed chunk token accounting rejected
    {
        let mut bad = base_valid_candidate_run();
        bad.run_id = "nc2-streamed-chunks".into();
        bad.token_accounting = TokenAccountingSource::StreamedChunks;
        let e = evaluate_interleaved_candidate(&bad, &policy, &ceilings)?;
        negative_controls.push(NegativeControl {
            case_id: "nc2_streamed_chunks_token_accounting_rejected".into(),
            outcome: e.outcome,
            detail: "streamed SSE chunk counts rejected in favor of server usage / tokenizer"
                .into(),
        });
    }

    // NC3: missing unified-memory residency on unified-memory host rejected
    {
        let mut bad = base_valid_candidate_run();
        bad.run_id = "nc3-missing-unified-mem".into();
        bad.unified_memory_residency_mib = None;
        let e = evaluate_interleaved_candidate(&bad, &policy, &ceilings)?;
        negative_controls.push(NegativeControl {
            case_id: "nc3_unified_memory_missing_residency_rejected".into(),
            outcome: e.outcome,
            detail: "unified-memory host requires residency measurement beyond cgroups".into(),
        });
    }

    // NC4: candidate beating only one neighboring control rejected
    {
        let mut bad = base_valid_candidate_run();
        bad.run_id = "nc4-one-sided-control".into();
        bad.control_after_decode_samples = vec![48.5, 48.7, 48.6, 48.8, 48.4, 48.6];
        let e = evaluate_interleaved_candidate(&bad, &policy, &ceilings)?;
        negative_controls.push(NegativeControl {
            case_id: "nc4_beats_only_one_neighbor_control_rejected".into(),
            outcome: e.outcome,
            detail: "candidate failing to beat control_after rejected as drift artifact".into(),
        });
    }

    // NC5: secondary metric regression > 3.0% rejected
    {
        let mut bad = base_valid_candidate_run();
        bad.run_id = "nc5-secondary-regression".into();
        bad.secondary_metrics[0].candidate = 1710.0; // -6.04% prefill regression!
        let e = evaluate_interleaved_candidate(&bad, &policy, &ceilings)?;
        negative_controls.push(NegativeControl {
            case_id: "nc5_secondary_metric_regression_over_3pct_rejected".into(),
            outcome: e.outcome,
            detail: format!(
                "secondary metric regressed {}% (> 3.0% bound)",
                opt_f64(e.worst_secondary_regression_pct)
            ),
        });
    }

    // NC6: fast looping/empty/failed soak run rejected
    {
        let mut bad = base_valid_candidate_run();
        bad.run_id = "nc6-fast-looping-soak".into();
        bad.candidate_decode_samples = vec![55.0, 55.2, 55.1, 55.3, 55.0, 55.1];
        bad.soak.runs[1].repeated_ngram_ratio = 0.82;
        bad.soak.runs[2].output_text = "   ".into();
        let e = evaluate_interleaved_candidate(&bad, &policy, &ceilings)?;
        negative_controls.push(NegativeControl {
            case_id: "nc6_fast_looping_or_empty_soak_run_rejected".into(),
            outcome: e.outcome,
            detail: "suspiciously fast looping/empty concurrent KV soak run rejected".into(),
        });
    }

    // NC7: win disappearing on clean-checkout rebuild recheck rejected
    {
        let mut disappeared = clean_run.clone();
        disappeared.run_id = "nc7-clean-rebuild-regression".into();
        disappeared.candidate_decode_samples = vec![43.8, 44.0, 43.9, 44.1, 43.7, 44.0];
        let recheck = evaluate_clean_rebuild_recheck(
            &entry1,
            &disappeared,
            true,
            true,
            true,
            &policy,
            &ceilings,
        )?;
        negative_controls.push(NegativeControl {
            case_id: "nc7_clean_rebuild_disappearing_win_rejected".into(),
            outcome: recheck.final_acceptance_outcome,
            detail: "candidate win that disappears on clean pinned rebuild is not accepted".into(),
        });
    }

    // NC8: importing B390/B70/3090/H200/2.1x author claims as pass/fail gates rejected
    {
        let outcome = validate_external_benchmark_gate_policy(true);
        negative_controls.push(NegativeControl {
            case_id: "nc8_importing_b390_b70_3090_h200_2_1x_claims_rejected".into(),
            outcome,
            detail:
                "authors' B390/B70/3090/H200/2.1x figures cannot be imported as pass/fail gates"
                    .into(),
        });
    }

    Ok(ServingHillClimbReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "Admitted local LLM server host with OpenAI-compatible engine and pinned checkpoint",
            "Ceiling calculator, Control-Candidate-Control median evaluator, ledger, needle/soak/greedy gates, and clean-rebuild recheck verified on CPU; physical serving host did not execute locally",
        ),
        synthetic_contract_fixture: true,
        default_enabled: DEFAULT_ENABLED,
        skill_repo: SKILL_REPO.into(),
        skill_commit: SKILL_COMMIT.into(),
        skill_license: SKILL_LICENSE.into(),
        workload_identity,
        protocol_digest,
        ceilings,
        ledger,
        clean_rebuild_recheck,
        negative_controls,
    })
}

fn ledger_entry_to_json(e: &LedgerEntry) -> String {
    format!(
        concat!(
            "{{\"run_id\":{},\"harness_id\":{},\"thinking_enabled\":{},\"warmup_enabled\":{},",
            "\"levers_changed\":{},\"control_before_median\":{},\"candidate_median\":{},",
            "\"control_after_median\":{},\"worst_secondary_regression_pct\":{},",
            "\"quality_class\":{},\"reported_as_quality_trade\":{},",
            "\"headline_speed_win_eligible\":{},\"needle_passed\":{},\"soak_passed\":{},",
            "\"greedy_passed\":{},\"outcome\":{},\"rejection_reasons\":{}}}"
        ),
        quote(&e.run_id),
        quote(&e.harness_id),
        e.thinking_enabled,
        e.warmup_enabled,
        str_array(&e.levers_changed),
        opt_f64(e.control_before_median),
        opt_f64(e.candidate_median),
        opt_f64(e.control_after_median),
        opt_f64(e.worst_secondary_regression_pct),
        quote(e.quality_class.as_str()),
        e.reported_as_quality_trade,
        e.headline_speed_win_eligible,
        e.needle_passed,
        e.soak_passed,
        e.greedy_passed,
        quote(e.outcome.as_str()),
        str_array(&e.rejection_reasons),
    )
}

pub fn inputs_json(report: &ServingHillClimbReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.serving-hill-climb.inputs.v1\",\n",
            "  \"issue\": 53,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"skill_repo\": {},\n",
            "  \"skill_commit\": {},\n",
            "  \"skill_license\": {},\n",
            "  \"reuses_issue\": 7,\n",
            "  \"min_decode_samples\": {},\n",
            "  \"max_secondary_regression_pct\": {},\n",
            "  \"protocol_digest\": {},\n",
            "  \"workload_identity\": {},\n",
            "  \"synthetic_contract_fixture\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        quote(&report.skill_repo),
        quote(&report.skill_commit),
        quote(&report.skill_license),
        MIN_DECODE_SAMPLES,
        finite_f64(MAX_SECONDARY_REGRESSION_PCT),
        report.protocol_digest.to_json(),
        report.workload_identity.to_json(),
        report.synthetic_contract_fixture,
    )
}

pub fn results_json(report: &ServingHillClimbReport) -> String {
    let ledger_json: Vec<String> = report.ledger.iter().map(ledger_entry_to_json).collect();
    let controls: Vec<String> = report
        .negative_controls
        .iter()
        .map(|c| {
            format!(
                "{{\"case_id\":{},\"outcome\":{},\"detail\":{}}}",
                quote(&c.case_id),
                quote(c.outcome.as_str()),
                quote(&c.detail)
            )
        })
        .collect();
    let r = &report.clean_rebuild_recheck;
    let c = &report.ceilings;
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.serving-hill-climb.results.v1\",\n",
            "  \"issue\": 53,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"ceilings\": {{\"theoretical_decode_bw_ceiling_tok_s\":{},\"practical_decode_ceiling_tok_s\":{},\"theoretical_prefill_compute_ceiling_tok_s\":{},\"practical_prefill_ceiling_tok_s\":{},\"requested_target_decode_tok_s\":{},\"target_exceeds_practical_ceiling\":{},\"target_feasibility_outcome\":{}}},\n",
            "  \"ledger\": [{}],\n",
            "  \"clean_rebuild_recheck\": {{\"candidate_run_id\":{},\"clean_checkout_verified\":{},\"pinned_engine_digest_matches\":{},\"pinned_weights_digest_matches\":{},\"harness_id\":{},\"thinking_enabled\":{},\"warmup_enabled\":{},\"tables_labeled_and_not_mixed\":{},\"recheck_entry\":{},\"final_acceptance_outcome\":{}}},\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.synthetic_contract_fixture,
        report.default_enabled,
        finite_f64(c.theoretical_decode_bw_ceiling_tok_s),
        finite_f64(c.practical_decode_ceiling_tok_s),
        finite_f64(c.theoretical_prefill_compute_ceiling_tok_s),
        finite_f64(c.practical_prefill_ceiling_tok_s),
        opt_f64(c.requested_target_decode_tok_s),
        c.target_exceeds_practical_ceiling,
        quote(c.target_feasibility_outcome.as_str()),
        ledger_json.join(","),
        quote(&r.candidate_run_id),
        r.clean_checkout_verified,
        r.pinned_engine_digest_matches,
        r.pinned_weights_digest_matches,
        quote(&r.harness_id),
        r.thinking_enabled,
        r.warmup_enabled,
        r.tables_labeled_and_not_mixed,
        ledger_entry_to_json(&r.recheck_entry),
        quote(r.final_acceptance_outcome.as_str()),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &ServingHillClimbReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #53: Ceiling-First Interleaved-Control Serving Hill-Climb"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Execution class: `{}` (`synthetic_contract_fixture: {}`, `default_enabled: {}`)",
        report.execution_class.as_str(),
        report.synthetic_contract_fixture,
        report.default_enabled
    )
    .unwrap();
    writeln!(
        out,
        "- Hardware gate: `{}` ({})",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(
        out,
        "- Pinned skill: `{}@{}` (`{}`; authors' B390/B70/3090/H200/2.1x figures are non-gating)",
        report.skill_repo, report.skill_commit, report.skill_license
    )
    .unwrap();
    writeln!(
        out,
        "- Pre-search ceilings: decode `practical={:.2} tok/s` (`theoretical={:.2} tok/s`), prefill `practical={:.2} tok/s`, target feasibility `{}`",
        report.ceilings.practical_decode_ceiling_tok_s,
        report.ceilings.theoretical_decode_bw_ceiling_tok_s,
        report.ceilings.practical_prefill_ceiling_tok_s,
        report.ceilings.target_feasibility_outcome.as_str()
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Hill-Climb Ledger (All Accepted & Rejected Runs Retained)"
    )
    .unwrap();
    for e in &report.ledger {
        writeln!(
            out,
            "- `{}` (`harness={}`, `thinking={}`, `warmup={}`): `ctrl_before={}`, `cand={}`, `ctrl_after={}`, `quality_trade={}`, `headline_eligible={}`, `outcome={}`",
            e.run_id,
            e.harness_id,
            e.thinking_enabled,
            e.warmup_enabled,
            opt_f64(e.control_before_median),
            opt_f64(e.candidate_median),
            opt_f64(e.control_after_median),
            e.reported_as_quality_trade,
            e.headline_speed_win_eligible,
            e.outcome.as_str(),
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Clean-Checkout Rebuild Recheck: `{}` -> `{}` (harness `{}`)",
        report.clean_rebuild_recheck.candidate_run_id,
        report
            .clean_rebuild_recheck
            .final_acceptance_outcome
            .as_str(),
        report.clean_rebuild_recheck.harness_id
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Negative Controls ({} / {} rejected)",
        report
            .negative_controls
            .iter()
            .filter(|c| c.outcome != OutcomeStatus::Accepted)
            .count(),
        report.negative_controls.len()
    )
    .unwrap();
    for c in &report.negative_controls {
        writeln!(
            out,
            "- `{}` -> `{}` ({})",
            c.case_id,
            c.outcome.as_str(),
            c.detail
        )
        .unwrap();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_median_rejects_nan_and_computes_exact_even_and_odd() {
        assert!(checked_median(&[]).is_err());
        assert!(checked_median(&[1.0, f64::NAN, 2.0]).is_err());
        assert_eq!(checked_median(&[5.0, 1.0, 3.0]).unwrap(), 3.0);
        assert_eq!(
            checked_median(&[10.0, 2.0, 8.0, 4.0, 6.0, 12.0]).unwrap(),
            7.0
        );
    }

    #[test]
    fn multi_lever_fewer_than_six_samples_and_lossless_greedy_failure_rejected() {
        let report = run_default_harness().unwrap();
        let policy = HostPolicy {
            is_unified_memory_host: true,
            memory_cap_mib: 28_672.0,
            required_context_tokens: 32_768,
            required_soak_concurrency: 4,
            max_secondary_regression_pct: MAX_SECONDARY_REGRESSION_PCT,
        };

        // Multi-lever change rejected
        let mut multi = base_valid_candidate_run();
        multi.levers_changed = vec!["lever_a=1".into(), "lever_b=2".into()];
        let e_multi = evaluate_interleaved_candidate(&multi, &policy, &report.ceilings).unwrap();
        assert_eq!(e_multi.outcome, OutcomeStatus::Invalid);

        // Fewer than 6 samples rejected
        let mut few = base_valid_candidate_run();
        few.candidate_decode_samples = vec![48.0, 48.1, 47.9, 48.0, 48.2];
        let e_few = evaluate_interleaved_candidate(&few, &policy, &report.ceilings).unwrap();
        assert_eq!(e_few.outcome, OutcomeStatus::Invalid);

        // Claimed lossless failing greedy equivalence rejected
        let mut bad_greedy = base_valid_candidate_run();
        bad_greedy.quality.greedy_exact_match = false;
        bad_greedy.quality.greedy_mismatched_tokens = 2;
        let e_greedy =
            evaluate_interleaved_candidate(&bad_greedy, &policy, &report.ceilings).unwrap();
        assert_eq!(e_greedy.outcome, OutcomeStatus::Invalid);

        // Unchecked NaN policy / ceiling thresholds must fail with Err instead of silently bypassing
        let valid_run = base_valid_candidate_run();
        let mut nan_cap = policy.clone();
        nan_cap.memory_cap_mib = f64::NAN;
        assert!(evaluate_interleaved_candidate(&valid_run, &nan_cap, &report.ceilings).is_err());

        let mut nan_reg = policy.clone();
        nan_reg.max_secondary_regression_pct = f64::NAN;
        assert!(evaluate_interleaved_candidate(&valid_run, &nan_reg, &report.ceilings).is_err());

        let mut nan_ceil = report.ceilings.clone();
        nan_ceil.theoretical_decode_bw_ceiling_tok_s = f64::NAN;
        assert!(evaluate_interleaved_candidate(&valid_run, &policy, &nan_ceil).is_err());

        // Flipped thinking_enabled or warmup_enabled on clean rebuild must fail tables_labeled_and_not_mixed
        let initial =
            evaluate_interleaved_candidate(&valid_run, &policy, &report.ceilings).unwrap();
        assert_eq!(initial.outcome, OutcomeStatus::Accepted);

        let mut flipped_thinking = valid_run.clone();
        flipped_thinking.thinking_enabled = !initial.thinking_enabled;
        let r_think = evaluate_clean_rebuild_recheck(
            &initial,
            &flipped_thinking,
            true,
            true,
            true,
            &policy,
            &report.ceilings,
        )
        .unwrap();
        assert!(!r_think.tables_labeled_and_not_mixed);
        assert_eq!(r_think.final_acceptance_outcome, OutcomeStatus::Invalid);

        let mut flipped_warmup = valid_run.clone();
        flipped_warmup.warmup_enabled = !initial.warmup_enabled;
        let r_warm = evaluate_clean_rebuild_recheck(
            &initial,
            &flipped_warmup,
            true,
            true,
            true,
            &policy,
            &report.ceilings,
        )
        .unwrap();
        assert!(!r_warm.tables_labeled_and_not_mixed);
        assert_eq!(r_warm.final_acceptance_outcome, OutcomeStatus::Invalid);
    }

    #[test]
    fn default_harness_enforces_ceilings_ledger_clean_rebuild_and_all_ncs() {
        let report = run_default_harness().unwrap();
        assert_eq!(
            report.execution_class,
            ExecutionClass::BoundedContractHarness
        );
        assert_eq!(
            report.hardware_gate.status,
            ExecutionClass::UnavailableHardwareRuntime
        );
        assert!(report.synthetic_contract_fixture);
        assert!(!report.default_enabled);
        assert_eq!(report.skill_commit, SKILL_COMMIT);
        assert_eq!(report.skill_license, "MIT");
        assert_eq!(
            report.ceilings.target_feasibility_outcome,
            OutcomeStatus::Accepted
        );
        assert_eq!(report.ledger.len(), 3);
        assert_eq!(report.ledger[0].outcome, OutcomeStatus::Accepted);
        assert_eq!(report.ledger[1].outcome, OutcomeStatus::Invalid);
        // Lossy lever must be marked as quality trade and not headline-speed-win eligible
        assert!(report.ledger[2].reported_as_quality_trade);
        assert!(!report.ledger[2].headline_speed_win_eligible);
        assert_eq!(
            report.clean_rebuild_recheck.final_acceptance_outcome,
            OutcomeStatus::Accepted
        );
        assert_eq!(report.negative_controls.len(), 8);
        for nc in &report.negative_controls {
            assert_ne!(
                nc.outcome,
                OutcomeStatus::Accepted,
                "negative control {} must not be accepted",
                nc.case_id
            );
        }
        let ij = inputs_json(&report);
        let rj = results_json(&report);
        let md = summary_markdown(&report);
        assert!(ij.ends_with('\n'));
        assert!(rj.ends_with('\n'));
        assert!(ij.contains(SKILL_COMMIT));
        assert!(rj.contains("clean_rebuild_recheck"));
        assert!(md.contains("Issue #53"));
    }
}
