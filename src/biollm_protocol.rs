//! Issue #46: Benchmark biological-recording adapters against matched synthetic and shuffled controls.
//!
//! Pinned external references (strictly unverified non-gating related work; do NOT assume current
//! BioLLM-8B benchmarks exist!):
//! - `https://biollm.com/adapter`
//! - `https://github.com/4R7I5T/CL1_LLM_Encoder/blob/main/SCIENTIFIC_RESULTS.md`
//!
//! Provides:
//! - Reproducible 5-arm protocol schema comparing at **strictly equal training budgets**:
//!   1. `base_model`
//!   2. `text_only_lora`
//!   3. `bio_paired_adapter` (correctly paired biological recordings)
//!   4. `bio_shuffled_control` (shuffled recording-to-prompt pairing)
//!   5. `matched_synthetic_control` (dimension- and statistics-matched synthetic signals: mean,
//!      variance, lag-1 autocorrelation)
//! - Strict split-hygiene validator holding out BOTH task families AND recording sessions/cultures
//!   (disjoint train vs held-out sets), plus explicit separation between `MeasuredTiming` (hardware
//!   clock with uncertainty bounds) and `ReconstructedTiming` (interpolated/resampled).
//! - Deterministic pure-Rust signal statistics verifier and miniature 5-arm evaluation runner
//!   computing accuracy, retention, calibration (Expected Calibration Error `ECE` + Brier score),
//!   compute, and latency, plus falsifiable biological-attribution criteria and 8 negative controls.

use std::collections::BTreeSet;
use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, sigmoid};
use crate::json::{finite_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BioArmKind {
    BaseModel,
    TextOnlyLora,
    BioPairedAdapter,
    BioShuffledControl,
    MatchedSyntheticControl,
}

impl BioArmKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BaseModel => "base_model",
            Self::TextOnlyLora => "text_only_lora",
            Self::BioPairedAdapter => "bio_paired_adapter",
            Self::BioShuffledControl => "bio_shuffled_control",
            Self::MatchedSyntheticControl => "matched_synthetic_control",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimingProvenance {
    MeasuredHardwareClock { uncertainty_us: u64 },
    ReconstructedInterpolated,
}

impl TimingProvenance {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MeasuredHardwareClock { .. } => "measured_hardware_clock",
            Self::ReconstructedInterpolated => "reconstructed_interpolated",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingBudgetSpec {
    pub steps: usize,
    pub tokens: usize,
    pub trainable_params: usize,
    pub wall_budget_secs: u64,
}

#[derive(Debug, Clone)]
pub struct BioDatasetSplit {
    pub train_task_families: Vec<String>,
    pub heldout_task_families: Vec<String>,
    pub train_cultures_sessions: Vec<String>,
    pub heldout_cultures_sessions: Vec<String>,
    pub timing_provenance: TimingProvenance,
    pub claimed_measured_timing: bool,
}

pub fn validate_split_and_timing(split: &BioDatasetSplit) -> Result<(), String> {
    if split.heldout_task_families.is_empty() || split.heldout_cultures_sessions.is_empty() {
        return Err(
            "both held-out task families and held-out cultures/sessions are required".into(),
        );
    }
    let tr_t: BTreeSet<&str> = split
        .train_task_families
        .iter()
        .map(String::as_str)
        .collect();
    let ho_t: BTreeSet<&str> = split
        .heldout_task_families
        .iter()
        .map(String::as_str)
        .collect();
    if tr_t.intersection(&ho_t).next().is_some() {
        return Err("task-family leakage between train and held-out splits".into());
    }
    let tr_c: BTreeSet<&str> = split
        .train_cultures_sessions
        .iter()
        .map(String::as_str)
        .collect();
    let ho_c: BTreeSet<&str> = split
        .heldout_cultures_sessions
        .iter()
        .map(String::as_str)
        .collect();
    if tr_c.intersection(&ho_c).next().is_some() {
        return Err("recording culture/session leakage between train and held-out splits".into());
    }
    if split.claimed_measured_timing
        && matches!(
            split.timing_provenance,
            TimingProvenance::ReconstructedInterpolated
        )
    {
        return Err("reconstructed timing cannot be labeled as measured hardware timing".into());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct SignalMoments {
    pub dim: usize,
    pub mean: f64,
    pub variance: f64,
    pub lag1_autocorr: f64,
}

pub fn compute_signal_moments(series: &[f64]) -> Result<SignalMoments, String> {
    if series.len() < 4 {
        return Err("signal length must be >= 4".into());
    }
    if series.iter().any(|x| !x.is_finite()) {
        return Err("non-finite value in signal series".into());
    }
    let n = series.len() as f64;
    let mean = series.iter().sum::<f64>() / n;
    let variance = series.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    let mut cov = 0.0_f64;
    for i in 0..series.len() - 1 {
        cov += (series[i] - mean) * (series[i + 1] - mean);
    }
    let lag1 = if variance > 1.0e-12 {
        (cov / ((series.len() - 1) as f64)) / variance
    } else {
        0.0
    };
    Ok(SignalMoments {
        dim: series.len(),
        mean,
        variance,
        lag1_autocorr: lag1,
    })
}

/// Construct a deterministic synthetic signal matching dimension, mean, variance, and lag-1 autocorr
/// of a reference biological signal via an AR(1) standardized sequence.
pub fn synthesize_matched_signal(
    ref_moments: SignalMoments,
    seed: usize,
) -> Result<Vec<f64>, String> {
    if ref_moments.dim < 4
        || !ref_moments.mean.is_finite()
        || !ref_moments.variance.is_finite()
        || !ref_moments.lag1_autocorr.is_finite()
        || ref_moments.variance <= 0.0
    {
        return Err("invalid or non-finite reference moments for synthetic signal".into());
    }
    let phi = ref_moments.lag1_autocorr.clamp(-0.85, 0.85);
    let mut raw = vec![0.0_f64; ref_moments.dim];
    raw[0] = det_weight(4601 + seed * 11, 0, 1.0);
    for t in 1..ref_moments.dim {
        let eps = det_weight(4601 + seed * 11, t, 1.0);
        raw[t] = phi * raw[t - 1] + (1.0 - phi * phi).sqrt() * eps;
    }
    let m0 = raw.iter().sum::<f64>() / (raw.len() as f64);
    let v0 = raw.iter().map(|x| (x - m0) * (x - m0)).sum::<f64>() / (raw.len() as f64);
    if !m0.is_finite() || !v0.is_finite() || v0 <= 1.0e-12 {
        return Err("degenerate or non-finite synthesized signal variance".into());
    }
    let std_ratio = (ref_moments.variance / v0).sqrt();
    for x in &mut raw {
        *x = (*x - m0) * std_ratio + ref_moments.mean;
        if !x.is_finite() {
            return Err("non-finite synthesized signal element".into());
        }
    }
    Ok(raw)
}

pub fn verify_synthetic_match(
    bio: SignalMoments,
    synth: SignalMoments,
    tol_mean: f64,
    tol_var: f64,
    tol_ac: f64,
) -> Result<(), String> {
    if !bio.mean.is_finite()
        || !bio.variance.is_finite()
        || !bio.lag1_autocorr.is_finite()
        || !synth.mean.is_finite()
        || !synth.variance.is_finite()
        || !synth.lag1_autocorr.is_finite()
        || !tol_mean.is_finite()
        || !tol_var.is_finite()
        || !tol_ac.is_finite()
        || tol_mean < 0.0
        || tol_var < 0.0
        || tol_ac < 0.0
    {
        return Err("non-finite or negative moment/tolerance in synthetic signal check".into());
    }
    if bio.dim != synth.dim {
        return Err(format!("dimension mismatch: {} vs {}", bio.dim, synth.dim));
    }
    if (bio.mean - synth.mean).abs() > tol_mean
        || (bio.variance - synth.variance).abs() > tol_var
        || (bio.lag1_autocorr - synth.lag1_autocorr).abs() > tol_ac
    {
        return Err("synthetic signal failed dimension/statistics match tolerance".into());
    }
    Ok(())
}

/// Expected Calibration Error (4 uniform bins) and Brier score.
pub fn compute_calibration_metrics(preds: &[(f64, bool)]) -> Result<(f64, f64), String> {
    if preds.is_empty() {
        return Err("empty predictions".into());
    }
    let mut brier = 0.0_f64;
    let mut bin_count = [0usize; 4];
    let mut bin_conf = [0.0_f64; 4];
    let mut bin_acc = [0.0_f64; 4];

    for &(p, y) in preds {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err("non-finite or out-of-range probability in calibration check".into());
        }
        let yf = if y { 1.0 } else { 0.0 };
        brier += (p - yf) * (p - yf);
        let b = ((p * 4.0).floor() as usize).min(3);
        bin_count[b] += 1;
        bin_conf[b] += p;
        bin_acc[b] += yf;
    }
    let n = preds.len() as f64;
    brier /= n;
    let mut ece = 0.0_f64;
    for b in 0..4 {
        if bin_count[b] > 0 {
            let c = bin_count[b] as f64;
            ece += (c / n) * ((bin_acc[b] / c) - (bin_conf[b] / c)).abs();
        }
    }
    Ok((ece, brier))
}

#[derive(Debug, Clone)]
pub struct BioArmResult {
    pub arm: BioArmKind,
    pub training_budget: TrainingBudgetSpec,
    pub heldout_task_accuracy: f64,
    pub retention_accuracy: f64,
    pub ece: f64,
    pub brier_score: f64,
    pub inference_latency_ms_synthetic: f64,
    pub compute_gflops_synthetic: f64,
}

/// Falsifiable decision criterion: `bio_paired_adapter` establishes biological-recording value
/// ONLY if budgets match across all adapted arms AND `bio_paired_adapter` beats ALL THREE controls
/// (`text_only_lora`, `bio_shuffled_control`, `matched_synthetic_control`) by at least `margin`
/// without violating retention or calibration bounds.
pub fn evaluate_falsifiable_bio_criterion(
    arms: &[BioArmResult],
    margin: f64,
    min_retention: f64,
    max_ece: f64,
) -> Result<(&'static str, OutcomeStatus), String> {
    if !margin.is_finite()
        || !min_retention.is_finite()
        || !max_ece.is_finite()
        || margin < 0.0
        || min_retention < 0.0
        || max_ece < 0.0
    {
        return Err("non-finite or negative decision threshold".into());
    }
    let find = |k: BioArmKind| {
        arms.iter()
            .find(|a| a.arm == k)
            .ok_or_else(|| format!("missing arm {}", k.as_str()))
    };
    let base = find(BioArmKind::BaseModel)?;
    let text = find(BioArmKind::TextOnlyLora)?;
    let paired = find(BioArmKind::BioPairedAdapter)?;
    let shuffled = find(BioArmKind::BioShuffledControl)?;
    let synth = find(BioArmKind::MatchedSyntheticControl)?;

    if base.training_budget.steps != 0
        || base.training_budget.tokens != 0
        || base.training_budget.trainable_params != 0
    {
        return Err("base_model must have zero training budget".into());
    }

    // Equal training budgets across all 4 trained arms
    for ctrl in [text, shuffled, synth] {
        if ctrl.training_budget != paired.training_budget {
            return Err("unequal training budgets across adapter arms".into());
        }
    }
    for a in arms {
        if !a.heldout_task_accuracy.is_finite()
            || !a.retention_accuracy.is_finite()
            || !a.ece.is_finite()
            || !a.brier_score.is_finite()
        {
            return Err("non-finite metric in BioLLM arm evaluation".into());
        }
    }

    let beats_all = paired.heldout_task_accuracy >= base.heldout_task_accuracy + margin
        && paired.heldout_task_accuracy >= text.heldout_task_accuracy + margin
        && paired.heldout_task_accuracy >= shuffled.heldout_task_accuracy + margin
        && paired.heldout_task_accuracy >= synth.heldout_task_accuracy + margin;
    let retention_ok = paired.retention_accuracy >= min_retention;
    let cal_ok = paired.ece <= max_ece;

    if beats_all && retention_ok && cal_ok {
        Ok(("SUPPORTED_ON_FIXTURE", OutcomeStatus::Accepted))
    } else {
        Ok(("FALSIFIED_OR_NULL", OutcomeStatus::Invalid))
    }
}

/// Rejects assuming unverified BioLLM-8B benchmarks exist when no local artifacts are present.
pub fn validate_external_benchmark_assumption(
    assumes_biollm_8b_exists: bool,
    local_verified_artifact_present: bool,
) -> OutcomeStatus {
    if assumes_biollm_8b_exists && !local_verified_artifact_present {
        OutcomeStatus::BlockedOnArtifacts
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
pub struct BioLlmProtocolReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub synthetic_contract_fixture: bool,
    pub bio_signal_moments: SignalMoments,
    pub synth_signal_moments: SignalMoments,
    pub arms: Vec<BioArmResult>,
    pub fixture_criterion_verdict: String,
    pub assumes_biollm_8b_benchmarks_exist: bool,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<BioLlmProtocolReport, String> {
    let split = BioDatasetSplit {
        train_task_families: vec!["qa_train".into(), "logic_train".into()],
        heldout_task_families: vec!["reasoning_heldout".into(), "code_heldout".into()],
        train_cultures_sessions: vec!["culture_a_sess1".into(), "culture_b_sess1".into()],
        heldout_cultures_sessions: vec!["culture_c_sess1".into(), "culture_d_sess1".into()],
        timing_provenance: TimingProvenance::MeasuredHardwareClock {
            uncertainty_us: 250,
        },
        claimed_measured_timing: true,
    };
    validate_split_and_timing(&split)?;

    // Build deterministic synthetic biological-like AR(1) recording and verify matched synthetic control
    let mut bio_series = vec![0.0_f64; 32];
    bio_series[0] = 0.25;
    for t in 1..32 {
        bio_series[t] = 0.55 * bio_series[t - 1] + 0.6 * det_weight(4620, t, 0.8);
    }
    let bio_moments = compute_signal_moments(&bio_series)?;
    let synth_series = synthesize_matched_signal(bio_moments, 7)?;
    let synth_moments = compute_signal_moments(&synth_series)?;
    verify_synthetic_match(bio_moments, synth_moments, 1.0e-9, 1.0e-9, 0.35)?;

    let equal_budget = TrainingBudgetSpec {
        steps: 1000,
        tokens: 262_144,
        trainable_params: 1_048_576,
        wall_budget_secs: 600,
    };
    let zero_budget = TrainingBudgetSpec {
        steps: 0,
        tokens: 0,
        trainable_params: 0,
        wall_budget_secs: 0,
    };

    // Deterministic synthetic fixture predictions across all 5 arms (demonstrating null/non-superior
    // biological fixture where matched synthetic & text LoRA tie paired bio!)
    let make_arm = |arm: BioArmKind,
                    budget: TrainingBudgetSpec,
                    logits: &[f64],
                    labels: &[bool],
                    ret: f64,
                    lat: f64,
                    gflops: f64|
     -> Result<BioArmResult, String> {
        let mut preds = Vec::with_capacity(logits.len());
        let mut correct = 0usize;
        for (&l, &y) in logits.iter().zip(labels.iter()) {
            let p = sigmoid(l);
            if (p >= 0.5) == y {
                correct += 1;
            }
            preds.push((p, y));
        }
        let (ece, brier) = compute_calibration_metrics(&preds)?;
        Ok(BioArmResult {
            arm,
            training_budget: budget,
            heldout_task_accuracy: (correct as f64) / (labels.len() as f64),
            retention_accuracy: ret,
            ece,
            brier_score: brier,
            inference_latency_ms_synthetic: lat,
            compute_gflops_synthetic: gflops,
        })
    };

    let labels = [true, false, true, true, false, true, false, false];
    let arms = vec![
        make_arm(
            BioArmKind::BaseModel,
            zero_budget,
            &[0.8, -0.7, 0.4, -0.2, 0.3, 0.6, -0.9, 0.1],
            &labels,
            0.92,
            12.0,
            14.0,
        )?,
        make_arm(
            BioArmKind::TextOnlyLora,
            equal_budget.clone(),
            &[1.2, -1.1, 0.9, 0.7, -0.8, 0.9, -1.0, 0.2],
            &labels,
            0.90,
            12.8,
            14.8,
        )?,
        make_arm(
            BioArmKind::BioPairedAdapter,
            equal_budget.clone(),
            &[1.1, -1.0, 0.8, 0.6, -0.7, 0.9, -0.9, 0.3],
            &labels,
            0.89,
            16.5,
            17.2,
        )?,
        make_arm(
            BioArmKind::BioShuffledControl,
            equal_budget.clone(),
            &[1.0, -0.9, 0.7, 0.5, -0.6, 0.8, -0.8, 0.2],
            &labels,
            0.88,
            16.5,
            17.2,
        )?,
        make_arm(
            BioArmKind::MatchedSyntheticControl,
            equal_budget.clone(),
            &[1.1, -1.0, 0.8, 0.6, -0.7, 0.9, -0.9, 0.2],
            &labels,
            0.89,
            16.5,
            17.2,
        )?,
    ];

    let (verdict, _) = evaluate_falsifiable_bio_criterion(&arms, 0.05, 0.85, 0.25)?;

    let mut negative_controls = Vec::new();

    // NC1: unequal training budget across arms rejected
    {
        let mut bad_arms = arms.clone();
        bad_arms[2].training_budget.steps = 2000;
        negative_controls.push(NegativeControl {
            case_id: "nc1_unequal_training_budget_rejected".into(),
            outcome: if evaluate_falsifiable_bio_criterion(&bad_arms, 0.05, 0.85, 0.25).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "giving bio_paired_adapter 2x training steps rejected".into(),
        });
    }

    // NC2: recording culture/session leakage between train and held-out rejected
    {
        let mut bad = split.clone();
        bad.heldout_cultures_sessions.push("culture_a_sess1".into());
        negative_controls.push(NegativeControl {
            case_id: "nc2_culture_session_leakage_rejected".into(),
            outcome: if validate_split_and_timing(&bad).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "shared culture/session between train and eval splits rejected".into(),
        });
    }

    // NC3: task-family leakage between train and held-out rejected
    {
        let mut bad = split.clone();
        bad.heldout_task_families.push("qa_train".into());
        negative_controls.push(NegativeControl {
            case_id: "nc3_task_family_leakage_rejected".into(),
            outcome: if validate_split_and_timing(&bad).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "shared task family across train and held-out splits rejected".into(),
        });
    }

    // NC4: reconstructed timing mislabeled as measured timing rejected
    {
        let mut bad = split.clone();
        bad.timing_provenance = TimingProvenance::ReconstructedInterpolated;
        bad.claimed_measured_timing = true;
        negative_controls.push(NegativeControl {
            case_id: "nc4_reconstructed_timing_mislabeled_measured".into(),
            outcome: if validate_split_and_timing(&bad).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail:
                "reconstructed/interpolated timing refused from claiming measured hardware timing"
                    .into(),
        });
    }

    // NC5: mismatched synthetic signal statistics rejected
    {
        let mut bad_m = synth_moments;
        bad_m.variance *= 4.0;
        negative_controls.push(NegativeControl {
            case_id: "nc5_unmatched_synthetic_moments_rejected".into(),
            outcome: if verify_synthetic_match(bio_moments, bad_m, 1.0e-9, 1.0e-9, 0.35).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "synthetic control failing variance match rejected".into(),
        });
    }

    // NC6: claiming biological advantage when matched synthetic/shuffled ties paired bio fails criterion
    {
        let (_v, st) = evaluate_falsifiable_bio_criterion(&arms, 0.05, 0.85, 0.25)?;
        negative_controls.push(NegativeControl {
            case_id: "nc6_tied_synthetic_control_falsifies_bio_claim".into(),
            outcome: st,
            detail: "paired bio tying matched synthetic & text LoRA fails falsifiable criterion"
                .into(),
        });
    }

    // NC7: non-finite probability in calibration check rejected
    {
        let res = compute_calibration_metrics(&[(f64::NAN, true)]);
        negative_controls.push(NegativeControl {
            case_id: "nc7_nonfinite_calibration_prob_rejected".into(),
            outcome: if res.is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "NaN predicted probability rejected cleanly in ECE/Brier reducer".into(),
        });
    }

    // NC8: assuming BioLLM-8B benchmarks exist locally rejected
    {
        let st = validate_external_benchmark_assumption(true, false);
        negative_controls.push(NegativeControl {
            case_id: "nc8_unverified_biollm8b_benchmark_assumption".into(),
            outcome: st,
            detail: "do not assume current BioLLM-8B benchmarks exist; stays blocked_on_artifacts"
                .into(),
        });
    }

    Ok(BioLlmProtocolReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::blocked_on_artifacts(
            "Paired CL1 biological recordings & BioLLM-8B adapter checkpoints",
            "Equal-budget 5-arm protocol, split/timing hygiene, moment matcher, calibration runner, and falsifiable criteria verified on CPU; physical biological recordings and BioLLM-8B artifacts are absent locally",
        ),
        synthetic_contract_fixture: true,
        bio_signal_moments: bio_moments,
        synth_signal_moments: synth_moments,
        arms,
        fixture_criterion_verdict: verdict.into(),
        assumes_biollm_8b_benchmarks_exist: false,
        negative_controls,
    })
}

pub fn inputs_json(report: &BioLlmProtocolReport) -> String {
    let digest = DigestRef::sha256(b"biollm-5arm-equal-budget-protocol-v1");
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.biollm-protocol.inputs.v1\",\n",
            "  \"issue\": 46,\n",
            "  \"execution_class\": {},\n",
            "  \"protocol_digest\": {},\n",
            "  \"assumes_biollm_8b_benchmarks_exist\": {},\n",
            "  \"synthetic_contract_fixture\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        digest.to_json(),
        report.assumes_biollm_8b_benchmarks_exist,
        report.synthetic_contract_fixture,
    )
}

pub fn results_json(report: &BioLlmProtocolReport) -> String {
    let arms: Vec<String> = report
        .arms
        .iter()
        .map(|a| {
            format!(
                "{{\"arm\":{},\"steps\":{},\"tokens\":{},\"trainable_params\":{},\"heldout_task_accuracy\":{},\"retention_accuracy\":{},\"ece\":{},\"brier_score\":{},\"latency_ms_synthetic\":{},\"compute_gflops_synthetic\":{}}}",
                quote(a.arm.as_str()),
                a.training_budget.steps,
                a.training_budget.tokens,
                a.training_budget.trainable_params,
                finite_f64(a.heldout_task_accuracy),
                finite_f64(a.retention_accuracy),
                finite_f64(a.ece),
                finite_f64(a.brier_score),
                finite_f64(a.inference_latency_ms_synthetic),
                finite_f64(a.compute_gflops_synthetic),
            )
        })
        .collect();
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
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.biollm-protocol.results.v1\",\n",
            "  \"issue\": 46,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"assumes_biollm_8b_benchmarks_exist\": {},\n",
            "  \"bio_moments\": {{\"dim\":{},\"mean\":{},\"variance\":{},\"lag1_autocorr\":{}}},\n",
            "  \"synth_moments\": {{\"dim\":{},\"mean\":{},\"variance\":{},\"lag1_autocorr\":{}}},\n",
            "  \"fixture_criterion_verdict\": {},\n",
            "  \"arms\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.synthetic_contract_fixture,
        report.assumes_biollm_8b_benchmarks_exist,
        report.bio_signal_moments.dim,
        finite_f64(report.bio_signal_moments.mean),
        finite_f64(report.bio_signal_moments.variance),
        finite_f64(report.bio_signal_moments.lag1_autocorr),
        report.synth_signal_moments.dim,
        finite_f64(report.synth_signal_moments.mean),
        finite_f64(report.synth_signal_moments.variance),
        finite_f64(report.synth_signal_moments.lag1_autocorr),
        quote(&report.fixture_criterion_verdict),
        arms.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &BioLlmProtocolReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #46: Biological-Recording Adapters vs Matched Synthetic & Shuffled Controls"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Execution class: `{}` (`synthetic_contract_fixture: {}`)",
        report.execution_class.as_str(),
        report.synthetic_contract_fixture
    )
    .unwrap();
    writeln!(
        out,
        "- Artifact gate: `{}` ({})",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(
        out,
        "- Assumes BioLLM-8B benchmarks exist: `{}`; Fixture criterion verdict: `{}`",
        report.assumes_biollm_8b_benchmarks_exist, report.fixture_criterion_verdict
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Equal-Budget 5-Arm Evaluation (Synthetic Contract Fixture)"
    )
    .unwrap();
    for a in &report.arms {
        writeln!(
            out,
            "- `{}` (`steps={}`): `acc={:.3}`, `ret={:.3}`, `ECE={:.4}`, `Brier={:.4}`, `lat={:.1}ms`",
            a.arm.as_str(),
            a.training_budget.steps,
            a.heldout_task_accuracy,
            a.retention_accuracy,
            a.ece,
            a.brier_score,
            a.inference_latency_ms_synthetic
        )
        .unwrap();
    }
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
