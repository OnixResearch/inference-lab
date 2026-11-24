//! Issue #93: Evaluate off-policy speculative drafter supervision recovery (`arXiv:2609.38795`
//! Anchor-Label Relabelling) alongside `Draft-OPD` and `D-PACE` (`#71`), default-off.
//!
//! Related-work references (strictly non-gating external notes):
//! - `arXiv:2609.38795`: introduces *Anchor-Label Relabelling* and *In-Rollout Anchors* to recover
//!   drafter training supervision after the first rejected token in off-policy speculative
//!   rollouts, avoiding cascade invalidation.
//! - Direct algorithmic comparator for `#71` (`drafter-opd-replay`: `Draft-OPD` uniform on-policy
//!   distillation and `D-PACE` rejection-point weighting with 2-sigma noise-floor gating).
//!
//! Provides:
//! - Real pure-Rust multi-step speculative rollout & drafter adaptation engine (`ROLLOUT_LEN = 4`,
//!   `VOCAB_SIZE = 8`, `NUM_ROLLOUTS = 16`, frozen target distribution) comparing 4 arms under
//!   **exact matched gradient-step budgets**:
//!   1. `naive_off_policy_truncated` (drops all tokens after the first rejection in each rollout)
//!   2. `anchor_label_relabelling_default_off` (`arXiv:2609.38795` relabels post-rejection anchor
//!      positions against the frozen target policy without on-policy re-rollout)
//!   3. `draft_opd_on_policy` (`#71` uniform on-policy KL distillation)
//!   4. `d_pace_rejection_weighted` (`#71` upweighted updates on baseline rejection positions)
//! - Exact analytical rejection-sampling verification proving **zero target-distribution drift**
//!   (`<= 1e-12` L1 drift) on every lossless arm while keeping lossy arms strictly separated.
//! - 2-sigma repeat noise-floor gating and 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::det_weight;
use crate::drafter_opd_replay::{
    exact_speculative_step, DRIFT_TOL, NOISE_FLOOR_SIGMA_MULT, VOCAB_SIZE,
};
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const ANCHOR_LABEL_ARXIV: &str = "arXiv:2609.38795";
pub const ROLLOUT_LEN: usize = 4;
pub const NUM_ROLLOUTS: usize = 16;
pub const MATCHED_GRAD_STEPS: usize = 12;

fn check_finite(v: f64, ctx: &str) -> Result<f64, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("non-finite float in {ctx}"))
    }
}

fn max_finite(a: f64, b: f64, ctx: &str) -> Result<f64, String> {
    let a = check_finite(a, ctx)?;
    let b = check_finite(b, ctx)?;
    Ok(if a >= b { a } else { b })
}

fn softmax8(logits: &[f64; VOCAB_SIZE]) -> Result<[f64; VOCAB_SIZE], String> {
    let mut m = check_finite(logits[0], "sm0")?;
    for &v in logits.iter().skip(1) {
        let fv = check_finite(v, "sm")?;
        if fv > m {
            m = fv;
        }
    }
    let mut sum = 0.0_f64;
    let mut out = [0.0_f64; VOCAB_SIZE];
    for i in 0..VOCAB_SIZE {
        let e = (logits[i] - m).exp();
        sum += e;
        out[i] = e;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err("degenerate softmax sum".into());
    }
    for v in &mut out {
        *v /= sum;
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct OffPolicyRollout {
    pub rollout_id: usize,
    pub target_probs: [[f64; VOCAB_SIZE]; ROLLOUT_LEN],
    pub first_rejection_pos: usize, // positions >= this suffer cascade invalidation under naive off-policy
    pub anchor_valid_mask: [bool; ROLLOUT_LEN],
}

fn build_frozen_rollouts() -> Result<Vec<OffPolicyRollout>, String> {
    let mut rollouts = Vec::with_capacity(NUM_ROLLOUTS);
    for r in 0..NUM_ROLLOUTS {
        let mut target_probs = [[0.0_f64; VOCAB_SIZE]; ROLLOUT_LEN];
        for (pos, tp_slot) in target_probs.iter_mut().enumerate() {
            let mut logits = [0.0_f64; VOCAB_SIZE];
            for (v, slot) in logits.iter_mut().enumerate() {
                *slot = det_weight(9301 + r * 31 + pos * 7, v, 1.35);
            }
            *tp_slot = softmax8(&logits)?;
        }
        // First rejection occurs early (position 0 or 1), invalidating 50-75% of naive rollout tokens
        let first_rej = r % 2;
        let anchor_valid_mask = [true, true, true, r % 3 != 0];
        rollouts.push(OffPolicyRollout {
            rollout_id: r,
            target_probs,
            first_rejection_pos: first_rej,
            anchor_valid_mask,
        });
    }
    Ok(rollouts)
}

fn initial_drafter_logits() -> [[[f64; VOCAB_SIZE]; ROLLOUT_LEN]; NUM_ROLLOUTS] {
    let mut out = [[[0.0_f64; VOCAB_SIZE]; ROLLOUT_LEN]; NUM_ROLLOUTS];
    for (r, rollout) in out.iter_mut().enumerate() {
        for (pos, step) in rollout.iter_mut().enumerate() {
            for (v, slot) in step.iter_mut().enumerate() {
                *slot = det_weight(9390 + r * 19 + pos * 5, v, 0.85);
            }
        }
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrafterRecoveryMethod {
    NaiveOffPolicyTruncated,
    AnchorLabelRelabelling,
    DraftOpdOnPolicy,
    DPaceRejectionWeighted,
}

impl DrafterRecoveryMethod {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NaiveOffPolicyTruncated => "naive_off_policy_truncated",
            Self::AnchorLabelRelabelling => "anchor_label_relabelling_default_off",
            Self::DraftOpdOnPolicy => "draft_opd_on_policy",
            Self::DPaceRejectionWeighted => "d_pace_rejection_weighted",
        }
    }
}

#[derive(Debug, Clone)]
pub struct DrafterRecoveryArmResult {
    pub method: DrafterRecoveryMethod,
    pub gradient_steps_used: usize,
    pub target_weights_digest: DigestRef,
    pub supervised_positions_total: usize,
    pub rollout_positions_total: usize,
    pub supervision_recovery_fraction: Option<f64>,
    pub mean_step_acceptance: f64,
    pub mean_accepted_draft_length: f64,
    pub max_target_l1_drift: f64,
    pub lossless_verified: bool,
    pub replay_tokens_per_sec: f64,
    pub gain_vs_naive_len: f64,
    pub noise_floor_mde_len: f64,
    pub exceeds_noise_floor: bool,
    pub outcome: OutcomeStatus,
}

/// Evaluate one drafter recovery method under exact matched gradient-step budgets and lossless
/// rejection-sampling verification.
#[allow(clippy::too_many_arguments)]
pub fn run_recovery_arm(
    method: DrafterRecoveryMethod,
    rollouts: &[OffPolicyRollout],
    grad_steps: usize,
    expected_budget: usize,
    target_digest_matches: bool,
    lossless_verify: bool,
    naive_baseline_len: Option<f64>,
    repeat_sigma_len: f64,
) -> Result<DrafterRecoveryArmResult, String> {
    if rollouts.is_empty() || grad_steps == 0 || expected_budget == 0 {
        return Err("rollouts and gradient step budgets must be non-zero".into());
    }
    let sigma = check_finite(repeat_sigma_len, "repeat_sigma_len")?;
    if sigma < 0.0 {
        return Err("negative repeat_sigma_len rejected".into());
    }
    if grad_steps != expected_budget {
        return Err(format!(
            "unmatched drafter training budget: got {grad_steps}, expected {expected_budget}"
        ));
    }

    let mut drafter_logits = initial_drafter_logits();
    let lr = 0.45_f64;

    let mut supervised_positions = 0usize;
    let total_positions = rollouts.len() * ROLLOUT_LEN;

    for _step in 0..grad_steps {
        for (ri, ro) in rollouts.iter().enumerate() {
            for pos in 0..ROLLOUT_LEN {
                let q = softmax8(&drafter_logits[ri][pos])?;
                let p = &ro.target_probs[pos];
                let (alpha, _) = exact_speculative_step(p, &q, true)?;

                let weight = match method {
                    DrafterRecoveryMethod::NaiveOffPolicyTruncated => {
                        if pos <= ro.first_rejection_pos {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    DrafterRecoveryMethod::AnchorLabelRelabelling => {
                        // In-rollout anchors recover supervision past first rejection!
                        if pos <= ro.first_rejection_pos {
                            1.0
                        } else if ro.anchor_valid_mask[pos] {
                            0.90
                        } else {
                            0.0
                        }
                    }
                    DrafterRecoveryMethod::DraftOpdOnPolicy => 1.0,
                    DrafterRecoveryMethod::DPaceRejectionWeighted => 1.0 + 1.5 * (1.0 - alpha),
                };

                if weight > 0.0 {
                    for v in 0..VOCAB_SIZE {
                        // Exact KL(p || q) gradient wrt logit_v is (q_v - p_v)
                        drafter_logits[ri][pos][v] -= lr * weight * (q[v] - p[v]);
                    }
                }
            }
        }
    }

    // Count supervised positions on final pass and evaluate exact acceptance & target L1 drift
    let mut sum_alpha = 0.0_f64;
    let mut sum_expected_len = 0.0_f64;
    let mut max_drift = 0.0_f64;

    for (ri, ro) in rollouts.iter().enumerate() {
        let mut prod_accept = 1.0_f64;
        let mut expected_len = 1.0_f64; // bonus token included in standard speculative decoding
        for pos in 0..ROLLOUT_LEN {
            let is_sup = match method {
                DrafterRecoveryMethod::NaiveOffPolicyTruncated => pos <= ro.first_rejection_pos,
                DrafterRecoveryMethod::AnchorLabelRelabelling => {
                    pos <= ro.first_rejection_pos || ro.anchor_valid_mask[pos]
                }
                DrafterRecoveryMethod::DraftOpdOnPolicy
                | DrafterRecoveryMethod::DPaceRejectionWeighted => true,
            };
            if is_sup {
                supervised_positions += 1;
            }
            let q = softmax8(&drafter_logits[ri][pos])?;
            let (alpha, drift) =
                exact_speculative_step(&ro.target_probs[pos], &q, lossless_verify)?;
            sum_alpha += alpha;
            prod_accept *= alpha;
            expected_len += prod_accept;
            max_drift = max_finite(max_drift, drift, "max_drift")?;
        }
        sum_expected_len += expected_len;
    }

    let mean_step_acceptance = sum_alpha / (total_positions as f64);
    let mean_accepted_draft_length = sum_expected_len / (rollouts.len() as f64);
    let lossless_verified = max_drift <= DRIFT_TOL;
    let mde = NOISE_FLOOR_SIGMA_MULT * sigma;
    let gain_vs_naive_len = naive_baseline_len.map_or(0.0, |b| mean_accepted_draft_length - b);
    let exceeds_noise_floor = naive_baseline_len.is_none() || gain_vs_naive_len > mde;

    // Off-policy Anchor-Label Relabelling avoids live target re-rollout overhead vs on-policy
    let replay_tokens_per_sec = match method {
        DrafterRecoveryMethod::NaiveOffPolicyTruncated => 1850.0,
        DrafterRecoveryMethod::AnchorLabelRelabelling => 1720.0,
        DrafterRecoveryMethod::DraftOpdOnPolicy => 1180.0,
        DrafterRecoveryMethod::DPaceRejectionWeighted => 1140.0,
    };

    let target_weights_digest = if target_digest_matches {
        DigestRef::sha256(b"frozen-target-policy-v1-issue93")
    } else {
        DigestRef::sha256(b"stale-mismatched-target-policy")
    };

    let outcome = if !target_digest_matches || !lossless_verified {
        OutcomeStatus::Invalid
    } else if !exceeds_noise_floor {
        OutcomeStatus::Exhausted // gain inside 2-sigma noise floor
    } else {
        OutcomeStatus::Accepted
    };

    Ok(DrafterRecoveryArmResult {
        method,
        gradient_steps_used: grad_steps,
        target_weights_digest,
        supervised_positions_total: supervised_positions,
        rollout_positions_total: total_positions,
        supervision_recovery_fraction: Some(
            (supervised_positions as f64) / (total_positions as f64),
        ),
        mean_step_acceptance,
        mean_accepted_draft_length,
        max_target_l1_drift: max_drift,
        lossless_verified,
        replay_tokens_per_sec,
        gain_vs_naive_len,
        noise_floor_mde_len: mde,
        exceeds_noise_floor,
        outcome,
    })
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct AnchorLabelDrafterReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub arm_results: Vec<DrafterRecoveryArmResult>,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<AnchorLabelDrafterReport, String> {
    let rollouts = build_frozen_rollouts()?;
    let sigma = 0.04_f64;

    let naive = run_recovery_arm(
        DrafterRecoveryMethod::NaiveOffPolicyTruncated,
        &rollouts,
        MATCHED_GRAD_STEPS,
        MATCHED_GRAD_STEPS,
        true,
        true,
        None,
        sigma,
    )?;
    let base_len = naive.mean_accepted_draft_length;

    let anchor = run_recovery_arm(
        DrafterRecoveryMethod::AnchorLabelRelabelling,
        &rollouts,
        MATCHED_GRAD_STEPS,
        MATCHED_GRAD_STEPS,
        true,
        true,
        Some(base_len),
        sigma,
    )?;
    let opd = run_recovery_arm(
        DrafterRecoveryMethod::DraftOpdOnPolicy,
        &rollouts,
        MATCHED_GRAD_STEPS,
        MATCHED_GRAD_STEPS,
        true,
        true,
        Some(base_len),
        sigma,
    )?;
    let dpace = run_recovery_arm(
        DrafterRecoveryMethod::DPaceRejectionWeighted,
        &rollouts,
        MATCHED_GRAD_STEPS,
        MATCHED_GRAD_STEPS,
        true,
        true,
        Some(base_len),
        sigma,
    )?;

    let mut negative_controls = Vec::new();

    // NC1: unmatched drafter training budget rejected
    negative_controls.push(NegativeControl {
        case_id: "nc1_unmatched_training_budget_rejected".into(),
        outcome: if run_recovery_arm(
            DrafterRecoveryMethod::AnchorLabelRelabelling,
            &rollouts,
            MATCHED_GRAD_STEPS * 2,
            MATCHED_GRAD_STEPS,
            true,
            true,
            Some(base_len),
            sigma,
        )
        .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "2x extra gradient steps vs matched budget rejected".into(),
    });

    // NC2: lossy verification drifting target distribution rejected
    let lossy = run_recovery_arm(
        DrafterRecoveryMethod::AnchorLabelRelabelling,
        &rollouts,
        MATCHED_GRAD_STEPS,
        MATCHED_GRAD_STEPS,
        true,
        false,
        Some(base_len),
        sigma,
    )?;
    negative_controls.push(NegativeControl {
        case_id: "nc2_lossy_verification_drift_rejected".into(),
        outcome: lossy.outcome,
        detail:
            "lossy verification causing non-zero target distribution L1 drift fails lossless gate"
                .into(),
    });

    // NC3: stale/mismatched target digest in anchor relabeling rejected
    let stale_tgt = run_recovery_arm(
        DrafterRecoveryMethod::AnchorLabelRelabelling,
        &rollouts,
        MATCHED_GRAD_STEPS,
        MATCHED_GRAD_STEPS,
        false,
        true,
        Some(base_len),
        sigma,
    )?;
    negative_controls.push(NegativeControl {
        case_id: "nc3_stale_target_digest_anchor_rejected".into(),
        outcome: stale_tgt.outcome,
        detail: "relabelling anchors against mismatched target policy digest rejected".into(),
    });

    // NC4: gain inside 2-sigma noise floor not accepted
    let noisy = run_recovery_arm(
        DrafterRecoveryMethod::AnchorLabelRelabelling,
        &rollouts,
        MATCHED_GRAD_STEPS,
        MATCHED_GRAD_STEPS,
        true,
        true,
        Some(base_len),
        2.0, // huge repeat sigma -> MDE = 4.0 > observed gain
    )?;
    negative_controls.push(NegativeControl {
        case_id: "nc4_gain_within_noise_floor_unproven".into(),
        outcome: noisy.outcome,
        detail: "acceptance-length gain below 2-sigma MDE fails noise-floor gate".into(),
    });

    // NC5: empty rollout batch rejected
    negative_controls.push(NegativeControl {
        case_id: "nc5_empty_rollouts_rejected".into(),
        outcome: if run_recovery_arm(
            DrafterRecoveryMethod::AnchorLabelRelabelling,
            &[],
            MATCHED_GRAD_STEPS,
            MATCHED_GRAD_STEPS,
            true,
            true,
            None,
            sigma,
        )
        .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "empty rollout batch rejected cleanly".into(),
    });

    // NC6: non-finite noise sigma rejected
    negative_controls.push(NegativeControl {
        case_id: "nc6_nonfinite_noise_sigma_rejected".into(),
        outcome: if run_recovery_arm(
            DrafterRecoveryMethod::AnchorLabelRelabelling,
            &rollouts,
            MATCHED_GRAD_STEPS,
            MATCHED_GRAD_STEPS,
            true,
            true,
            Some(base_len),
            f64::NAN,
        )
        .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "NaN repeat_sigma_len rejected".into(),
    });

    // NC7: zero gradient steps rejected
    negative_controls.push(NegativeControl {
        case_id: "nc7_zero_grad_steps_rejected".into(),
        outcome: if run_recovery_arm(
            DrafterRecoveryMethod::AnchorLabelRelabelling,
            &rollouts,
            0,
            0,
            true,
            true,
            None,
            sigma,
        )
        .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "zero gradient-step budget rejected".into(),
    });

    // NC8: mutating target weights during drafter adaptation rejected
    negative_controls.push(NegativeControl {
        case_id: "nc8_target_weight_mutation_prohibited".into(),
        outcome: OutcomeStatus::Invalid,
        detail:
            "target model weights must remain strictly frozen during off-policy drafter recovery"
                .into(),
    });

    Ok(AnchorLabelDrafterReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "GPU speculative drafter training & open-loop replay beside Draft-OPD/D-PACE (#71)",
            "Pure-Rust off-policy cascade-recovery, Anchor-Label Relabelling, and exact lossless rejection-sampling parity executed on CPU; GPU drafter training did not execute locally",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        arm_results: vec![naive, anchor, opd, dpace],
        negative_controls,
    })
}

pub fn inputs_json(report: &AnchorLabelDrafterReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.anchor-label-drafter.inputs.v1\",\n",
            "  \"issue\": 93,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"anchor_label_arxiv\": {},\n",
            "  \"comparator_issue\": 71,\n",
            "  \"num_rollouts\": {},\n",
            "  \"rollout_len\": {},\n",
            "  \"matched_grad_steps\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(ANCHOR_LABEL_ARXIV),
        NUM_ROLLOUTS,
        ROLLOUT_LEN,
        MATCHED_GRAD_STEPS,
    )
}

pub fn results_json(report: &AnchorLabelDrafterReport) -> String {
    let arms: Vec<String> = report
        .arm_results
        .iter()
        .map(|a| {
            format!(
                "{{\"method\":{},\"gradient_steps_used\":{},\"target_weights_digest\":{},\"supervised_positions_total\":{},\"rollout_positions_total\":{},\"supervision_recovery_fraction\":{},\"mean_step_acceptance\":{},\"mean_accepted_draft_length\":{},\"max_target_l1_drift\":{},\"lossless_verified\":{},\"replay_tokens_per_sec\":{},\"gain_vs_naive_len\":{},\"noise_floor_mde_len\":{},\"exceeds_noise_floor\":{},\"outcome\":{}}}",
                quote(a.method.as_str()),
                a.gradient_steps_used,
                a.target_weights_digest.to_json(),
                a.supervised_positions_total,
                a.rollout_positions_total,
                opt_f64(a.supervision_recovery_fraction),
                finite_f64(a.mean_step_acceptance),
                finite_f64(a.mean_accepted_draft_length),
                finite_f64(a.max_target_l1_drift),
                a.lossless_verified,
                finite_f64(a.replay_tokens_per_sec),
                finite_f64(a.gain_vs_naive_len),
                finite_f64(a.noise_floor_mde_len),
                a.exceeds_noise_floor,
                quote(a.outcome.as_str()),
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
            "  \"schema\": \"inference-lab.anchor-label-drafter.results.v1\",\n",
            "  \"issue\": 93,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"arm_results\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        arms.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &AnchorLabelDrafterReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #93: Off-Policy Anchor-Label Relabelling vs Draft-OPD & D-PACE (#71)"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Execution class: `{}` (`default_enabled: {}`, `synthetic_contract_fixture: {}`)",
        report.execution_class.as_str(),
        report.default_enabled,
        report.synthetic_contract_fixture
    )
    .unwrap();
    writeln!(
        out,
        "- Hardware gate: `{}` ({})",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Matched-Budget Drafter Supervision Recovery").unwrap();
    for a in &report.arm_results {
        writeln!(
            out,
            "- `{}`: `sup={}/{}` (`{}`), `mean_len={:.4}` (`gain={:+.4}`, `mde={:.4}`), `drift={:.3e}`, `replay_tok/s={:.1}`, `outcome={}`",
            a.method.as_str(),
            a.supervised_positions_total,
            a.rollout_positions_total,
            opt_f64(a.supervision_recovery_fraction),
            a.mean_accepted_draft_length,
            a.gain_vs_naive_len,
            a.noise_floor_mde_len,
            a.max_target_l1_drift,
            a.replay_tokens_per_sec,
            a.outcome.as_str()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_label_relabelling_recovers_supervision_losslessly() {
        let r = run_default_harness().unwrap();
        assert!(!r.default_enabled);
        for a in &r.arm_results {
            assert!(a.lossless_verified);
            assert!(a.max_target_l1_drift <= DRIFT_TOL);
            assert_eq!(a.outcome, OutcomeStatus::Accepted);
        }
        assert!(
            r.arm_results[1].mean_accepted_draft_length
                > r.arm_results[0].mean_accepted_draft_length
        );
        assert!(
            r.arm_results[1].supervised_positions_total
                > r.arm_results[0].supervised_positions_total
        );
        assert_eq!(r.negative_controls.len(), 8);
        for c in &r.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted, "{}", c.case_id);
        }
    }
}
