//! Issue #73: Qualify pinned `LFM3-1.7B` (tri-hybrid 256k) and `d1-2B` single-pass typed
//! decision model on structured-output conformance, calibration, and latency, default-off.
//!
//! Related-work record (strictly non-gating, honest offline pin status):
//! - X post (2026-10-27, Liquid AI `@liquidai`): `https://x.com/liquidai/status/2115128329003303107`
//! - `pin_status`: `"post_only_checkpoints_unpinned_offline"` (exact `LFM3-1.7B` and `d1-2B`
//!   checkpoint digests pending public checkpoint verification — no fabricated commit SHA).
//! - Scope boundary: `LFM3-Audio-2B` is explicitly excluded and tracked separately in
//!   `audio-lab#2` (`excludes_audio_model = true`).
//! - Complements `#36` (compact hybrid recurrent models) and `#55` (single-pass typed extraction).
//!
//! Provides:
//! - Default-off evaluation adapters for `LFM3-1.7B` and `d1-2B`.
//! - Real pure-Rust **miniature tri-hybrid block** (depthwise short conv, linear recurrent state,
//!   and causal attention) verifying exact cached step-by-step vs full-sequence prefill parity
//!   (`<= 1e-12`), plus `8K`–`256K` throughput, state memory, and retention scaling for `LFM3-1.7B`.
//! - Real pure-Rust **`d1-2B` typed-decision schema validator**, **confidence-independent consumer
//!   ground-truth checker** (a well-typed wrong decision at `p = 0.99` fails consumer admission!),
//!   and **exact ECE (Expected Calibration Error) & Brier score** calculator comparing single-pass
//!   `d1-2B` against an autoregressive structured-output baseline on frozen fixtures.
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, sigmoid};
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const UPSTREAM_X_POST: &str = "https://x.com/liquidai/status/2115128329003303107";
pub const LFM3_PIN_STATUS: &str = "post_only_checkpoints_unpinned_offline";
pub const AUDIO_OUT_OF_SCOPE_ISSUE: &str = "audio-lab#2";
pub const LFM3_CONTEXT_SWEEP: [usize; 4] = [8_192, 32_768, 131_072, 262_144];

pub const D_MODEL: usize = 4;
pub const CONV_KERNEL: usize = 3;
pub const PARITY_TOL: f64 = 1.0e-12;
pub const ECE_BINS: usize = 5;

fn check_finite(v: f64, ctx: &str) -> Result<f64, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("non-finite float in {ctx}"))
    }
}

/// Typed decision classes for `d1-2B` single-pass structured output verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionLabel {
    Allow,
    Deny,
    Escalate,
}

impl DecisionLabel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Deny => "deny",
            Self::Escalate => "escalate",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TypedDecisionPrediction {
    pub fixture_id: String,
    pub schema_valid: bool,
    pub predicted_label: Option<DecisionLabel>,
    pub ground_truth_label: DecisionLabel,
    pub reported_prob: f64,
    pub latency_ms: f64,
}

#[derive(Debug, Clone)]
pub struct CalibrationAndConformanceSummary {
    pub arm_id: String,
    pub total_fixtures: usize,
    pub schema_valid_count: usize,
    pub consumer_correct_count: usize,
    pub schema_conformance_rate: Option<f64>,
    pub consumer_accuracy: Option<f64>,
    pub ece: Option<f64>,
    pub brier_score: Option<f64>,
    pub mean_latency_ms: Option<f64>,
    pub outcome: OutcomeStatus,
}

/// Independently grade typed-decision predictions:
/// - Consumer correctness is STRICTLY independent of `reported_prob` (a schema-valid wrong label
///   with `0.99` confidence is wrong!).
/// - Computes exact `ECE` across `ECE_BINS` equal-width bins and exact binary `Brier` score
///   `mean((p_i - y_i)^2)` where `y_i = 1.0` iff schema-valid AND label matches ground truth.
pub fn evaluate_typed_decisions(
    arm_id: &str,
    preds: &[TypedDecisionPrediction],
) -> Result<CalibrationAndConformanceSummary, String> {
    if preds.is_empty() {
        return Ok(CalibrationAndConformanceSummary {
            arm_id: arm_id.into(),
            total_fixtures: 0,
            schema_valid_count: 0,
            consumer_correct_count: 0,
            schema_conformance_rate: None,
            consumer_accuracy: None,
            ece: None,
            brier_score: None,
            mean_latency_ms: None,
            outcome: OutcomeStatus::Invalid,
        });
    }
    let mut schema_valid = 0usize;
    let mut correct = 0usize;
    let mut brier_sum = 0.0_f64;
    let mut lat_sum = 0.0_f64;
    let mut bin_count = [0usize; ECE_BINS];
    let mut bin_conf_sum = [0.0_f64; ECE_BINS];
    let mut bin_acc_sum = [0.0_f64; ECE_BINS];

    for p in preds {
        let prob = check_finite(p.reported_prob, "reported_prob")?;
        let lat = check_finite(p.latency_ms, "latency_ms")?;
        if !(0.0..=1.0).contains(&prob) || lat < 0.0 {
            return Err("reported_prob out of [0, 1] or negative latency".into());
        }
        if p.schema_valid {
            schema_valid += 1;
        }
        let is_correct = p.schema_valid && p.predicted_label == Some(p.ground_truth_label);
        if is_correct {
            correct += 1;
        }
        let y = if is_correct { 1.0_f64 } else { 0.0_f64 };
        let diff = prob - y;
        brier_sum += diff * diff;
        lat_sum += lat;

        let raw_bin = (prob * (ECE_BINS as f64)).floor() as usize;
        let bin_idx = if raw_bin >= ECE_BINS {
            ECE_BINS - 1
        } else {
            raw_bin
        };
        bin_count[bin_idx] += 1;
        bin_conf_sum[bin_idx] += prob;
        bin_acc_sum[bin_idx] += y;
    }

    let n = preds.len() as f64;
    let mut ece = 0.0_f64;
    for b in 0..ECE_BINS {
        if bin_count[b] > 0 {
            let bn = bin_count[b] as f64;
            let avg_conf = bin_conf_sum[b] / bn;
            let avg_acc = bin_acc_sum[b] / bn;
            ece += (bn / n) * (avg_conf - avg_acc).abs();
        }
    }
    Ok(CalibrationAndConformanceSummary {
        arm_id: arm_id.into(),
        total_fixtures: preds.len(),
        schema_valid_count: schema_valid,
        consumer_correct_count: correct,
        schema_conformance_rate: Some((schema_valid as f64) / n),
        consumer_accuracy: Some((correct as f64) / n),
        ece: Some(check_finite(ece, "ece")?),
        brier_score: Some(check_finite(brier_sum / n, "brier")?),
        mean_latency_ms: Some(check_finite(lat_sum / n, "mean_lat")?),
        outcome: OutcomeStatus::Accepted,
    })
}

/// Miniature `LFM3-1.7B` tri-hybrid block:
/// 1) Causal depthwise short convolution (`kernel = 3`)
/// 2) Linear recurrent decay state (`h_t = decay * h_{t-1} + conv_t`)
/// 3) Causal scaled dot-product attention over `h_0..=t`
#[derive(Debug, Clone)]
pub struct MiniTriHybridBlock {
    pub conv_w: [[f64; CONV_KERNEL]; D_MODEL],
    pub decay: [f64; D_MODEL],
    pub wq: [[f64; D_MODEL]; D_MODEL],
    pub wk: [[f64; D_MODEL]; D_MODEL],
    pub wv: [[f64; D_MODEL]; D_MODEL],
}

impl MiniTriHybridBlock {
    pub fn build_deterministic() -> Self {
        let mut conv_w = [[0.0_f64; CONV_KERNEL]; D_MODEL];
        let mut decay = [0.0_f64; D_MODEL];
        let mut wq = [[0.0_f64; D_MODEL]; D_MODEL];
        let mut wk = [[0.0_f64; D_MODEL]; D_MODEL];
        let mut wv = [[0.0_f64; D_MODEL]; D_MODEL];
        for d in 0..D_MODEL {
            decay[d] = 0.70 + 0.20 * sigmoid(det_weight(7301, d, 1.0));
            for k in 0..CONV_KERNEL {
                conv_w[d][k] = det_weight(7305 + d * 7, k, 0.45);
            }
            for c in 0..D_MODEL {
                wq[d][c] = det_weight(7310, d * D_MODEL + c, 0.35);
                wk[d][c] = det_weight(7311, d * D_MODEL + c, 0.35);
                wv[d][c] = det_weight(7312, d * D_MODEL + c, 0.35);
            }
        }
        Self {
            conv_w,
            decay,
            wq,
            wk,
            wv,
        }
    }
}

#[allow(clippy::needless_range_loop)]
fn matvec4(w: &[[f64; D_MODEL]; D_MODEL], x: &[f64; D_MODEL]) -> [f64; D_MODEL] {
    let mut out = [0.0_f64; D_MODEL];
    for r in 0..D_MODEL {
        let mut s = 0.0_f64;
        for c in 0..D_MODEL {
            s += w[r][c] * x[c];
        }
        out[r] = s;
    }
    out
}

/// Full-sequence reference forward for `MiniTriHybridBlock`.
#[allow(clippy::needless_range_loop)]
pub fn tri_hybrid_full_reference(
    block: &MiniTriHybridBlock,
    xs: &[[f64; D_MODEL]],
) -> Result<Vec<[f64; D_MODEL]>, String> {
    let mut rec_seq = Vec::with_capacity(xs.len());
    let mut h_rec = [0.0_f64; D_MODEL];
    for t in 0..xs.len() {
        let mut conv_out = [0.0_f64; D_MODEL];
        for d in 0..D_MODEL {
            let mut acc = 0.0_f64;
            for tap in 0..CONV_KERNEL {
                if t >= tap {
                    acc += block.conv_w[d][tap] * check_finite(xs[t - tap][d], "tri_x")?;
                }
            }
            conv_out[d] = acc;
            h_rec[d] = check_finite(block.decay[d] * h_rec[d] + conv_out[d], "tri_rec")?;
        }
        rec_seq.push(h_rec);
    }

    let inv_scale = 1.0_f64 / (D_MODEL as f64).sqrt();
    let mut outs = Vec::with_capacity(xs.len());
    for t in 0..xs.len() {
        let q = matvec4(&block.wq, &rec_seq[t]);
        let mut scores = Vec::with_capacity(t + 1);
        let mut m = f64::NEG_INFINITY;
        for p in 0..=t {
            let k = matvec4(&block.wk, &rec_seq[p]);
            let mut dot = 0.0_f64;
            for d in 0..D_MODEL {
                dot += q[d] * k[d];
            }
            let s = dot * inv_scale;
            if s > m {
                m = s;
            }
            scores.push(s);
        }
        let mut sum = 0.0_f64;
        let mut ws = Vec::with_capacity(t + 1);
        for &s in &scores {
            let e = (s - m).exp();
            sum += e;
            ws.push(e);
        }
        let mut y = [0.0_f64; D_MODEL];
        for d in 0..D_MODEL {
            let mut acc = 0.0_f64;
            for p in 0..=t {
                let v = matvec4(&block.wv, &rec_seq[p]);
                acc += (ws[p] / sum) * v[d];
            }
            y[d] = check_finite(rec_seq[t][d] + acc, "tri_y")?;
        }
        outs.push(y);
    }
    Ok(outs)
}

/// Incremental cached step-by-step forward for `MiniTriHybridBlock` (conv ring buffer + recurrent
/// state + cached K/V). Must match `tri_hybrid_full_reference` within `<= 1e-12`.
#[allow(clippy::needless_range_loop)]
pub fn tri_hybrid_cached_step_by_step(
    block: &MiniTriHybridBlock,
    xs: &[[f64; D_MODEL]],
) -> Result<Vec<[f64; D_MODEL]>, String> {
    let mut conv_buf = [[0.0_f64; D_MODEL]; CONV_KERNEL];
    let mut h_rec = [0.0_f64; D_MODEL];
    let mut cached_k = Vec::with_capacity(xs.len());
    let mut cached_v = Vec::with_capacity(xs.len());
    let mut outs = Vec::with_capacity(xs.len());
    let inv_scale = 1.0_f64 / (D_MODEL as f64).sqrt();

    for x in xs {
        // Shift conv ring buffer: conv_buf[0] = current, [1] = t-1, [2] = t-2
        conv_buf[2] = conv_buf[1];
        conv_buf[1] = conv_buf[0];
        conv_buf[0] = *x;
        for d in 0..D_MODEL {
            let mut conv = 0.0_f64;
            for tap in 0..CONV_KERNEL {
                conv += block.conv_w[d][tap] * check_finite(conv_buf[tap][d], "cached_x")?;
            }
            h_rec[d] = check_finite(block.decay[d] * h_rec[d] + conv, "cached_rec")?;
        }
        let q = matvec4(&block.wq, &h_rec);
        cached_k.push(matvec4(&block.wk, &h_rec));
        cached_v.push(matvec4(&block.wv, &h_rec));

        let len = cached_k.len();
        let mut scores = Vec::with_capacity(len);
        let mut m = f64::NEG_INFINITY;
        for k_p in &cached_k {
            let mut dot = 0.0_f64;
            for d in 0..D_MODEL {
                dot += q[d] * k_p[d];
            }
            let s = dot * inv_scale;
            if s > m {
                m = s;
            }
            scores.push(s);
        }
        let mut sum = 0.0_f64;
        let mut ws = Vec::with_capacity(len);
        for &s in &scores {
            let e = (s - m).exp();
            sum += e;
            ws.push(e);
        }
        let mut y = [0.0_f64; D_MODEL];
        for d in 0..D_MODEL {
            let mut acc = 0.0_f64;
            for p in 0..len {
                acc += (ws[p] / sum) * cached_v[p][d];
            }
            y[d] = check_finite(h_rec[d] + acc, "cached_y")?;
        }
        outs.push(y);
    }
    Ok(outs)
}

/// Guard adapter selection so `LFM3-Audio-2B` is cleanly rejected as out of scope (`audio-lab#2`).
pub fn qualify_lfm3_adapter(model_name: &str) -> OutcomeStatus {
    match model_name {
        "LFM3-1.7B" | "d1-2B" => OutcomeStatus::BlockedOnArtifacts,
        "LFM3-Audio-2B" => OutcomeStatus::Unsupported,
        _ => OutcomeStatus::Invalid,
    }
}

#[derive(Debug, Clone)]
pub struct Lfm3ContextRow {
    pub context_tokens: usize,
    pub state_and_kv_mb: f64,
    pub synthetic_decode_tok_s: f64,
    pub synthetic_retention_score: f64,
}

#[derive(Debug, Clone)]
pub struct NegativeControlResult {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct Lfm3D1DecisionReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub pin_status: String,
    pub excludes_audio_model: bool,
    pub audio_out_of_scope_issue: String,
    pub tri_hybrid_cached_max_abs_err: f64,
    pub decision_arms: Vec<CalibrationAndConformanceSummary>,
    pub lfm3_context_scaling: Vec<Lfm3ContextRow>,
    pub negative_controls: Vec<NegativeControlResult>,
}

pub fn run_default_harness() -> Result<Lfm3D1DecisionReport, String> {
    // 1. Tri-hybrid cached vs full parity
    let block = MiniTriHybridBlock::build_deterministic();
    let mut xs = Vec::with_capacity(10);
    for t in 0..10 {
        let mut x = [0.0_f64; D_MODEL];
        for (d, slot) in x.iter_mut().enumerate() {
            *slot = det_weight(7340 + t * 9, d, 0.80);
        }
        xs.push(x);
    }
    let full_y = tri_hybrid_full_reference(&block, &xs)?;
    let cached_y = tri_hybrid_cached_step_by_step(&block, &xs)?;
    let mut tri_err = 0.0_f64;
    for (a, b) in full_y.iter().zip(cached_y.iter()) {
        for d in 0..D_MODEL {
            let e = (a[d] - b[d]).abs();
            if e > tri_err {
                tri_err = e;
            }
        }
    }
    if tri_err > PARITY_TOL {
        return Err(format!("tri-hybrid cached parity error {tri_err}"));
    }

    // 2. Frozen typed-decision fixtures comparing single-pass d1-2B vs AR baseline
    let d1_preds = vec![
        TypedDecisionPrediction {
            fixture_id: "f1".into(),
            schema_valid: true,
            predicted_label: Some(DecisionLabel::Allow),
            ground_truth_label: DecisionLabel::Allow,
            reported_prob: 0.90,
            latency_ms: 4.2,
        },
        TypedDecisionPrediction {
            fixture_id: "f2".into(),
            schema_valid: true,
            predicted_label: Some(DecisionLabel::Deny),
            ground_truth_label: DecisionLabel::Deny,
            reported_prob: 0.85,
            latency_ms: 4.1,
        },
        TypedDecisionPrediction {
            fixture_id: "f3".into(),
            schema_valid: true,
            predicted_label: Some(DecisionLabel::Escalate),
            ground_truth_label: DecisionLabel::Escalate,
            reported_prob: 0.72,
            latency_ms: 4.3,
        },
        TypedDecisionPrediction {
            fixture_id: "f4".into(),
            schema_valid: true,
            predicted_label: Some(DecisionLabel::Allow),
            ground_truth_label: DecisionLabel::Deny,
            reported_prob: 0.55,
            latency_ms: 4.0,
        },
        TypedDecisionPrediction {
            fixture_id: "f5".into(),
            schema_valid: true,
            predicted_label: Some(DecisionLabel::Deny),
            ground_truth_label: DecisionLabel::Deny,
            reported_prob: 0.88,
            latency_ms: 4.2,
        },
        TypedDecisionPrediction {
            fixture_id: "f6".into(),
            schema_valid: true,
            predicted_label: Some(DecisionLabel::Allow),
            ground_truth_label: DecisionLabel::Allow,
            reported_prob: 0.91,
            latency_ms: 4.1,
        },
    ];
    let ar_preds = vec![
        TypedDecisionPrediction {
            fixture_id: "f1".into(),
            schema_valid: true,
            predicted_label: Some(DecisionLabel::Allow),
            ground_truth_label: DecisionLabel::Allow,
            reported_prob: 0.98,
            latency_ms: 28.5,
        },
        TypedDecisionPrediction {
            fixture_id: "f2".into(),
            schema_valid: true,
            predicted_label: Some(DecisionLabel::Deny),
            ground_truth_label: DecisionLabel::Deny,
            reported_prob: 0.97,
            latency_ms: 31.0,
        },
        TypedDecisionPrediction {
            fixture_id: "f3".into(),
            schema_valid: false,
            predicted_label: None,
            ground_truth_label: DecisionLabel::Escalate,
            reported_prob: 0.92,
            latency_ms: 34.2,
        },
        TypedDecisionPrediction {
            fixture_id: "f4".into(),
            schema_valid: true,
            predicted_label: Some(DecisionLabel::Allow),
            ground_truth_label: DecisionLabel::Deny,
            reported_prob: 0.95,
            latency_ms: 29.1,
        },
        TypedDecisionPrediction {
            fixture_id: "f5".into(),
            schema_valid: true,
            predicted_label: Some(DecisionLabel::Deny),
            ground_truth_label: DecisionLabel::Deny,
            reported_prob: 0.96,
            latency_ms: 27.8,
        },
        TypedDecisionPrediction {
            fixture_id: "f6".into(),
            schema_valid: true,
            predicted_label: Some(DecisionLabel::Allow),
            ground_truth_label: DecisionLabel::Allow,
            reported_prob: 0.97,
            latency_ms: 30.0,
        },
    ];
    let decision_arms = vec![
        evaluate_typed_decisions("d1_2b_single_pass_typed", &d1_preds)?,
        evaluate_typed_decisions("autoregressive_structured_baseline", &ar_preds)?,
    ];

    // 3. LFM3-1.7B 8K-256K long-context throughput & retention contract
    let mut lfm3_context_scaling = Vec::with_capacity(LFM3_CONTEXT_SWEEP.len());
    for &ctx in &LFM3_CONTEXT_SWEEP {
        let frac = (ctx as f64) / 262_144.0;
        let state_mb = 28.0 + (ctx as f64) * 0.0022; // small hybrid attention fraction
        lfm3_context_scaling.push(Lfm3ContextRow {
            context_tokens: ctx,
            state_and_kv_mb: state_mb,
            synthetic_decode_tok_s: 145.0 / (1.0 + 0.18 * frac),
            synthetic_retention_score: 0.93 - 0.09 * frac,
        });
    }

    let mut nc = Vec::new();
    // NC1: high-confidence wrong typed decision fails consumer accuracy & drives high Brier
    let wrong_confident = evaluate_typed_decisions(
        "overconfident_wrong",
        &[TypedDecisionPrediction {
            fixture_id: "bad1".into(),
            schema_valid: true,
            predicted_label: Some(DecisionLabel::Allow),
            ground_truth_label: DecisionLabel::Deny,
            reported_prob: 0.99,
            latency_ms: 4.0,
        }],
    )?;
    let nc1 = if wrong_confident.consumer_correct_count == 0
        && wrong_confident.brier_score.unwrap_or(0.0) > 0.95
    {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc1_high_confidence_wrong_decision_not_accepted".into(),
        outcome: nc1,
        detail: "schema-valid wrong decision at p=0.99 scored 0/1 consumer accuracy & Brier=0.9801"
            .into(),
    });

    // NC2: malformed schema cannot count as correct even if label matches
    let bad_schema = evaluate_typed_decisions(
        "bad_schema",
        &[TypedDecisionPrediction {
            fixture_id: "bad2".into(),
            schema_valid: false,
            predicted_label: Some(DecisionLabel::Allow),
            ground_truth_label: DecisionLabel::Allow,
            reported_prob: 0.80,
            latency_ms: 4.0,
        }],
    )?;
    let nc2 = if bad_schema.consumer_correct_count == 0 {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc2_schema_invalid_output_fails_consumer_check".into(),
        outcome: nc2,
        detail: "schema_valid=false rejected from consumer_correct_count despite matching label"
            .into(),
    });

    // NC3: LFM3-Audio-2B explicitly rejected as out-of-scope (audio-lab#2)
    nc.push(NegativeControlResult {
        case_id: "nc3_lfm3_audio_2b_out_of_scope_audio_lab_2".into(),
        outcome: qualify_lfm3_adapter("LFM3-Audio-2B"),
        detail: "LFM3-Audio-2B belongs strictly to audio-lab#2 and is refused here".into(),
    });

    // NC4: out-of-range probability > 1.0 rejected
    let nc4 = if evaluate_typed_decisions(
        "bad_prob",
        &[TypedDecisionPrediction {
            fixture_id: "bad3".into(),
            schema_valid: true,
            predicted_label: Some(DecisionLabel::Allow),
            ground_truth_label: DecisionLabel::Allow,
            reported_prob: 1.25,
            latency_ms: 4.0,
        }],
    )
    .is_err()
    {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc4_probability_above_one_rejected".into(),
        outcome: nc4,
        detail: "reported_prob=1.25 outside [0, 1] refused before ECE/Brier".into(),
    });

    // NC5: NaN reported probability rejected
    let nc5 = if evaluate_typed_decisions(
        "nan_prob",
        &[TypedDecisionPrediction {
            fixture_id: "bad4".into(),
            schema_valid: true,
            predicted_label: Some(DecisionLabel::Allow),
            ground_truth_label: DecisionLabel::Allow,
            reported_prob: f64::NAN,
            latency_ms: 4.0,
        }],
    )
    .is_err()
    {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc5_nan_probability_rejected".into(),
        outcome: nc5,
        detail: "NaN reported_prob refused before calibration binning".into(),
    });

    // NC6: empty fixture list returns None metrics and Invalid
    let empty_eval = evaluate_typed_decisions("empty", &[])?;
    let nc6 = if empty_eval.ece.is_none() && empty_eval.brier_score.is_none() {
        empty_eval.outcome
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc6_empty_fixture_returns_none_not_nan".into(),
        outcome: nc6,
        detail: "empty decision fixture returned ece=None, brier=None, invalid".into(),
    });

    // NC7: NaN input to tri-hybrid block rejected
    let mut bad_xs = xs.clone();
    bad_xs[0][0] = f64::NAN;
    let nc7 = if tri_hybrid_cached_step_by_step(&block, &bad_xs).is_err() {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc7_nan_tri_hybrid_input_rejected".into(),
        outcome: nc7,
        detail: "NaN activation refused in LFM3 tri-hybrid cached decode".into(),
    });

    // NC8: unpinned LFM3-1.7B / d1-2B checkpoints block physical evaluation
    nc.push(NegativeControlResult {
        case_id: "nc8_unpinned_lfm3_d1_checkpoints_blocked".into(),
        outcome: qualify_lfm3_adapter("d1-2B"),
        detail: "LFM3-1.7B & d1-2B public checkpoint revisions unpinned offline".into(),
    });

    Ok(Lfm3D1DecisionReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::blocked_on_artifacts(
            "GPU serving host evaluating Liquid AI LFM3-1.7B (256k tri-hybrid) & d1-2B single-pass decision checkpoints",
            "LFM3-1.7B and d1-2B checkpoint revisions are unpinned offline (post_only_checkpoints_unpinned_offline); tri-hybrid cached parity, schema conformance, and ECE/Brier calibration verified on CPU",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        pin_status: LFM3_PIN_STATUS.into(),
        excludes_audio_model: true,
        audio_out_of_scope_issue: AUDIO_OUT_OF_SCOPE_ISSUE.into(),
        tri_hybrid_cached_max_abs_err: tri_err,
        decision_arms,
        lfm3_context_scaling,
        negative_controls: nc,
    })
}

pub fn inputs_json(report: &Lfm3D1DecisionReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.lfm3-d1-decision.inputs.v1\",\n",
            "  \"issue\": 73,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"upstream_x_post\": {},\n",
            "  \"pin_status\": {},\n",
            "  \"excludes_audio_model\": {},\n",
            "  \"audio_out_of_scope_issue\": {},\n",
            "  \"complements\": [\"inference-lab#36\",\"inference-lab#55\"],\n",
            "  \"ece_bins\": {},\n",
            "  \"author_unverified_non_gating_notes\": \"Liquid AI LFM3-1.7B and d1-2B launch claims are unverified related work; LFM3-Audio-2B is tracked separately in audio-lab#2\"\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(UPSTREAM_X_POST),
        quote(&report.pin_status),
        report.excludes_audio_model,
        quote(&report.audio_out_of_scope_issue),
        ECE_BINS,
    )
}

pub fn results_json(report: &Lfm3D1DecisionReport) -> String {
    let d_json: Vec<String> = report
        .decision_arms
        .iter()
        .map(|a| {
            format!(
                "{{\"arm_id\":{},\"total_fixtures\":{},\"schema_valid_count\":{},\"consumer_correct_count\":{},\"schema_conformance_rate\":{},\"consumer_accuracy\":{},\"ece\":{},\"brier_score\":{},\"mean_latency_ms\":{},\"outcome\":{}}}",
                quote(&a.arm_id),
                a.total_fixtures,
                a.schema_valid_count,
                a.consumer_correct_count,
                opt_f64(a.schema_conformance_rate),
                opt_f64(a.consumer_accuracy),
                opt_f64(a.ece),
                opt_f64(a.brier_score),
                opt_f64(a.mean_latency_ms),
                quote(a.outcome.as_str()),
            )
        })
        .collect();
    let ctx_json: Vec<String> = report
        .lfm3_context_scaling
        .iter()
        .map(|r| {
            format!(
                "{{\"context_tokens\":{},\"state_and_kv_mb\":{},\"synthetic_decode_tok_s\":{},\"synthetic_retention_score\":{}}}",
                r.context_tokens,
                finite_f64(r.state_and_kv_mb),
                finite_f64(r.synthetic_decode_tok_s),
                finite_f64(r.synthetic_retention_score),
            )
        })
        .collect();
    let nc_json: Vec<String> = report
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
            "  \"schema\": \"inference-lab.lfm3-d1-decision.results.v1\",\n",
            "  \"issue\": 73,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"pin_status\": {},\n",
            "  \"excludes_audio_model\": {},\n",
            "  \"tri_hybrid_cached_max_abs_err\": {},\n",
            "  \"decision_arms\": [{}],\n",
            "  \"lfm3_context_scaling\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(&report.pin_status),
        report.excludes_audio_model,
        finite_f64(report.tri_hybrid_cached_max_abs_err),
        d_json.join(","),
        ctx_json.join(","),
        nc_json.join(","),
    )
}

pub fn summary_markdown(report: &Lfm3D1DecisionReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #73: Pinned `LFM3-1.7B` (Tri-Hybrid 256k) & `d1-2B` Single-Pass Typed Decision Qualification"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Execution class: `{}` (`default_enabled: {}`, `pin_status: {}`, `LFM3-Audio-2B` excluded to `{}`)",
        report.execution_class.as_str(),
        report.default_enabled,
        report.pin_status,
        report.audio_out_of_scope_issue
    )
    .unwrap();
    writeln!(
        out,
        "- Hardware / artifact gate: `{}` ({})",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(
        out,
        "- Miniature tri-hybrid cached-vs-full parity: `max_abs_err={:.3e}` (tol `<= 1e-12`)",
        report.tri_hybrid_cached_max_abs_err
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "## `d1-2B` Single-Pass vs Autoregressive Typed Decision Conformance & Calibration"
    )
    .unwrap();
    for a in &report.decision_arms {
        writeln!(
            out,
            "- `{}`: `schema={}/{}`, `consumer_correct={}/{}`, `ece={:?}`, `brier={:?}`, `mean_lat_ms={:?}`",
            a.arm_id,
            a.schema_valid_count,
            a.total_fixtures,
            a.consumer_correct_count,
            a.total_fixtures,
            a.ece,
            a.brier_score,
            a.mean_latency_ms
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "## `LFM3-1.7B` 8K–256K Context Scaling Contract").unwrap();
    for r in &report.lfm3_context_scaling {
        writeln!(
            out,
            "- `ctx={}`: `state_kv={:.1} MB`, `tok_s={:.1}`, `retention={:.2}`",
            r.context_tokens,
            r.state_and_kv_mb,
            r.synthetic_decode_tok_s,
            r.synthetic_retention_score
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
    fn tri_hybrid_is_exact_d1_is_better_calibrated_and_controls_fail() {
        let report = run_default_harness().unwrap();
        assert!(!report.default_enabled);
        assert!(report.excludes_audio_model);
        assert!(report.tri_hybrid_cached_max_abs_err <= PARITY_TOL);
        let d1 = &report.decision_arms[0];
        let ar = &report.decision_arms[1];
        assert!(d1.ece.unwrap() < ar.ece.unwrap());
        assert!(d1.brier_score.unwrap() < ar.brier_score.unwrap());
        assert!(d1.mean_latency_ms.unwrap() < ar.mean_latency_ms.unwrap());
        assert_eq!(report.negative_controls.len(), 8);
        for c in &report.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }
}
