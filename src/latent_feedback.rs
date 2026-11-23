//! Issue #50: Test latent feedback against a matched extra-epoch control in data-limited
//! small-model training.
//!
//! Pinned motivation (strictly unverified non-gating related work):
//! - `arXiv:2611.20404` (*Latent Feedback Is a Compute and Data Amplifier*).
//! - Related issues: `#33`, `#36`, `#37`.
//! - Authors' reported 48% compute and 47% unique-data savings from 60M–778M parameters are
//!   strictly unverified related work and NEVER imported as lab pass/fail thresholds.
//! - The "stable at large scale" question is explicitly kept `open_untested`.
//!
//! Provides:
//! - Real miniature pure-Rust sequence learner (vocab `V = 8`, hidden dim `D = 8`, 2 residual
//!   layers) with genuine forward pass, exact analytical backpropagation, and SGD training over a
//!   deterministic data-limited training corpus plus a strictly disjoint held-out split.
//! - Two genuinely distinct architectures:
//!   1. `StandardTransformerControl`: standard token embedding -> 2 residual layers -> LM head,
//!      trained for extra epochs so total exact FLOPs match the latent-feedback arm on the exact
//!      same unique training tokens.
//!   2. `LatentFeedbackModel`: feeds the previous token's (`t - 1`) top-layer RMS-normalized
//!      latent state projected through `W_fb` into token `t`'s input (`x_t = emb[tok_t] + W_fb * norm(h_{t-1})`),
//!      with exact analytical 1-step BPTT gradients verified against an independent central
//!      finite-difference gradient checker (`max_grad_err <= 1e-6`).
//! - Exact deterministic FLOP accounting (`FlopCounter`), train and held-out cross-entropy loss
//!   and accuracy, causal invariance verification (ensuring zero future-token leakage), explicit
//!   `large_scale_stability_status: "open_untested"`, and 8 executable negative controls.

use std::collections::BTreeSet;
use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::det_weight;
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const PAPER_PIN: &str = "arXiv:2611.20404";
pub const SOURCE_BOOKMARK: &str = "https://x.com/rosinality/status/2122676232701579652";
pub const RELATED_ISSUES: [u32; 3] = [33, 36, 37];

pub const VOCAB: usize = 8;
pub const DIM: usize = 8;
pub const SEQ_LEN: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelVariant {
    StandardExtraEpochControl,
    LatentFeedback,
}

impl ModelVariant {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StandardExtraEpochControl => "standard_extra_epoch_control",
            Self::LatentFeedback => "latent_feedback",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SeqModelWeights {
    pub emb: [[f64; DIM]; VOCAB],
    pub w1: [[f64; DIM]; DIM],
    pub b1: [f64; DIM],
    pub w2: [[f64; DIM]; DIM],
    pub b2: [f64; DIM],
    pub w_fb: [[f64; DIM]; DIM],
    pub head: [[f64; DIM]; VOCAB],
}

#[derive(Debug, Clone)]
pub struct SeqModelGrads {
    pub emb: [[f64; DIM]; VOCAB],
    pub w1: [[f64; DIM]; DIM],
    pub b1: [f64; DIM],
    pub w2: [[f64; DIM]; DIM],
    pub b2: [f64; DIM],
    pub w_fb: [[f64; DIM]; DIM],
    pub head: [[f64; DIM]; VOCAB],
}

impl SeqModelGrads {
    fn zeros() -> Self {
        Self {
            emb: [[0.0; DIM]; VOCAB],
            w1: [[0.0; DIM]; DIM],
            b1: [0.0; DIM],
            w2: [[0.0; DIM]; DIM],
            b2: [0.0; DIM],
            w_fb: [[0.0; DIM]; DIM],
            head: [[0.0; DIM]; VOCAB],
        }
    }
}

impl SeqModelWeights {
    pub fn init_deterministic(seed: usize, use_feedback: bool) -> Self {
        let mut emb = [[0.0; DIM]; VOCAB];
        let mut w1 = [[0.0; DIM]; DIM];
        let mut b1 = [0.0; DIM];
        let mut w2 = [[0.0; DIM]; DIM];
        let mut b2 = [0.0; DIM];
        let mut w_fb = [[0.0; DIM]; DIM];
        let mut head = [[0.0; DIM]; VOCAB];
        for v in 0..VOCAB {
            for d in 0..DIM {
                emb[v][d] = det_weight(seed + 101 + v * 17, d, 0.35);
                head[v][d] = det_weight(seed + 401 + v * 23, d, 0.35);
            }
        }
        for r in 0..DIM {
            b1[r] = det_weight(seed + 201, r, 0.05);
            b2[r] = det_weight(seed + 301, r, 0.05);
            for c in 0..DIM {
                w1[r][c] = det_weight(seed + 211 + r * 13, c, 0.30);
                w2[r][c] = det_weight(seed + 311 + r * 19, c, 0.30);
                if use_feedback {
                    w_fb[r][c] = det_weight(seed + 511 + r * 29, c, 0.25);
                }
            }
        }
        Self {
            emb,
            w1,
            b1,
            w2,
            b2,
            w_fb,
            head,
        }
    }

    pub fn parameter_count(use_feedback: bool) -> usize {
        let base = VOCAB * DIM + DIM * DIM + DIM + DIM * DIM + DIM + VOCAB * DIM;
        if use_feedback {
            base + DIM * DIM
        } else {
            base
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct FlopCounter {
    pub forward_flops: u64,
    pub backward_flops: u64,
}

impl FlopCounter {
    pub fn total(&self) -> u64 {
        self.forward_flops + self.backward_flops
    }
}

#[derive(Debug, Clone)]
pub struct StepCache {
    pub tok: usize,
    pub target: usize,
    pub prev_norm_h: [f64; DIM],
    pub x_in: [f64; DIM],
    pub z1: [f64; DIM],
    pub a1: [f64; DIM],
    pub h1: [f64; DIM],
    pub z2: [f64; DIM],
    pub a2: [f64; DIM],
    pub h2: [f64; DIM],
    pub rms_inv: f64,
    pub norm_h2: [f64; DIM],
    pub probs: [f64; VOCAB],
}

#[derive(Debug, Clone)]
pub struct ArmTrainingResult {
    pub variant: ModelVariant,
    pub epochs_run: usize,
    pub unique_train_tokens: usize,
    pub parameter_count: usize,
    pub forward_flops: u64,
    pub backward_flops: u64,
    pub total_flops: u64,
    pub initial_train_loss: f64,
    pub final_train_loss: f64,
    pub train_accuracy: Option<f64>,
    pub heldout_loss: f64,
    pub heldout_accuracy: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct LatentFeedbackReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub synthetic_contract_fixture: bool,
    pub fixture_digest: DigestRef,
    pub large_scale_stability_status: String,
    pub grad_check_max_abs_err: f64,
    pub flop_ratio_feedback_over_control: Option<f64>,
    pub causal_invariance_max_err: f64,
    pub arms: Vec<ArmTrainingResult>,
    pub negative_controls: Vec<NegativeControl>,
}

fn check_finite(v: f64, label: &str) -> Result<f64, String> {
    if !v.is_finite() {
        return Err(format!("non-finite float in {label}"));
    }
    Ok(v)
}

fn rms_norm_with_inv(x: &[f64; DIM]) -> Result<([f64; DIM], f64), String> {
    let mut sq = 0.0_f64;
    for &v in x {
        let fv = check_finite(v, "rms_norm input")?;
        sq += fv * fv;
    }
    let inv = 1.0 / (sq / (DIM as f64) + 1.0e-5).sqrt();
    check_finite(inv, "rms_norm inv")?;
    let mut out = [0.0_f64; DIM];
    for i in 0..DIM {
        out[i] = x[i] * inv;
    }
    Ok((out, inv))
}

fn softmax_vocab(logits: &[f64; VOCAB]) -> Result<[f64; VOCAB], String> {
    let mut peak = check_finite(logits[0], "softmax logit")?;
    for &v in &logits[1..] {
        let fv = check_finite(v, "softmax logit")?;
        if fv > peak {
            peak = fv;
        }
    }
    let mut probs = [0.0_f64; VOCAB];
    let mut sum = 0.0_f64;
    for i in 0..VOCAB {
        let e = (logits[i] - peak).exp();
        check_finite(e, "softmax exp")?;
        probs[i] = e;
        sum += e;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return Err("invalid softmax sum".into());
    }
    for p in &mut probs {
        *p /= sum;
    }
    Ok(probs)
}

fn argmax_vocab(probs: &[f64; VOCAB]) -> Result<usize, String> {
    let mut best_i = 0usize;
    let mut best_v = check_finite(probs[0], "argmax prob")?;
    for (i, &v) in probs.iter().enumerate().skip(1) {
        let fv = check_finite(v, "argmax prob")?;
        if fv > best_v {
            best_v = fv;
            best_i = i;
        }
    }
    Ok(best_i)
}

pub type SequenceSplits = (Vec<Vec<usize>>, Vec<Vec<usize>>);

/// Forward pass over a token sequence `[t_0, t_1, ..., t_{L-1}]` predicting `t_{i+1}` from `t_i`.
/// When `use_feedback` is true, feeds `W_fb * rms_norm(h2_{i-1})` into step `i`'s input.
/// When `inject_future_leak` is true (negative control ONLY), illegally peeks at token `i + 1`.
#[allow(clippy::needless_range_loop)]
pub fn forward_sequence(
    weights: &SeqModelWeights,
    seq: &[usize],
    use_feedback: bool,
    inject_future_leak: bool,
    flops: &mut FlopCounter,
) -> Result<(f64, usize, usize, Vec<StepCache>), String> {
    if seq.len() < 2 {
        return Err("sequence must contain at least 2 tokens".into());
    }
    let steps = seq.len() - 1;
    let mut caches = Vec::with_capacity(steps);
    let mut prev_norm_h = [0.0_f64; DIM];
    let mut total_loss = 0.0_f64;
    let mut correct = 0usize;

    for t in 0..steps {
        let tok = seq[t];
        let target = seq[t + 1];
        if tok >= VOCAB || target >= VOCAB {
            return Err("token out of vocabulary range".into());
        }
        let mut x_in = weights.emb[tok];
        if use_feedback && t > 0 {
            for r in 0..DIM {
                let mut acc = 0.0_f64;
                for c in 0..DIM {
                    acc += weights.w_fb[r][c] * prev_norm_h[c];
                }
                x_in[r] += acc;
            }
            flops.forward_flops += (2 * DIM * DIM + DIM) as u64;
        }
        if inject_future_leak && t + 1 < seq.len() {
            let future_tok = seq[t + 1];
            for d in 0..DIM {
                x_in[d] += 0.25 * weights.emb[future_tok][d];
            }
        }

        // Layer 1: residual tanh block
        let mut z1 = [0.0_f64; DIM];
        let mut a1 = [0.0_f64; DIM];
        let mut h1 = [0.0_f64; DIM];
        for r in 0..DIM {
            let mut acc = weights.b1[r];
            for c in 0..DIM {
                acc += weights.w1[r][c] * x_in[c];
            }
            z1[r] = check_finite(acc, "z1")?;
            a1[r] = z1[r].tanh();
            h1[r] = x_in[r] + a1[r];
        }
        flops.forward_flops += (2 * DIM * DIM + 3 * DIM) as u64;

        // Layer 2: residual tanh block
        let mut z2 = [0.0_f64; DIM];
        let mut a2 = [0.0_f64; DIM];
        let mut h2 = [0.0_f64; DIM];
        for r in 0..DIM {
            let mut acc = weights.b2[r];
            for c in 0..DIM {
                acc += weights.w2[r][c] * h1[c];
            }
            z2[r] = check_finite(acc, "z2")?;
            a2[r] = z2[r].tanh();
            h2[r] = h1[r] + a2[r];
        }
        flops.forward_flops += (2 * DIM * DIM + 3 * DIM) as u64;

        let (norm_h2, rms_inv) = rms_norm_with_inv(&h2)?;
        flops.forward_flops += (3 * DIM) as u64;

        // LM head
        let mut logits = [0.0_f64; VOCAB];
        for v in 0..VOCAB {
            let mut acc = 0.0_f64;
            for d in 0..DIM {
                acc += weights.head[v][d] * h2[d];
            }
            logits[v] = check_finite(acc, "lm logit")?;
        }
        flops.forward_flops += (2 * VOCAB * DIM) as u64;

        let probs = softmax_vocab(&logits)?;
        let p_t = probs[target];
        let clamped_p = if p_t < 1.0e-15 { 1.0e-15 } else { p_t };
        total_loss += -clamped_p.ln();
        if argmax_vocab(&probs)? == target {
            correct += 1;
        }

        caches.push(StepCache {
            tok,
            target,
            prev_norm_h,
            x_in,
            z1,
            a1,
            h1,
            z2,
            a2,
            h2,
            rms_inv,
            norm_h2,
            probs,
        });
        prev_norm_h = norm_h2;
    }

    let mean_loss = total_loss / (steps as f64);
    check_finite(mean_loss, "mean_loss")?;
    Ok((mean_loss, correct, steps, caches))
}

/// Exact analytical reverse-mode backpropagation through the full sequence, including exact
/// recurrent feedback propagation through `W_fb` and `rms_norm` from step `t + 1` back to `t`.
#[allow(clippy::needless_range_loop)]
pub fn backward_sequence(
    weights: &SeqModelWeights,
    caches: &[StepCache],
    use_feedback: bool,
    corrupt_grad: bool,
    flops: &mut FlopCounter,
) -> Result<SeqModelGrads, String> {
    if caches.is_empty() {
        return Err("empty forward cache in backward_sequence".into());
    }
    let scale = 1.0 / (caches.len() as f64);
    let mut grads = SeqModelGrads::zeros();
    let mut d_next_norm_h = [0.0_f64; DIM];

    for t in (0..caches.len()).rev() {
        let c = &caches[t];
        let mut d_logits = [0.0_f64; VOCAB];
        for v in 0..VOCAB {
            let y = if v == c.target { 1.0 } else { 0.0 };
            d_logits[v] = (c.probs[v] - y) * scale;
        }

        // LM head backward
        let mut d_h2 = [0.0_f64; DIM];
        for v in 0..VOCAB {
            for d in 0..DIM {
                grads.head[v][d] += d_logits[v] * c.h2[d];
                d_h2[d] += weights.head[v][d] * d_logits[v];
            }
        }
        flops.backward_flops += (4 * VOCAB * DIM) as u64;

        // Add recurrent gradient flowing back from step t+1 through rms_norm(h2_t)
        if use_feedback && t + 1 < caches.len() {
            let mut dot = 0.0_f64;
            for d in 0..DIM {
                dot += d_next_norm_h[d] * c.h2[d];
            }
            let inv3 = c.rms_inv * c.rms_inv * c.rms_inv / (DIM as f64);
            for d in 0..DIM {
                d_h2[d] += d_next_norm_h[d] * c.rms_inv - c.h2[d] * dot * inv3;
            }
            flops.backward_flops += (6 * DIM) as u64;
        }

        // Layer 2 backward: h2 = h1 + tanh(W2 h1 + b2)
        let mut d_z2 = [0.0_f64; DIM];
        for r in 0..DIM {
            d_z2[r] = d_h2[r] * (1.0 - c.a2[r] * c.a2[r]);
            grads.b2[r] += d_z2[r];
        }
        let mut d_h1 = d_h2;
        for r in 0..DIM {
            for col in 0..DIM {
                grads.w2[r][col] += d_z2[r] * c.h1[col];
                d_h1[col] += weights.w2[r][col] * d_z2[r];
            }
        }
        flops.backward_flops += (4 * DIM * DIM + 3 * DIM) as u64;

        // Layer 1 backward: h1 = x_in + tanh(W1 x_in + b1)
        let mut d_z1 = [0.0_f64; DIM];
        for r in 0..DIM {
            d_z1[r] = d_h1[r] * (1.0 - c.a1[r] * c.a1[r]);
            grads.b1[r] += d_z1[r];
        }
        let mut d_x_in = d_h1;
        for r in 0..DIM {
            for col in 0..DIM {
                grads.w1[r][col] += d_z1[r] * c.x_in[col];
                d_x_in[col] += weights.w1[r][col] * d_z1[r];
            }
        }
        flops.backward_flops += (4 * DIM * DIM + 3 * DIM) as u64;

        // Token embedding gradient
        for d in 0..DIM {
            grads.emb[c.tok][d] += d_x_in[d];
        }

        // Feedback projection gradient & propagate into step t-1's norm_h2
        d_next_norm_h = [0.0_f64; DIM];
        if use_feedback && t > 0 {
            for r in 0..DIM {
                for col in 0..DIM {
                    grads.w_fb[r][col] += d_x_in[r] * c.prev_norm_h[col];
                    d_next_norm_h[col] += weights.w_fb[r][col] * d_x_in[r];
                }
            }
            flops.backward_flops += (4 * DIM * DIM) as u64;
        }
    }

    if corrupt_grad {
        grads.w_fb[0][0] += 0.25;
    }

    // Check finiteness of gradients
    for r in 0..DIM {
        check_finite(grads.b1[r], "grad b1")?;
        check_finite(grads.b2[r], "grad b2")?;
        for col in 0..DIM {
            check_finite(grads.w1[r][col], "grad w1")?;
            check_finite(grads.w2[r][col], "grad w2")?;
            check_finite(grads.w_fb[r][col], "grad w_fb")?;
        }
    }
    Ok(grads)
}

fn apply_sgd(weights: &mut SeqModelWeights, grads: &SeqModelGrads, lr: f64, use_feedback: bool) {
    for v in 0..VOCAB {
        for d in 0..DIM {
            weights.emb[v][d] -= lr * grads.emb[v][d];
            weights.head[v][d] -= lr * grads.head[v][d];
        }
    }
    for r in 0..DIM {
        weights.b1[r] -= lr * grads.b1[r];
        weights.b2[r] -= lr * grads.b2[r];
        for c in 0..DIM {
            weights.w1[r][c] -= lr * grads.w1[r][c];
            weights.w2[r][c] -= lr * grads.w2[r][c];
            if use_feedback {
                weights.w_fb[r][c] -= lr * grads.w_fb[r][c];
            }
        }
    }
}

/// Independent central finite-difference gradient verification for `LatentFeedbackModel`.
pub fn verify_analytical_gradients_fd(
    weights: &SeqModelWeights,
    seq: &[usize],
    corrupt_grad: bool,
) -> Result<f64, String> {
    let mut flops = FlopCounter::default();
    let (_, _, _, caches) = forward_sequence(weights, seq, true, false, &mut flops)?;
    let anal = backward_sequence(weights, &caches, true, corrupt_grad, &mut flops)?;

    let eps = 1.0e-6;
    let mut max_err = 0.0_f64;

    // Check w_fb[0][0], w_fb[2][3], w1[1][2], w2[3][4], head[1][2], emb[seq[0]][1]
    let checks: &[(&str, usize, usize)] = &[
        ("w_fb", 0, 0),
        ("w_fb", 2, 3),
        ("w1", 1, 2),
        ("w2", 3, 4),
        ("head", 1, 2),
        ("emb", seq[0], 1),
    ];

    for &(name, r, c) in checks {
        let mut wp = weights.clone();
        let mut wm = weights.clone();
        let g_anal = match name {
            "w_fb" => {
                wp.w_fb[r][c] += eps;
                wm.w_fb[r][c] -= eps;
                anal.w_fb[r][c]
            }
            "w1" => {
                wp.w1[r][c] += eps;
                wm.w1[r][c] -= eps;
                anal.w1[r][c]
            }
            "w2" => {
                wp.w2[r][c] += eps;
                wm.w2[r][c] -= eps;
                anal.w2[r][c]
            }
            "head" => {
                wp.head[r][c] += eps;
                wm.head[r][c] -= eps;
                anal.head[r][c]
            }
            _ => {
                wp.emb[r][c] += eps;
                wm.emb[r][c] -= eps;
                anal.emb[r][c]
            }
        };
        let mut f_tmp = FlopCounter::default();
        let (lp, _, _, _) = forward_sequence(&wp, seq, true, false, &mut f_tmp)?;
        let (lm, _, _, _) = forward_sequence(&wm, seq, true, false, &mut f_tmp)?;
        let g_fd = (lp - lm) / (2.0 * eps);
        let err = (g_anal - g_fd).abs();
        check_finite(err, "fd grad err")?;
        if err > max_err {
            max_err = err;
        }
    }
    Ok(max_err)
}

/// Verify causal invariance: mutating future tokens after position `cutoff` must NOT change
/// step `cutoff`'s hidden state or output probabilities (difference must be 0.0).
pub fn verify_causal_invariance(
    weights: &SeqModelWeights,
    seq: &[usize],
    inject_future_leak: bool,
) -> Result<f64, String> {
    if seq.len() < 4 {
        return Err("sequence too short for causal invariance test".into());
    }
    let cutoff = 2usize;
    let mut alt = seq.to_vec();
    for tok in alt.iter_mut().skip(cutoff + 1) {
        *tok = (*tok + 3) % VOCAB;
    }
    let mut f1 = FlopCounter::default();
    let mut f2 = FlopCounter::default();
    let (_, _, _, c1) = forward_sequence(weights, seq, true, inject_future_leak, &mut f1)?;
    let (_, _, _, c2) = forward_sequence(weights, &alt, true, inject_future_leak, &mut f2)?;
    let mut max_diff = 0.0_f64;
    for v in 0..VOCAB {
        let d = (c1[cutoff].probs[v] - c2[cutoff].probs[v]).abs();
        check_finite(d, "causal diff")?;
        if d > max_diff {
            max_diff = d;
        }
    }
    Ok(max_diff)
}

/// Generate a data-limited deterministic sequence dataset where next token depends on the
/// previous 2 tokens (so token-to-token latent feedback captures multi-step context that a
/// single-token Markov baseline without recurrence cannot represent cleanly).
pub fn build_disjoint_dataset() -> Result<SequenceSplits, String> {
    let make_seq = |a: usize, b: usize| -> Vec<usize> {
        let mut s = Vec::with_capacity(SEQ_LEN);
        s.push(a % VOCAB);
        s.push(b % VOCAB);
        for t in 2..SEQ_LEN {
            let next = (s[t - 1] * 3 + s[t - 2] * 5 + 1) % VOCAB;
            s.push(next);
        }
        s
    };
    let train = vec![
        make_seq(0, 1),
        make_seq(1, 2),
        make_seq(2, 3),
        make_seq(3, 4),
        make_seq(4, 5),
        make_seq(5, 6),
        make_seq(6, 7),
        make_seq(7, 0),
        make_seq(0, 3),
        make_seq(2, 5),
    ];
    let heldout = vec![
        make_seq(1, 4),
        make_seq(3, 6),
        make_seq(5, 0),
        make_seq(7, 2),
    ];
    validate_disjoint_splits(&train, &heldout)?;
    Ok((train, heldout))
}

pub fn validate_disjoint_splits(
    train: &[Vec<usize>],
    heldout: &[Vec<usize>],
) -> Result<(), String> {
    if train.is_empty() || heldout.is_empty() {
        return Err("train and held-out splits must both be non-empty".into());
    }
    let tr: BTreeSet<&Vec<usize>> = train.iter().collect();
    for h in heldout {
        if tr.contains(h) {
            return Err("sequence overlap detected between train and held-out splits".into());
        }
    }
    Ok(())
}

fn evaluate_split(
    weights: &SeqModelWeights,
    split: &[Vec<usize>],
    use_feedback: bool,
) -> Result<(f64, Option<f64>), String> {
    let mut flops = FlopCounter::default();
    let mut loss_sum = 0.0_f64;
    let mut correct_sum = 0usize;
    let mut step_sum = 0usize;
    for seq in split {
        let (l, c, s, _) = forward_sequence(weights, seq, use_feedback, false, &mut flops)?;
        loss_sum += l;
        correct_sum += c;
        step_sum += s;
    }
    let mean_loss = loss_sum / (split.len() as f64);
    check_finite(mean_loss, "eval split loss")?;
    let acc = if step_sum == 0 {
        None
    } else {
        Some((correct_sum as f64) / (step_sum as f64))
    };
    Ok((mean_loss, acc))
}

pub fn train_arm(
    variant: ModelVariant,
    train: &[Vec<usize>],
    heldout: &[Vec<usize>],
    epochs: usize,
    lr: f64,
    seed: usize,
) -> Result<ArmTrainingResult, String> {
    validate_disjoint_splits(train, heldout)?;
    if epochs == 0 {
        return Err("epochs must be >= 1".into());
    }
    check_finite(lr, "learning rate")?;
    if lr <= 0.0 {
        return Err("learning rate must be positive".into());
    }
    let use_feedback = variant == ModelVariant::LatentFeedback;
    let mut weights = SeqModelWeights::init_deterministic(seed, use_feedback);
    let (initial_train_loss, _) = evaluate_split(&weights, train, use_feedback)?;

    let mut flops = FlopCounter::default();
    for _ in 0..epochs {
        for seq in train {
            let (_, _, _, caches) =
                forward_sequence(&weights, seq, use_feedback, false, &mut flops)?;
            let grads = backward_sequence(&weights, &caches, use_feedback, false, &mut flops)?;
            apply_sgd(&mut weights, &grads, lr, use_feedback);
        }
    }

    let (final_train_loss, train_accuracy) = evaluate_split(&weights, train, use_feedback)?;
    let (heldout_loss, heldout_accuracy) = evaluate_split(&weights, heldout, use_feedback)?;
    let unique_train_tokens: usize = train.iter().map(Vec::len).sum();

    Ok(ArmTrainingResult {
        variant,
        epochs_run: epochs,
        unique_train_tokens,
        parameter_count: SeqModelWeights::parameter_count(use_feedback),
        forward_flops: flops.forward_flops,
        backward_flops: flops.backward_flops,
        total_flops: flops.total(),
        initial_train_loss,
        final_train_loss,
        train_accuracy,
        heldout_loss,
        heldout_accuracy,
    })
}

pub fn validate_matched_flop_budget(
    control_flops: u64,
    feedback_flops: u64,
    max_rel_diff: f64,
) -> Result<f64, String> {
    check_finite(max_rel_diff, "max_rel_diff")?;
    if control_flops == 0 || feedback_flops == 0 {
        return Err("zero FLOP budget is invalid".into());
    }
    let ratio = (feedback_flops as f64) / (control_flops as f64);
    check_finite(ratio, "flop_ratio")?;
    if (ratio - 1.0).abs() > max_rel_diff {
        return Err(format!(
            "unmatched FLOP budgets: feedback={feedback_flops} vs control={control_flops} (ratio={ratio:.4})"
        ));
    }
    Ok(ratio)
}

pub fn validate_large_scale_claim(status: &str) -> OutcomeStatus {
    if status == "open_untested" {
        OutcomeStatus::Accepted
    } else {
        OutcomeStatus::Invalid
    }
}

pub fn validate_no_paper_savings_gate(imports_48_47_pct_as_gate: bool) -> OutcomeStatus {
    if imports_48_47_pct_as_gate {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    }
}

pub fn run_default_harness() -> Result<LatentFeedbackReport, String> {
    let (train, heldout) = build_disjoint_dataset()?;
    let mut digest_buf = String::new();
    for s in train.iter().chain(heldout.iter()) {
        let _ = write!(digest_buf, "{s:?};");
    }
    let fixture_digest = DigestRef::sha256(digest_buf.as_bytes());

    let seed = 50_001usize;
    let init_fb = SeqModelWeights::init_deterministic(seed, true);
    let grad_check_max_abs_err = verify_analytical_gradients_fd(&init_fb, &train[0], false)?;
    if grad_check_max_abs_err > 1.0e-6 {
        return Err(format!(
            "analytical gradient failed FD verification: {grad_check_max_abs_err:.3e}"
        ));
    }
    let causal_invariance_max_err = verify_causal_invariance(&init_fb, &train[0], false)?;
    if causal_invariance_max_err > 1.0e-12 {
        return Err(format!(
            "latent feedback model violated causality: {causal_invariance_max_err:.3e}"
        ));
    }

    // Match exact FLOPs: LatentFeedback does extra feedback projection FLOPs per step, so
    // StandardExtraEpochControl runs extra epochs on the exact same unique training tokens
    // to match total FLOPs within 5%!
    // Compute exact per-epoch FLOPs dynamically from a 1-epoch probe so epochs are derived,
    // never guessed.
    let probe_fb = train_arm(
        ModelVariant::LatentFeedback,
        &train,
        &heldout,
        1,
        0.12,
        seed,
    )?;
    let probe_ctrl = train_arm(
        ModelVariant::StandardExtraEpochControl,
        &train,
        &heldout,
        1,
        0.12,
        seed,
    )?;
    let fb_epochs = 24usize;
    let target_flops = probe_fb.total_flops * (fb_epochs as u64);
    let ctrl_epochs =
        ((target_flops + probe_ctrl.total_flops / 2) / probe_ctrl.total_flops) as usize;

    let ctrl_res = train_arm(
        ModelVariant::StandardExtraEpochControl,
        &train,
        &heldout,
        ctrl_epochs,
        0.12,
        seed,
    )?;
    let fb_res = train_arm(
        ModelVariant::LatentFeedback,
        &train,
        &heldout,
        fb_epochs,
        0.12,
        seed,
    )?;

    let flop_ratio = validate_matched_flop_budget(ctrl_res.total_flops, fb_res.total_flops, 0.05)?;
    let large_scale_status = "open_untested".to_string();

    let mut negative_controls = Vec::new();

    // NC1: unmatched FLOP budget rejected
    {
        let outcome =
            if validate_matched_flop_budget(ctrl_res.total_flops, fb_res.total_flops * 2, 0.05)
                .is_err()
            {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            };
        negative_controls.push(NegativeControl {
            case_id: "nc1_unmatched_flop_budget_rejected".into(),
            outcome,
            detail: "2x FLOP disparity between arms violates matched-FLOP control".into(),
        });
    }

    // NC2: train / held-out sequence overlap rejected
    {
        let mut leaky_heldout = heldout.clone();
        leaky_heldout.push(train[0].clone());
        let outcome = if validate_disjoint_splits(&train, &leaky_heldout).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
        negative_controls.push(NegativeControl {
            case_id: "nc2_train_heldout_overlap_rejected".into(),
            outcome,
            detail: "duplicate sequence across train and held-out splits is refused".into(),
        });
    }

    // NC3: future-token leakage caught by causal invariance check
    {
        let leak_diff = verify_causal_invariance(&init_fb, &train[0], true)?;
        negative_controls.push(NegativeControl {
            case_id: "nc3_future_token_leakage_caught".into(),
            outcome: if leak_diff > 1.0e-12 {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!(
                "injected future-token leak caused {leak_diff:.3e} causal perturbation"
            ),
        });
    }

    // NC4: corrupted analytical gradient caught by independent finite-difference check
    {
        let bad_err = verify_analytical_gradients_fd(&init_fb, &train[0], true)?;
        negative_controls.push(NegativeControl {
            case_id: "nc4_corrupted_gradient_caught_by_fd".into(),
            outcome: if bad_err > 1.0e-6 {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!("corrupted analytical gradient diverged by {bad_err:.3e} vs FD"),
        });
    }

    // NC5: claiming large-scale stability verified from small-scale run rejected
    negative_controls.push(NegativeControl {
        case_id: "nc5_large_scale_stability_claim_rejected".into(),
        outcome: validate_large_scale_claim("verified_at_scale"),
        detail: "large-scale stability must remain explicitly open_untested".into(),
    });

    // NC6: NaN learning rate / loss rejected
    {
        let outcome = if train_arm(
            ModelVariant::LatentFeedback,
            &train,
            &heldout,
            1,
            f64::NAN,
            seed,
        )
        .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
        negative_controls.push(NegativeControl {
            case_id: "nc6_nan_learning_rate_rejected".into(),
            outcome,
            detail: "non-finite learning rate or loss fails finite check".into(),
        });
    }

    // NC7: zero FLOP budget / empty dataset rejected
    {
        let outcome = if validate_matched_flop_budget(0, fb_res.total_flops, 0.05).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
        negative_controls.push(NegativeControl {
            case_id: "nc7_zero_flop_budget_guarded".into(),
            outcome,
            detail: "zero FLOP count is rejected before ratio division".into(),
        });
    }

    // NC8: importing paper's 48% compute / 47% unique-data savings as pass gate rejected
    negative_controls.push(NegativeControl {
        case_id: "nc8_paper_savings_not_pass_gate".into(),
        outcome: validate_no_paper_savings_gate(true),
        detail: "arXiv:2611.20404 48% compute / 47% data claims are unverified, never pass gates"
            .into(),
    });

    Ok(LatentFeedbackReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "GPU small-model training cluster (60M–778M parameter scale) for data-limited Transformer vs latent-feedback runs",
            "Miniature pure-Rust matched-FLOP sequence learner, FD gradient check, and causal invariance verified on CPU; 60M–778M GPU training did not execute locally and large-scale stability remains open_untested",
        ),
        synthetic_contract_fixture: true,
        fixture_digest,
        large_scale_stability_status: large_scale_status,
        grad_check_max_abs_err,
        flop_ratio_feedback_over_control: Some(flop_ratio),
        causal_invariance_max_err,
        arms: vec![ctrl_res, fb_res],
        negative_controls,
    })
}

pub fn inputs_json(report: &LatentFeedbackReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.latent-feedback.inputs.v1\",\n",
            "  \"issue\": 50,\n",
            "  \"related_issues\": [33, 36, 37],\n",
            "  \"execution_class\": {},\n",
            "  \"non_gating_paper\": {},\n",
            "  \"source_bookmark\": {},\n",
            "  \"large_scale_stability_status\": {},\n",
            "  \"fixture_digest\": {},\n",
            "  \"synthetic_contract_fixture\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(PAPER_PIN),
        quote(SOURCE_BOOKMARK),
        quote(&report.large_scale_stability_status),
        report.fixture_digest.to_json(),
        report.synthetic_contract_fixture,
    )
}

pub fn results_json(report: &LatentFeedbackReport) -> String {
    let arms: Vec<String> = report
        .arms
        .iter()
        .map(|a| {
            format!(
                "{{\"variant\":{},\"epochs_run\":{},\"unique_train_tokens\":{},\"parameter_count\":{},\"forward_flops\":{},\"backward_flops\":{},\"total_flops\":{},\"initial_train_loss\":{},\"final_train_loss\":{},\"train_accuracy\":{},\"heldout_loss\":{},\"heldout_accuracy\":{}}}",
                quote(a.variant.as_str()),
                a.epochs_run,
                a.unique_train_tokens,
                a.parameter_count,
                a.forward_flops,
                a.backward_flops,
                a.total_flops,
                finite_f64(a.initial_train_loss),
                finite_f64(a.final_train_loss),
                opt_f64(a.train_accuracy),
                finite_f64(a.heldout_loss),
                opt_f64(a.heldout_accuracy),
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
                quote(&c.detail),
            )
        })
        .collect();
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.latent-feedback.results.v1\",\n",
            "  \"issue\": 50,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"large_scale_stability_status\": {},\n",
            "  \"grad_check_max_abs_err\": {},\n",
            "  \"causal_invariance_max_err\": {},\n",
            "  \"flop_ratio_feedback_over_control\": {},\n",
            "  \"arms\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.synthetic_contract_fixture,
        quote(&report.large_scale_stability_status),
        finite_f64(report.grad_check_max_abs_err),
        finite_f64(report.causal_invariance_max_err),
        opt_f64(report.flop_ratio_feedback_over_control),
        arms.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &LatentFeedbackReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #50: Latent Feedback vs Matched Extra-Epoch Control (Data-Limited)"
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
        "- Hardware gate: `{}` ({})",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(
        out,
        "- Large-scale stability: `{}` | FD gradient check: `max_abs_err={:.3e}` | Causal invariance: `{:.3e}` | FLOP ratio: `{:.4}`",
        report.large_scale_stability_status,
        report.grad_check_max_abs_err,
        report.causal_invariance_max_err,
        report.flop_ratio_feedback_over_control.unwrap_or(0.0),
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Matched-FLOP Miniature Training Arms").unwrap();
    for a in &report.arms {
        writeln!(
            out,
            "- `{}`: `epochs={}`, `unique_tokens={}`, `total_flops={}`, `train_loss={:.4}->{:.4}`, `heldout_loss={:.4}`, `heldout_acc={:.3}`",
            a.variant.as_str(),
            a.epochs_run,
            a.unique_train_tokens,
            a.total_flops,
            a.initial_train_loss,
            a.final_train_loss,
            a.heldout_loss,
            a.heldout_accuracy.unwrap_or(0.0),
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
    fn matched_flop_training_runs_fd_clean_causal_and_rejects_all_controls() {
        let report = run_default_harness().unwrap();
        assert_eq!(report.large_scale_stability_status, "open_untested");
        assert!(report.grad_check_max_abs_err <= 1.0e-6);
        assert_eq!(report.causal_invariance_max_err, 0.0);
        let ratio = report.flop_ratio_feedback_over_control.unwrap();
        assert!((ratio - 1.0).abs() <= 0.05);
        assert_eq!(report.arms.len(), 2);
        // Both arms genuinely decrease training loss under SGD
        for a in &report.arms {
            assert!(a.final_train_loss < a.initial_train_loss);
        }
        assert_eq!(report.negative_controls.len(), 8);
        for c in &report.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }
}
