//! Issue #51: Measure SimpleHC against mHC training overhead beside the attention-residual ablations.
//!
//! Pinned source basis (strictly unverified non-gating motivation):
//! - Source bookmark: X post `2119613538515960212` reporting DeepSeek mHC overhead and introducing
//!   SimpleHC. Primary paper/repo was NOT located in the sweep; `primary_source_pin_status` is
//!   explicitly recorded as `"post_only_unlocated_paper_repo"` and `hardware_gate` is
//!   `HardwareGate::blocked_on_artifacts(...)`.
//! - Related issue: `#33` (`BlockAttnResRef` beside `src/inkling_residuals.rs`).
//! - The post's 59-100% PyTorch overhead figures are unverified external claims and are NOT
//!   imported as local pass/fail gates.
//!
//! Provides:
//! - Real miniature multi-stream (`N_STREAMS = 4`, `D_MODEL = 6`, `NUM_LAYERS = 4`) hyper-connection
//!   implementations comparing `DeepSeekMhc` (real iterative Sinkhorn-Knopp Birkhoff polytope
//!   projection with verified row/col sum error `<= 1e-6`), `SimpleHc` (lightweight single-pass
//!   row-normalized stream mixing), and `BlockAttnResRef` (beside `#33`) at identically matched
//!   parameter counts and identically matched training tokens.
//! - Independent transparent scalar references vs candidate forward paths (`<= 1e-12` parity) for
//!   both `DeepSeekMhc` and `SimpleHc`.
//! - Exact analytical backpropagation (including exact reverse-mode differentiation through
//!   iterative Sinkhorn-Knopp) + real multi-step SGD training, exact op/FLOP accounting
//!   (`sinkhorn_ops`, `total_step_flops`, `mhc_overhead_vs_simplehc_pct`), train & held-out loss,
//!   and 8 executable negative controls.

#![allow(clippy::needless_range_loop)]

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, sigmoid};
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const SOURCE_BOOKMARK_POST_ID: &str = "2119613538515960212";
pub const PRIMARY_SOURCE_PIN_STATUS_UNLOCATED: &str = "post_only_unlocated_paper_repo";
pub const RELATED_ISSUES: &[u32] = &[33];

pub const N_STREAMS: usize = 4;
pub const D_MODEL: usize = 6;
pub const NUM_LAYERS: usize = 4;
pub const BLOCK_LAYERS: usize = 2;
pub const BLOCK_MLP_HIDDEN: usize = 7;
pub const VOCAB_SIZE: usize = 6;
pub const SINKHORN_ITERS: usize = 20;
const RMS_EPS: f64 = 1.0e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HcVariant {
    DeepSeekMhc,
    SimpleHc,
}

impl HcVariant {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DeepSeekMhc => "deepseek_mhc",
            Self::SimpleHc => "simple_hc",
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CandidateFault {
    pub corrupt_stream_mix: bool,
}

#[derive(Debug, Clone, Default)]
pub struct OpCounter {
    pub sinkhorn_ops: u64,
    pub total_flops: u64,
}

impl OpCounter {
    fn add_sinkhorn(&mut self, ops: u64) {
        self.sinkhorn_ops = self.sinkhorn_ops.saturating_add(ops);
        self.total_flops = self.total_flops.saturating_add(ops);
    }

    fn add_flops(&mut self, ops: u64) {
        self.total_flops = self.total_flops.saturating_add(ops);
    }
}

#[derive(Debug, Clone)]
pub struct SinkhornCache {
    pub a_hist: [[[f64; N_STREAMS]; N_STREAMS]; SINKHORN_ITERS + 1],
    pub r_hist: [[[f64; N_STREAMS]; N_STREAMS]; SINKHORN_ITERS],
    pub row_sums: [[f64; N_STREAMS]; SINKHORN_ITERS],
    pub col_sums: [[f64; N_STREAMS]; SINKHORN_ITERS],
}

/// Real iterative Sinkhorn-Knopp projection onto the Birkhoff polytope of doubly-stochastic matrices.
pub fn sinkhorn_knopp_forward(
    logits: &[[f64; N_STREAMS]; N_STREAMS],
    iters: usize,
    ops: &mut OpCounter,
) -> Result<([[f64; N_STREAMS]; N_STREAMS], SinkhornCache), String> {
    if iters > SINKHORN_ITERS {
        return Err(format!(
            "sinkhorn iters {iters} exceeds max {SINKHORN_ITERS}"
        ));
    }
    let mut max_l = f64::NEG_INFINITY;
    for row in logits {
        for &v in row {
            if !v.is_finite() {
                return Err("non-finite logit in Sinkhorn-Knopp projection".into());
            }
            if v > max_l {
                max_l = v;
            }
        }
    }
    let mut cache = SinkhornCache {
        a_hist: [[[0.0; N_STREAMS]; N_STREAMS]; SINKHORN_ITERS + 1],
        r_hist: [[[0.0; N_STREAMS]; N_STREAMS]; SINKHORN_ITERS],
        row_sums: [[0.0; N_STREAMS]; SINKHORN_ITERS],
        col_sums: [[0.0; N_STREAMS]; SINKHORN_ITERS],
    };
    for i in 0..N_STREAMS {
        for j in 0..N_STREAMS {
            let e = (logits[i][j] - max_l).exp();
            if !e.is_finite() || e <= 0.0 {
                return Err("non-positive or non-finite exp in Sinkhorn-Knopp".into());
            }
            cache.a_hist[0][i][j] = e;
        }
    }
    ops.add_sinkhorn((2 * N_STREAMS * N_STREAMS) as u64);

    for k in 0..iters {
        for i in 0..N_STREAMS {
            let mut rs = 0.0_f64;
            for j in 0..N_STREAMS {
                rs += cache.a_hist[k][i][j];
            }
            if !rs.is_finite() || rs <= 0.0 {
                return Err("non-positive row sum in Sinkhorn-Knopp".into());
            }
            cache.row_sums[k][i] = rs;
            for j in 0..N_STREAMS {
                cache.r_hist[k][i][j] = cache.a_hist[k][i][j] / rs;
            }
        }
        for j in 0..N_STREAMS {
            let mut cs = 0.0_f64;
            for i in 0..N_STREAMS {
                cs += cache.r_hist[k][i][j];
            }
            if !cs.is_finite() || cs <= 0.0 {
                return Err("non-positive col sum in Sinkhorn-Knopp".into());
            }
            cache.col_sums[k][j] = cs;
            for i in 0..N_STREAMS {
                cache.a_hist[k + 1][i][j] = cache.r_hist[k][i][j] / cs;
            }
        }
        ops.add_sinkhorn((4 * N_STREAMS * N_STREAMS) as u64);
    }

    Ok((cache.a_hist[iters], cache))
}

fn sinkhorn_knopp_backward(
    cache: &SinkhornCache,
    iters: usize,
    d_out: &[[f64; N_STREAMS]; N_STREAMS],
    ops: &mut OpCounter,
) -> [[f64; N_STREAMS]; N_STREAMS] {
    let mut da = *d_out;
    for k in (0..iters).rev() {
        let mut dr = [[0.0_f64; N_STREAMS]; N_STREAMS];
        for j in 0..N_STREAMS {
            let mut dot_c = 0.0_f64;
            for i in 0..N_STREAMS {
                dot_c += da[i][j] * cache.a_hist[k + 1][i][j];
            }
            let cs = cache.col_sums[k][j];
            for i in 0..N_STREAMS {
                dr[i][j] = (da[i][j] - dot_c) / cs;
            }
        }
        let mut da_prev = [[0.0_f64; N_STREAMS]; N_STREAMS];
        for i in 0..N_STREAMS {
            let mut dot_r = 0.0_f64;
            for j in 0..N_STREAMS {
                dot_r += dr[i][j] * cache.r_hist[k][i][j];
            }
            let rs = cache.row_sums[k][i];
            for j in 0..N_STREAMS {
                da_prev[i][j] = (dr[i][j] - dot_r) / rs;
            }
        }
        da = da_prev;
        ops.add_sinkhorn((8 * N_STREAMS * N_STREAMS) as u64);
    }
    let mut d_logits = [[0.0_f64; N_STREAMS]; N_STREAMS];
    for i in 0..N_STREAMS {
        for j in 0..N_STREAMS {
            d_logits[i][j] = da[i][j] * cache.a_hist[0][i][j];
        }
    }
    ops.add_sinkhorn((2 * N_STREAMS * N_STREAMS) as u64);
    d_logits
}

/// Verifies doubly-stochastic Birkhoff polytope invariants (non-negative entries, all row and col
/// sums within `tol` of 1.0).
pub fn verify_doubly_stochastic(
    mat: &[[f64; N_STREAMS]; N_STREAMS],
    tol: f64,
) -> Result<(f64, OutcomeStatus), String> {
    if !tol.is_finite() || tol < 0.0 {
        return Err("non-finite or negative doubly-stochastic tolerance".into());
    }
    let mut max_err = 0.0_f64;
    let mut all_non_neg = true;
    for row in mat {
        let mut rs = 0.0_f64;
        for &v in row {
            if !v.is_finite() {
                return Err("non-finite entry in Birkhoff matrix".into());
            }
            if v < -1.0e-12 {
                all_non_neg = false;
            }
            rs += v;
        }
        let err = (rs - 1.0).abs();
        if err > max_err {
            max_err = err;
        }
    }
    for j in 0..N_STREAMS {
        let mut cs = 0.0_f64;
        for row in mat {
            cs += row[j];
        }
        let err = (cs - 1.0).abs();
        if err > max_err {
            max_err = err;
        }
    }
    let status = if all_non_neg && max_err <= tol {
        OutcomeStatus::Accepted
    } else {
        OutcomeStatus::Invalid
    };
    Ok((max_err, status))
}

fn softmax_vec(logits: &[f64]) -> Result<(Vec<f64>, f64), String> {
    if logits.is_empty() {
        return Err("empty softmax input".into());
    }
    let mut max_l = f64::NEG_INFINITY;
    for &v in logits {
        if !v.is_finite() {
            return Err("non-finite logit in softmax".into());
        }
        if v > max_l {
            max_l = v;
        }
    }
    let mut exps = Vec::with_capacity(logits.len());
    let mut sum_e = 0.0_f64;
    for &v in logits {
        let e = (v - max_l).exp();
        sum_e += e;
        exps.push(e);
    }
    if !sum_e.is_finite() || sum_e <= 0.0 {
        return Err("non-positive softmax denominator".into());
    }
    let lse = max_l + sum_e.ln();
    for e in &mut exps {
        *e /= sum_e;
    }
    Ok((exps, lse))
}

fn rms_norm_d(x: &[f64; D_MODEL], ops: &mut OpCounter) -> Result<([f64; D_MODEL], f64), String> {
    let mut sq = 0.0_f64;
    for &v in x {
        if !v.is_finite() {
            return Err("non-finite input to RMSNorm".into());
        }
        sq += v * v;
    }
    let inv = 1.0 / (sq / (D_MODEL as f64) + RMS_EPS).sqrt();
    if !inv.is_finite() {
        return Err("non-finite RMSNorm scale".into());
    }
    let mut y = [0.0_f64; D_MODEL];
    for i in 0..D_MODEL {
        y[i] = x[i] * inv;
    }
    ops.add_flops((3 * D_MODEL) as u64);
    Ok((y, inv))
}

fn rms_norm_d_bwd(
    x: &[f64; D_MODEL],
    inv: f64,
    dy: &[f64; D_MODEL],
    ops: &mut OpCounter,
) -> [f64; D_MODEL] {
    let mut dot = 0.0_f64;
    for i in 0..D_MODEL {
        dot += dy[i] * x[i];
    }
    let coeff = (inv * inv * inv / (D_MODEL as f64)) * dot;
    let mut dx = [0.0_f64; D_MODEL];
    for i in 0..D_MODEL {
        dx[i] = dy[i] * inv - x[i] * coeff;
    }
    ops.add_flops((6 * D_MODEL) as u64);
    dx
}

/// Lightweight single-pass row-normalized mixing for SimpleHC (no iterative Sinkhorn-Knopp).
fn simplehc_row_mix_forward(
    logits: &[[f64; N_STREAMS]; N_STREAMS],
    ops: &mut OpCounter,
) -> Result<[[f64; N_STREAMS]; N_STREAMS], String> {
    let mut out = [[0.0_f64; N_STREAMS]; N_STREAMS];
    for i in 0..N_STREAMS {
        let (row_p, _) = softmax_vec(&logits[i])?;
        out[i].copy_from_slice(&row_p[..N_STREAMS]);
    }
    ops.add_flops((3 * N_STREAMS * N_STREAMS) as u64);
    Ok(out)
}

fn simplehc_row_mix_backward(
    h_res: &[[f64; N_STREAMS]; N_STREAMS],
    d_hres: &[[f64; N_STREAMS]; N_STREAMS],
    ops: &mut OpCounter,
) -> [[f64; N_STREAMS]; N_STREAMS] {
    let mut d_logits = [[0.0_f64; N_STREAMS]; N_STREAMS];
    for i in 0..N_STREAMS {
        let mut dot = 0.0_f64;
        for j in 0..N_STREAMS {
            dot += d_hres[i][j] * h_res[i][j];
        }
        for j in 0..N_STREAMS {
            d_logits[i][j] = h_res[i][j] * (d_hres[i][j] - dot);
        }
    }
    ops.add_flops((4 * N_STREAMS * N_STREAMS) as u64);
    d_logits
}

#[derive(Debug, Clone)]
pub struct HcLayerParams {
    pub h_res_logits: [[f64; N_STREAMS]; N_STREAMS],
    pub h_pre_logits: [f64; N_STREAMS],
    pub h_post_logits: [f64; N_STREAMS],
    pub w1: [[f64; D_MODEL]; D_MODEL],
    pub w2: [[f64; D_MODEL]; D_MODEL],
}

impl HcLayerParams {
    fn zeros() -> Self {
        Self {
            h_res_logits: [[0.0; N_STREAMS]; N_STREAMS],
            h_pre_logits: [0.0; N_STREAMS],
            h_post_logits: [0.0; N_STREAMS],
            w1: [[0.0; D_MODEL]; D_MODEL],
            w2: [[0.0; D_MODEL]; D_MODEL],
        }
    }

    fn init(seed: usize) -> Self {
        let mut lp = Self::zeros();
        for i in 0..N_STREAMS {
            for j in 0..N_STREAMS {
                let diag = if i == j { 0.8 } else { 0.0 };
                lp.h_res_logits[i][j] = diag + det_weight(seed + i * 11, j, 0.35);
            }
            lp.h_pre_logits[i] = det_weight(seed + 100, i, 0.4);
            lp.h_post_logits[i] = det_weight(seed + 150, i, 0.4);
        }
        for r in 0..D_MODEL {
            for c in 0..D_MODEL {
                lp.w1[r][c] = det_weight(seed + 200 + r * 13, c, 0.45);
                lp.w2[r][c] = det_weight(seed + 400 + r * 17, c, 0.35);
            }
        }
        lp
    }
}

#[derive(Debug, Clone)]
pub struct HyperConnModel {
    pub variant: HcVariant,
    pub embed: [[f64; D_MODEL]; VOCAB_SIZE],
    pub layers: [HcLayerParams; NUM_LAYERS],
    pub out_scale: [f64; D_MODEL],
    pub w_head: [[f64; D_MODEL]; VOCAB_SIZE],
}

impl HyperConnModel {
    pub fn init(seed: usize, variant: HcVariant) -> Self {
        let mut embed = [[0.0; D_MODEL]; VOCAB_SIZE];
        for v in 0..VOCAB_SIZE {
            for d in 0..D_MODEL {
                embed[v][d] = det_weight(seed + v * 19, d, 0.5);
            }
        }
        let layers = [
            HcLayerParams::init(seed + 1000),
            HcLayerParams::init(seed + 2000),
            HcLayerParams::init(seed + 3000),
            HcLayerParams::init(seed + 4000),
        ];
        let mut out_scale = [1.0; D_MODEL];
        for (d, s) in out_scale.iter_mut().enumerate() {
            *s = 1.0 + det_weight(seed + 5000, d, 0.1);
        }
        let mut w_head = [[0.0; D_MODEL]; VOCAB_SIZE];
        for v in 0..VOCAB_SIZE {
            for d in 0..D_MODEL {
                w_head[v][d] = det_weight(seed + 6000 + v * 23, d, 0.4);
            }
        }
        Self {
            variant,
            embed,
            layers,
            out_scale,
            w_head,
        }
    }

    pub fn zeros_like(&self) -> Self {
        Self {
            variant: self.variant,
            embed: [[0.0; D_MODEL]; VOCAB_SIZE],
            layers: [
                HcLayerParams::zeros(),
                HcLayerParams::zeros(),
                HcLayerParams::zeros(),
                HcLayerParams::zeros(),
            ],
            out_scale: [0.0; D_MODEL],
            w_head: [[0.0; D_MODEL]; VOCAB_SIZE],
        }
    }

    pub fn param_count(&self) -> usize {
        let per_layer = N_STREAMS * N_STREAMS + 2 * N_STREAMS + 2 * D_MODEL * D_MODEL;
        VOCAB_SIZE * D_MODEL + NUM_LAYERS * per_layer + D_MODEL + VOCAB_SIZE * D_MODEL
    }

    pub fn apply_sgd(&mut self, grad: &HyperConnModel, lr: f64) {
        for v in 0..VOCAB_SIZE {
            for d in 0..D_MODEL {
                self.embed[v][d] -= lr * grad.embed[v][d];
                self.w_head[v][d] -= lr * grad.w_head[v][d];
            }
        }
        for d in 0..D_MODEL {
            self.out_scale[d] -= lr * grad.out_scale[d];
        }
        for l in 0..NUM_LAYERS {
            let lp = &mut self.layers[l];
            let lg = &grad.layers[l];
            for i in 0..N_STREAMS {
                for j in 0..N_STREAMS {
                    lp.h_res_logits[i][j] -= lr * lg.h_res_logits[i][j];
                }
                lp.h_pre_logits[i] -= lr * lg.h_pre_logits[i];
                lp.h_post_logits[i] -= lr * lg.h_post_logits[i];
            }
            for r in 0..D_MODEL {
                for c in 0..D_MODEL {
                    lp.w1[r][c] -= lr * lg.w1[r][c];
                    lp.w2[r][c] -= lr * lg.w2[r][c];
                }
            }
        }
    }
}

/// Completely independent transparent scalar reference forward pass for DeepSeek mHC and SimpleHC
/// over a compositional 2-token input `(tok_a, tok_b)` (`0.5 * (embed[tok_a] + embed[tok_b])`).
pub fn forward_hc_scalar_ref(
    model: &HyperConnModel,
    input_pair: (usize, usize),
) -> Result<[f64; VOCAB_SIZE], String> {
    let (tok_a, tok_b) = input_pair;
    if tok_a >= VOCAB_SIZE || tok_b >= VOCAB_SIZE {
        return Err("token out of range".into());
    }
    let mut x = [[0.0_f64; D_MODEL]; N_STREAMS];
    for s in 0..N_STREAMS {
        for d in 0..D_MODEL {
            let va = model.embed[tok_a][d];
            let vb = model.embed[tok_b][d];
            if !va.is_finite() || !vb.is_finite() {
                return Err("non-finite embed in scalar ref".into());
            }
            x[s][d] = 0.5 * (va + vb);
        }
    }
    for l in 0..NUM_LAYERS {
        let lp = &model.layers[l];
        let h_res = match model.variant {
            HcVariant::DeepSeekMhc => {
                // Independent Sinkhorn loop
                let mut m = f64::NEG_INFINITY;
                for row in &lp.h_res_logits {
                    for &v in row {
                        if !v.is_finite() {
                            return Err("non-finite h_res logit in scalar ref".into());
                        }
                        if v > m {
                            m = v;
                        }
                    }
                }
                let mut a = [[0.0_f64; N_STREAMS]; N_STREAMS];
                for i in 0..N_STREAMS {
                    for j in 0..N_STREAMS {
                        a[i][j] = (lp.h_res_logits[i][j] - m).exp();
                    }
                }
                for _ in 0..SINKHORN_ITERS {
                    for row in &mut a {
                        let rs: f64 = row.iter().sum();
                        if !rs.is_finite() || rs <= 0.0 {
                            return Err("invalid row sum in scalar ref".into());
                        }
                        for v in row.iter_mut() {
                            *v /= rs;
                        }
                    }
                    for j in 0..N_STREAMS {
                        let mut cs = 0.0_f64;
                        for row in &a {
                            cs += row[j];
                        }
                        if !cs.is_finite() || cs <= 0.0 {
                            return Err("invalid col sum in scalar ref".into());
                        }
                        for row in &mut a {
                            row[j] /= cs;
                        }
                    }
                }
                a
            }
            HcVariant::SimpleHc => {
                let mut a = [[0.0_f64; N_STREAMS]; N_STREAMS];
                for i in 0..N_STREAMS {
                    let mut m = f64::NEG_INFINITY;
                    for &v in &lp.h_res_logits[i] {
                        if !v.is_finite() {
                            return Err("non-finite simplehc logit".into());
                        }
                        if v > m {
                            m = v;
                        }
                    }
                    let mut s = 0.0_f64;
                    for j in 0..N_STREAMS {
                        a[i][j] = (lp.h_res_logits[i][j] - m).exp();
                        s += a[i][j];
                    }
                    for j in 0..N_STREAMS {
                        a[i][j] /= s;
                    }
                }
                a
            }
        };

        // h_pre softmax
        let mut m_pre = f64::NEG_INFINITY;
        for &v in &lp.h_pre_logits {
            if !v.is_finite() {
                return Err("non-finite h_pre logit".into());
            }
            if v > m_pre {
                m_pre = v;
            }
        }
        let mut h_pre = [0.0_f64; N_STREAMS];
        let mut sum_pre = 0.0_f64;
        for s in 0..N_STREAMS {
            h_pre[s] = (lp.h_pre_logits[s] - m_pre).exp();
            sum_pre += h_pre[s];
        }
        for v in &mut h_pre {
            *v /= sum_pre;
        }

        // h_post = 2 * sigmoid
        let mut h_post = [0.0_f64; N_STREAMS];
        for s in 0..N_STREAMS {
            if !lp.h_post_logits[s].is_finite() {
                return Err("non-finite h_post logit".into());
            }
            h_post[s] = 2.0 * sigmoid(lp.h_post_logits[s]);
        }

        // Read u = sum_s h_pre[s] * x[s]
        let mut u = [0.0_f64; D_MODEL];
        for s in 0..N_STREAMS {
            for d in 0..D_MODEL {
                u[d] += h_pre[s] * x[s][d];
            }
        }
        // RMSNorm(u)
        let sq: f64 = u.iter().map(|v| v * v).sum();
        let inv = 1.0 / (sq / (D_MODEL as f64) + RMS_EPS).sqrt();
        let mut u_n = [0.0_f64; D_MODEL];
        for d in 0..D_MODEL {
            u_n[d] = u[d] * inv;
        }
        // Sublayer MLP
        let mut act = [0.0_f64; D_MODEL];
        for r in 0..D_MODEL {
            let mut acc = 0.0_f64;
            for c in 0..D_MODEL {
                acc += lp.w1[r][c] * u_n[c];
            }
            act[r] = acc.tanh();
        }
        let mut delta = [0.0_f64; D_MODEL];
        for r in 0..D_MODEL {
            let mut acc = 0.0_f64;
            for c in 0..D_MODEL {
                acc += lp.w2[r][c] * act[c];
            }
            delta[r] = acc;
        }

        // Mix + write back
        let mut x_next = [[0.0_f64; D_MODEL]; N_STREAMS];
        for s in 0..N_STREAMS {
            for d in 0..D_MODEL {
                let mut m_val = 0.0_f64;
                for r in 0..N_STREAMS {
                    m_val += h_res[s][r] * x[r][d];
                }
                x_next[s][d] = m_val + h_post[s] * delta[d];
            }
        }
        x = x_next;
    }

    let mut avg = [0.0_f64; D_MODEL];
    for s in 0..N_STREAMS {
        for d in 0..D_MODEL {
            avg[d] += x[s][d] / (N_STREAMS as f64);
        }
    }
    let sq: f64 = avg.iter().map(|v| v * v).sum();
    let inv = 1.0 / (sq / (D_MODEL as f64) + RMS_EPS).sqrt();
    let mut z = [0.0_f64; D_MODEL];
    for d in 0..D_MODEL {
        z[d] = avg[d] * inv * model.out_scale[d];
    }
    let mut logits = [0.0_f64; VOCAB_SIZE];
    for c in 0..VOCAB_SIZE {
        let mut acc = 0.0_f64;
        for d in 0..D_MODEL {
            acc += model.w_head[c][d] * z[d];
        }
        if !acc.is_finite() {
            return Err("non-finite scalar ref logit".into());
        }
        logits[c] = acc;
    }
    Ok(logits)
}

#[derive(Debug, Clone)]
struct HcLayerCache {
    x_in: [[f64; D_MODEL]; N_STREAMS],
    h_res: [[f64; N_STREAMS]; N_STREAMS],
    sinkhorn: Option<SinkhornCache>,
    h_pre: [f64; N_STREAMS],
    h_post_sig: [f64; N_STREAMS],
    h_post: [f64; N_STREAMS],
    u: [f64; D_MODEL],
    u_norm: [f64; D_MODEL],
    u_inv: f64,
    act: [f64; D_MODEL],
    delta: [f64; D_MODEL],
}

#[derive(Debug, Clone)]
pub struct HcForwardOut {
    pub logits: [f64; VOCAB_SIZE],
    pub max_birkhoff_err: Option<f64>,
    pub ops: OpCounter,
}

type HcInternalForward = (
    HcForwardOut,
    Vec<HcLayerCache>,
    [f64; D_MODEL],
    [f64; D_MODEL],
    f64,
);

fn forward_hc_candidate_internal(
    model: &HyperConnModel,
    input_pair: (usize, usize),
    fault: CandidateFault,
) -> Result<HcInternalForward, String> {
    let (tok_a, tok_b) = input_pair;
    if tok_a >= VOCAB_SIZE || tok_b >= VOCAB_SIZE {
        return Err("token out of range".into());
    }
    let mut ops = OpCounter::default();
    let mut x = [[0.0_f64; D_MODEL]; N_STREAMS];
    for s in 0..N_STREAMS {
        for d in 0..D_MODEL {
            let va = model.embed[tok_a][d];
            let vb = model.embed[tok_b][d];
            if !va.is_finite() || !vb.is_finite() {
                return Err("non-finite embed weight".into());
            }
            x[s][d] = 0.5 * (va + vb);
        }
    }

    let mut caches = Vec::with_capacity(NUM_LAYERS);
    let mut max_birkhoff_err: Option<f64> = None;

    for l in 0..NUM_LAYERS {
        let lp = &model.layers[l];
        let (h_res, sinkhorn_cache) = match model.variant {
            HcVariant::DeepSeekMhc => {
                let (m, sc) = sinkhorn_knopp_forward(&lp.h_res_logits, SINKHORN_ITERS, &mut ops)?;
                let (err, status) = verify_doubly_stochastic(&m, 1.0e-6)?;
                if status != OutcomeStatus::Accepted {
                    return Err(format!("Sinkhorn Birkhoff error {err:.3e} exceeded 1e-6"));
                }
                max_birkhoff_err = Some(match max_birkhoff_err {
                    None => err,
                    Some(prev) => {
                        if err > prev {
                            err
                        } else {
                            prev
                        }
                    }
                });
                (m, Some(sc))
            }
            HcVariant::SimpleHc => (simplehc_row_mix_forward(&lp.h_res_logits, &mut ops)?, None),
        };

        let (h_pre_vec, _) = softmax_vec(&lp.h_pre_logits)?;
        let mut h_pre = [0.0_f64; N_STREAMS];
        h_pre.copy_from_slice(&h_pre_vec[..N_STREAMS]);
        ops.add_flops((3 * N_STREAMS) as u64);

        let mut h_post_sig = [0.0_f64; N_STREAMS];
        let mut h_post = [0.0_f64; N_STREAMS];
        for s in 0..N_STREAMS {
            if !lp.h_post_logits[s].is_finite() {
                return Err("non-finite h_post logit".into());
            }
            let sig = sigmoid(lp.h_post_logits[s]);
            h_post_sig[s] = sig;
            h_post[s] = 2.0 * sig;
        }
        ops.add_flops((3 * N_STREAMS) as u64);

        let mut u = [0.0_f64; D_MODEL];
        for s in 0..N_STREAMS {
            for d in 0..D_MODEL {
                u[d] += h_pre[s] * x[s][d];
            }
        }
        ops.add_flops((2 * N_STREAMS * D_MODEL) as u64);

        let (u_norm, u_inv) = rms_norm_d(&u, &mut ops)?;

        let mut act = [0.0_f64; D_MODEL];
        for r in 0..D_MODEL {
            let mut acc = 0.0_f64;
            for c in 0..D_MODEL {
                if !lp.w1[r][c].is_finite() {
                    return Err("non-finite w1 weight".into());
                }
                acc += lp.w1[r][c] * u_norm[c];
            }
            act[r] = acc.tanh();
        }
        ops.add_flops((2 * D_MODEL * D_MODEL + 2 * D_MODEL) as u64);

        let mut delta = [0.0_f64; D_MODEL];
        for r in 0..D_MODEL {
            let mut acc = 0.0_f64;
            for c in 0..D_MODEL {
                if !lp.w2[r][c].is_finite() {
                    return Err("non-finite w2 weight".into());
                }
                acc += lp.w2[r][c] * act[c];
            }
            delta[r] = acc;
        }
        ops.add_flops((2 * D_MODEL * D_MODEL) as u64);

        let x_in = x;
        let mut x_next = [[0.0_f64; D_MODEL]; N_STREAMS];
        for s in 0..N_STREAMS {
            for d in 0..D_MODEL {
                let mut m_val = 0.0_f64;
                for r in 0..N_STREAMS {
                    m_val += h_res[s][r] * x_in[r][d];
                }
                if fault.corrupt_stream_mix && l == 0 && s == 0 {
                    m_val *= 0.85; // Deliberate candidate bug for NC6!
                }
                x_next[s][d] = m_val + h_post[s] * delta[d];
            }
        }
        ops.add_flops((2 * N_STREAMS * N_STREAMS * D_MODEL + 2 * N_STREAMS * D_MODEL) as u64);

        caches.push(HcLayerCache {
            x_in,
            h_res,
            sinkhorn: sinkhorn_cache,
            h_pre,
            h_post_sig,
            h_post,
            u,
            u_norm,
            u_inv,
            act,
            delta,
        });
        x = x_next;
    }

    let mut avg = [0.0_f64; D_MODEL];
    for s in 0..N_STREAMS {
        for d in 0..D_MODEL {
            avg[d] += x[s][d] / (N_STREAMS as f64);
        }
    }
    let (avg_norm, avg_inv) = rms_norm_d(&avg, &mut ops)?;
    let mut z = [0.0_f64; D_MODEL];
    for d in 0..D_MODEL {
        if !model.out_scale[d].is_finite() {
            return Err("non-finite out_scale".into());
        }
        z[d] = avg_norm[d] * model.out_scale[d];
    }
    let mut logits = [0.0_f64; VOCAB_SIZE];
    for c in 0..VOCAB_SIZE {
        let mut acc = 0.0_f64;
        for d in 0..D_MODEL {
            if !model.w_head[c][d].is_finite() {
                return Err("non-finite w_head weight".into());
            }
            acc += model.w_head[c][d] * z[d];
        }
        if !acc.is_finite() {
            return Err("non-finite candidate logit".into());
        }
        logits[c] = acc;
    }
    ops.add_flops((2 * VOCAB_SIZE * D_MODEL) as u64);

    Ok((
        HcForwardOut {
            logits,
            max_birkhoff_err,
            ops,
        },
        caches,
        avg,
        avg_norm,
        avg_inv,
    ))
}

pub fn forward_hc_candidate(
    model: &HyperConnModel,
    input_pair: (usize, usize),
    fault: CandidateFault,
) -> Result<HcForwardOut, String> {
    let (out, _, _, _, _) = forward_hc_candidate_internal(model, input_pair, fault)?;
    Ok(out)
}

/// Exact analytical forward + backward step for `DeepSeekMhc` and `SimpleHc`.
pub fn forward_backward_hc(
    model: &HyperConnModel,
    input_pair: (usize, usize),
    target_tok: usize,
) -> Result<(f64, HyperConnModel, OpCounter, Option<f64>), String> {
    if target_tok >= VOCAB_SIZE {
        return Err("target token out of range".into());
    }
    let (fwd, caches, avg, avg_norm, avg_inv) =
        forward_hc_candidate_internal(model, input_pair, CandidateFault::default())?;
    let mut ops = fwd.ops;
    let mut grad = model.zeros_like();

    let (probs, lse) = softmax_vec(&fwd.logits)?;
    let loss = lse - fwd.logits[target_tok];
    if !loss.is_finite() {
        return Err("non-finite cross-entropy loss".into());
    }

    let mut d_logits = [0.0_f64; VOCAB_SIZE];
    for c in 0..VOCAB_SIZE {
        d_logits[c] = probs[c] - if c == target_tok { 1.0 } else { 0.0 };
    }
    let mut d_z = [0.0_f64; D_MODEL];
    for c in 0..VOCAB_SIZE {
        for d in 0..D_MODEL {
            let zd = avg_norm[d] * model.out_scale[d];
            grad.w_head[c][d] += d_logits[c] * zd;
            d_z[d] += model.w_head[c][d] * d_logits[c];
        }
    }
    ops.add_flops((4 * VOCAB_SIZE * D_MODEL) as u64);

    let mut d_avg_norm = [0.0_f64; D_MODEL];
    for d in 0..D_MODEL {
        grad.out_scale[d] += d_z[d] * avg_norm[d];
        d_avg_norm[d] = d_z[d] * model.out_scale[d];
    }
    let d_avg = rms_norm_d_bwd(&avg, avg_inv, &d_avg_norm, &mut ops);
    let mut dx = [[0.0_f64; D_MODEL]; N_STREAMS];
    for s in 0..N_STREAMS {
        for d in 0..D_MODEL {
            dx[s][d] = d_avg[d] / (N_STREAMS as f64);
        }
    }

    for l in (0..NUM_LAYERS).rev() {
        let lp = &model.layers[l];
        let lg = &mut grad.layers[l];
        let c = &caches[l];

        // Backward through x_next[s][d] = sum_r h_res[s][r] * x_in[r][d] + h_post[s] * delta[d]
        let mut d_hres = [[0.0_f64; N_STREAMS]; N_STREAMS];
        let mut dx_prev = [[0.0_f64; D_MODEL]; N_STREAMS];
        let mut d_hpost = [0.0_f64; N_STREAMS];
        let mut d_delta = [0.0_f64; D_MODEL];
        for s in 0..N_STREAMS {
            for d in 0..D_MODEL {
                let g = dx[s][d];
                d_hpost[s] += g * c.delta[d];
                d_delta[d] += g * c.h_post[s];
                for r in 0..N_STREAMS {
                    d_hres[s][r] += g * c.x_in[r][d];
                    dx_prev[r][d] += c.h_res[s][r] * g;
                }
            }
        }
        ops.add_flops((4 * N_STREAMS * N_STREAMS * D_MODEL + 4 * N_STREAMS * D_MODEL) as u64);

        // h_post_logits backward: h_post = 2 * sigmoid
        for s in 0..N_STREAMS {
            let sig = c.h_post_sig[s];
            lg.h_post_logits[s] += d_hpost[s] * 2.0 * sig * (1.0 - sig);
        }

        // h_res_logits backward
        let d_hres_logits = match model.variant {
            HcVariant::DeepSeekMhc => {
                let sc = c
                    .sinkhorn
                    .as_ref()
                    .ok_or_else(|| "missing Sinkhorn cache".to_string())?;
                sinkhorn_knopp_backward(sc, SINKHORN_ITERS, &d_hres, &mut ops)
            }
            HcVariant::SimpleHc => simplehc_row_mix_backward(&c.h_res, &d_hres, &mut ops),
        };
        for i in 0..N_STREAMS {
            for j in 0..N_STREAMS {
                lg.h_res_logits[i][j] += d_hres_logits[i][j];
            }
        }

        // Sublayer MLP backward
        let mut d_act = [0.0_f64; D_MODEL];
        for r in 0..D_MODEL {
            for col in 0..D_MODEL {
                lg.w2[r][col] += d_delta[r] * c.act[col];
                d_act[col] += lp.w2[r][col] * d_delta[r];
            }
        }
        let mut d_pre = [0.0_f64; D_MODEL];
        for r in 0..D_MODEL {
            d_pre[r] = d_act[r] * (1.0 - c.act[r] * c.act[r]);
        }
        let mut d_unorm = [0.0_f64; D_MODEL];
        for r in 0..D_MODEL {
            for col in 0..D_MODEL {
                lg.w1[r][col] += d_pre[r] * c.u_norm[col];
                d_unorm[col] += lp.w1[r][col] * d_pre[r];
            }
        }
        ops.add_flops((8 * D_MODEL * D_MODEL + 2 * D_MODEL) as u64);

        let d_u = rms_norm_d_bwd(&c.u, c.u_inv, &d_unorm, &mut ops);

        // Read backward: u[d] = sum_s h_pre[s] * x_in[s][d]
        let mut d_hpre = [0.0_f64; N_STREAMS];
        for s in 0..N_STREAMS {
            for d in 0..D_MODEL {
                d_hpre[s] += d_u[d] * c.x_in[s][d];
                dx_prev[s][d] += c.h_pre[s] * d_u[d];
            }
        }
        ops.add_flops((4 * N_STREAMS * D_MODEL) as u64);

        // h_pre softmax backward
        let mut dot_pre = 0.0_f64;
        for s in 0..N_STREAMS {
            dot_pre += d_hpre[s] * c.h_pre[s];
        }
        for s in 0..N_STREAMS {
            lg.h_pre_logits[s] += c.h_pre[s] * (d_hpre[s] - dot_pre);
        }

        dx = dx_prev;
    }

    let (tok_a, tok_b) = input_pair;
    for s in 0..N_STREAMS {
        for d in 0..D_MODEL {
            grad.embed[tok_a][d] += 0.5 * dx[s][d];
            grad.embed[tok_b][d] += 0.5 * dx[s][d];
        }
    }

    Ok((loss, grad, ops, fwd.max_birkhoff_err))
}

// ---------- BlockAttnResRef (beside #33) at identically matched parameter count ----------

#[derive(Debug, Clone)]
pub struct BlockAttnResLayer {
    pub depth_query: [f64; D_MODEL],
    pub rms_gain: [f64; D_MODEL],
    pub w1: [[f64; D_MODEL]; BLOCK_MLP_HIDDEN],
    pub w2: [[f64; BLOCK_MLP_HIDDEN]; D_MODEL],
}

impl BlockAttnResLayer {
    fn zeros() -> Self {
        Self {
            depth_query: [0.0; D_MODEL],
            rms_gain: [0.0; D_MODEL],
            w1: [[0.0; D_MODEL]; BLOCK_MLP_HIDDEN],
            w2: [[0.0; BLOCK_MLP_HIDDEN]; D_MODEL],
        }
    }

    fn init(seed: usize) -> Self {
        let mut lp = Self::zeros();
        for d in 0..D_MODEL {
            lp.depth_query[d] = det_weight(seed + 50, d, 0.5);
            lp.rms_gain[d] = 1.0 + det_weight(seed + 100, d, 0.1);
        }
        for r in 0..BLOCK_MLP_HIDDEN {
            for c in 0..D_MODEL {
                lp.w1[r][c] = det_weight(seed + 200 + r * 13, c, 0.42);
            }
        }
        for r in 0..D_MODEL {
            for c in 0..BLOCK_MLP_HIDDEN {
                lp.w2[r][c] = det_weight(seed + 400 + r * 17, c, 0.32);
            }
        }
        lp
    }
}

#[derive(Debug, Clone)]
pub struct BlockAttnResRef {
    pub embed: [[f64; D_MODEL]; VOCAB_SIZE],
    pub layers: [BlockAttnResLayer; NUM_LAYERS],
    pub final_depth_query: [f64; D_MODEL],
    pub w_head: [[f64; D_MODEL]; VOCAB_SIZE],
}

impl BlockAttnResRef {
    pub fn init(seed: usize) -> Self {
        let mut embed = [[0.0; D_MODEL]; VOCAB_SIZE];
        for v in 0..VOCAB_SIZE {
            for d in 0..D_MODEL {
                embed[v][d] = det_weight(seed + v * 19, d, 0.5);
            }
        }
        let layers = [
            BlockAttnResLayer::init(seed + 1000),
            BlockAttnResLayer::init(seed + 2000),
            BlockAttnResLayer::init(seed + 3000),
            BlockAttnResLayer::init(seed + 4000),
        ];
        let mut final_depth_query = [0.0; D_MODEL];
        for (d, q) in final_depth_query.iter_mut().enumerate() {
            *q = det_weight(seed + 5000, d, 0.5);
        }
        let mut w_head = [[0.0; D_MODEL]; VOCAB_SIZE];
        for v in 0..VOCAB_SIZE {
            for d in 0..D_MODEL {
                w_head[v][d] = det_weight(seed + 6000 + v * 23, d, 0.4);
            }
        }
        Self {
            embed,
            layers,
            final_depth_query,
            w_head,
        }
    }

    pub fn zeros_like(&self) -> Self {
        Self {
            embed: [[0.0; D_MODEL]; VOCAB_SIZE],
            layers: [
                BlockAttnResLayer::zeros(),
                BlockAttnResLayer::zeros(),
                BlockAttnResLayer::zeros(),
                BlockAttnResLayer::zeros(),
            ],
            final_depth_query: [0.0; D_MODEL],
            w_head: [[0.0; D_MODEL]; VOCAB_SIZE],
        }
    }

    pub fn param_count(&self) -> usize {
        let per_layer = 2 * D_MODEL + 2 * D_MODEL * BLOCK_MLP_HIDDEN;
        VOCAB_SIZE * D_MODEL + NUM_LAYERS * per_layer + D_MODEL + VOCAB_SIZE * D_MODEL
    }

    pub fn apply_sgd(&mut self, grad: &BlockAttnResRef, lr: f64) {
        for v in 0..VOCAB_SIZE {
            for d in 0..D_MODEL {
                self.embed[v][d] -= lr * grad.embed[v][d];
                self.w_head[v][d] -= lr * grad.w_head[v][d];
            }
        }
        for d in 0..D_MODEL {
            self.final_depth_query[d] -= lr * grad.final_depth_query[d];
        }
        for l in 0..NUM_LAYERS {
            let lp = &mut self.layers[l];
            let lg = &grad.layers[l];
            for d in 0..D_MODEL {
                lp.depth_query[d] -= lr * lg.depth_query[d];
                lp.rms_gain[d] -= lr * lg.rms_gain[d];
            }
            for r in 0..BLOCK_MLP_HIDDEN {
                for c in 0..D_MODEL {
                    lp.w1[r][c] -= lr * lg.w1[r][c];
                }
            }
            for r in 0..D_MODEL {
                for c in 0..BLOCK_MLP_HIDDEN {
                    lp.w2[r][c] -= lr * lg.w2[r][c];
                }
            }
        }
    }
}

struct DepthMixCache {
    cands: Vec<[f64; D_MODEL]>,
    normed: Vec<[f64; D_MODEL]>,
    invs: Vec<f64>,
    weights: Vec<f64>,
}

fn depth_mix_fwd(
    query: &[f64; D_MODEL],
    cands: &[[f64; D_MODEL]],
    ops: &mut OpCounter,
) -> Result<([f64; D_MODEL], DepthMixCache), String> {
    let mut normed = Vec::with_capacity(cands.len());
    let mut invs = Vec::with_capacity(cands.len());
    let mut scores = Vec::with_capacity(cands.len());
    for c in cands {
        let (nc, inv) = rms_norm_d(c, ops)?;
        let mut dot = 0.0_f64;
        for d in 0..D_MODEL {
            dot += query[d] * nc[d];
        }
        normed.push(nc);
        invs.push(inv);
        scores.push(dot);
    }
    let (weights, _) = softmax_vec(&scores)?;
    let mut out = [0.0_f64; D_MODEL];
    for (w, c) in weights.iter().zip(cands.iter()) {
        for d in 0..D_MODEL {
            out[d] += w * c[d];
        }
    }
    ops.add_flops((4 * cands.len() * D_MODEL) as u64);
    Ok((
        out,
        DepthMixCache {
            cands: cands.to_vec(),
            normed,
            invs,
            weights,
        },
    ))
}

fn depth_mix_bwd(
    query: &[f64; D_MODEL],
    cache: &DepthMixCache,
    d_out: &[f64; D_MODEL],
    d_query: &mut [f64; D_MODEL],
    ops: &mut OpCounter,
) -> Vec<[f64; D_MODEL]> {
    let m = cache.cands.len();
    let mut d_w = vec![0.0_f64; m];
    let mut d_cands = vec![[0.0_f64; D_MODEL]; m];
    for i in 0..m {
        for d in 0..D_MODEL {
            d_w[i] += d_out[d] * cache.cands[i][d];
            d_cands[i][d] += cache.weights[i] * d_out[d];
        }
    }
    let mut dot = 0.0_f64;
    for i in 0..m {
        dot += d_w[i] * cache.weights[i];
    }
    for i in 0..m {
        let d_score = cache.weights[i] * (d_w[i] - dot);
        let mut d_nc = [0.0_f64; D_MODEL];
        for d in 0..D_MODEL {
            d_query[d] += d_score * cache.normed[i][d];
            d_nc[d] = d_score * query[d];
        }
        let d_c_from_norm = rms_norm_d_bwd(&cache.cands[i], cache.invs[i], &d_nc, ops);
        for d in 0..D_MODEL {
            d_cands[i][d] += d_c_from_norm[d];
        }
    }
    ops.add_flops((6 * m * D_MODEL) as u64);
    d_cands
}

pub fn forward_block_attn_res(
    model: &BlockAttnResRef,
    input_pair: (usize, usize),
) -> Result<([f64; VOCAB_SIZE], OpCounter), String> {
    let (tok_a, tok_b) = input_pair;
    if tok_a >= VOCAB_SIZE || tok_b >= VOCAB_SIZE {
        return Err("token out of range".into());
    }
    let mut ops = OpCounter::default();
    let mut init_emb = [0.0_f64; D_MODEL];
    for d in 0..D_MODEL {
        let va = model.embed[tok_a][d];
        let vb = model.embed[tok_b][d];
        if !va.is_finite() || !vb.is_finite() {
            return Err("non-finite embed in block attn res".into());
        }
        init_emb[d] = 0.5 * (va + vb);
    }
    let mut completed = vec![init_emb];
    let mut active = [0.0_f64; D_MODEL];
    for l in 0..NUM_LAYERS {
        let mut cands = completed.clone();
        if l % BLOCK_LAYERS != 0 {
            cands.push(active);
        }
        let (layer_in, _) = depth_mix_fwd(&model.layers[l].depth_query, &cands, &mut ops)?;
        let (u_n, _) = rms_norm_d(&layer_in, &mut ops)?;
        let mut u = [0.0_f64; D_MODEL];
        for d in 0..D_MODEL {
            u[d] = u_n[d] * model.layers[l].rms_gain[d];
        }
        let mut act = [0.0_f64; BLOCK_MLP_HIDDEN];
        for r in 0..BLOCK_MLP_HIDDEN {
            let mut acc = 0.0_f64;
            for c in 0..D_MODEL {
                acc += model.layers[l].w1[r][c] * u[c];
            }
            act[r] = acc.tanh();
        }
        let mut delta = [0.0_f64; D_MODEL];
        for r in 0..D_MODEL {
            let mut acc = 0.0_f64;
            for c in 0..BLOCK_MLP_HIDDEN {
                acc += model.layers[l].w2[r][c] * act[c];
            }
            delta[r] = acc;
        }
        ops.add_flops((4 * D_MODEL * BLOCK_MLP_HIDDEN + 2 * BLOCK_MLP_HIDDEN) as u64);
        for d in 0..D_MODEL {
            active[d] += delta[d];
        }
        if (l + 1) % BLOCK_LAYERS == 0 {
            completed.push(active);
            active = [0.0; D_MODEL];
        }
    }
    let (final_h, _) = depth_mix_fwd(&model.final_depth_query, &completed, &mut ops)?;
    let (z, _) = rms_norm_d(&final_h, &mut ops)?;
    let mut logits = [0.0_f64; VOCAB_SIZE];
    for c in 0..VOCAB_SIZE {
        let mut acc = 0.0_f64;
        for d in 0..D_MODEL {
            acc += model.w_head[c][d] * z[d];
        }
        if !acc.is_finite() {
            return Err("non-finite block attn res logit".into());
        }
        logits[c] = acc;
    }
    ops.add_flops((2 * VOCAB_SIZE * D_MODEL) as u64);
    Ok((logits, ops))
}

pub fn forward_backward_block_attn_res(
    model: &BlockAttnResRef,
    input_pair: (usize, usize),
    target_tok: usize,
) -> Result<(f64, BlockAttnResRef, OpCounter), String> {
    let (tok_a, tok_b) = input_pair;
    if tok_a >= VOCAB_SIZE || tok_b >= VOCAB_SIZE || target_tok >= VOCAB_SIZE {
        return Err("token out of range".into());
    }
    let mut ops = OpCounter::default();
    let mut init_emb = [0.0_f64; D_MODEL];
    for d in 0..D_MODEL {
        init_emb[d] = 0.5 * (model.embed[tok_a][d] + model.embed[tok_b][d]);
    }
    let mut completed = vec![init_emb];
    let mut active = [0.0_f64; D_MODEL];

    struct LayerRec {
        mix_cache: DepthMixCache,
        has_active_cand: bool,
        completed_len_at_start: usize,
        layer_in: [f64; D_MODEL],
        u_n: [f64; D_MODEL],
        u_inv: f64,
        u: [f64; D_MODEL],
        act: [f64; BLOCK_MLP_HIDDEN],
        pushes_block: bool,
    }
    let mut recs = Vec::with_capacity(NUM_LAYERS);

    for l in 0..NUM_LAYERS {
        let completed_len_at_start = completed.len();
        let has_active_cand = l % BLOCK_LAYERS != 0;
        let mut cands = completed.clone();
        if has_active_cand {
            cands.push(active);
        }
        let (layer_in, mix_cache) = depth_mix_fwd(&model.layers[l].depth_query, &cands, &mut ops)?;
        let (u_n, u_inv) = rms_norm_d(&layer_in, &mut ops)?;
        let mut u = [0.0_f64; D_MODEL];
        for d in 0..D_MODEL {
            u[d] = u_n[d] * model.layers[l].rms_gain[d];
        }
        let mut act = [0.0_f64; BLOCK_MLP_HIDDEN];
        for r in 0..BLOCK_MLP_HIDDEN {
            let mut acc = 0.0_f64;
            for c in 0..D_MODEL {
                acc += model.layers[l].w1[r][c] * u[c];
            }
            act[r] = acc.tanh();
        }
        let mut delta = [0.0_f64; D_MODEL];
        for r in 0..D_MODEL {
            let mut acc = 0.0_f64;
            for c in 0..BLOCK_MLP_HIDDEN {
                acc += model.layers[l].w2[r][c] * act[c];
            }
            delta[r] = acc;
        }
        ops.add_flops((4 * D_MODEL * BLOCK_MLP_HIDDEN + 2 * BLOCK_MLP_HIDDEN) as u64);
        for d in 0..D_MODEL {
            active[d] += delta[d];
        }
        let pushes_block = (l + 1) % BLOCK_LAYERS == 0;
        if pushes_block {
            completed.push(active);
            active = [0.0; D_MODEL];
        }
        recs.push(LayerRec {
            mix_cache,
            has_active_cand,
            completed_len_at_start,
            layer_in,
            u_n,
            u_inv,
            u,
            act,
            pushes_block,
        });
    }

    let (final_h, final_mix) = depth_mix_fwd(&model.final_depth_query, &completed, &mut ops)?;
    let (z, z_inv) = rms_norm_d(&final_h, &mut ops)?;
    let mut logits = [0.0_f64; VOCAB_SIZE];
    for c in 0..VOCAB_SIZE {
        let mut acc = 0.0_f64;
        for d in 0..D_MODEL {
            acc += model.w_head[c][d] * z[d];
        }
        logits[c] = acc;
    }
    ops.add_flops((2 * VOCAB_SIZE * D_MODEL) as u64);

    let (probs, lse) = softmax_vec(&logits)?;
    let loss = lse - logits[target_tok];
    if !loss.is_finite() {
        return Err("non-finite block attn res loss".into());
    }

    let mut grad = model.zeros_like();
    let mut d_z = [0.0_f64; D_MODEL];
    for c in 0..VOCAB_SIZE {
        let dl = probs[c] - if c == target_tok { 1.0 } else { 0.0 };
        for d in 0..D_MODEL {
            grad.w_head[c][d] += dl * z[d];
            d_z[d] += model.w_head[c][d] * dl;
        }
    }
    ops.add_flops((4 * VOCAB_SIZE * D_MODEL) as u64);

    let d_final_h = rms_norm_d_bwd(&final_h, z_inv, &d_z, &mut ops);
    let mut d_completed = depth_mix_bwd(
        &model.final_depth_query,
        &final_mix,
        &d_final_h,
        &mut grad.final_depth_query,
        &mut ops,
    );
    let mut d_active = [0.0_f64; D_MODEL];

    for l in (0..NUM_LAYERS).rev() {
        let rec = &recs[l];
        let lp = &model.layers[l];
        let lg = &mut grad.layers[l];
        if rec.pushes_block {
            d_active = d_completed
                .pop()
                .ok_or_else(|| "missing completed block grad".to_string())?;
        }
        // delta was added to active_block
        let d_delta = d_active;

        let mut d_act = [0.0_f64; BLOCK_MLP_HIDDEN];
        for r in 0..D_MODEL {
            for c in 0..BLOCK_MLP_HIDDEN {
                lg.w2[r][c] += d_delta[r] * rec.act[c];
                d_act[c] += lp.w2[r][c] * d_delta[r];
            }
        }
        let mut d_pre = [0.0_f64; BLOCK_MLP_HIDDEN];
        for r in 0..BLOCK_MLP_HIDDEN {
            d_pre[r] = d_act[r] * (1.0 - rec.act[r] * rec.act[r]);
        }
        let mut d_u = [0.0_f64; D_MODEL];
        for r in 0..BLOCK_MLP_HIDDEN {
            for c in 0..D_MODEL {
                lg.w1[r][c] += d_pre[r] * rec.u[c];
                d_u[c] += lp.w1[r][c] * d_pre[r];
            }
        }
        ops.add_flops((8 * D_MODEL * BLOCK_MLP_HIDDEN + 2 * BLOCK_MLP_HIDDEN) as u64);

        let mut d_un = [0.0_f64; D_MODEL];
        for d in 0..D_MODEL {
            lg.rms_gain[d] += d_u[d] * rec.u_n[d];
            d_un[d] = d_u[d] * lp.rms_gain[d];
        }
        let d_layer_in = rms_norm_d_bwd(&rec.layer_in, rec.u_inv, &d_un, &mut ops);
        let d_cands = depth_mix_bwd(
            &lp.depth_query,
            &rec.mix_cache,
            &d_layer_in,
            &mut lg.depth_query,
            &mut ops,
        );
        for i in 0..rec.completed_len_at_start {
            for d in 0..D_MODEL {
                d_completed[i][d] += d_cands[i][d];
            }
        }
        if rec.has_active_cand {
            let a_idx = rec.completed_len_at_start;
            for d in 0..D_MODEL {
                d_active[d] += d_cands[a_idx][d];
            }
        } else {
            d_active = [0.0; D_MODEL];
        }
    }

    for d in 0..D_MODEL {
        grad.embed[tok_a][d] += 0.5 * d_completed[0][d];
        grad.embed[tok_b][d] += 0.5 * d_completed[0][d];
    }

    Ok((loss, grad, ops))
}

fn max_logit_diff(a: &[f64; VOCAB_SIZE], b: &[f64; VOCAB_SIZE]) -> Result<f64, String> {
    let mut m = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return Err("non-finite logit in parity check".into());
        }
        let d = (x - y).abs();
        if d > m {
            m = d;
        }
    }
    Ok(m)
}

pub fn compute_overhead_pct(mhc_flops: u64, simplehc_flops: u64) -> Option<f64> {
    if simplehc_flops == 0 {
        return None;
    }
    let pct = ((mhc_flops as f64 - simplehc_flops as f64) / (simplehc_flops as f64)) * 100.0;
    if pct.is_finite() {
        Some(pct)
    } else {
        None
    }
}

pub fn validate_primary_source_pin(pin_status: &str) -> OutcomeStatus {
    if pin_status == PRIMARY_SOURCE_PIN_STATUS_UNLOCATED {
        OutcomeStatus::BlockedOnArtifacts
    } else {
        OutcomeStatus::Invalid
    }
}

pub fn validate_matched_comparison(param_counts: &[usize], token_counts: &[u64]) -> OutcomeStatus {
    if param_counts.is_empty() || token_counts.is_empty() {
        return OutcomeStatus::Invalid;
    }
    let p0 = param_counts[0];
    let t0 = token_counts[0];
    if p0 == 0 || t0 == 0 {
        return OutcomeStatus::Invalid;
    }
    if param_counts.iter().all(|&p| p == p0) && token_counts.iter().all(|&t| t == t0) {
        OutcomeStatus::Accepted
    } else {
        OutcomeStatus::Invalid
    }
}

pub fn validate_overhead_gate_policy(import_post_59_100_as_gate: bool) -> OutcomeStatus {
    if import_post_59_100_as_gate {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    }
}

#[derive(Debug, Clone)]
pub struct VariantArmSummary {
    pub arm_id: String,
    pub param_count: usize,
    pub training_tokens: u64,
    pub sinkhorn_ops_per_step: u64,
    pub total_step_flops: u64,
    pub total_training_flops: u64,
    pub scalar_ref_parity_max_err: Option<f64>,
    pub max_birkhoff_err: Option<f64>,
    pub initial_train_loss: f64,
    pub final_train_loss: f64,
    pub final_heldout_loss: f64,
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct SimpleHcMhcReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub synthetic_contract_fixture: bool,
    pub primary_source_pin_status: String,
    pub fixture_digest: DigestRef,
    pub matched_params_and_tokens_outcome: OutcomeStatus,
    pub mhc_overhead_vs_simplehc_pct: Option<f64>,
    pub arms: Vec<VariantArmSummary>,
    pub negative_controls: Vec<NegativeControl>,
}

type CompositionalExample = ((usize, usize), usize);
type TokenPairSplits = (Vec<CompositionalExample>, Vec<CompositionalExample>);

fn canonical_pair(p: (usize, usize)) -> (usize, usize) {
    if p.0 <= p.1 {
        p
    } else {
        (p.1, p.0)
    }
}

pub fn validate_disjoint_token_pairs(
    train: &[CompositionalExample],
    heldout: &[CompositionalExample],
) -> Result<(), String> {
    if train.is_empty() || heldout.is_empty() {
        return Err("train and held-out compositional splits must be non-empty".into());
    }
    let mut train_keys = std::collections::BTreeSet::new();
    for &(inp, _) in train {
        train_keys.insert(canonical_pair(inp));
    }
    for &(inp, _) in heldout {
        if train_keys.contains(&canonical_pair(inp)) {
            return Err(format!(
                "train/held-out overlap detected for input pair {:?}",
                inp
            ));
        }
    }
    Ok(())
}

fn build_pairs() -> Result<TokenPairSplits, String> {
    // Partition the 21 distinct unordered pairs {a, b} (0 <= a <= b < 6) into 15 train and 6
    // strictly disjoint held-out compositional examples with target (a + b + 1) mod VOCAB_SIZE.
    let heldout_inputs: [(usize, usize); 6] = [(0, 3), (1, 2), (2, 5), (3, 4), (4, 5), (5, 5)];
    let mut train = Vec::with_capacity(15);
    let mut heldout = Vec::with_capacity(6);
    for a in 0..VOCAB_SIZE {
        for b in a..VOCAB_SIZE {
            let tgt = (a + b + 1) % VOCAB_SIZE;
            if heldout_inputs.contains(&(a, b)) {
                heldout.push(((a, b), tgt));
            } else {
                train.push(((a, b), tgt));
            }
        }
    }
    validate_disjoint_token_pairs(&train, &heldout)?;
    Ok((train, heldout))
}

fn eval_hc_loss(model: &HyperConnModel, pairs: &[CompositionalExample]) -> Result<f64, String> {
    if pairs.is_empty() {
        return Err("empty eval pairs".into());
    }
    let mut sum = 0.0_f64;
    for &(inp, tgt) in pairs {
        let fwd = forward_hc_candidate(model, inp, CandidateFault::default())?;
        let (_, lse) = softmax_vec(&fwd.logits)?;
        sum += lse - fwd.logits[tgt];
    }
    Ok(sum / (pairs.len() as f64))
}

fn eval_block_res_loss(
    model: &BlockAttnResRef,
    pairs: &[CompositionalExample],
) -> Result<f64, String> {
    if pairs.is_empty() {
        return Err("empty eval pairs".into());
    }
    let mut sum = 0.0_f64;
    for &(inp, tgt) in pairs {
        let (logits, _) = forward_block_attn_res(model, inp)?;
        let (_, lse) = softmax_vec(&logits)?;
        sum += lse - logits[tgt];
    }
    Ok(sum / (pairs.len() as f64))
}

pub fn run_default_harness() -> Result<SimpleHcMhcReport, String> {
    let (train_pairs, heldout_pairs) = build_pairs()?;
    let epochs = 25usize;
    let lr = 0.08_f64;

    let mut digest_buf = String::new();
    for &((a, b), t) in &train_pairs {
        let _ = write!(digest_buf, "tr:{a}+{b}->{t};");
    }
    for &((a, b), t) in &heldout_pairs {
        let _ = write!(digest_buf, "ho:{a}+{b}->{t};");
    }
    let fixture_digest = DigestRef::sha256(digest_buf.as_bytes());

    let mut mhc = HyperConnModel::init(5101, HcVariant::DeepSeekMhc);
    let mut simplehc = HyperConnModel::init(5101, HcVariant::SimpleHc);
    let mut block_res = BlockAttnResRef::init(5101);

    // Independent scalar reference vs candidate parity checks before training
    let mhc_ref = forward_hc_scalar_ref(&mhc, (2, 4))?;
    let mhc_cand = forward_hc_candidate(&mhc, (2, 4), CandidateFault::default())?;
    let mhc_parity_err = max_logit_diff(&mhc_ref, &mhc_cand.logits)?;
    if mhc_parity_err > 1.0e-12 {
        return Err(format!("mHC scalar parity failed: {mhc_parity_err:.3e}"));
    }

    let shc_ref = forward_hc_scalar_ref(&simplehc, (2, 4))?;
    let shc_cand = forward_hc_candidate(&simplehc, (2, 4), CandidateFault::default())?;
    let shc_parity_err = max_logit_diff(&shc_ref, &shc_cand.logits)?;
    if shc_parity_err > 1.0e-12 {
        return Err(format!(
            "SimpleHC scalar parity failed: {shc_parity_err:.3e}"
        ));
    }

    // Train DeepSeekMhc
    let mhc_init_loss = eval_hc_loss(&mhc, &train_pairs)?;
    let mut mhc_total_ops = OpCounter::default();
    let mut mhc_step_ops = OpCounter::default();
    let mut mhc_max_birkhoff = 0.0_f64;
    let mut mhc_tokens = 0u64;
    for _ in 0..epochs {
        for &(inp, tgt) in &train_pairs {
            let (_, grad, step_ops, b_err) = forward_backward_hc(&mhc, inp, tgt)?;
            mhc.apply_sgd(&grad, lr);
            mhc_step_ops = step_ops.clone();
            mhc_total_ops.sinkhorn_ops = mhc_total_ops
                .sinkhorn_ops
                .saturating_add(step_ops.sinkhorn_ops);
            mhc_total_ops.total_flops = mhc_total_ops
                .total_flops
                .saturating_add(step_ops.total_flops);
            if let Some(e) = b_err {
                if e > mhc_max_birkhoff {
                    mhc_max_birkhoff = e;
                }
            }
            mhc_tokens = mhc_tokens.saturating_add(1);
        }
    }
    let mhc_final_train = eval_hc_loss(&mhc, &train_pairs)?;
    let mhc_final_heldout = eval_hc_loss(&mhc, &heldout_pairs)?;

    // Train SimpleHc
    let shc_init_loss = eval_hc_loss(&simplehc, &train_pairs)?;
    let mut shc_total_ops = OpCounter::default();
    let mut shc_step_ops = OpCounter::default();
    let mut shc_tokens = 0u64;
    for _ in 0..epochs {
        for &(inp, tgt) in &train_pairs {
            let (_, grad, step_ops, _) = forward_backward_hc(&simplehc, inp, tgt)?;
            simplehc.apply_sgd(&grad, lr);
            shc_step_ops = step_ops.clone();
            shc_total_ops.sinkhorn_ops = shc_total_ops
                .sinkhorn_ops
                .saturating_add(step_ops.sinkhorn_ops);
            shc_total_ops.total_flops = shc_total_ops
                .total_flops
                .saturating_add(step_ops.total_flops);
            shc_tokens = shc_tokens.saturating_add(1);
        }
    }
    let shc_final_train = eval_hc_loss(&simplehc, &train_pairs)?;
    let shc_final_heldout = eval_hc_loss(&simplehc, &heldout_pairs)?;

    // Train BlockAttnResRef (beside #33)
    let br_init_loss = eval_block_res_loss(&block_res, &train_pairs)?;
    let mut br_total_ops = OpCounter::default();
    let mut br_step_ops = OpCounter::default();
    let mut br_tokens = 0u64;
    for _ in 0..epochs {
        for &(inp, tgt) in &train_pairs {
            let (_, grad, step_ops) = forward_backward_block_attn_res(&block_res, inp, tgt)?;
            block_res.apply_sgd(&grad, lr);
            br_step_ops = step_ops.clone();
            br_total_ops.total_flops = br_total_ops
                .total_flops
                .saturating_add(step_ops.total_flops);
            br_tokens = br_tokens.saturating_add(1);
        }
    }
    let br_final_train = eval_block_res_loss(&block_res, &train_pairs)?;
    let br_final_heldout = eval_block_res_loss(&block_res, &heldout_pairs)?;

    let matched_params_and_tokens_outcome = validate_matched_comparison(
        &[
            mhc.param_count(),
            simplehc.param_count(),
            block_res.param_count(),
        ],
        &[mhc_tokens, shc_tokens, br_tokens],
    );

    let mhc_overhead_vs_simplehc_pct =
        compute_overhead_pct(mhc_step_ops.total_flops, shc_step_ops.total_flops);

    let arms = vec![
        VariantArmSummary {
            arm_id: "deepseek_mhc".into(),
            param_count: mhc.param_count(),
            training_tokens: mhc_tokens,
            sinkhorn_ops_per_step: mhc_step_ops.sinkhorn_ops,
            total_step_flops: mhc_step_ops.total_flops,
            total_training_flops: mhc_total_ops.total_flops,
            scalar_ref_parity_max_err: Some(mhc_parity_err),
            max_birkhoff_err: Some(mhc_max_birkhoff),
            initial_train_loss: mhc_init_loss,
            final_train_loss: mhc_final_train,
            final_heldout_loss: mhc_final_heldout,
        },
        VariantArmSummary {
            arm_id: "simple_hc".into(),
            param_count: simplehc.param_count(),
            training_tokens: shc_tokens,
            sinkhorn_ops_per_step: shc_step_ops.sinkhorn_ops,
            total_step_flops: shc_step_ops.total_flops,
            total_training_flops: shc_total_ops.total_flops,
            scalar_ref_parity_max_err: Some(shc_parity_err),
            max_birkhoff_err: None,
            initial_train_loss: shc_init_loss,
            final_train_loss: shc_final_train,
            final_heldout_loss: shc_final_heldout,
        },
        VariantArmSummary {
            arm_id: "block_attn_res_ref_33".into(),
            param_count: block_res.param_count(),
            training_tokens: br_tokens,
            sinkhorn_ops_per_step: 0,
            total_step_flops: br_step_ops.total_flops,
            total_training_flops: br_total_ops.total_flops,
            scalar_ref_parity_max_err: None,
            max_birkhoff_err: None,
            initial_train_loss: br_init_loss,
            final_train_loss: br_final_train,
            final_heldout_loss: br_final_heldout,
        },
    ];

    // 8 executable negative controls
    let mut negative_controls = Vec::new();

    // NC1: claiming primary repo pinned when unlocated
    {
        let outcome = validate_primary_source_pin("pinned_primary_repo_sha");
        negative_controls.push(NegativeControl {
            case_id: "nc1_claiming_primary_repo_pinned_rejected".into(),
            outcome,
            detail: "claiming unlocated SimpleHC primary repo is pinned rejected".into(),
        });
    }

    // NC2: Sinkhorn doubly-stochastic violation caught
    {
        let bad_mat = [
            [0.8, 0.2, 0.0, 0.0],
            [0.7, 0.3, 0.0, 0.0],
            [0.1, 0.1, 0.4, 0.4],
            [0.1, 0.1, 0.4, 0.4],
        ];
        let (err, outcome) = verify_doubly_stochastic(&bad_mat, 1.0e-6)?;
        negative_controls.push(NegativeControl {
            case_id: "nc2_sinkhorn_doubly_stochastic_violation".into(),
            outcome,
            detail: format!("non-doubly-stochastic matrix rejected (max_err={err:.3e})"),
        });
    }

    // NC3: mismatched parameter counts rejected
    {
        let outcome = validate_matched_comparison(&[462, 512, 462], &[300, 300, 300]);
        negative_controls.push(NegativeControl {
            case_id: "nc3_mismatched_params_rejected".into(),
            outcome,
            detail: "arms with mismatched parameter counts rejected".into(),
        });
    }

    // NC4: mismatched token budgets rejected
    {
        let outcome = validate_matched_comparison(&[462, 462, 462], &[300, 150, 300]);
        negative_controls.push(NegativeControl {
            case_id: "nc4_mismatched_tokens_rejected".into(),
            outcome,
            detail: "arms with mismatched training token budgets rejected".into(),
        });
    }

    // NC5: NaN in Sinkhorn logits or weights rejected
    {
        let mut bad_logits = mhc.layers[0].h_res_logits;
        bad_logits[0][0] = f64::NAN;
        let mut tmp_ops = OpCounter::default();
        let res = sinkhorn_knopp_forward(&bad_logits, SINKHORN_ITERS, &mut tmp_ops);
        let outcome = if res.is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
        negative_controls.push(NegativeControl {
            case_id: "nc5_nan_in_sinkhorn_or_weights_rejected".into(),
            outcome,
            detail: "NaN in Sinkhorn-Knopp logits rejected cleanly".into(),
        });
    }

    // NC6: faulty candidate caught by independent scalar reference
    {
        let ref_out = forward_hc_scalar_ref(&mhc, (2, 4))?;
        let bad_cand = forward_hc_candidate(
            &mhc,
            (2, 4),
            CandidateFault {
                corrupt_stream_mix: true,
            },
        )?;
        let err = max_logit_diff(&ref_out, &bad_cand.logits)?;
        let outcome = if err > 1.0e-12 {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
        negative_controls.push(NegativeControl {
            case_id: "nc6_faulty_candidate_caught_by_scalar_ref".into(),
            outcome,
            detail: format!("faulty stream-mix candidate caught by scalar ref (err={err:.3e})"),
        });
    }

    // NC7: zero-FLOP divisor guard
    {
        let ov = compute_overhead_pct(1000, 0);
        let outcome = if ov.is_none() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
        negative_controls.push(NegativeControl {
            case_id: "nc7_zero_flop_divisor_guard".into(),
            outcome,
            detail: "zero SimpleHC FLOP divisor returns None and refuses division by zero".into(),
        });
    }

    // NC8: importing post's 59-100% overhead as local pass/fail gate rejected
    {
        let outcome = validate_overhead_gate_policy(true);
        negative_controls.push(NegativeControl {
            case_id: "nc8_importing_post_59_100_overhead_as_gate_rejected".into(),
            outcome,
            detail: "post's 59-100% overhead claim cannot be imported as a pass/fail gate".into(),
        });
    }

    Ok(SimpleHcMhcReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::blocked_on_artifacts(
            "PyTorch/GPU training comparison against pinned SimpleHC primary source",
            "Primary SimpleHC paper/code repository was not located in sweep (post-only bookmark); miniature matched-param CPU harness verified locally",
        ),
        synthetic_contract_fixture: true,
        primary_source_pin_status: PRIMARY_SOURCE_PIN_STATUS_UNLOCATED.into(),
        fixture_digest,
        matched_params_and_tokens_outcome,
        mhc_overhead_vs_simplehc_pct,
        arms,
        negative_controls,
    })
}

pub fn inputs_json(report: &SimpleHcMhcReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.simplehc-mhc.inputs.v1\",\n",
            "  \"issue\": 51,\n",
            "  \"execution_class\": {},\n",
            "  \"source_bookmark_post_id\": {},\n",
            "  \"primary_source_pin_status\": {},\n",
            "  \"related_issues\": [33],\n",
            "  \"n_streams\": {},\n",
            "  \"d_model\": {},\n",
            "  \"num_layers\": {},\n",
            "  \"sinkhorn_iters\": {},\n",
            "  \"fixture_digest\": {},\n",
            "  \"synthetic_contract_fixture\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(SOURCE_BOOKMARK_POST_ID),
        quote(&report.primary_source_pin_status),
        N_STREAMS,
        D_MODEL,
        NUM_LAYERS,
        SINKHORN_ITERS,
        report.fixture_digest.to_json(),
        report.synthetic_contract_fixture,
    )
}

pub fn results_json(report: &SimpleHcMhcReport) -> String {
    let arms: Vec<String> = report
        .arms
        .iter()
        .map(|a| {
            format!(
                concat!(
                    "{{\"arm_id\":{},\"param_count\":{},\"training_tokens\":{},",
                    "\"sinkhorn_ops_per_step\":{},\"total_step_flops\":{},\"total_training_flops\":{},",
                    "\"scalar_ref_parity_max_err\":{},\"max_birkhoff_err\":{},",
                    "\"initial_train_loss\":{},\"final_train_loss\":{},\"final_heldout_loss\":{}}}"
                ),
                quote(&a.arm_id),
                a.param_count,
                a.training_tokens,
                a.sinkhorn_ops_per_step,
                a.total_step_flops,
                a.total_training_flops,
                opt_f64(a.scalar_ref_parity_max_err),
                opt_f64(a.max_birkhoff_err),
                finite_f64(a.initial_train_loss),
                finite_f64(a.final_train_loss),
                finite_f64(a.final_heldout_loss),
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
            "  \"schema\": \"inference-lab.simplehc-mhc.results.v1\",\n",
            "  \"issue\": 51,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"primary_source_pin_status\": {},\n",
            "  \"matched_params_and_tokens_outcome\": {},\n",
            "  \"mhc_overhead_vs_simplehc_pct\": {},\n",
            "  \"arms\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.synthetic_contract_fixture,
        quote(&report.primary_source_pin_status),
        quote(report.matched_params_and_tokens_outcome.as_str()),
        opt_f64(report.mhc_overhead_vs_simplehc_pct),
        arms.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &SimpleHcMhcReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #51: SimpleHC vs DeepSeek mHC Beside Block Attention Residuals (#33)"
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
        "- Primary source pin status: `{}` (post `{}`; 59-100% overhead claim is non-gating)",
        report.primary_source_pin_status, SOURCE_BOOKMARK_POST_ID
    )
    .unwrap();
    writeln!(
        out,
        "- Matched params & tokens: `{}`; mHC step-FLOP overhead vs SimpleHC: `{}%`",
        report.matched_params_and_tokens_outcome.as_str(),
        opt_f64(report.mhc_overhead_vs_simplehc_pct)
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Hyper-Connection & Attention-Residual Arms").unwrap();
    for a in &report.arms {
        writeln!(
            out,
            "- `{}`: `params={}`, `tokens={}`, `sinkhorn_ops/step={}`, `step_flops={}`, `loss={:.4}->{:.4}`, `heldout={:.4}`",
            a.arm_id,
            a.param_count,
            a.training_tokens,
            a.sinkhorn_ops_per_step,
            a.total_step_flops,
            a.initial_train_loss,
            a.final_train_loss,
            a.final_heldout_loss,
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
    fn sinkhorn_backward_matches_finite_differences() {
        let model = HyperConnModel::init(5101, HcVariant::DeepSeekMhc);
        let (_, grad, _, _) = forward_backward_hc(&model, (2, 4), 5).unwrap();
        let eps = 1.0e-6;
        let mut max_err = 0.0_f64;
        for i in 0..N_STREAMS {
            for j in 0..N_STREAMS {
                let mut m_plus = model.clone();
                m_plus.layers[0].h_res_logits[i][j] += eps;
                let (lp, _, _, _) = forward_backward_hc(&m_plus, (2, 4), 5).unwrap();
                let mut m_minus = model.clone();
                m_minus.layers[0].h_res_logits[i][j] -= eps;
                let (lm, _, _, _) = forward_backward_hc(&m_minus, (2, 4), 5).unwrap();
                let num = (lp - lm) / (2.0 * eps);
                let ana = grad.layers[0].h_res_logits[i][j];
                let err = (num - ana).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        assert!(
            max_err <= 1.0e-6,
            "Sinkhorn FD grad error too high: {max_err}"
        );
    }

    #[test]
    fn block_attn_res_backward_matches_finite_differences() {
        let model = BlockAttnResRef::init(5101);
        let (_, grad, _) = forward_backward_block_attn_res(&model, (2, 4), 5).unwrap();
        let eps = 1.0e-6;
        let mut max_err = 0.0_f64;
        for d in 0..D_MODEL {
            let mut m_plus = model.clone();
            m_plus.layers[1].depth_query[d] += eps;
            let (lp, _, _) = forward_backward_block_attn_res(&m_plus, (2, 4), 5).unwrap();
            let mut m_minus = model.clone();
            m_minus.layers[1].depth_query[d] -= eps;
            let (lm, _, _) = forward_backward_block_attn_res(&m_minus, (2, 4), 5).unwrap();
            let err = ((lp - lm) / (2.0 * eps) - grad.layers[1].depth_query[d]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err <= 1.0e-6,
            "BlockAttnRes FD grad error too high: {max_err}"
        );
    }

    #[test]
    fn compositional_splits_are_disjoint_and_reject_overlap() {
        let (tr, ho) = build_pairs().unwrap();
        assert_eq!(tr.len(), 15);
        assert_eq!(ho.len(), 6);
        assert!(validate_disjoint_token_pairs(&tr, &ho).is_ok());
        // Reversed pair (b, a) of a train pair (a, b) must also be caught as overlapping!
        let mut leaky = ho.clone();
        let ((a, b), tgt) = tr[1];
        leaky.push(((b, a), tgt));
        assert!(validate_disjoint_token_pairs(&tr, &leaky).is_err());
    }

    #[test]
    fn default_harness_matches_params_verifies_sinkhorn_and_rejects_all_ncs() {
        let report = run_default_harness().unwrap();
        assert_eq!(
            report.execution_class,
            ExecutionClass::BoundedContractHarness
        );
        assert_eq!(
            report.hardware_gate.status,
            ExecutionClass::BlockedOnArtifacts
        );
        assert!(report.synthetic_contract_fixture);
        assert_eq!(
            report.primary_source_pin_status,
            "post_only_unlocated_paper_repo"
        );
        assert_eq!(
            report.matched_params_and_tokens_outcome,
            OutcomeStatus::Accepted
        );
        assert_eq!(report.arms.len(), 3);
        let p0 = report.arms[0].param_count;
        for a in &report.arms {
            assert_eq!(a.param_count, p0);
            assert!(a.final_train_loss < a.initial_train_loss);
            assert!(
                (a.final_train_loss - a.final_heldout_loss).abs() > 1.0e-6,
                "train and disjoint held-out losses must not be identical for {}",
                a.arm_id
            );
        }
        assert!(report.arms[0].sinkhorn_ops_per_step > 0);
        assert_eq!(report.arms[1].sinkhorn_ops_per_step, 0);
        assert!(report.arms[0].max_birkhoff_err.unwrap() <= 1.0e-6);
        assert!(report.arms[0].scalar_ref_parity_max_err.unwrap() <= 1.0e-12);
        assert!(report.arms[1].scalar_ref_parity_max_err.unwrap() <= 1.0e-12);
        assert!(report.mhc_overhead_vs_simplehc_pct.unwrap() > 0.0);
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
        assert!(ij.contains("post_only_unlocated_paper_repo"));
        assert!(rj.contains("deepseek_mhc"));
        assert!(md.contains("Issue #51"));
    }
}
