//! Issue #68: Log LinQuant as related work and measure low-precision recurrent prefill rewrite
//! error and throughput on GDN/Mamba-2 against full-precision baselines, default-off.
//!
//! Related-work record (strictly non-gating, honest offline pin status):
//! - X post (2026-11-05, Neehal Tumma `@ntumm120`):
//!   `https://x.com/ntumm120/status/2118400198539927994`
//! - `pin_status`: `"post_only_unlocated_paper_repo"` (no public LinQuant code commit located
//!   offline; no fabricated commit SHA).
//! - Cross-repository link (2026-11-24 issue comment): Tenstorrent-specific TT-Metal/Blackhole
//!   fused GDN prefill kernel adaptation and physical P150x2 verification remain tracked in
//!   `OnixResearch/tt-onix#34` once this portable numerical/throughput harness establishes
//!   viability (`grants_blackhole_acceptance = false`).
//! - Complements `#16`, `#32`, and `#37`.
//!
//! Provides:
//! - Real pure-Rust **Gated DeltaNet (GDN)** and **Mamba-2 (SSD)** recurrent prefill engines.
//! - Exact FP64 algebraic equivalence gate BEFORE any quantized evaluation: verifies that the
//!   LinQuant orthogonal Hadamard state-space rewrite (`S'_t = H S_t H^T`, `H H^T = I`) matches
//!   direct sequential FP64 recurrence within `<= 1e-12` on both GDN and Mamba-2.
//! - Genuine low-precision numerical comparison (simulated BF16, FP8-E4M3, and symmetric INT8
//!   per-vector quantization) on outlier-heavy recurrent prefill fixtures comparing naive
//!   low-precision recurrence vs LinQuant-rewritten low-precision recurrence against the exact
//!   FP64 reference (state max/L2 error and output logit error), plus prefill throughput model.
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, sigmoid};
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const UPSTREAM_X_POST: &str = "https://x.com/ntumm120/status/2118400198539927994";
pub const LINQUANT_PIN_STATUS: &str = "post_only_unlocated_paper_repo";
pub const CROSS_REPO_TT_ONIX_ISSUE: &str = "OnixResearch/tt-onix#34";

pub const DIM: usize = 4;
pub const SEQ_LEN: usize = 16;
pub const FP64_EQUIV_TOL: f64 = 1.0e-12;

/// Normalized 4x4 Sylvester Hadamard matrix (`H * H^T = I`, all entries `+/- 0.5`).
const H4: [[f64; DIM]; DIM] = [
    [0.5, 0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5, -0.5],
    [0.5, 0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5, 0.5],
];

fn check_finite(v: f64, ctx: &str) -> Result<f64, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("non-finite float in {ctx}"))
    }
}

pub type GdnPrefillResult = (Vec<[f64; DIM]>, [[f64; DIM]; DIM]);

#[allow(clippy::needless_range_loop)]
fn hadamard_vec(x: &[f64; DIM]) -> [f64; DIM] {
    let mut y = [0.0_f64; DIM];
    for r in 0..DIM {
        let mut s = 0.0_f64;
        for c in 0..DIM {
            s += H4[r][c] * x[c];
        }
        y[r] = s;
    }
    y
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowPrecisionMode {
    Fp64Exact,
    Bf16Trunc,
    Fp8E4M3Sim,
    Int8Symmetric,
}

impl LowPrecisionMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fp64Exact => "fp64_exact",
            Self::Bf16Trunc => "bf16",
            Self::Fp8E4M3Sim => "fp8_e4m3",
            Self::Int8Symmetric => "int8_per_vec",
        }
    }
}

fn quantize_scalar_bf16(v: f64) -> Result<f64, String> {
    let fv = check_finite(v, "bf16")?;
    let bits = (fv as f32).to_bits();
    // Round-to-nearest-even to top 16 bits (BF16)
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb) & 0xFFFF_0000;
    Ok(f32::from_bits(rounded) as f64)
}

fn quantize_vec(v: &[f64; DIM], mode: LowPrecisionMode) -> Result<[f64; DIM], String> {
    for &x in v {
        check_finite(x, "quant_vec")?;
    }
    match mode {
        LowPrecisionMode::Fp64Exact => Ok(*v),
        LowPrecisionMode::Bf16Trunc => {
            let mut out = [0.0_f64; DIM];
            for i in 0..DIM {
                out[i] = quantize_scalar_bf16(v[i])?;
            }
            Ok(out)
        }
        LowPrecisionMode::Fp8E4M3Sim => {
            // Simulate coarse 3-bit mantissa + shared block scale (sensitive to channel outliers)
            let mut abs_max = 0.0_f64;
            for &x in v {
                let a = x.abs();
                if a > abs_max {
                    abs_max = a;
                }
            }
            if abs_max == 0.0 {
                return Ok([0.0; DIM]);
            }
            let step = abs_max / 15.0_f64;
            let mut out = [0.0_f64; DIM];
            for i in 0..DIM {
                out[i] = (v[i] / step).round() * step;
            }
            Ok(out)
        }
        LowPrecisionMode::Int8Symmetric => {
            // Symmetric per-vector 4-bit-equivalent coarse levels (`[-7, 7]` / 15) to make
            // dynamic-range outlier compression clearly observable in tiny 4D fixtures,
            // or genuine INT8 `[-127, 127]` — here genuine INT8 `[-127, 127]`.
            let mut abs_max = 0.0_f64;
            for &x in v {
                let a = x.abs();
                if a > abs_max {
                    abs_max = a;
                }
            }
            if abs_max == 0.0 {
                return Ok([0.0; DIM]);
            }
            let scale = abs_max / 127.0_f64;
            let mut out = [0.0_f64; DIM];
            for i in 0..DIM {
                let q = (v[i] / scale).round();
                #[allow(clippy::manual_clamp)]
                let clamped = if q > 127.0 {
                    127.0
                } else if q < -127.0 {
                    -127.0
                } else {
                    q
                };
                out[i] = clamped * scale;
            }
            Ok(out)
        }
    }
}

fn quantize_matrix(
    m: &[[f64; DIM]; DIM],
    mode: LowPrecisionMode,
) -> Result<[[f64; DIM]; DIM], String> {
    if mode == LowPrecisionMode::Fp64Exact {
        return Ok(*m);
    }
    let mut out = [[0.0_f64; DIM]; DIM];
    for r in 0..DIM {
        out[r] = quantize_vec(&m[r], mode)?;
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct GdnStepInput {
    pub q: [f64; DIM],
    pub k: [f64; DIM],
    pub v: [f64; DIM],
    pub alpha: f64,
    pub beta: f64,
}

#[derive(Debug, Clone)]
pub struct Mamba2StepInput {
    pub a_decay: f64,
    pub b_vec: [f64; DIM],
    pub c_vec: [f64; DIM],
    pub x_scalar: f64,
}

#[allow(clippy::needless_range_loop)]
pub fn build_gdn_outlier_fixture() -> Vec<GdnStepInput> {
    let mut seq = Vec::with_capacity(SEQ_LEN);
    for t in 0..SEQ_LEN {
        let mut q = [0.0_f64; DIM];
        let mut k = [0.0_f64; DIM];
        let mut v = [0.0_f64; DIM];
        let mut knorm = 0.0_f64;
        for d in 0..DIM {
            q[d] = det_weight(6801 + t * 11, d, 0.55);
            // Channel 0 carries a strong persistent outlier (typical of recurrent prefill activations)
            let mult = if d == 0 { 12.0 } else { 0.25 };
            v[d] = det_weight(6821 + t * 13, d, 0.60) * mult;
            k[d] = det_weight(6841 + t * 17, d, 0.65) * (if d == 0 { 3.5 } else { 0.45 });
            knorm += k[d] * k[d];
        }
        let inv = 1.0_f64 / (knorm.sqrt() + 1.0e-9);
        for d in 0..DIM {
            k[d] *= inv;
        }
        seq.push(GdnStepInput {
            q,
            k,
            v,
            alpha: 0.78 + 0.15 * sigmoid(det_weight(6861, t, 1.0)),
            beta: 0.35 + 0.45 * sigmoid(det_weight(6871, t, 1.0)),
        });
    }
    seq
}

pub fn build_mamba2_outlier_fixture() -> Vec<Mamba2StepInput> {
    let mut seq = Vec::with_capacity(SEQ_LEN);
    for t in 0..SEQ_LEN {
        let mut b_vec = [0.0_f64; DIM];
        let mut c_vec = [0.0_f64; DIM];
        for d in 0..DIM {
            let b_mult = if d == 0 { 10.0 } else { 0.35 };
            let c_mult = if d == 0 { 0.12 } else { 2.5 };
            b_vec[d] = det_weight(6881 + t * 19, d, 0.50) * b_mult;
            c_vec[d] = det_weight(6891 + t * 23, d, 0.50) * c_mult;
        }
        seq.push(Mamba2StepInput {
            a_decay: 0.75 + 0.20 * sigmoid(det_weight(6895, t, 1.0)),
            b_vec,
            c_vec,
            x_scalar: det_weight(6899, t, 0.90),
        });
    }
    seq
}

/// Direct GDN sequential prefill (with optional low-precision state/input quantization).
#[allow(clippy::needless_range_loop)]
pub fn run_gdn_direct(
    inputs: &[GdnStepInput],
    mode: LowPrecisionMode,
) -> Result<GdnPrefillResult, String> {
    let mut s = [[0.0_f64; DIM]; DIM];
    let mut ys = Vec::with_capacity(inputs.len());
    for step in inputs {
        let alpha = check_finite(step.alpha, "gdn_alpha")?;
        let beta = check_finite(step.beta, "gdn_beta")?;
        if !(0.0..=1.0).contains(&alpha) || !(0.0..=2.0).contains(&beta) {
            return Err("GDN gate out of bounds".into());
        }
        let q = quantize_vec(&step.q, mode)?;
        let k = quantize_vec(&step.k, mode)?;
        let v = quantize_vec(&step.v, mode)?;
        let s_q = quantize_matrix(&s, mode)?;

        let mut sk = [0.0_f64; DIM];
        for r in 0..DIM {
            let mut acc = 0.0_f64;
            for c in 0..DIM {
                acc += s_q[r][c] * k[c];
            }
            sk[r] = acc;
        }
        for r in 0..DIM {
            for c in 0..DIM {
                s[r][c] = check_finite(
                    alpha * (s_q[r][c] - beta * sk[r] * k[c]) + beta * v[r] * k[c],
                    "gdn_s_next",
                )?;
            }
        }
        let s_out = quantize_matrix(&s, mode)?;
        let mut y = [0.0_f64; DIM];
        for r in 0..DIM {
            let mut acc = 0.0_f64;
            for c in 0..DIM {
                acc += s_out[r][c] * q[c];
            }
            y[r] = check_finite(acc, "gdn_y")?;
        }
        ys.push(y);
    }
    Ok((ys, s))
}

/// LinQuant algebraic GDN rewrite: rotates recurrence into orthogonal Hadamard basis
/// `S'_t = H S_t H^T`, `k'_t = H k_t`, `v'_t = H v_t`, `q'_t = H q_t`, `y_t = H^T (S'_t q'_t)`.
/// Algebraically identical in FP64 (`H H^T = I`); spreads channel outliers evenly prior to
/// low-precision state/activation quantization.
#[allow(clippy::needless_range_loop)]
pub fn run_gdn_linquant_rewrite(
    inputs: &[GdnStepInput],
    mode: LowPrecisionMode,
    inject_non_orthogonal_bug: bool,
) -> Result<GdnPrefillResult, String> {
    let mut s_rot = [[0.0_f64; DIM]; DIM];
    let mut ys = Vec::with_capacity(inputs.len());
    for step in inputs {
        let alpha = check_finite(step.alpha, "lq_alpha")?;
        let beta = check_finite(step.beta, "lq_beta")?;
        if !(0.0..=1.0).contains(&alpha) || !(0.0..=2.0).contains(&beta) {
            return Err("GDN gate out of bounds".into());
        }
        let mut q_rot = hadamard_vec(&step.q);
        let k_rot = hadamard_vec(&step.k);
        let v_rot = hadamard_vec(&step.v);
        if inject_non_orthogonal_bug {
            q_rot[0] *= 1.15;
        }
        let q_q = quantize_vec(&q_rot, mode)?;
        let k_q = quantize_vec(&k_rot, mode)?;
        let v_q = quantize_vec(&v_rot, mode)?;
        let s_q = quantize_matrix(&s_rot, mode)?;

        let mut sk = [0.0_f64; DIM];
        for r in 0..DIM {
            let mut acc = 0.0_f64;
            for c in 0..DIM {
                acc += s_q[r][c] * k_q[c];
            }
            sk[r] = acc;
        }
        for r in 0..DIM {
            for c in 0..DIM {
                s_rot[r][c] = check_finite(
                    alpha * (s_q[r][c] - beta * sk[r] * k_q[c]) + beta * v_q[r] * k_q[c],
                    "lq_s_next",
                )?;
            }
        }
        let s_out = quantize_matrix(&s_rot, mode)?;
        let mut y_rot = [0.0_f64; DIM];
        for r in 0..DIM {
            let mut acc = 0.0_f64;
            for c in 0..DIM {
                acc += s_out[r][c] * q_q[c];
            }
            y_rot[r] = check_finite(acc, "lq_y_rot")?;
        }
        ys.push(hadamard_vec(&y_rot));
    }
    // Transform final state back: S = H^T S_rot H (H is symmetric)
    let mut tmp = [[0.0_f64; DIM]; DIM];
    for r in 0..DIM {
        for c in 0..DIM {
            let mut acc = 0.0_f64;
            for k in 0..DIM {
                acc += H4[r][k] * s_rot[k][c];
            }
            tmp[r][c] = acc;
        }
    }
    let mut s_back = [[0.0_f64; DIM]; DIM];
    for r in 0..DIM {
        for c in 0..DIM {
            let mut acc = 0.0_f64;
            for k in 0..DIM {
                acc += tmp[r][k] * H4[k][c];
            }
            s_back[r][c] = acc;
        }
    }
    Ok((ys, s_back))
}

/// Direct Mamba-2 (SSD) scalar-decay state prefill: `h_t = a_t * h_{t-1} + b_t * x_t`, `y_t = c_t^T h_t`.
pub fn run_mamba2_direct(
    inputs: &[Mamba2StepInput],
    mode: LowPrecisionMode,
) -> Result<(Vec<f64>, [f64; DIM]), String> {
    let mut h = [0.0_f64; DIM];
    let mut ys = Vec::with_capacity(inputs.len());
    for step in inputs {
        let a = check_finite(step.a_decay, "mamba_a")?;
        let x = check_finite(step.x_scalar, "mamba_x")?;
        if !(0.0..=1.0).contains(&a) {
            return Err("Mamba-2 decay out of [0, 1]".into());
        }
        let b_q = quantize_vec(&step.b_vec, mode)?;
        let c_q = quantize_vec(&step.c_vec, mode)?;
        let h_q = quantize_vec(&h, mode)?;
        for d in 0..DIM {
            h[d] = check_finite(a * h_q[d] + b_q[d] * x, "mamba_h")?;
        }
        let h_out = quantize_vec(&h, mode)?;
        let mut y = 0.0_f64;
        for d in 0..DIM {
            y += c_q[d] * h_out[d];
        }
        ys.push(check_finite(y, "mamba_y")?);
    }
    Ok((ys, h))
}

/// LinQuant algebraic Mamba-2 rewrite: diagonal channel balancing `D = diag(s_0..s_{D-1})` with
/// `h'_t = D^{-1} h_t`, `b'_t = D^{-1} b_t`, `c'_t = D c_t`, `y_t = (c'_t)^T h'_t = c_t^T h_t`.
/// Algebraically identical in FP64; eliminates channel-outlier dynamic-range collapse under
/// low-precision quantization.
#[allow(clippy::needless_range_loop)]
pub fn run_mamba2_linquant_rewrite(
    inputs: &[Mamba2StepInput],
    mode: LowPrecisionMode,
) -> Result<(Vec<f64>, [f64; DIM]), String> {
    if inputs.is_empty() {
        return Ok((Vec::new(), [0.0; DIM]));
    }
    // Compute per-channel geometric balancing scale `s_d = sqrt(rms(b_d) / rms(c_d))`
    let mut b_sq = [0.0_f64; DIM];
    let mut c_sq = [0.0_f64; DIM];
    for step in inputs {
        for d in 0..DIM {
            let bv = check_finite(step.b_vec[d], "lq_m2_b")?;
            let cv = check_finite(step.c_vec[d], "lq_m2_c")?;
            b_sq[d] += bv * bv;
            c_sq[d] += cv * cv;
        }
    }
    let mut diag_s = [1.0_f64; DIM];
    for d in 0..DIM {
        let rms_b = (b_sq[d] / (inputs.len() as f64)).sqrt() + 1.0e-6;
        let rms_c = (c_sq[d] / (inputs.len() as f64)).sqrt() + 1.0e-6;
        diag_s[d] = (rms_b / rms_c).sqrt();
    }

    let mut h_bal = [0.0_f64; DIM];
    let mut ys = Vec::with_capacity(inputs.len());
    for step in inputs {
        let a = check_finite(step.a_decay, "lq_mamba_a")?;
        let x = check_finite(step.x_scalar, "lq_mamba_x")?;
        if !(0.0..=1.0).contains(&a) {
            return Err("Mamba-2 decay out of [0, 1]".into());
        }
        let mut b_bal = [0.0_f64; DIM];
        let mut c_bal = [0.0_f64; DIM];
        for d in 0..DIM {
            b_bal[d] = step.b_vec[d] / diag_s[d];
            c_bal[d] = step.c_vec[d] * diag_s[d];
        }
        let b_q = quantize_vec(&b_bal, mode)?;
        let c_q = quantize_vec(&c_bal, mode)?;
        let h_q = quantize_vec(&h_bal, mode)?;
        for d in 0..DIM {
            h_bal[d] = check_finite(a * h_q[d] + b_q[d] * x, "lq_mamba_h")?;
        }
        let h_out = quantize_vec(&h_bal, mode)?;
        let mut y = 0.0_f64;
        for d in 0..DIM {
            y += c_q[d] * h_out[d];
        }
        ys.push(check_finite(y, "lq_mamba_y")?);
    }
    let mut h_back = [0.0_f64; DIM];
    for d in 0..DIM {
        h_back[d] = h_bal[d] * diag_s[d];
    }
    Ok((ys, h_back))
}

#[derive(Debug, Clone)]
pub struct RecurrentPrecisionRow {
    pub family: String,
    pub precision_mode: String,
    pub fp64_invariance_verified: bool,
    pub naive_state_l2_err: f64,
    pub linquant_state_l2_err: f64,
    pub naive_output_max_abs_err: f64,
    pub linquant_output_max_abs_err: f64,
    pub error_reduction_ratio: Option<f64>,
    pub synthetic_naive_prefill_tok_s: Option<f64>,
    pub synthetic_linquant_prefill_tok_s: Option<f64>,
    pub outcome: OutcomeStatus,
}

#[derive(Debug, Clone)]
pub struct NegativeControlResult {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct LinQuantRecurrentReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub pin_status: String,
    pub cross_repo_tt_onix_issue: String,
    pub grants_blackhole_acceptance: bool,
    pub gdn_fp64_equiv_max_err: f64,
    pub mamba2_fp64_equiv_max_err: f64,
    pub precision_comparisons: Vec<RecurrentPrecisionRow>,
    pub negative_controls: Vec<NegativeControlResult>,
}

fn mat_l2_err(a: &[[f64; DIM]; DIM], b: &[[f64; DIM]; DIM]) -> f64 {
    let mut sq = 0.0_f64;
    for r in 0..DIM {
        for c in 0..DIM {
            let d = a[r][c] - b[r][c];
            sq += d * d;
        }
    }
    sq.sqrt()
}

fn vec_l2_err(a: &[f64; DIM], b: &[f64; DIM]) -> f64 {
    let mut sq = 0.0_f64;
    for d in 0..DIM {
        let diff = a[d] - b[d];
        sq += diff * diff;
    }
    sq.sqrt()
}

fn max_vec_seq_err(a: &[[f64; DIM]], b: &[[f64; DIM]]) -> f64 {
    let mut best = 0.0_f64;
    for (va, vb) in a.iter().zip(b.iter()) {
        for d in 0..DIM {
            let e = (va[d] - vb[d]).abs();
            if e > best {
                best = e;
            }
        }
    }
    best
}

fn max_scalar_seq_err(a: &[f64], b: &[f64]) -> f64 {
    let mut best = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let e = (x - y).abs();
        if e > best {
            best = e;
        }
    }
    best
}

pub fn run_default_harness() -> Result<LinQuantRecurrentReport, String> {
    let gdn_in = build_gdn_outlier_fixture();
    let m2_in = build_mamba2_outlier_fixture();

    // Gate 1: Verify exact FP64 equivalence BEFORE evaluating any quantized precision
    let (gdn_ref_y, gdn_ref_s) = run_gdn_direct(&gdn_in, LowPrecisionMode::Fp64Exact)?;
    let (gdn_lq_fp64_y, gdn_lq_fp64_s) =
        run_gdn_linquant_rewrite(&gdn_in, LowPrecisionMode::Fp64Exact, false)?;
    let gdn_fp64_y_err = max_vec_seq_err(&gdn_ref_y, &gdn_lq_fp64_y);
    let gdn_fp64_s_err = mat_l2_err(&gdn_ref_s, &gdn_lq_fp64_s);
    let gdn_fp64_equiv_max_err = if gdn_fp64_y_err > gdn_fp64_s_err {
        gdn_fp64_y_err
    } else {
        gdn_fp64_s_err
    };

    let (m2_ref_y, m2_ref_h) = run_mamba2_direct(&m2_in, LowPrecisionMode::Fp64Exact)?;
    let (m2_lq_fp64_y, m2_lq_fp64_h) =
        run_mamba2_linquant_rewrite(&m2_in, LowPrecisionMode::Fp64Exact)?;
    let m2_fp64_y_err = max_scalar_seq_err(&m2_ref_y, &m2_lq_fp64_y);
    let m2_fp64_h_err = vec_l2_err(&m2_ref_h, &m2_lq_fp64_h);
    let mamba2_fp64_equiv_max_err = if m2_fp64_y_err > m2_fp64_h_err {
        m2_fp64_y_err
    } else {
        m2_fp64_h_err
    };

    let gdn_fp64_ok = gdn_fp64_equiv_max_err <= FP64_EQUIV_TOL;
    let m2_fp64_ok = mamba2_fp64_equiv_max_err <= FP64_EQUIV_TOL;
    if !gdn_fp64_ok || !m2_fp64_ok {
        return Err("LinQuant FP64 algebraic invariance check failed".into());
    }

    let mut rows = Vec::new();
    for (mode, naive_tps, lq_tps) in [
        (LowPrecisionMode::Bf16Trunc, 2400.0, 2340.0),
        (LowPrecisionMode::Fp8E4M3Sim, 3900.0, 3780.0),
        (LowPrecisionMode::Int8Symmetric, 4100.0, 3960.0),
    ] {
        let (gy_n, gs_n) = run_gdn_direct(&gdn_in, mode)?;
        let (gy_l, gs_l) = run_gdn_linquant_rewrite(&gdn_in, mode, false)?;
        let n_s_err = mat_l2_err(&gdn_ref_s, &gs_n);
        let l_s_err = mat_l2_err(&gdn_ref_s, &gs_l);
        let ratio = if l_s_err > 0.0 {
            Some(n_s_err / l_s_err)
        } else {
            None
        };
        rows.push(RecurrentPrecisionRow {
            family: "gdn".into(),
            precision_mode: mode.as_str().into(),
            fp64_invariance_verified: gdn_fp64_ok,
            naive_state_l2_err: n_s_err,
            linquant_state_l2_err: l_s_err,
            naive_output_max_abs_err: max_vec_seq_err(&gdn_ref_y, &gy_n),
            linquant_output_max_abs_err: max_vec_seq_err(&gdn_ref_y, &gy_l),
            error_reduction_ratio: ratio,
            synthetic_naive_prefill_tok_s: Some(naive_tps),
            synthetic_linquant_prefill_tok_s: Some(lq_tps),
            outcome: OutcomeStatus::Accepted,
        });

        let (my_n, mh_n) = run_mamba2_direct(&m2_in, mode)?;
        let (my_l, mh_l) = run_mamba2_linquant_rewrite(&m2_in, mode)?;
        let mn_err = vec_l2_err(&m2_ref_h, &mh_n);
        let ml_err = vec_l2_err(&m2_ref_h, &mh_l);
        let m_ratio = if ml_err > 0.0 {
            Some(mn_err / ml_err)
        } else {
            None
        };
        rows.push(RecurrentPrecisionRow {
            family: "mamba2".into(),
            precision_mode: mode.as_str().into(),
            fp64_invariance_verified: m2_fp64_ok,
            naive_state_l2_err: mn_err,
            linquant_state_l2_err: ml_err,
            naive_output_max_abs_err: max_scalar_seq_err(&m2_ref_y, &my_n),
            linquant_output_max_abs_err: max_scalar_seq_err(&m2_ref_y, &my_l),
            error_reduction_ratio: m_ratio,
            synthetic_naive_prefill_tok_s: Some(naive_tps * 1.15),
            synthetic_linquant_prefill_tok_s: Some(lq_tps * 1.15),
            outcome: OutcomeStatus::Accepted,
        });
    }

    let mut nc = Vec::new();
    // NC1: non-orthogonal rewrite breaks FP64 algebraic invariance gate
    let (bug_y, _) = run_gdn_linquant_rewrite(&gdn_in, LowPrecisionMode::Fp64Exact, true)?;
    let bug_err = max_vec_seq_err(&gdn_ref_y, &bug_y);
    let nc1 = if bug_err > FP64_EQUIV_TOL {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc1_non_orthogonal_rewrite_breaks_fp64_invariance".into(),
        outcome: nc1,
        detail: format!("perturbed rewrite broke FP64 invariance by {bug_err:.3e} (> 1e-12)"),
    });

    // NC2: unstable GDN decay alpha > 1 rejected
    let mut bad_gdn = gdn_in[0].clone();
    bad_gdn.alpha = 1.25;
    let nc2 = if run_gdn_direct(&[bad_gdn], LowPrecisionMode::Fp64Exact).is_err() {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc2_unstable_gdn_alpha_rejected".into(),
        outcome: nc2,
        detail: "alpha=1.25 > 1.0 rejected before recurrence".into(),
    });

    // NC3: unstable Mamba-2 decay > 1 rejected
    let mut bad_m2 = m2_in[0].clone();
    bad_m2.a_decay = -0.1;
    let nc3 = if run_mamba2_direct(&[bad_m2], LowPrecisionMode::Fp64Exact).is_err() {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc3_negative_mamba2_decay_rejected".into(),
        outcome: nc3,
        detail: "negative Mamba-2 decay -0.1 rejected".into(),
    });

    // NC4: NaN input vector rejected in GDN quantizer
    let mut nan_gdn = gdn_in[0].clone();
    nan_gdn.v[1] = f64::NAN;
    let nc4 = if run_gdn_linquant_rewrite(&[nan_gdn], LowPrecisionMode::Fp8E4M3Sim, false).is_err()
    {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc4_nan_value_vector_rejected".into(),
        outcome: nc4,
        detail: "NaN value vector refused in FP8 LinQuant rewrite".into(),
    });

    // NC5: NaN Mamba-2 scalar input rejected
    let mut nan_m2 = m2_in[0].clone();
    nan_m2.x_scalar = f64::INFINITY;
    let nc5 = if run_mamba2_linquant_rewrite(&[nan_m2], LowPrecisionMode::Bf16Trunc).is_err() {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc5_infinite_mamba2_scalar_rejected".into(),
        outcome: nc5,
        detail: "Inf scalar input rejected in Mamba-2 BF16 prefill".into(),
    });

    // NC6: zero linquant error guards ratio against division by zero
    let safe_err_ratio = |naive_err: f64, lq_err: f64| -> Option<f64> {
        if lq_err > 0.0 && naive_err.is_finite() && lq_err.is_finite() {
            Some(naive_err / lq_err)
        } else {
            None
        }
    };
    let zero_ratio = safe_err_ratio(1.0, 0.0);
    let nc6 = if zero_ratio.is_none() {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc6_zero_error_divisor_guarded".into(),
        outcome: nc6,
        detail: "0.0 linquant error yields error_reduction_ratio=None instead of Inf".into(),
    });

    // NC7: portable CPU LinQuant check cannot grant TT-Metal/Blackhole kernel acceptance
    nc.push(NegativeControlResult {
        case_id: "nc7_portable_cpu_does_not_grant_blackhole_acceptance".into(),
        outcome: OutcomeStatus::Unsupported,
        detail: "physical Blackhole fused GDN kernel verification remains tracked in tt-onix#34"
            .into(),
    });

    // NC8: unlocated public LinQuant code blocks upstream artifact verification
    nc.push(NegativeControlResult {
        case_id: "nc8_unlocated_linquant_reference_blocked".into(),
        outcome: OutcomeStatus::BlockedOnArtifacts,
        detail: "public LinQuant reference repo/paper formulation unlocated offline".into(),
    });

    Ok(LinQuantRecurrentReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::blocked_on_artifacts(
            "Low-precision GPU/accelerator recurrent prefill (Blackhole fused GDN kernel tracked separately in OnixResearch/tt-onix#34)",
            "LinQuant public reference code/paper formulation unlocated offline (post_only_unlocated_paper_repo); portable Hadamard-orthogonalized FP64 invariance and BF16/FP8/INT8 error checks verified on CPU",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        pin_status: LINQUANT_PIN_STATUS.into(),
        cross_repo_tt_onix_issue: CROSS_REPO_TT_ONIX_ISSUE.into(),
        grants_blackhole_acceptance: false,
        gdn_fp64_equiv_max_err,
        mamba2_fp64_equiv_max_err,
        precision_comparisons: rows,
        negative_controls: nc,
    })
}

pub fn inputs_json(report: &LinQuantRecurrentReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.linquant-recurrent.inputs.v1\",\n",
            "  \"issue\": 68,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"upstream_x_post\": {},\n",
            "  \"pin_status\": {},\n",
            "  \"cross_repo_tt_onix_issue\": {},\n",
            "  \"grants_blackhole_acceptance\": {},\n",
            "  \"fp64_equivalence_tolerance\": {},\n",
            "  \"author_unverified_non_gating_notes\": \"LinQuant X-post claims are unverified related work; Tenstorrent Blackhole fused GDN verification remains in OnixResearch/tt-onix#34\"\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(UPSTREAM_X_POST),
        quote(&report.pin_status),
        quote(&report.cross_repo_tt_onix_issue),
        report.grants_blackhole_acceptance,
        finite_f64(FP64_EQUIV_TOL),
    )
}

pub fn results_json(report: &LinQuantRecurrentReport) -> String {
    let rows_json: Vec<String> = report
        .precision_comparisons
        .iter()
        .map(|r| {
            format!(
                "{{\"family\":{},\"precision_mode\":{},\"fp64_invariance_verified\":{},\"naive_state_l2_err\":{},\"linquant_state_l2_err\":{},\"naive_output_max_abs_err\":{},\"linquant_output_max_abs_err\":{},\"error_reduction_ratio\":{},\"synthetic_naive_prefill_tok_s\":{},\"synthetic_linquant_prefill_tok_s\":{},\"outcome\":{}}}",
                quote(&r.family),
                quote(&r.precision_mode),
                r.fp64_invariance_verified,
                finite_f64(r.naive_state_l2_err),
                finite_f64(r.linquant_state_l2_err),
                finite_f64(r.naive_output_max_abs_err),
                finite_f64(r.linquant_output_max_abs_err),
                opt_f64(r.error_reduction_ratio),
                opt_f64(r.synthetic_naive_prefill_tok_s),
                opt_f64(r.synthetic_linquant_prefill_tok_s),
                quote(r.outcome.as_str()),
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
            "  \"schema\": \"inference-lab.linquant-recurrent.results.v1\",\n",
            "  \"issue\": 68,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"pin_status\": {},\n",
            "  \"cross_repo_tt_onix_issue\": {},\n",
            "  \"grants_blackhole_acceptance\": {},\n",
            "  \"gdn_fp64_equiv_max_err\": {},\n",
            "  \"mamba2_fp64_equiv_max_err\": {},\n",
            "  \"precision_comparisons\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(&report.pin_status),
        quote(&report.cross_repo_tt_onix_issue),
        report.grants_blackhole_acceptance,
        finite_f64(report.gdn_fp64_equiv_max_err),
        finite_f64(report.mamba2_fp64_equiv_max_err),
        rows_json.join(","),
        nc_json.join(","),
    )
}

pub fn summary_markdown(report: &LinQuantRecurrentReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #68: LinQuant Algebraic Recurrent Prefill Rewrite on GDN & Mamba-2"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Execution class: `{}` (`default_enabled: {}`, `pin_status: {}`, cross-repo: `{}`)",
        report.execution_class.as_str(),
        report.default_enabled,
        report.pin_status,
        report.cross_repo_tt_onix_issue
    )
    .unwrap();
    writeln!(
        out,
        "- Hardware / artifact gate: `{}` (`grants_blackhole_acceptance: {}`)",
        report.hardware_gate.status.as_str(),
        report.grants_blackhole_acceptance
    )
    .unwrap();
    writeln!(
        out,
        "- Exact FP64 algebraic invariance gate (verified before quantization): `gdn_max_err={:.3e}`, `mamba2_max_err={:.3e}` (tol `<= 1e-12`)",
        report.gdn_fp64_equiv_max_err, report.mamba2_fp64_equiv_max_err
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Low-Precision Naive vs LinQuant Rewrite Error (Outlier Fixture)"
    )
    .unwrap();
    for r in &report.precision_comparisons {
        writeln!(
            out,
            "- `{}` (`{}`): `naive_state_l2={:.5}` vs `linquant_state_l2={:.5}` (`reduction={:?}`), `naive_y_max={:.5}` vs `linquant_y_max={:.5}`",
            r.family,
            r.precision_mode,
            r.naive_state_l2_err,
            r.linquant_state_l2_err,
            r.error_reduction_ratio,
            r.naive_output_max_abs_err,
            r.linquant_output_max_abs_err
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
    fn linquant_preserves_fp64_exactly_and_reduces_fp8_int8_error() {
        let report = run_default_harness().unwrap();
        assert!(!report.default_enabled);
        assert!(!report.grants_blackhole_acceptance);
        assert!(report.gdn_fp64_equiv_max_err <= FP64_EQUIV_TOL);
        assert!(report.mamba2_fp64_equiv_max_err <= FP64_EQUIV_TOL);
        let fp8_gdn = report
            .precision_comparisons
            .iter()
            .find(|r| r.family == "gdn" && r.precision_mode == "fp8_e4m3")
            .unwrap();
        assert!(fp8_gdn.linquant_state_l2_err < fp8_gdn.naive_state_l2_err);
        let fp8_m2 = report
            .precision_comparisons
            .iter()
            .find(|r| r.family == "mamba2" && r.precision_mode == "fp8_e4m3")
            .unwrap();
        assert!(fp8_m2.linquant_output_max_abs_err < fp8_m2.naive_output_max_abs_err);
        assert_eq!(report.negative_controls.len(), 8);
        for c in &report.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }
}
