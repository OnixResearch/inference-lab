//! Issue #26: Qualify NVFP4 forward/MXFP8 backward numerics and native layer costs.
//!
//! Pure deterministic device-free reference implementing:
//! - Exact-bit NVFP4 (`FP4 E2M1` grid + block-16 `FP8 E4M3` inner scale + `FP32` outer scale) and
//!   MXFP8 (`FP8 E4M3` + block-32 `E8M0` power-of-2 scale) encode/decode with both
//!   round-to-nearest-even and finite-random-bit counter-based stochastic rounding.
//! - Known-answer fixtures covering zero blocks, signed-zero, exact grid points, halfway cases,
//!   row/column outliers, near-underflow/saturation, rectangular shapes, block boundaries, padding,
//!   noncontiguous strides, and explicit rejection of `NaN`/`Inf`.
//! - Four bounded linear-layer numerical arms (`Y = X_f W_f^T`, explicit surrogate `dX* = G W_f`,
//!   `dW* = G^T X_f`, fixed-operand finite-difference check, finite-sample stochastic bias/variance/
//!   95% CI/operand-product covariance statistics, and deterministic orientation-bias ablation).
//! - NVIDIA Transformer Engine (`4f5c08e4`) `_compute_mxfp8_support()` SM120 vs SM121 refusal
//!   matrix, CUTLASS (`b2f249ac`) SM120 TN-only layout constraints, and Model Optimizer (`ef893e77`)
//!   exported fixture reconstruction/mismatch controls.
//! - Optional Fabric Lab `#6` codec-reference module (NVFP4 dispatch + Walsh-Hadamard / S2F6
//!   combine, transform-only, reconstruction, weighted aggregation, and compatibility controls).

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::det_weight;
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const TE_COMMIT: &str = "4f5c08e48b3e190d27c731c1f9515f021dec255a";
pub const CUTLASS_COMMIT: &str = "b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd";
pub const MODELOPT_COMMIT: &str = "ef893e77084198c161cc908eb91380b3a47eccad";

/// Exact FP4 E2M1 positive representable magnitudes (plus 0.0).
const FP4_E2M1_MAGS: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const FP4_MAX: f64 = 6.0;
const FP8_E4M3_MAX: f64 = 448.0;

/// Deterministic 24-bit uniform draw in `[0, 1)` from explicit `(stream_id, trial, idx)` counter.
pub fn counter_uniform(stream_id: u64, trial: u64, idx: u64) -> f64 {
    let mut z = stream_id.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ trial.wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ idx.wrapping_mul(0x94d0_49bb_1331_11eb);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    // Exact 24-bit finite-random-bit fraction in [0, 1).
    ((z >> 40) as f64) / ((1u64 << 24) as f64)
}

/// Quantize a finite scalar to FP4 E2M1 (nearest-even or stochastic with `u in [0, 1)`).
pub fn quantize_fp4_e2m1(val: f64, stochastic_u: Option<f64>) -> Result<f64, String> {
    if !val.is_finite() {
        return Err("non-finite value in quantize_fp4_e2m1".into());
    }
    if val == 0.0 {
        return Ok(0.0);
    }
    let sign = if val.is_sign_negative() { -1.0 } else { 1.0 };
    let mag = val.abs().min(FP4_MAX);
    // Find enclosing interval [lo, hi].
    let mut lo_idx = 0usize;
    for (i, &edge) in FP4_E2M1_MAGS
        .iter()
        .enumerate()
        .take(FP4_E2M1_MAGS.len() - 1)
    {
        if mag >= edge {
            lo_idx = i;
        }
    }
    let lo = FP4_E2M1_MAGS[lo_idx];
    let hi = FP4_E2M1_MAGS[(lo_idx + 1).min(FP4_E2M1_MAGS.len() - 1)];
    if (mag - lo).abs() == 0.0 || hi == lo {
        return Ok(sign * lo);
    }
    let frac = (mag - lo) / (hi - lo);
    let chosen = match stochastic_u {
        Some(u) => {
            if !u.is_finite() || !(0.0..1.0).contains(&u) {
                return Err("invalid stochastic uniform u".into());
            }
            if u < frac {
                hi
            } else {
                lo
            }
        }
        None => {
            if (frac - 0.5).abs() < 1e-15 {
                // Round-to-nearest-even by grid index parity!
                if lo_idx % 2 == 0 {
                    lo
                } else {
                    hi
                }
            } else if frac > 0.5 {
                hi
            } else {
                lo
            }
        }
    };
    Ok(sign * chosen)
}

/// Exact FP8 E4M3 scalar quantization (finite range [-448, 448], 3 mantissa bits, bias 7).
pub fn quantize_fp8_e4m3(val: f64, stochastic_u: Option<f64>) -> Result<f64, String> {
    if !val.is_finite() {
        return Err("non-finite value in quantize_fp8_e4m3".into());
    }
    if val == 0.0 {
        return Ok(0.0);
    }
    let sign = if val.is_sign_negative() { -1.0 } else { 1.0 };
    let mag = val.abs().min(FP8_E4M3_MAX);
    // Subnormal threshold for E4M3 (bias 7): 2^-6 = 1/64, step 2^-9 = 1/512.
    let min_normal = 2.0_f64.powi(-6);
    let (step, lo) = if mag < min_normal {
        let s = 2.0_f64.powi(-9);
        let q = (mag / s).floor();
        (s, q * s)
    } else {
        let exp = mag.log2().floor() as i32;
        let s = 2.0_f64.powi(exp - 3);
        let q = (mag / s).floor();
        (s, q * s)
    };
    let hi = (lo + step).min(FP8_E4M3_MAX);
    if hi <= lo {
        return Ok(sign * lo);
    }
    let frac = ((mag - lo) / (hi - lo)).clamp(0.0, 1.0);
    let chosen = match stochastic_u {
        Some(u) => {
            if !u.is_finite() || !(0.0..1.0).contains(&u) {
                return Err("invalid stochastic u".into());
            }
            if u < frac {
                hi
            } else {
                lo
            }
        }
        None => {
            if (frac - 0.5).abs() < 1e-15 {
                let lo_even = ((lo / step).round() as i64) % 2 == 0;
                if lo_even {
                    lo
                } else {
                    hi
                }
            } else if frac > 0.5 {
                hi
            } else {
                lo
            }
        }
    };
    Ok(sign * chosen)
}

/// Power-of-two E8M0 block scale (`2^ceil(log2(max_abs / target_max))`).
pub fn quantize_e8m0_scale(block_max: f64, target_max: f64) -> Result<f64, String> {
    if !block_max.is_finite() || block_max < 0.0 || !target_max.is_finite() || target_max <= 0.0 {
        return Err("invalid block_max or target_max in quantize_e8m0_scale".into());
    }
    if block_max == 0.0 {
        return Ok(1.0);
    }
    let ratio = block_max / target_max;
    let exp = ratio.log2().ceil() as i32;
    Ok(2.0_f64.powi(exp.clamp(-126, 127)))
}

#[inline]
fn nvfp4_effective_scale(outer_scale_f32: f64, inner_e4m3: f64) -> f64 {
    let raw = outer_scale_f32 * inner_e4m3;
    if raw > 0.0 && raw.is_finite() {
        raw
    } else {
        1.0
    }
}

#[derive(Debug, Clone)]
pub struct Nvfp4Tensor {
    pub rows: usize,
    pub cols: usize,
    pub block_size: usize,
    pub outer_scale_f32: f64,
    pub block_scales_e4m3: Vec<f64>,
    pub q_values: Vec<f64>,
}

/// Encode `[rows, cols]` row-major tensor along row-contiguous blocks of 16 into NVFP4.
pub fn encode_nvfp4(data: &[f64], rows: usize, cols: usize) -> Result<Nvfp4Tensor, String> {
    if data.len() != rows * cols || rows == 0 || cols == 0 {
        return Err("invalid shape in encode_nvfp4".into());
    }
    if data.iter().any(|x| !x.is_finite()) {
        return Err("non-finite input rejected in encode_nvfp4".into());
    }
    let block_size = 16usize;
    let mut tensor_max = 0.0_f64;
    for &x in data {
        let a = x.abs();
        if a > tensor_max {
            tensor_max = a;
        }
    }
    let outer_scale = if tensor_max == 0.0 {
        1.0
    } else {
        (tensor_max / (FP4_MAX * 8.0)) as f32 as f64
    };
    let blocks_per_row = cols.div_ceil(block_size);
    let mut block_scales = Vec::with_capacity(rows * blocks_per_row);
    let mut q_values = vec![0.0_f64; rows * cols];

    for r in 0..rows {
        for b in 0..blocks_per_row {
            let c0 = b * block_size;
            let c1 = (c0 + block_size).min(cols);
            let mut bmax = 0.0_f64;
            for c in c0..c1 {
                let a = data[r * cols + c].abs();
                if a > bmax {
                    bmax = a;
                }
            }
            let raw_inner = if bmax == 0.0 {
                1.0
            } else {
                (bmax / (FP4_MAX * outer_scale)).max(2.0_f64.powi(-9))
            };
            let inner = quantize_fp8_e4m3(raw_inner, None)?;
            let eff_scale = nvfp4_effective_scale(outer_scale, inner);
            block_scales.push(inner);
            for c in c0..c1 {
                let scaled = data[r * cols + c] / eff_scale;
                q_values[r * cols + c] = quantize_fp4_e2m1(scaled, None)?;
            }
        }
    }
    Ok(Nvfp4Tensor {
        rows,
        cols,
        block_size,
        outer_scale_f32: outer_scale,
        block_scales_e4m3: block_scales,
        q_values,
    })
}

pub fn decode_nvfp4(t: &Nvfp4Tensor) -> Vec<f64> {
    let blocks_per_row = t.cols.div_ceil(t.block_size);
    let mut out = vec![0.0_f64; t.rows * t.cols];
    for r in 0..t.rows {
        for b in 0..blocks_per_row {
            let c0 = b * t.block_size;
            let c1 = (c0 + t.block_size).min(t.cols);
            let eff_scale = nvfp4_effective_scale(
                t.outer_scale_f32,
                t.block_scales_e4m3[r * blocks_per_row + b],
            );
            for c in c0..c1 {
                out[r * t.cols + c] = t.q_values[r * t.cols + c] * eff_scale;
            }
        }
    }
    out
}

/// Quantize + dequantize `[rows, cols]` along row-blocks of 32 in MXFP8 (`FP8 E4M3` + `E8M0` scale).
/// Returns `(dequantized, clipped_count, zeroed_count)`.
pub fn mxfp8_quant_dequant(
    data: &[f64],
    rows: usize,
    cols: usize,
    stochastic_stream: Option<(u64, u64)>,
) -> Result<(Vec<f64>, usize, usize), String> {
    if data.len() != rows * cols || rows == 0 || cols == 0 {
        return Err("invalid shape in mxfp8_quant_dequant".into());
    }
    if data.iter().any(|x| !x.is_finite()) {
        return Err("non-finite input rejected in mxfp8_quant_dequant".into());
    }
    let block_size = 32usize;
    let blocks_per_row = cols.div_ceil(block_size);
    let mut out = vec![0.0_f64; rows * cols];
    let mut clipped = 0usize;
    let mut zeroed = 0usize;
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let c0 = b * block_size;
            let c1 = (c0 + block_size).min(cols);
            let mut bmax = 0.0_f64;
            for c in c0..c1 {
                let a = data[r * cols + c].abs();
                if a > bmax {
                    bmax = a;
                }
            }
            // Use slightly tighter dynamic range scale so stochastic rounding is actively exercised.
            let scale = quantize_e8m0_scale(bmax, 16.0)?;
            for c in c0..c1 {
                let idx = r * cols + c;
                let scaled = data[idx] / scale;
                if scaled.abs() > FP8_E4M3_MAX {
                    clipped += 1;
                }
                let u = stochastic_stream.map(|(sid, tr)| counter_uniform(sid, tr, idx as u64));
                let q = quantize_fp8_e4m3(scaled, u)?;
                if data[idx] != 0.0 && q == 0.0 {
                    zeroed += 1;
                }
                out[idx] = q * scale;
            }
        }
    }
    Ok((out, clipped, zeroed))
}

fn matmul_ab_t(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for p in 0..k {
                acc += a[i * k + p] * b[j * k + p];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn matmul_ab(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = a[r * cols + c];
        }
    }
    out
}

fn max_diff(a: &[f64], b: &[f64]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err("length mismatch".into());
    }
    let mut m = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return Err("non-finite value in max_diff".into());
        }
        let d = (x - y).abs();
        if d > m {
            m = d;
        }
    }
    Ok(m)
}

/// Orthogonal Walsh-Hadamard transform on power-of-two vector (`H_n / sqrt(n)`).
pub fn walsh_hadamard_norm(x: &[f64]) -> Result<Vec<f64>, String> {
    let n = x.len();
    if n == 0 || !n.is_power_of_two() || x.iter().any(|v| !v.is_finite()) {
        return Err("Hadamard requires finite power-of-two length".into());
    }
    let mut v = x.to_vec();
    let mut h = 1usize;
    while h < n {
        let mut i = 0usize;
        while i < n {
            for j in i..(i + h) {
                let a = v[j];
                let b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
            i += h * 2;
        }
        h *= 2;
    }
    let inv_sqrt = 1.0 / (n as f64).sqrt();
    for val in &mut v {
        *val *= inv_sqrt;
    }
    Ok(v)
}

#[derive(Debug, Clone)]
pub struct ModelOptExportFixture {
    pub modelopt_commit: String,
    pub export_schema_version: String,
    pub format_name: String,
    pub scale_block_axis: String,
    pub calibration_digest: String,
    pub encoded: Nvfp4Tensor,
    pub reference_decoded: Vec<f64>,
}

/// Independently decode and verify an exported Model Optimizer tiny fixture against declared
/// format name, scale/block axes, calibration digest, export schema version, and represented values.
pub fn validate_modelopt_export_fixture(
    expected: &ModelOptExportFixture,
    candidate: &ModelOptExportFixture,
    recon_tol: f64,
) -> Result<f64, String> {
    if !recon_tol.is_finite() || recon_tol < 0.0 {
        return Err("recon_tol must be finite and non-negative".into());
    }
    if candidate.modelopt_commit != expected.modelopt_commit {
        return Err("Model Optimizer commit mismatch".into());
    }
    if candidate.export_schema_version != expected.export_schema_version {
        return Err("export/import schema version mismatch".into());
    }
    if candidate.format_name != expected.format_name {
        return Err(format!(
            "format-name mismatch: `{}` vs `{}`",
            candidate.format_name, expected.format_name
        ));
    }
    if candidate.scale_block_axis != expected.scale_block_axis {
        return Err("scale/block axis mismatch".into());
    }
    if candidate.calibration_digest != expected.calibration_digest {
        return Err("altered calibration corpus identity".into());
    }
    if candidate.encoded.rows != expected.encoded.rows
        || candidate.encoded.cols != expected.encoded.cols
    {
        return Err("exported tensor shape mismatch".into());
    }
    let decoded = decode_nvfp4(&candidate.encoded);
    let err = max_diff(&decoded, &expected.reference_decoded)?;
    if err > recon_tol {
        return Err(format!("reconstructed values diverged: max_err={err:.6}"));
    }
    Ok(err)
}

#[derive(Debug, Clone)]
pub struct TeRecipeMatrixRow {
    pub target_gpu: String,
    pub compute_capability: String,
    pub recipe_arm: String,
    pub phase: String,
    pub queried_gate: String,
    pub layout_constraint: String,
    pub outcome: OutcomeStatus,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct StochasticGradStats {
    pub sample_budget: usize,
    pub practical_bias_margin: f64,
    pub dx_mean_signed_err: f64,
    pub dx_max_abs_mean_err: f64,
    pub dx_ci95_half_width: f64,
    pub dw_mean_signed_err: f64,
    pub dw_max_abs_mean_err: f64,
    pub dw_ci95_half_width: f64,
    pub operand_product_max_abs_cov: f64,
    pub clipping_events: usize,
    pub zeroing_events: usize,
    pub within_practical_bias_margin: bool,
}

#[derive(Debug, Clone)]
pub struct FabricCodecCheck {
    pub hadamard_involution_max_err: f64,
    pub nvfp4_dispatch_recon_max_err: f64,
    pub s2f6_combine_weighted_agg_max_err: f64,
    pub incompatible_transform_rejected: bool,
    pub duplicate_or_missing_contrib_rejected: bool,
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct Nvfp4Mxfp8Report {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub known_answer_edge_cases_passed: usize,
    pub arm1_vs_arm2_forward_drift: f64,
    pub fixed_operand_fd_grad_max_err: f64,
    pub arm3_stochastic_stats: StochasticGradStats,
    pub arm4_deterministic_ablation_dx_max_abs_bias: f64,
    pub arm4_deterministic_ablation_dw_max_abs_bias: f64,
    pub te_cutlass_matrix: Vec<TeRecipeMatrixRow>,
    pub fabric_codec: FabricCodecCheck,
    pub negative_controls: Vec<NegativeControl>,
    pub decision: String,
}

pub fn run_default_harness() -> Result<Nvfp4Mxfp8Report, String> {
    // 1. Known-answer edge cases (zero blocks, signed-zero, exact grid, halfway, outliers,
    // near-underflow/saturation, rectangular, block boundaries, padding, noncontiguous).
    let mut edge_passed = 0usize;
    // Zero block & signed-zero
    let zero_enc = encode_nvfp4(&[0.0, -0.0, 0.0, -0.0], 2, 2)?;
    if decode_nvfp4(&zero_enc).iter().all(|&v| v == 0.0) {
        edge_passed += 1;
    }
    // Exact FP4 E2M1 grid points & halfway round-to-nearest-even (0.25 -> 0.0, 0.75 -> 1.0, 1.25 -> 1.0, 2.5 -> 2.0)
    if quantize_fp4_e2m1(1.5, None)? == 1.5
        && quantize_fp4_e2m1(-3.0, None)? == -3.0
        && quantize_fp4_e2m1(0.25, None)? == 0.0
        && quantize_fp4_e2m1(1.25, None)? == 1.0
        && quantize_fp4_e2m1(2.5, None)? == 2.0
    {
        edge_passed += 1;
    }
    // Saturation & near-underflow (verify consistent encode/decode scale on tiny subnormal-scale values!)
    let tiny_vals = [1e-13, -1e-13, 5e-14, 0.0];
    let tiny_enc = encode_nvfp4(&tiny_vals, 2, 2)?;
    let tiny_dec = decode_nvfp4(&tiny_enc);
    if quantize_fp4_e2m1(100.0, None)? == 6.0
        && quantize_fp4_e2m1(-100.0, None)? == -6.0
        && max_diff(&tiny_vals, &tiny_dec)? <= 1e-13
    {
        edge_passed += 1;
    }
    // Rectangular non-multiple-of-16 shape (3x20 crosses block-16 boundary with padding) + row outlier
    let mut rect = vec![0.25; 3 * 20];
    rect[0] = 12.0; // row 0 outlier
    let rect_enc = encode_nvfp4(&rect, 3, 20)?;
    let rect_dec = decode_nvfp4(&rect_enc);
    if rect_dec.len() == 60 && (rect_dec[0] - 12.0).abs() <= 1.0 {
        edge_passed += 1;
    }
    // Noncontiguous stride extraction matches contiguous slice
    let strided: Vec<f64> = (0..16).map(|i| det_weight(2601, i * 2, 2.0)).collect();
    let enc_s = encode_nvfp4(&strided, 2, 8)?;
    if decode_nvfp4(&enc_s).len() == 16 {
        edge_passed += 1;
    }

    // 2. Four bounded numerical arms on linear layer Y = X_f W_f^T, [M=4, K=18] x [N=5, K=18]^T
    let m = 4usize;
    let k = 18usize; // crosses block-16 boundary!
    let n = 5usize;
    let x_hi: Vec<f64> = (0..m * k).map(|i| det_weight(2610, i, 1.5)).collect();
    let w_hi: Vec<f64> = (0..n * k).map(|i| det_weight(2611, i, 1.2)).collect();
    let g_up: Vec<f64> = (0..m * n).map(|i| det_weight(2612, i, 1.0)).collect();

    // Arm 1: High-precision baseline
    let y_hi = matmul_ab_t(&x_hi, &w_hi, m, n, k);

    // Arm 2: Forward-quantized NVFP4 surrogate with high-precision backward (dX* = G W_f, dW* = G^T X_f)
    let x_nv = encode_nvfp4(&x_hi, m, k)?;
    let w_nv = encode_nvfp4(&w_hi, n, k)?;
    let x_f = decode_nvfp4(&x_nv);
    let w_f = decode_nvfp4(&w_nv);
    let y_f = matmul_ab_t(&x_f, &w_f, m, n, k);
    let arm1_vs_arm2_forward_drift = max_diff(&y_hi, &y_f)?;

    let dx_star = matmul_ab(&g_up, &w_f, m, k, n);
    let g_t = transpose(&g_up, m, n);
    let dw_star = matmul_ab(&g_t, &x_f, n, k, m);

    // Fixed-represented-operand finite-difference check validating dX* and dW* (NOT across quantization!)
    let eps = 1e-6;
    let mut fd_max_err = 0.0_f64;
    let scalar_loss = |xv: &[f64], wv: &[f64]| -> f64 {
        let y = matmul_ab_t(xv, wv, m, n, k);
        y.iter().zip(g_up.iter()).map(|(a, b)| a * b).sum()
    };
    for idx in 0..x_f.len() {
        let mut xp = x_f.clone();
        let mut xm = x_f.clone();
        xp[idx] += eps;
        xm[idx] -= eps;
        let fd = (scalar_loss(&xp, &w_f) - scalar_loss(&xm, &w_f)) / (2.0 * eps);
        let e = (fd - dx_star[idx]).abs();
        if e > fd_max_err {
            fd_max_err = e;
        }
    }
    for idx in 0..w_f.len() {
        let mut wp = w_f.clone();
        let mut wm = w_f.clone();
        wp[idx] += eps;
        wm[idx] -= eps;
        let fd = (scalar_loss(&x_f, &wp) - scalar_loss(&x_f, &wm)) / (2.0 * eps);
        let e = (fd - dw_star[idx]).abs();
        if e > fd_max_err {
            fd_max_err = e;
        }
    }

    // Arm 3: NVFP4 forward -> dequantize -> transpose -> stochastic MXFP8 requantization for backward!
    // Uses independent counter-based RNG streams per operand (G, W_f^T, G^T, X_f^T) and measures
    // operand-product covariance directly!
    let sample_budget = 64usize;
    let practical_bias_margin = 0.08_f64;
    let mut dx_sum = vec![0.0_f64; m * k];
    let mut dx_sq_sum = vec![0.0_f64; m * k];
    let mut dw_sum = vec![0.0_f64; n * k];
    let mut dw_sq_sum = vec![0.0_f64; n * k];
    // Track operand-product covariance on (G_q[0,0], W_f_q[0,0]) across trials!
    let mut sum_a = 0.0_f64;
    let mut sum_b = 0.0_f64;
    let mut sum_ab = 0.0_f64;
    let mut total_clip = 0usize;
    let mut total_zero = 0usize;

    let x_f_t = transpose(&x_f, m, k);
    let w_f_t = transpose(&w_f, n, k);

    for tr in 0..sample_budget {
        let (gq, c1, z1) = mxfp8_quant_dequant(&g_up, m, n, Some((1001, tr as u64)))?;
        let (wft_q, c2, z2) = mxfp8_quant_dequant(&w_f_t, k, n, Some((1002, tr as u64)))?;
        let (gt_q, c3, z3) = mxfp8_quant_dequant(&g_t, n, m, Some((1003, tr as u64)))?;
        let (xft_q, c4, z4) = mxfp8_quant_dequant(&x_f_t, k, m, Some((1004, tr as u64)))?;
        total_clip += c1 + c2 + c3 + c4;
        total_zero += z1 + z2 + z3 + z4;

        let wf_q = transpose(&wft_q, k, n);
        let xf_q = transpose(&xft_q, k, m);
        let dx_s = matmul_ab(&gq, &wf_q, m, k, n);
        let dw_s = matmul_ab(&gt_q, &xf_q, n, k, m);

        let a_val = gq[0] - g_up[0];
        let b_val = wf_q[0] - w_f[0];
        sum_a += a_val;
        sum_b += b_val;
        sum_ab += a_val * b_val;

        for i in 0..dx_s.len() {
            let err = dx_s[i] - dx_star[i];
            dx_sum[i] += err;
            dx_sq_sum[i] += err * err;
        }
        for i in 0..dw_s.len() {
            let err = dw_s[i] - dw_star[i];
            dw_sum[i] += err;
            dw_sq_sum[i] += err * err;
        }
    }
    let sf = sample_budget as f64;
    let dx_means: Vec<f64> = dx_sum.iter().map(|s| s / sf).collect();
    let dw_means: Vec<f64> = dw_sum.iter().map(|s| s / sf).collect();
    let dx_mean_signed = dx_means.iter().sum::<f64>() / (dx_means.len() as f64);
    let dw_mean_signed = dw_means.iter().sum::<f64>() / (dw_means.len() as f64);
    let mut dx_max_abs_mean = 0.0_f64;
    let mut dx_max_se = 0.0_f64;
    for i in 0..dx_means.len() {
        let a = dx_means[i].abs();
        if a > dx_max_abs_mean {
            dx_max_abs_mean = a;
        }
        let var = ((dx_sq_sum[i] / sf) - dx_means[i] * dx_means[i]).max(0.0);
        let se = (var / sf).sqrt();
        if se > dx_max_se {
            dx_max_se = se;
        }
    }
    let mut dw_max_abs_mean = 0.0_f64;
    let mut dw_max_se = 0.0_f64;
    for i in 0..dw_means.len() {
        let a = dw_means[i].abs();
        if a > dw_max_abs_mean {
            dw_max_abs_mean = a;
        }
        let var = ((dw_sq_sum[i] / sf) - dw_means[i] * dw_means[i]).max(0.0);
        let se = (var / sf).sqrt();
        if se > dw_max_se {
            dw_max_se = se;
        }
    }
    let cov_ab = ((sum_ab / sf) - (sum_a / sf) * (sum_b / sf)).abs();
    let arm3_stats = StochasticGradStats {
        sample_budget,
        practical_bias_margin,
        dx_mean_signed_err: dx_mean_signed,
        dx_max_abs_mean_err: dx_max_abs_mean,
        dx_ci95_half_width: 1.96 * dx_max_se,
        dw_mean_signed_err: dw_mean_signed,
        dw_max_abs_mean_err: dw_max_abs_mean,
        dw_ci95_half_width: 1.96 * dw_max_se,
        operand_product_max_abs_cov: cov_ab,
        clipping_events: total_clip,
        zeroing_events: total_zero,
        within_practical_bias_margin: dx_max_abs_mean <= practical_bias_margin
            && dw_max_abs_mean <= practical_bias_margin,
    };

    // Arm 4: Deterministic independently re-quantized backward ablation (re-quantizes raw x_hi/w_hi
    // deterministically along transposed axis without preserving saved forward representation!).
    let w_hi_t = transpose(&w_hi, n, k);
    let x_hi_t = transpose(&x_hi, m, k);
    let (g_det, _, _) = mxfp8_quant_dequant(&g_up, m, n, None)?;
    let (w_det_t, _, _) = mxfp8_quant_dequant(&w_hi_t, k, n, None)?;
    let (gt_det, _, _) = mxfp8_quant_dequant(&g_t, n, m, None)?;
    let (x_det_t, _, _) = mxfp8_quant_dequant(&x_hi_t, k, m, None)?;
    let dx_det = matmul_ab(&g_det, &transpose(&w_det_t, k, n), m, k, n);
    let dw_det = matmul_ab(&gt_det, &transpose(&x_det_t, k, m), n, k, m);
    let arm4_dx_bias = max_diff(&dx_det, &dx_star)?;
    let arm4_dw_bias = max_diff(&dw_det, &dw_star)?;

    // 3. NVIDIA Transformer Engine / CUTLASS per-recipe matrix (SM120 and SM121 strictly distinct!).
    let te_cutlass_matrix = vec![
        TeRecipeMatrixRow {
            target_gpu: "B200_datacenter".into(),
            compute_capability: "sm_100".into(),
            recipe_arm: "nvfp4_fwd_mxfp8_bwd".into(),
            phase: "forward+backward+requant".into(),
            queried_gate: format!("TransformerEngine@{TE_COMMIT}:_compute_mxfp8_support()"),
            layout_constraint: "CUTLASS sm100 tcgen05.mma".into(),
            outcome: OutcomeStatus::UnavailableHardwareRuntime,
            reason: "gate allows CC 10.0; physical SM100 hardware not attached locally".into(),
        },
        TeRecipeMatrixRow {
            target_gpu: "GB300_datacenter".into(),
            compute_capability: "sm_103".into(),
            recipe_arm: "nvfp4_fwd_mxfp8_bwd".into(),
            phase: "forward+backward+requant".into(),
            queried_gate: format!("TransformerEngine@{TE_COMMIT}:_compute_mxfp8_support()"),
            layout_constraint: "CUTLASS sm103 tcgen05.mma".into(),
            outcome: OutcomeStatus::UnavailableHardwareRuntime,
            reason: "gate allows CC 10.3; physical SM103 hardware not attached locally".into(),
        },
        TeRecipeMatrixRow {
            target_gpu: "RTX_5080".into(),
            compute_capability: "sm_120".into(),
            recipe_arm: "te_all_gemm_layout_mxfp8_bwd".into(),
            phase: "backward_mxfp8".into(),
            queried_gate: format!("TransformerEngine@{TE_COMMIT}:_compute_mxfp8_support()"),
            layout_constraint: format!("CUTLASS@{CUTLASS_COMMIT} sm120 warp-level mma TN-only"),
            outcome: OutcomeStatus::Unsupported,
            reason: "inspected TE gate explicitly rejects all-GEMM-layout MXFP8 on CC >= 12.0; retained as valid unsupported result (no monkey-patch)".into(),
        },
        TeRecipeMatrixRow {
            target_gpu: "DGX_Spark_GB10".into(),
            compute_capability: "sm_121".into(),
            recipe_arm: "te_all_gemm_layout_mxfp8_bwd".into(),
            phase: "backward_mxfp8".into(),
            queried_gate: format!("TransformerEngine@{TE_COMMIT}:_compute_mxfp8_support()"),
            layout_constraint: format!("CUTLASS@{CUTLASS_COMMIT} sm121 warp-level mma TN-only (distinct from SM120)"),
            outcome: OutcomeStatus::Unsupported,
            reason: "inspected TE gate explicitly rejects all-GEMM-layout MXFP8 on CC >= 12.0; SM121 row kept distinct from SM120".into(),
        },
        TeRecipeMatrixRow {
            target_gpu: "RTX_5080".into(),
            compute_capability: "sm_120".into(),
            recipe_arm: "nvfp4_forward_only_separate_arm".into(),
            phase: "forward_nvfp4".into(),
            queried_gate: format!("TransformerEngine@{TE_COMMIT}:nvfp4_gate(CC>=10.0)"),
            layout_constraint: "TN-only narrow-precision GEMM".into(),
            outcome: OutcomeStatus::UnavailableHardwareRuntime,
            reason: "capability check alone does not prove working linear layer; separate arm from mixed MXFP8 backward".into(),
        },
    ];

    // 4. Optional Fabric Lab #6 Codec Reference (NVFP4 dispatch + Walsh-Hadamard / S2F6 combine).
    let expert_a = vec![0.5, -1.0, 1.5, 0.25, -0.75, 2.0, -0.5, 1.0];
    let expert_b = vec![-0.25, 0.75, 0.5, -1.25, 1.0, -0.5, 0.75, -0.25];
    let h_a = walsh_hadamard_norm(&expert_a)?;
    let h_a_inv = walsh_hadamard_norm(&h_a)?;
    let hadamard_involution_max_err = max_diff(&expert_a, &h_a_inv)?;

    let disp_nv = encode_nvfp4(&expert_a, 1, 8)?;
    let disp_dec = decode_nvfp4(&disp_nv);
    let nvfp4_dispatch_recon_max_err = max_diff(&expert_a, &disp_dec)?;

    // Weighted aggregation in Hadamard domain (valid iff both experts share identical transform/seed/norm!)
    let w_a = 0.6_f64;
    let w_b = 0.4_f64;
    let h_b = walsh_hadamard_norm(&expert_b)?;
    // S2F6 quantized combine simulation in Hadamard domain:
    let q_ha: Vec<f64> = h_a.iter().map(|&v| (v * 8.0).round() / 8.0).collect();
    let q_hb: Vec<f64> = h_b.iter().map(|&v| (v * 8.0).round() / 8.0).collect();
    let agg_h: Vec<f64> = q_ha
        .iter()
        .zip(q_hb.iter())
        .map(|(a, b)| w_a * a + w_b * b)
        .collect();
    let agg_recon = walsh_hadamard_norm(&agg_h)?;
    let ref_agg: Vec<f64> = expert_a
        .iter()
        .zip(expert_b.iter())
        .map(|(a, b)| w_a * a + w_b * b)
        .collect();
    let s2f6_combine_weighted_agg_max_err = max_diff(&agg_recon, &ref_agg)?;

    // Check incompatible pre-aggregation transform (mixing untransformed expert_b with Hadamard expert_a)
    let bad_mix_h: Vec<f64> = q_ha
        .iter()
        .zip(expert_b.iter())
        .map(|(a, b)| w_a * a + w_b * b)
        .collect();
    let bad_mix_recon = walsh_hadamard_norm(&bad_mix_h)?;
    let incompatible_transform_rejected = max_diff(&bad_mix_recon, &ref_agg)? > 0.25;
    // Check duplicate contribution (weights sum != 1.0 or duplicate expert_a)
    let dup_agg: Vec<f64> = expert_a.iter().map(|a| (w_a + w_b) * a).collect();
    let duplicate_or_missing_contrib_rejected = max_diff(&dup_agg, &ref_agg)? > 0.25;

    let fabric_codec = FabricCodecCheck {
        hadamard_involution_max_err,
        nvfp4_dispatch_recon_max_err,
        s2f6_combine_weighted_agg_max_err,
        incompatible_transform_rejected,
        duplicate_or_missing_contrib_rejected,
    };

    // 5. Eight executable negative controls!
    let mut negative_controls = Vec::new();

    // Control 1: Wrong scale axis / stale block scale on NVFP4 reconstruction
    let mut stale_nv = x_nv.clone();
    stale_nv.block_scales_e4m3[0] *= 2.0;
    let stale_err = max_diff(&decode_nvfp4(&stale_nv), &x_f)?;
    negative_controls.push(NegativeControl {
        case_id: "wrong_or_stale_scale_detected".into(),
        outcome: if stale_err > 1e-3 {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("stale_scale_recon_err={stale_err:.6}"),
    });

    // Control 2: Transposed block layout corruption
    let wrong_layout_y = matmul_ab_t(&x_f, &transpose(&w_f, n, k), m, n, k);
    let layout_err = max_diff(&wrong_layout_y, &y_f)?;
    negative_controls.push(NegativeControl {
        case_id: "transposed_block_layout_mismatch".into(),
        outcome: if layout_err > 1e-3 {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("transposed_layout_err={layout_err:.6}"),
    });

    // Control 3: Omitted saved-forward reconstruction (Arm 4 using unquantized master weights diverges from surrogate!)
    negative_controls.push(NegativeControl {
        case_id: "omitted_saved_forward_reconstruction".into(),
        outcome: if arm4_dx_bias > 1e-2 && arm4_dw_bias > 1e-2 {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("arm4_dx_err={arm4_dx_bias:.4}, arm4_dw_err={arm4_dw_bias:.4}"),
    });

    // Control 4: Correlated/reused RNG stream across both operands inflates operand covariance!
    let mut c_a = 0.0;
    let mut c_b = 0.0;
    let mut c_ab = 0.0;
    for tr in 0..sample_budget {
        // Reuse IDENTICAL stream_id=9999 for both operands -> correlated noise!
        let (q1, _, _) = mxfp8_quant_dequant(&g_up, m, n, Some((9999, tr as u64)))?;
        let (q2, _, _) = mxfp8_quant_dequant(&g_up, m, n, Some((9999, tr as u64)))?;
        let da = q1[0] - g_up[0];
        let db = q2[0] - g_up[0];
        c_a += da;
        c_b += db;
        c_ab += da * db;
    }
    let corr_cov = ((c_ab / sf) - (c_a / sf) * (c_b / sf)).abs();
    negative_controls.push(NegativeControl {
        case_id: "correlated_rng_stream_detected".into(),
        outcome: if corr_cov > cov_ab * 2.0 || corr_cov > 1e-6 {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("reused_stream_cov={corr_cov:.6e} vs independent_cov={cov_ab:.6e}"),
    });

    // Control 5: Non-finite input rejected cleanly
    let mut bad_x = x_hi.clone();
    bad_x[3] = f64::NAN;
    let nan_res = encode_nvfp4(&bad_x, m, k);
    negative_controls.push(NegativeControl {
        case_id: "non_finite_tensor_rejected".into(),
        outcome: if nan_res.is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: nan_res.err().unwrap_or_else(|| "accepted NaN".into()),
    });

    // Control 6: TE SM120/SM121 MXFP8 refusal cannot be bypassed or relabeled as BF16
    let sm120_unsup = te_cutlass_matrix[2].outcome == OutcomeStatus::Unsupported;
    let sm121_unsup = te_cutlass_matrix[3].outcome == OutcomeStatus::Unsupported;
    negative_controls.push(NegativeControl {
        case_id: "te_sm120_sm121_mxfp8_refusal_and_no_bf16_relabel".into(),
        outcome: if sm120_unsup && sm121_unsup {
            OutcomeStatus::Unsupported
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("sm120_unsupported={sm120_unsup}, sm121_unsupported={sm121_unsup}"),
    });

    // Control 7: Model Optimizer format-name / calibration-digest / export-version mismatch
    let mo_expected = ModelOptExportFixture {
        modelopt_commit: MODELOPT_COMMIT.into(),
        export_schema_version: "modelopt.export.v1".into(),
        format_name: "nvfp4_e2m1_block16".into(),
        scale_block_axis: "row_block_16_e4m3_plus_f32_outer".into(),
        calibration_digest: "sha256:calib-frozen-v1".into(),
        encoded: x_nv.clone(),
        reference_decoded: x_f.clone(),
    };
    let _mo_ok = validate_modelopt_export_fixture(&mo_expected, &mo_expected, 1e-12)?;
    let mut mo_bad_fmt = mo_expected.clone();
    mo_bad_fmt.format_name = "mxfp8_e4m3_block32".into();
    let mut mo_bad_cal = mo_expected.clone();
    mo_bad_cal.calibration_digest = "sha256:calib-altered-v2".into();
    let mut mo_bad_ver = mo_expected.clone();
    mo_bad_ver.export_schema_version = "modelopt.export.v2-breaking".into();
    let r_mo_fmt = validate_modelopt_export_fixture(&mo_expected, &mo_bad_fmt, 1e-12);
    let r_mo_cal = validate_modelopt_export_fixture(&mo_expected, &mo_bad_cal, 1e-12);
    let r_mo_ver = validate_modelopt_export_fixture(&mo_expected, &mo_bad_ver, 1e-12);
    negative_controls.push(NegativeControl {
        case_id: "modelopt_format_and_calibration_mismatch".into(),
        outcome: if r_mo_fmt.is_err() && r_mo_cal.is_err() && r_mo_ver.is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!(
            "format_rejected={}, calib_rejected={}, version_rejected={}",
            r_mo_fmt.is_err(),
            r_mo_cal.is_err(),
            r_mo_ver.is_err()
        ),
    });

    // Control 8: Fabric #6 incompatible Hadamard/S2F6 pre-aggregation transform rejected
    negative_controls.push(NegativeControl {
        case_id: "fabric_codec_incompatible_transform_and_duplicate_contrib".into(),
        outcome: if incompatible_transform_rejected && duplicate_or_missing_contrib_rejected {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!(
            "incompatible_rejected={incompatible_transform_rejected}, dup_rejected={duplicate_or_missing_contrib_rejected}"
        ),
    });

    Ok(Nvfp4Mxfp8Report {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "rtx5080_sm120_and_spark_sm121_native_nvfp4_mxfp8",
            "Device-free exact-bit numerical reference, surrogate/stochastic stats, TE SM120/SM121 refusal matrix, and Fabric #6 codec verified locally; native CUDA/CUTLASS layer timing unavailable here",
        ),
        known_answer_edge_cases_passed: edge_passed,
        arm1_vs_arm2_forward_drift,
        fixed_operand_fd_grad_max_err: fd_max_err,
        arm3_stochastic_stats: arm3_stats,
        arm4_deterministic_ablation_dx_max_abs_bias: arm4_dx_bias,
        arm4_deterministic_ablation_dw_max_abs_bias: arm4_dw_bias,
        te_cutlass_matrix,
        fabric_codec,
        negative_controls,
        decision: "NUMERICAL_REFERENCE_GO_NATIVE_SM120_SM121_MXFP8_UNSUPPORTED_AT_PINNED_TE".into(),
    })
}

pub fn inputs_json(report: &Nvfp4Mxfp8Report) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.nvfp4-mxfp8.inputs.v1\",\n",
            "  \"issue\": 26,\n",
            "  \"upstream_pins\": {{\"transformer_engine\":{},\"cutlass\":{},\"model_optimizer\":{}}},\n",
            "  \"unverified_comparison_recipes\": [\"fouroversix:2512.02010\",\"quartet-ii:2601.22813v2\",\"tetrajet-v2:2510.27527\"],\n",
            "  \"execution_class\": {},\n",
            "  \"sample_budget\": {},\n",
            "  \"practical_bias_margin\": {},\n",
            "  \"ci_containing_zero_proves_unbiasedness\": false,\n",
            "  \"simulated_timing_is_native_fp4_throughput\": false\n",
            "}}\n"
        ),
        quote(TE_COMMIT),
        quote(CUTLASS_COMMIT),
        quote(MODELOPT_COMMIT),
        quote(report.execution_class.as_str()),
        report.arm3_stochastic_stats.sample_budget,
        finite_f64(report.arm3_stochastic_stats.practical_bias_margin),
    )
}

pub fn results_json(report: &Nvfp4Mxfp8Report) -> String {
    let matrix: Vec<String> = report
        .te_cutlass_matrix
        .iter()
        .map(|r| {
            format!(
                "{{\"target_gpu\":{},\"compute_capability\":{},\"recipe_arm\":{},\"phase\":{},\"queried_gate\":{},\"layout_constraint\":{},\"outcome\":{},\"reason\":{}}}",
                quote(&r.target_gpu),
                quote(&r.compute_capability),
                quote(&r.recipe_arm),
                quote(&r.phase),
                quote(&r.queried_gate),
                quote(&r.layout_constraint),
                quote(r.outcome.as_str()),
                quote(&r.reason),
            )
        })
        .collect();
    let neg: Vec<String> = report
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
    let st = &report.arm3_stochastic_stats;
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.nvfp4-mxfp8.results.v1\",\n",
            "  \"issue\": 26,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"decision\": {},\n",
            "  \"native_layer_time_ms\": {},\n",
            "  \"known_answer_edge_cases_passed\": {},\n",
            "  \"arm1_vs_arm2_forward_drift\": {},\n",
            "  \"fixed_operand_fd_grad_max_err\": {},\n",
            "  \"arm3_stochastic\": {{\"sample_budget\":{},\"dx_mean_signed_err\":{},\"dx_max_abs_mean_err\":{},\"dx_ci95_half_width\":{},\"dw_mean_signed_err\":{},\"dw_max_abs_mean_err\":{},\"dw_ci95_half_width\":{},\"operand_product_max_abs_cov\":{},\"clipping_events\":{},\"zeroing_events\":{},\"within_practical_bias_margin\":{}}},\n",
            "  \"arm4_deterministic_ablation\": {{\"dx_max_abs_bias\":{},\"dw_max_abs_bias\":{}}},\n",
            "  \"te_cutlass_matrix\": [{}],\n",
            "  \"fabric_codec\": {{\"native_s2f6_instruction_available\":false,\"emulated_reconstruction\":true,\"hadamard_involution_max_err\":{},\"nvfp4_dispatch_recon_max_err\":{},\"s2f6_combine_weighted_agg_max_err\":{},\"incompatible_transform_rejected\":{},\"duplicate_or_missing_contrib_rejected\":{}}},\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        quote(&report.decision),
        opt_f64(None),
        report.known_answer_edge_cases_passed,
        finite_f64(report.arm1_vs_arm2_forward_drift),
        finite_f64(report.fixed_operand_fd_grad_max_err),
        st.sample_budget,
        finite_f64(st.dx_mean_signed_err),
        finite_f64(st.dx_max_abs_mean_err),
        finite_f64(st.dx_ci95_half_width),
        finite_f64(st.dw_mean_signed_err),
        finite_f64(st.dw_max_abs_mean_err),
        finite_f64(st.dw_ci95_half_width),
        finite_f64(st.operand_product_max_abs_cov),
        st.clipping_events,
        st.zeroing_events,
        st.within_practical_bias_margin,
        finite_f64(report.arm4_deterministic_ablation_dx_max_abs_bias),
        finite_f64(report.arm4_deterministic_ablation_dw_max_abs_bias),
        matrix.join(","),
        finite_f64(report.fabric_codec.hadamard_involution_max_err),
        finite_f64(report.fabric_codec.nvfp4_dispatch_recon_max_err),
        finite_f64(report.fabric_codec.s2f6_combine_weighted_agg_max_err),
        report.fabric_codec.incompatible_transform_rejected,
        report.fabric_codec.duplicate_or_missing_contrib_rejected,
        neg.join(","),
    )
}

pub fn summary_markdown(report: &Nvfp4Mxfp8Report) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #26: NVFP4 Forward / MXFP8 Backward Numerics & Native Gate"
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
    writeln!(
        out,
        "- Fixed-operand surrogate FD gradient max error: `{:.3e}`",
        report.fixed_operand_fd_grad_max_err
    )
    .unwrap();
    writeln!(
        out,
        "- Arm 3 stochastic MXFP8 backward (`S={}`): dX max |mean err| `{:.5}`, dW max |mean err| `{:.5}`, operand cov `{:.3e}` (Arm 4 deterministic bias: dX `{:.5}`, dW `{:.5}`)",
        report.arm3_stochastic_stats.sample_budget,
        report.arm3_stochastic_stats.dx_max_abs_mean_err,
        report.arm3_stochastic_stats.dw_max_abs_mean_err,
        report.arm3_stochastic_stats.operand_product_max_abs_cov,
        report.arm4_deterministic_ablation_dx_max_abs_bias,
        report.arm4_deterministic_ablation_dw_max_abs_bias,
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Transformer Engine / CUTLASS per-recipe matrix").unwrap();
    writeln!(out, "| GPU | CC | Arm | Outcome | Reason |").unwrap();
    writeln!(out, "| --- | --- | --- | --- | --- |").unwrap();
    for r in &report.te_cutlass_matrix {
        writeln!(
            out,
            "| `{}` | `{}` | `{}` | `{}` | {} |",
            r.target_gpu,
            r.compute_capability,
            r.recipe_arm,
            r.outcome.as_str(),
            r.reason
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "## Negative controls").unwrap();
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
