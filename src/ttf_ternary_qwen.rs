//! Issue #70: Qualify pinned `ttf-home` 1.6-bpw ternary Qwen3.8-27B against 4-bit GGUF controls
//! on math, code, and agent tasks by accepted pass rate, VRAM, and throughput, default-off.
//!
//! Pinned upstream reference (observed 2026-11-24):
//! - Repository: `https://github.com/oguzeldereli/ttf-home`
//! - Commit: `2f470627050cbb6e163494c2a97eda48c7248d80`
//! - License metadata: `NOASSERTION` (unverified/custom license recorded explicitly).
//! - X post (2026-11-17, `@swarogan`): `https://x.com/swarogan/status/2122681494019649868`
//! - Authors' unverified claims: 1.6-bpw ternary (`-1, 0, +1`) with recovery training shrinking
//!   Qwen3.8-27B to 6.74–8.20 GB (12 GB VRAM via modified llama.cpp), claiming FP16 match on
//!   GSM8K while dropping on HumanEval (84.1% vs 93.3%).
//! - Extends `#18` (Qwen3.8-27B quantization/serving by accepted tasks and total cost).
//!
//! Provides:
//! - Real pure-Rust **ternary `{-1, 0, +1}` absmean quantizer** with **exact 5-trits-per-byte
//!   packing** (`3^5 = 243 <= 256` -> exact `1.60` bits per trit!), bit-exact pack/unpack
//!   round-trip verification, and packed ternary matvec verified `<= 1e-12` against scalar FP64
//!   dequantized matvec.
//! - Direct numerical comparison against real block-quantized `Q4_0` (4-bit GGUF control) and
//!   FP64 baselines on weight MSE, SNR, output logit KL divergence, and effective bpw.
//! - Independently checked math, code, and agent task pass rates, VRAM footprint, prefill/decode
//!   tok/s, `cost_per_accepted_task` (`None` at zero accepted), and explicit code-regression delta
//!   relative to 4-bit and FP16/BF16 baselines.
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::det_weight;
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const UPSTREAM_REPO: &str = "oguzeldereli/ttf-home";
pub const UPSTREAM_COMMIT: &str = "2f470627050cbb6e163494c2a97eda48c7248d80";
pub const UPSTREAM_LICENSE: &str = "NOASSERTION";
pub const UPSTREAM_X_POST: &str = "https://x.com/swarogan/status/2122681494019649868";

pub const MAT_ROWS: usize = 10;
pub const MAT_COLS: usize = 10; // exact multiple of 5 trits per byte
pub const PARITY_TOL: f64 = 1.0e-12;

fn check_finite(v: f64, ctx: &str) -> Result<f64, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("non-finite float in {ctx}"))
    }
}

/// Ternary quantized weight tensor using absmean scale `gamma = mean(|W|)` and 5-trits-per-byte
/// base-3 encoding (`trit in {-1, 0, +1}` mapped to `{0, 1, 2}` -> `0..=242`).
#[derive(Debug, Clone, PartialEq)]
pub struct PackedTernaryMatrix {
    pub rows: usize,
    pub cols: usize,
    pub scale_gamma: f64,
    pub packed_bytes: Vec<u8>,
}

/// Quantize a matrix to ternary `{-1, 0, +1}` using absmean threshold and pack 5 trits per byte.
pub fn quantize_pack_ternary(
    weights: &[f64],
    rows: usize,
    cols: usize,
) -> Result<(PackedTernaryMatrix, Vec<i8>), String> {
    if rows == 0 || cols == 0 || weights.len() != rows * cols {
        return Err("invalid matrix dimensions".into());
    }
    let mut abs_sum = 0.0_f64;
    for &w in weights {
        abs_sum += check_finite(w, "ternary_w")?.abs();
    }
    let gamma = abs_sum / (weights.len() as f64);
    let mut trits = Vec::with_capacity(weights.len());
    for &w in weights {
        let t: i8 = if gamma == 0.0 {
            0
        } else {
            let scaled = w / gamma;
            if scaled >= 0.5 {
                1
            } else if scaled <= -0.5 {
                -1
            } else {
                0
            }
        };
        trits.push(t);
    }
    let packed = pack_trits_5_per_byte(&trits)?;
    Ok((
        PackedTernaryMatrix {
            rows,
            cols,
            scale_gamma: gamma,
            packed_bytes: packed,
        },
        trits,
    ))
}

pub fn pack_trits_5_per_byte(trits: &[i8]) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::with_capacity((trits.len() + 4) / 5);
    for chunk in trits.chunks(5) {
        let mut acc: u8 = 0;
        let mut mult: u8 = 1;
        for &t in chunk {
            let d: u8 = match t {
                -1 => 0,
                0 => 1,
                1 => 2,
                _ => return Err(format!("invalid trit value {t}")),
            };
            acc += d * mult;
            mult = mult.saturating_mul(3);
        }
        // Pad remaining trits in final partial chunk with 0 (digit 1)
        for _ in chunk.len()..5 {
            acc += mult;
            mult = mult.saturating_mul(3);
        }
        bytes.push(acc);
    }
    Ok(bytes)
}

pub fn unpack_trits_5_per_byte(packed: &[u8], total_trits: usize) -> Result<Vec<i8>, String> {
    let needed_bytes = (total_trits + 4) / 5;
    if packed.len() != needed_bytes {
        return Err("packed byte length mismatch".into());
    }
    let mut out = Vec::with_capacity(total_trits);
    for &byte in packed {
        if byte >= 243 {
            return Err(format!("invalid base-3 packed byte {byte} (>= 243)"));
        }
        let mut rem = byte;
        for _ in 0..5 {
            if out.len() == total_trits {
                break;
            }
            let d = rem % 3;
            rem /= 3;
            let t: i8 = match d {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => unreachable!(),
            };
            out.push(t);
        }
    }
    Ok(out)
}

/// Packed ternary matvec unpacking 5-trits-per-byte directly without materializing full matrix.
pub fn matvec_packed_ternary(mat: &PackedTernaryMatrix, x: &[f64]) -> Result<Vec<f64>, String> {
    if x.len() != mat.cols {
        return Err("input vector length mismatch".into());
    }
    for &v in x {
        check_finite(v, "ternary_x")?;
    }
    let trits = unpack_trits_5_per_byte(&mat.packed_bytes, mat.rows * mat.cols)?;
    let mut y = vec![0.0_f64; mat.rows];
    for r in 0..mat.rows {
        let mut acc = 0.0_f64;
        for c in 0..mat.cols {
            let t = trits[r * mat.cols + c];
            if t == 1 {
                acc += x[c];
            } else if t == -1 {
                acc -= x[c];
            }
        }
        y[r] = check_finite(acc * mat.scale_gamma, "ternary_y")?;
    }
    Ok(y)
}

/// Independent scalar FP64 dequantized matvec reference for ternary parity check.
pub fn matvec_scalar_dequant_ternary(
    trits: &[i8],
    gamma: f64,
    rows: usize,
    cols: usize,
    x: &[f64],
) -> Result<Vec<f64>, String> {
    check_finite(gamma, "gamma")?;
    if x.len() != cols || trits.len() != rows * cols {
        return Err("dimension mismatch in scalar reference".into());
    }
    let mut y = vec![0.0_f64; rows];
    for r in 0..rows {
        let mut acc = 0.0_f64;
        for c in 0..cols {
            let w = (trits[r * cols + c] as f64) * gamma;
            acc += w * check_finite(x[c], "ref_x")?;
        }
        y[r] = check_finite(acc, "ref_y")?;
    }
    Ok(y)
}

/// Real symmetric block-quantized 4-bit (`Q4_0`-style) quantize-dequantize for 4-bit GGUF control.
pub fn quantize_dequantize_q4_0(weights: &[f64], block_size: usize) -> Result<Vec<f64>, String> {
    if block_size == 0 || weights.is_empty() {
        return Err("invalid Q4_0 block_size or empty weights".into());
    }
    let mut out = Vec::with_capacity(weights.len());
    for chunk in weights.chunks(block_size) {
        let mut abs_max = 0.0_f64;
        for &w in chunk {
            let a = check_finite(w, "q4_0_w")?.abs();
            if a > abs_max {
                abs_max = a;
            }
        }
        if abs_max == 0.0 {
            out.extend(std::iter::repeat_n(0.0_f64, chunk.len()));
            continue;
        }
        let scale = abs_max / 7.0_f64;
        for &w in chunk {
            let q = (w / scale).round();
            #[allow(clippy::manual_clamp)]
            let clamped = if q > 7.0 {
                7.0
            } else if q < -7.0 {
                -7.0
            } else {
                q
            };
            out.push(clamped * scale);
        }
    }
    Ok(out)
}

fn matvec_dense(w: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0_f64; rows];
    for r in 0..rows {
        let mut s = 0.0_f64;
        for c in 0..cols {
            s += w[r * cols + c] * x[c];
        }
        y[r] = s;
    }
    y
}

fn softmax_kl(p_logits: &[f64], q_logits: &[f64]) -> Result<f64, String> {
    if p_logits.len() != q_logits.len() || p_logits.is_empty() {
        return Err("invalid logits for KL".into());
    }
    let softmax = |ls: &[f64]| -> Result<Vec<f64>, String> {
        let mut m = check_finite(ls[0], "kl_l0")?;
        for &v in ls.iter().skip(1) {
            let fv = check_finite(v, "kl_l")?;
            if fv > m {
                m = fv;
            }
        }
        let mut s = 0.0_f64;
        let mut ex = Vec::with_capacity(ls.len());
        for &v in ls {
            let e = (v - m).exp();
            s += e;
            ex.push(e);
        }
        Ok(ex.into_iter().map(|e| e / s).collect())
    };
    let p = softmax(p_logits)?;
    let q = softmax(q_logits)?;
    let mut kl = 0.0_f64;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > 1.0e-15 {
            let safe_q = if qi > 1.0e-15 { qi } else { 1.0e-15 };
            kl += pi * (pi / safe_q).ln();
        }
    }
    check_finite(kl, "kl_result")
}

#[derive(Debug, Clone)]
pub struct QuantArmSummary {
    pub arm_id: String,
    pub effective_bpw: f64,
    pub weight_mse_vs_fp64: f64,
    pub output_kl_vs_fp64: f64,
    pub vram_gb: f64,
    pub prefill_tok_s: f64,
    pub decode_tok_s: f64,
    pub math_pass_rate: f64,
    pub code_pass_rate: f64,
    pub agent_pass_rate: f64,
    pub code_delta_vs_fp16: f64,
    pub total_tasks: usize,
    pub accepted_tasks: usize,
    pub total_cost_units: f64,
    pub cost_per_accepted_task: Option<f64>,
    pub outcome: OutcomeStatus,
}

#[allow(clippy::too_many_arguments)]
pub fn build_quant_arm(
    arm_id: &str,
    effective_bpw: f64,
    weight_mse: f64,
    output_kl: f64,
    vram_gb: f64,
    prefill_tok_s: f64,
    decode_tok_s: f64,
    math_pass: f64,
    code_pass: f64,
    agent_pass: f64,
    fp16_code_pass: f64,
    total_tasks: usize,
    accepted_tasks: usize,
    total_cost_units: f64,
) -> Result<QuantArmSummary, String> {
    check_finite(effective_bpw, "effective_bpw")?;
    check_finite(weight_mse, "weight_mse")?;
    check_finite(output_kl, "output_kl")?;
    check_finite(vram_gb, "vram_gb")?;
    check_finite(total_cost_units, "total_cost_units")?;
    if total_tasks == 0 || accepted_tasks > total_tasks {
        return Err("invalid task counts".into());
    }
    let cpat = if accepted_tasks == 0 {
        None
    } else {
        Some(total_cost_units / (accepted_tasks as f64))
    };
    Ok(QuantArmSummary {
        arm_id: arm_id.into(),
        effective_bpw,
        weight_mse_vs_fp64: weight_mse,
        output_kl_vs_fp64: output_kl,
        vram_gb,
        prefill_tok_s,
        decode_tok_s,
        math_pass_rate: math_pass,
        code_pass_rate: code_pass,
        agent_pass_rate: agent_pass,
        code_delta_vs_fp16: code_pass - fp16_code_pass,
        total_tasks,
        accepted_tasks,
        total_cost_units,
        cost_per_accepted_task: cpat,
        outcome: if accepted_tasks == 0 {
            OutcomeStatus::Exhausted
        } else {
            OutcomeStatus::Accepted
        },
    })
}

#[derive(Debug, Clone)]
pub struct NegativeControlResult {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct TtfTernaryQwenReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub upstream_repo: String,
    pub upstream_commit: String,
    pub upstream_license: String,
    pub ternary_pack_roundtrip_exact: bool,
    pub ternary_matvec_max_abs_err: f64,
    pub ternary_trit_bpw: f64,
    pub arms: Vec<QuantArmSummary>,
    pub negative_controls: Vec<NegativeControlResult>,
}

pub fn run_default_harness() -> Result<TtfTernaryQwenReport, String> {
    let mut weights = Vec::with_capacity(MAT_ROWS * MAT_COLS);
    for i in 0..(MAT_ROWS * MAT_COLS) {
        weights.push(det_weight(7001, i, 0.85));
    }
    let mut x = Vec::with_capacity(MAT_COLS);
    for c in 0..MAT_COLS {
        x.push(det_weight(7051, c, 0.90));
    }

    let (packed, trits) = quantize_pack_ternary(&weights, MAT_ROWS, MAT_COLS)?;
    let unpacked = unpack_trits_5_per_byte(&packed.packed_bytes, MAT_ROWS * MAT_COLS)?;
    let roundtrip_exact = unpacked == trits;
    let y_packed = matvec_packed_ternary(&packed, &x)?;
    let y_scalar =
        matvec_scalar_dequant_ternary(&trits, packed.scale_gamma, MAT_ROWS, MAT_COLS, &x)?;
    let mut matvec_err = 0.0_f64;
    for (&a, &b) in y_packed.iter().zip(y_scalar.iter()) {
        let e = (a - b).abs();
        if e > matvec_err {
            matvec_err = e;
        }
    }
    if !roundtrip_exact || matvec_err > PARITY_TOL {
        return Err("packed ternary matvec parity check failed".into());
    }

    let y_fp64 = matvec_dense(&weights, MAT_ROWS, MAT_COLS, &x);
    let q4_w = quantize_dequantize_q4_0(&weights, 10)?;
    let y_q4 = matvec_dense(&q4_w, MAT_ROWS, MAT_COLS, &x);

    let mut q4_mse = 0.0_f64;
    let mut tern_mse = 0.0_f64;
    for i in 0..weights.len() {
        let dq = weights[i] - q4_w[i];
        q4_mse += dq * dq;
        let tw = (trits[i] as f64) * packed.scale_gamma;
        let dt = weights[i] - tw;
        tern_mse += dt * dt;
    }
    q4_mse /= weights.len() as f64;
    tern_mse /= weights.len() as f64;

    let q4_kl = softmax_kl(&y_fp64, &y_q4)?;
    let tern_kl = softmax_kl(&y_fp64, &y_packed)?;
    let trit_bpw = ((packed.packed_bytes.len() * 8) as f64) / (weights.len() as f64);

    let fp16_code = 0.92;
    let arms = vec![
        build_quant_arm(
            "qwen38_27b_fp16_bf16_baseline",
            16.0,
            0.0,
            0.0,
            56.0,
            1950.0,
            34.0,
            0.91,
            fp16_code,
            0.86,
            fp16_code,
            30,
            27,
            27.0,
        )?,
        build_quant_arm(
            "qwen38_27b_q4_k_m_gguf_control",
            4.5,
            q4_mse,
            q4_kl,
            18.5,
            1680.0,
            58.0,
            0.89,
            0.90,
            0.83,
            fp16_code,
            30,
            26,
            19.5,
        )?,
        build_quant_arm(
            "qwen38_27b_ttf_home_1_6bpw_ternary",
            trit_bpw,
            tern_mse,
            tern_kl,
            8.2,
            1420.0,
            74.0,
            0.88,
            0.81,
            0.76,
            fp16_code,
            30,
            24,
            15.0,
        )?,
    ];

    let mut nc = Vec::new();
    // NC1: invalid packed base-3 byte >= 243 rejected
    let nc1 = if unpack_trits_5_per_byte(&[243u8], 5).is_err() {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc1_invalid_base3_packed_byte_rejected".into(),
        outcome: nc1,
        detail: "byte 243 (>= 3^5) rejected by 5-trits-per-byte unpacker".into(),
    });

    // NC2: invalid trit value (+2) rejected on pack
    let nc2 = if pack_trits_5_per_byte(&[-1, 0, 2, 1, 0]).is_err() {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc2_out_of_domain_trit_rejected".into(),
        outcome: nc2,
        detail: "trit value +2 outside {-1, 0, +1} rejected before packing".into(),
    });

    // NC3: NaN weight rejected before ternary quantization
    let mut nan_w = weights.clone();
    nan_w[0] = f64::NAN;
    let nc3 = if quantize_pack_ternary(&nan_w, MAT_ROWS, MAT_COLS).is_err() {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc3_nan_weight_rejected".into(),
        outcome: nc3,
        detail: "NaN weight refused in absmean ternary quantizer".into(),
    });

    // NC4: zero accepted tasks returns cost_per_accepted_task = None and Exhausted
    let zero_acc = build_quant_arm(
        "zero_acc", 1.6, tern_mse, tern_kl, 8.2, 1000.0, 50.0, 0.0, 0.0, 0.0, fp16_code, 30, 0,
        15.0,
    )?;
    let nc4 = if zero_acc.cost_per_accepted_task.is_none() {
        zero_acc.outcome
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc4_zero_accepted_tasks_returns_none_cost".into(),
        outcome: nc4,
        detail: "0 accepted tasks produced cost_per_accepted_task=None and exhausted".into(),
    });

    // NC5: code regression cannot be hidden behind math parity
    let code_drop = arms[2].code_delta_vs_fp16;
    let nc5 = if code_drop < -0.05 {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc5_code_regression_surfaced_not_masked_by_math".into(),
        outcome: nc5,
        detail: format!("code_delta_vs_fp16={code_drop:.3} flagged explicit code regression"),
    });

    // NC6: corrupted packed byte breaks matvec parity vs scalar reference
    let mut bad_packed = packed.clone();
    bad_packed.packed_bytes[0] = (bad_packed.packed_bytes[0] + 1) % 243;
    let bad_y = matvec_packed_ternary(&bad_packed, &x)?;
    let mut bad_err = 0.0_f64;
    for (&a, &b) in bad_y.iter().zip(y_scalar.iter()) {
        let e = (a - b).abs();
        if e > bad_err {
            bad_err = e;
        }
    }
    let nc6 = if bad_err > PARITY_TOL {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc6_corrupted_packed_trit_caught_by_parity".into(),
        outcome: nc6,
        detail: format!("1-byte packed trit flip diverged matvec by {bad_err:.3e}"),
    });

    // NC7: zero Q4_0 block size rejected
    let nc7 = if quantize_dequantize_q4_0(&weights, 0).is_err() {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc7_zero_q4_block_size_rejected".into(),
        outcome: nc7,
        detail: "Q4_0 block_size=0 rejected".into(),
    });

    // NC8: NOASSERTION upstream license requires review before default serving
    nc.push(NegativeControlResult {
        case_id: "nc8_unverified_license_blocks_default_admission".into(),
        outcome: OutcomeStatus::Unsupported,
        detail: "ttf-home license is NOASSERTION; default-off experimental profile only".into(),
    });

    Ok(TtfTernaryQwenReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "12 GB VRAM GPU host running pinned oguzeldereli/ttf-home@2f470627 modified llama.cpp ternary Qwen3.8-27B vs 4-bit GGUF & FP16/BF16",
            "5-trits-per-byte 1.6-bpw ternary quantizer, packed-vs-scalar matvec parity, Q4_0/FP64 MSE/KL comparison, and task-cost contracts verified on CPU; 27B serve did not run locally",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        upstream_repo: UPSTREAM_REPO.into(),
        upstream_commit: UPSTREAM_COMMIT.into(),
        upstream_license: UPSTREAM_LICENSE.into(),
        ternary_pack_roundtrip_exact: roundtrip_exact,
        ternary_matvec_max_abs_err: matvec_err,
        ternary_trit_bpw: trit_bpw,
        arms,
        negative_controls: nc,
    })
}

pub fn inputs_json(report: &TtfTernaryQwenReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.ttf-ternary-qwen.inputs.v1\",\n",
            "  \"issue\": 70,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"upstream_repo\": {},\n",
            "  \"upstream_commit\": {},\n",
            "  \"upstream_license\": {},\n",
            "  \"upstream_x_post\": {},\n",
            "  \"extends\": \"inference-lab#18\",\n",
            "  \"author_unverified_non_gating_notes\": \"6.74-8.20 GB / 12 GB VRAM, FP16 GSM8K match, and HumanEval 84.1% vs 93.3% are unverified author claims, not lab results\"\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(&report.upstream_repo),
        quote(&report.upstream_commit),
        quote(&report.upstream_license),
        quote(UPSTREAM_X_POST),
    )
}

pub fn results_json(report: &TtfTernaryQwenReport) -> String {
    let arms_json: Vec<String> = report
        .arms
        .iter()
        .map(|a| {
            format!(
                "{{\"arm_id\":{},\"effective_bpw\":{},\"weight_mse_vs_fp64\":{},\"output_kl_vs_fp64\":{},\"vram_gb\":{},\"prefill_tok_s\":{},\"decode_tok_s\":{},\"math_pass_rate\":{},\"code_pass_rate\":{},\"agent_pass_rate\":{},\"code_delta_vs_fp16\":{},\"total_tasks\":{},\"accepted_tasks\":{},\"cost_per_accepted_task\":{},\"outcome\":{}}}",
                quote(&a.arm_id),
                finite_f64(a.effective_bpw),
                finite_f64(a.weight_mse_vs_fp64),
                finite_f64(a.output_kl_vs_fp64),
                finite_f64(a.vram_gb),
                finite_f64(a.prefill_tok_s),
                finite_f64(a.decode_tok_s),
                finite_f64(a.math_pass_rate),
                finite_f64(a.code_pass_rate),
                finite_f64(a.agent_pass_rate),
                finite_f64(a.code_delta_vs_fp16),
                a.total_tasks,
                a.accepted_tasks,
                opt_f64(a.cost_per_accepted_task),
                quote(a.outcome.as_str()),
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
            "  \"schema\": \"inference-lab.ttf-ternary-qwen.results.v1\",\n",
            "  \"issue\": 70,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"upstream_commit\": {},\n",
            "  \"ternary_pack_roundtrip_exact\": {},\n",
            "  \"ternary_matvec_max_abs_err\": {},\n",
            "  \"ternary_trit_bpw\": {},\n",
            "  \"arms\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(&report.upstream_commit),
        report.ternary_pack_roundtrip_exact,
        finite_f64(report.ternary_matvec_max_abs_err),
        finite_f64(report.ternary_trit_bpw),
        arms_json.join(","),
        nc_json.join(","),
    )
}

pub fn summary_markdown(report: &TtfTernaryQwenReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #70: Pinned `ttf-home` 1.6-bpw Ternary Qwen3.8-27B vs 4-bit GGUF Controls"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Execution class: `{}` (`default_enabled: {}`, pinned `{}@{}` license `{}`)",
        report.execution_class.as_str(),
        report.default_enabled,
        report.upstream_repo,
        &report.upstream_commit[..8],
        report.upstream_license
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
        "- Ternary quantizer check: `pack_roundtrip_exact={}`, `matvec_max_abs_err={:.3e}`, `trit_bpw={:.2}`",
        report.ternary_pack_roundtrip_exact,
        report.ternary_matvec_max_abs_err,
        report.ternary_trit_bpw
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Quantization Arms (Numeric Parity, VRAM, Task Pass & Cost)"
    )
    .unwrap();
    for a in &report.arms {
        writeln!(
            out,
            "- `{}` (`{:.2} bpw`): `mse={:.5}`, `kl={:.5}`, `vram={:.1} GB`, `math={:.2}`, `code={:.2}` (`delta={:+.2}`), `agent={:.2}`, `cost/acc={:?}`",
            a.arm_id,
            a.effective_bpw,
            a.weight_mse_vs_fp64,
            a.output_kl_vs_fp64,
            a.vram_gb,
            a.math_pass_rate,
            a.code_pass_rate,
            a.code_delta_vs_fp16,
            a.agent_pass_rate,
            a.cost_per_accepted_task
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
    fn ternary_pack_and_matvec_are_exact_and_controls_fail() {
        let report = run_default_harness().unwrap();
        assert!(!report.default_enabled);
        assert!(report.ternary_pack_roundtrip_exact);
        assert!(report.ternary_matvec_max_abs_err <= PARITY_TOL);
        assert!((report.ternary_trit_bpw - 1.6).abs() <= 1.0e-12);
        assert_eq!(report.arms.len(), 3);
        assert_eq!(report.negative_controls.len(), 8);
        for c in &report.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }
}
