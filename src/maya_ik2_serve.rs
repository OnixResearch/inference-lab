//! Issue #59: Default-off `maya-ik2` serve of Inkling 2 Mini on a 24 GB `sm_86` card.
//!
//! Pinned upstream identities (observed 21 Nov 2026):
//! - Engine: `alesha-pro/maya-inkling-mini@c520fd14de3b21d67ddc6b7e7348b761aad26d52` (MIT)
//! - Baseline: `ggml-org/llama.cpp@34ff2e047ce520b1d83bd274bf3ff0bfbc72a69b` (`b12590`)
//! - Quantized GGUF: `unsloth/Inkling-2-Mini-GGUF@ae9327c5fbc7333116ba32eee78bb5d2bbefae4f`
//!   (`Inkling-2-Mini-UD-Q4_K_XL.gguf`, Apache-2.0)
//! - Official Mini reference (optional `mtp.safetensors` drafter, not required for first arm):
//!   `thinkingmachines/Inkling-2-Mini@6e7c8248f75265bdd4fbab5273304243b5d8c2cb`
//!
//! Scope & honesty contract:
//! - Strictly distinct from `#57` (`distinct_from_issue_57 = true`), `#33`, `#26`, `#35`, and `#18`.
//! - Default-off (`default_enabled = false`); NEVER installs `gpu-unlock` (`installs_gpu_unlock = false`).
//! - Separates 21 Nov X post claims (781 launches, 129.5 HTTP / 128.7 plain tok/s, 74/96 GSM8K,
//!   5.7e-5 KL) from the later pinned README claims (757 launches, 128 plain / 150-210 DSpark
//!   drafter tok/s, 74 plain / 77 drafter GSM8K) as distinct unverified non-gating notes.
//! - Enforces strict Linux + >=24 GB `sm_86` + CUDA 12.8 + `UD-Q4_K_XL` GGUF host gate, recording
//!   `gpu_unlock_present` without installing it and returning clean `Unsupported` when absent.
//! - Miniature packed 4-bit expert register-unpack parity vs independent scalar dequant reference:
//!   verifies exact register unpack (`<= 1e-12`) alongside dynamically computed IEEE-754 FP16
//!   register-unpack error and full-vocab KL divergence.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::det_weight;
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const MAYA_ENGINE_REPO: &str = "alesha-pro/maya-inkling-mini";
pub const MAYA_ENGINE_COMMIT: &str = "c520fd14de3b21d67ddc6b7e7348b761aad26d52";
pub const MAYA_ENGINE_LICENSE: &str = "MIT";
pub const LLAMA_CPP_REPO: &str = "ggml-org/llama.cpp";
pub const LLAMA_CPP_TAG: &str = "b12590";
pub const LLAMA_CPP_COMMIT: &str = "34ff2e047ce520b1d83bd274bf3ff0bfbc72a69b";
pub const UNSLOTH_GGUF_REPO: &str = "unsloth/Inkling-2-Mini-GGUF";
pub const UNSLOTH_GGUF_COMMIT: &str = "ae9327c5fbc7333116ba32eee78bb5d2bbefae4f";
pub const UNSLOTH_GGUF_FILE: &str = "Inkling-2-Mini-UD-Q4_K_XL.gguf";
pub const MODEL_WEIGHTS_LICENSE: &str = "Apache-2.0";
pub const X_POST_URL: &str = "https://x.com/superalesha/status/2124102269138346409";

pub const IN_DIM: usize = 32;
pub const HIDDEN_DIM: usize = 16;
pub const VOCAB_SIZE: usize = 32;
pub const BLOCK_SIZE: usize = 8; // 8 packed 4-bit nibbles = one 32-bit GPU register word

/// Explicitly non-gating record separating author X-post claims from later README claims.
#[derive(Debug, Clone)]
pub struct UnverifiedAuthorClaimNote {
    pub source_id: String,
    pub source_ref: String,
    pub claimed_launches_per_token: u32,
    pub claimed_plain_decode_tok_s: f64,
    pub claimed_http_or_drafter_note: String,
    pub verified_by_lab: bool,
    pub is_acceptance_gate: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MayaHostSpec {
    pub host_label: String,
    pub os: String,
    pub gpu_arch: String,
    pub vram_gb: u32,
    pub cuda_major: u32,
    pub cuda_minor: u32,
    pub gguf_quant: String,
    pub gpu_unlock_present: bool,
    pub binary_and_weights_available: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MayaHostGateDecision {
    pub admitted_host_requirements: bool,
    pub gpu_unlock_present_recorded: bool,
    pub installs_gpu_unlock: bool,
    pub outcome: OutcomeStatus,
    pub reason: String,
}

/// Strict Linux + 24 GB `sm_86` + CUDA 12.8 + `UD-Q4_K_XL` host qualification.
/// Records `gpu_unlock_present` purely as an observation; never installs `gpu-unlock`.
pub fn qualify_maya_host(spec: &MayaHostSpec) -> MayaHostGateDecision {
    if !spec.os.eq_ignore_ascii_case("linux") {
        return MayaHostGateDecision {
            admitted_host_requirements: false,
            gpu_unlock_present_recorded: spec.gpu_unlock_present,
            installs_gpu_unlock: false,
            outcome: OutcomeStatus::Unsupported,
            reason: format!(
                "maya-ik2 requires Linux (host {} reports OS {:?})",
                spec.host_label, spec.os
            ),
        };
    }
    if !spec.gpu_arch.eq_ignore_ascii_case("sm_86") {
        return MayaHostGateDecision {
            admitted_host_requirements: false,
            gpu_unlock_present_recorded: spec.gpu_unlock_present,
            installs_gpu_unlock: false,
            outcome: OutcomeStatus::Unsupported,
            reason: format!(
                "maya-ik2 requires sm_86 GPU (host {} reports {:?})",
                spec.host_label, spec.gpu_arch
            ),
        };
    }
    if spec.vram_gb < 24 {
        return MayaHostGateDecision {
            admitted_host_requirements: false,
            gpu_unlock_present_recorded: spec.gpu_unlock_present,
            installs_gpu_unlock: false,
            outcome: OutcomeStatus::Unsupported,
            reason: format!(
                "maya-ik2 requires >=24 GB VRAM (host {} has {} GB)",
                spec.host_label, spec.vram_gb
            ),
        };
    }
    if spec.cuda_major != 12 || spec.cuda_minor != 8 {
        return MayaHostGateDecision {
            admitted_host_requirements: false,
            gpu_unlock_present_recorded: spec.gpu_unlock_present,
            installs_gpu_unlock: false,
            outcome: OutcomeStatus::Unsupported,
            reason: format!(
                "maya-ik2 requires CUDA 12.8 (host {} has {}.{})",
                spec.host_label, spec.cuda_major, spec.cuda_minor
            ),
        };
    }
    if spec.gguf_quant != "UD-Q4_K_XL" {
        return MayaHostGateDecision {
            admitted_host_requirements: false,
            gpu_unlock_present_recorded: spec.gpu_unlock_present,
            installs_gpu_unlock: false,
            outcome: OutcomeStatus::Unsupported,
            reason: format!(
                "maya-ik2 reads only Unsloth UD-Q4_K_XL GGUF (got {:?})",
                spec.gguf_quant
            ),
        };
    }
    if !spec.binary_and_weights_available {
        return MayaHostGateDecision {
            admitted_host_requirements: true,
            gpu_unlock_present_recorded: spec.gpu_unlock_present,
            installs_gpu_unlock: false,
            outcome: OutcomeStatus::UnavailableHardwareRuntime,
            reason: format!(
                "host {} meets Linux/sm_86/24GB/CUDA12.8/UD-Q4_K_XL spec (gpu_unlock_present={}), but engine build/weights unavailable locally",
                spec.host_label, spec.gpu_unlock_present
            ),
        };
    }
    MayaHostGateDecision {
        admitted_host_requirements: true,
        gpu_unlock_present_recorded: spec.gpu_unlock_present,
        installs_gpu_unlock: false,
        outcome: OutcomeStatus::Accepted,
        reason: format!(
            "admitted Linux sm_86 >=24GB CUDA 12.8 host with UD-Q4_K_XL (gpu_unlock_present={})",
            spec.gpu_unlock_present
        ),
    }
}

/// Pure safe Rust IEEE-754 binary16 (FP16) round-to-nearest-even emulation on `f64`.
pub fn cast_fp16(x: f64) -> Result<f64, String> {
    if !x.is_finite() {
        return Err("non-finite value in cast_fp16".into());
    }
    if x == 0.0 {
        return Ok(0.0);
    }
    let sign = if x < 0.0 { -1.0_f64 } else { 1.0_f64 };
    let ax = x.abs();
    if ax >= 65504.0 {
        return Ok(sign * 65504.0);
    }
    let f = ax as f32;
    let bits = f.to_bits();
    let exp_unbiased = ((bits >> 23) & 0xff) as i32 - 127;
    let mant23 = bits & 0x007f_ffff;

    let (h_exp, h_mant): (u16, u16) = if exp_unbiased >= -14 {
        // Normal binary16 number: round 23-bit mantissa to 10-bit mantissa (RNE).
        let shifted = mant23 >> 13;
        let rem = mant23 & 0x1fff;
        let round_up = rem > 0x1000 || (rem == 0x1000 && (shifted & 1) == 1);
        let mut m10 = shifted + u32::from(round_up);
        let mut e5 = (exp_unbiased + 15) as u32;
        if m10 == 1024 {
            m10 = 0;
            e5 += 1;
        }
        if e5 >= 31 {
            return Ok(sign * 65504.0);
        }
        (e5 as u16, m10 as u16)
    } else if exp_unbiased >= -25 {
        // Subnormal binary16 number.
        let full_mant = (1u32 << 23) | mant23;
        let shift = (-14 - exp_unbiased + 13) as u32;
        let shifted = full_mant >> shift;
        let mask = (1u32 << shift) - 1;
        let half = 1u32 << (shift - 1);
        let rem = full_mant & mask;
        let round_up = rem > half || (rem == half && (shifted & 1) == 1);
        let m10 = shifted + u32::from(round_up);
        if m10 == 1024 {
            (1, 0)
        } else {
            (0, m10 as u16)
        }
    } else {
        (0, 0)
    };

    let mag = if h_exp == 0 {
        (h_mant as f64) * 2.0_f64.powi(-24)
    } else {
        (1.0 + (h_mant as f64) / 1024.0) * 2.0_f64.powi(i32::from(h_exp) - 15)
    };
    Ok(sign * mag)
}

/// One packed 4-bit quantized block of 8 weights stored as 4 bytes (`[u8; 4]`) or equivalently
/// one 32-bit GPU register (`u32`) plus block scale and zero-point offset (`w = scale * (q - zp)`).
#[derive(Debug, Clone)]
pub struct PackedQ4Block {
    pub packed_bytes: [u8; 4], // 8 nibbles: byte b holds low nibble 2*b and high nibble 2*b+1
    pub scale: f64,
    pub zero_point: f64,
}

#[derive(Debug, Clone)]
pub struct PackedQ4Matrix {
    pub rows: usize,
    pub cols: usize, // multiple of BLOCK_SIZE (8)
    pub blocks: Vec<PackedQ4Block>,
}

impl PackedQ4Matrix {
    pub fn build_deterministic(
        seed: usize,
        rows: usize,
        cols: usize,
        scale_mag: f64,
    ) -> Result<Self, String> {
        if cols % BLOCK_SIZE != 0 {
            return Err("cols must be a multiple of BLOCK_SIZE (8)".into());
        }
        let blocks_per_row = cols / BLOCK_SIZE;
        let mut blocks = Vec::with_capacity(rows * blocks_per_row);
        for r in 0..rows {
            for b in 0..blocks_per_row {
                let mut raw = [0.0_f64; BLOCK_SIZE];
                let mut min_v = f64::INFINITY;
                let mut max_v = f64::NEG_INFINITY;
                for (k, slot) in raw.iter_mut().enumerate() {
                    let v = det_weight(seed + r * 31, b * BLOCK_SIZE + k, scale_mag);
                    *slot = v;
                    if v < min_v {
                        min_v = v;
                    }
                    if v > max_v {
                        max_v = v;
                    }
                }
                let span = max_v - min_v;
                let scale = if span > 1.0e-12 { span / 15.0 } else { 1.0e-3 };
                let zero_point = -min_v / scale;
                let mut nibbles = [0u8; BLOCK_SIZE];
                for (k, &v) in raw.iter().enumerate() {
                    let qf = ((v - min_v) / scale).round();
                    if !qf.is_finite() {
                        return Err("non-finite quantized weight in PackedQ4Matrix".into());
                    }
                    let clamped = qf.clamp(0.0, 15.0);
                    nibbles[k] = clamped as u8;
                }
                let packed_bytes = [
                    (nibbles[0] & 0x0f) | ((nibbles[1] & 0x0f) << 4),
                    (nibbles[2] & 0x0f) | ((nibbles[3] & 0x0f) << 4),
                    (nibbles[4] & 0x0f) | ((nibbles[5] & 0x0f) << 4),
                    (nibbles[6] & 0x0f) | ((nibbles[7] & 0x0f) << 4),
                ];
                blocks.push(PackedQ4Block {
                    packed_bytes,
                    scale,
                    zero_point,
                });
            }
        }
        Ok(Self { rows, cols, blocks })
    }
}

/// Fault flags used strictly to test negative controls on the register-unpack kernel.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnpackFaults {
    pub swap_nibble_order: bool,
    pub corrupt_block_scale: bool,
}

/// Independent scalar dequantization reference: unpacks byte-by-byte into scalar `f64` matrix
/// and multiplies `y = W x` directly in FP64.
pub fn matvec_scalar_dequant_reference(
    mat: &PackedQ4Matrix,
    x: &[f64],
) -> Result<Vec<f64>, String> {
    if x.len() != mat.cols {
        return Err("input vector dimension mismatch".into());
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err("non-finite activation in scalar dequant reference".into());
    }
    let blocks_per_row = mat.cols / BLOCK_SIZE;
    let mut out = vec![0.0_f64; mat.rows];
    for (r, out_r) in out.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for b in 0..blocks_per_row {
            let blk = &mat.blocks[r * blocks_per_row + b];
            if !blk.scale.is_finite() || !blk.zero_point.is_finite() {
                return Err("non-finite block metadata".into());
            }
            for byte_idx in 0..4 {
                let byte = blk.packed_bytes[byte_idx];
                let lo = f64::from(byte & 0x0f);
                let hi = f64::from((byte >> 4) & 0x0f);
                let col_lo = b * BLOCK_SIZE + byte_idx * 2;
                let col_hi = col_lo + 1;
                let w_lo = blk.scale * (lo - blk.zero_point);
                let w_hi = blk.scale * (hi - blk.zero_point);
                acc += w_lo * x[col_lo] + w_hi * x[col_hi];
            }
        }
        if !acc.is_finite() {
            return Err("non-finite scalar matvec output".into());
        }
        *out_r = acc;
    }
    Ok(out)
}

/// Register-packed 32-bit word (`u32`) bit-shift unpack in FP64 (verifies exact bit-unpack parity).
pub fn matvec_register_unpack_exact(
    mat: &PackedQ4Matrix,
    x: &[f64],
    faults: UnpackFaults,
) -> Result<Vec<f64>, String> {
    if x.len() != mat.cols {
        return Err("input vector dimension mismatch".into());
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err("non-finite activation in register unpack exact".into());
    }
    let blocks_per_row = mat.cols / BLOCK_SIZE;
    let mut out = vec![0.0_f64; mat.rows];
    for (r, out_r) in out.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for b in 0..blocks_per_row {
            let blk = &mat.blocks[r * blocks_per_row + b];
            let reg_word = u32::from_le_bytes(blk.packed_bytes);
            let scale = if faults.corrupt_block_scale {
                blk.scale * 2.5
            } else {
                blk.scale
            };
            let base_col = b * BLOCK_SIZE;
            for lane in 0..BLOCK_SIZE {
                let shift_lane = if faults.swap_nibble_order {
                    lane ^ 1
                } else {
                    lane
                };
                let nibble = (reg_word >> (shift_lane * 4)) & 0x0f;
                let w = scale * (f64::from(nibble) - blk.zero_point);
                acc += w * x[base_col + lane];
            }
        }
        if !acc.is_finite() {
            return Err("non-finite register-unpack output".into());
        }
        *out_r = acc;
    }
    Ok(out)
}

/// `sm_86` FP16 register-unpack path: unpacks 4-bit nibbles from a 32-bit register into IEEE-754
/// FP16 weights and multiplies against FP16-rounded activations (matching lack of FP4 hardware
/// on RTX 3090 / `sm_86`).
pub fn matvec_register_unpack_fp16(
    mat: &PackedQ4Matrix,
    x: &[f64],
    faults: UnpackFaults,
) -> Result<Vec<f64>, String> {
    if x.len() != mat.cols {
        return Err("input vector dimension mismatch".into());
    }
    let mut x_fp16 = Vec::with_capacity(x.len());
    for &v in x {
        x_fp16.push(cast_fp16(v)?);
    }
    let blocks_per_row = mat.cols / BLOCK_SIZE;
    let mut out = vec![0.0_f64; mat.rows];
    for (r, out_r) in out.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for b in 0..blocks_per_row {
            let blk = &mat.blocks[r * blocks_per_row + b];
            let reg_word = u32::from_le_bytes(blk.packed_bytes);
            let scale_raw = if faults.corrupt_block_scale {
                blk.scale * 2.5
            } else {
                blk.scale
            };
            let scale_h = cast_fp16(scale_raw)?;
            let zp_h = cast_fp16(blk.zero_point)?;
            let base_col = b * BLOCK_SIZE;
            for lane in 0..BLOCK_SIZE {
                let shift_lane = if faults.swap_nibble_order {
                    lane ^ 1
                } else {
                    lane
                };
                let nibble = (reg_word >> (shift_lane * 4)) & 0x0f;
                let w_h = cast_fp16(scale_h * (f64::from(nibble) - zp_h))?;
                // Tensor-Core FP16*FP16 -> FP32 accumulate
                let prod = (w_h as f32) * (x_fp16[base_col + lane] as f32);
                acc += f64::from(prod);
            }
        }
        if !acc.is_finite() {
            return Err("non-finite FP16 register-unpack output".into());
        }
        *out_r = acc;
    }
    Ok(out)
}

fn log_softmax(logits: &[f64]) -> Result<Vec<f64>, String> {
    if logits.is_empty() {
        return Err("empty logits slice".into());
    }
    let mut max_l = logits[0];
    for &v in logits {
        if !v.is_finite() {
            return Err("non-finite logit in log_softmax".into());
        }
        if v > max_l {
            max_l = v;
        }
    }
    let mut sum_exp = 0.0_f64;
    for &v in logits {
        let e = (v - max_l).exp();
        if !e.is_finite() {
            return Err("non-finite exp in log_softmax".into());
        }
        sum_exp += e;
    }
    if sum_exp <= 0.0 || !sum_exp.is_finite() {
        return Err("invalid softmax normalizer".into());
    }
    let log_z = max_l + sum_exp.ln();
    Ok(logits.iter().map(|&v| v - log_z).collect())
}

/// Full-vocab KL divergence `KL(P_ref || P_cand) = sum_i P_ref(i) * (log P_ref(i) - log P_cand(i))`.
pub fn full_vocab_kl_divergence(ref_logits: &[f64], cand_logits: &[f64]) -> Result<f64, String> {
    if ref_logits.len() != cand_logits.len() || ref_logits.is_empty() {
        return Err("logit length mismatch in KL divergence".into());
    }
    let log_p = log_softmax(ref_logits)?;
    let log_q = log_softmax(cand_logits)?;
    let mut kl = 0.0_f64;
    for (&lp, &lq) in log_p.iter().zip(log_q.iter()) {
        let p = lp.exp();
        let term = p * (lp - lq);
        if !term.is_finite() {
            return Err("non-finite KL term".into());
        }
        kl += term;
    }
    // Clamp tiny floating-point negative zero to 0.0 without `.max()`
    Ok(if kl < 0.0 && kl > -1.0e-15 { 0.0 } else { kl })
}

fn argmax_finite(v: &[f64]) -> Result<usize, String> {
    if v.is_empty() {
        return Err("empty slice in argmax".into());
    }
    let mut best_i = 0usize;
    let mut best_v = v[0];
    if !best_v.is_finite() {
        return Err("non-finite value in argmax".into());
    }
    for (i, &x) in v.iter().enumerate().skip(1) {
        if !x.is_finite() {
            return Err("non-finite value in argmax".into());
        }
        if x > best_v {
            best_v = x;
            best_i = i;
        }
    }
    Ok(best_i)
}

fn vec_max_abs_diff(a: &[f64], b: &[f64]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err("vector length mismatch".into());
    }
    let mut m = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return Err("non-finite value in diff".into());
        }
        let d = (x - y).abs();
        if d > m {
            m = d;
        }
    }
    Ok(m)
}

/// Apply SwiGLU-style nonlinear expert step followed by LM head to get full-vocab logits.
fn run_mini_expert_to_logits(
    w_expert: &PackedQ4Matrix,
    w_lm_head: &PackedQ4Matrix,
    x: &[f64],
    mode: u8,
    faults: UnpackFaults,
) -> Result<Vec<f64>, String> {
    let hidden_pre = match mode {
        0 => matvec_scalar_dequant_reference(w_expert, x)?,
        1 => matvec_register_unpack_exact(w_expert, x, faults)?,
        _ => matvec_register_unpack_fp16(w_expert, x, faults)?,
    };
    let mut hidden_act = Vec::with_capacity(hidden_pre.len());
    for &h in &hidden_pre {
        // SiLU(h) = h / (1 + exp(-h))
        let sig = if h >= 0.0 {
            1.0 / (1.0 + (-h).exp())
        } else {
            let e = h.exp();
            e / (1.0 + e)
        };
        hidden_act.push(h * sig);
    }
    match mode {
        0 => matvec_scalar_dequant_reference(w_lm_head, &hidden_act),
        1 => matvec_register_unpack_exact(w_lm_head, &hidden_act, faults),
        _ => matvec_register_unpack_fp16(w_lm_head, &hidden_act, faults),
    }
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct MayaIk2ServeReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub synthetic_contract_fixture: bool,
    pub default_enabled: bool,
    pub distinct_from_issue_57: bool,
    pub installs_gpu_unlock: bool,
    pub x_post_unverified_claims: UnverifiedAuthorClaimNote,
    pub readme_unverified_claims: UnverifiedAuthorClaimNote,
    pub exact_register_unpack_max_abs_err: f64,
    pub exact_register_unpack_outcome: OutcomeStatus,
    pub fp16_register_unpack_max_abs_err: f64,
    pub fp16_full_vocab_kl_divergence: f64,
    pub fp16_greedy_top1_agreement: Option<f64>,
    pub local_host_gate_decision: MayaHostGateDecision,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<MayaIk2ServeReport, String> {
    let x_post_unverified_claims = UnverifiedAuthorClaimNote {
        source_id: "x_post_2026_11_21".into(),
        source_ref: X_POST_URL.into(),
        claimed_launches_per_token: 781,
        claimed_plain_decode_tok_s: 128.7,
        claimed_http_or_drafter_note:
            "129.5 HTTP tok/s vs 57.0 llama.cpp (7350 launches); 62.3 tok/s at 340K / 23.57 GiB; 74/96 GSM8K vs 73/96 scalar; 5.7e-5 KL (unverified author claim)"
                .into(),
        verified_by_lab: false,
        is_acceptance_gate: false,
    };
    let readme_unverified_claims = UnverifiedAuthorClaimNote {
        source_id: "pinned_readme_c520fd14".into(),
        source_ref: format!("{MAYA_ENGINE_REPO}@{MAYA_ENGINE_COMMIT}/README.md"),
        claimed_launches_per_token: 757,
        claimed_plain_decode_tok_s: 128.0,
        claimed_http_or_drafter_note:
            "128 plain tok/s, 150-210 tok/s with Thinking Machines DSpark drafter; 74 plain / 77 drafter GSM8K (unverified author claim)"
                .into(),
        verified_by_lab: false,
        is_acceptance_gate: false,
    };

    let w_expert = PackedQ4Matrix::build_deterministic(5901, HIDDEN_DIM, IN_DIM, 0.42)?;
    let w_lm_head = PackedQ4Matrix::build_deterministic(5977, VOCAB_SIZE, HIDDEN_DIM, 0.55)?;

    let num_positions = 8usize;
    let mut exact_max_err = 0.0_f64;
    let mut fp16_max_err = 0.0_f64;
    let mut kl_sum = 0.0_f64;
    let mut top1_matches = 0usize;

    let mut sample_input_0 = vec![0.0_f64; IN_DIM];
    let mut ref_logits_0 = Vec::new();

    for pos in 0..num_positions {
        let mut x = vec![0.0_f64; IN_DIM];
        for (i, slot) in x.iter_mut().enumerate() {
            *slot = det_weight(6100 + pos * 13, i, 0.85);
        }
        let ref_l =
            run_mini_expert_to_logits(&w_expert, &w_lm_head, &x, 0, UnpackFaults::default())?;
        let exact_l =
            run_mini_expert_to_logits(&w_expert, &w_lm_head, &x, 1, UnpackFaults::default())?;
        let fp16_l =
            run_mini_expert_to_logits(&w_expert, &w_lm_head, &x, 2, UnpackFaults::default())?;

        if pos == 0 {
            sample_input_0 = x;
            ref_logits_0 = ref_l.clone();
        }

        let e_exact = vec_max_abs_diff(&ref_l, &exact_l)?;
        if e_exact > exact_max_err {
            exact_max_err = e_exact;
        }
        let e_fp16 = vec_max_abs_diff(&ref_l, &fp16_l)?;
        if e_fp16 > fp16_max_err {
            fp16_max_err = e_fp16;
        }
        let kl = full_vocab_kl_divergence(&ref_l, &fp16_l)?;
        kl_sum += kl;
        if argmax_finite(&ref_l)? == argmax_finite(&fp16_l)? {
            top1_matches += 1;
        }
    }

    let mean_kl = kl_sum / (num_positions as f64);
    let top1_agreement = Some((top1_matches as f64) / (num_positions as f64));
    let exact_outcome = if exact_max_err <= 1.0e-12 {
        OutcomeStatus::Accepted
    } else {
        OutcomeStatus::Invalid
    };

    // Representative offline CPU/non-sm_86 host gate check (never installs gpu-unlock)
    let local_host_gate_decision = qualify_maya_host(&MayaHostSpec {
        host_label: "local-cpu-contract-runner".into(),
        os: "linux".into(),
        gpu_arch: "none".into(),
        vram_gb: 0,
        cuda_major: 0,
        cuda_minor: 0,
        gguf_quant: "UD-Q4_K_XL".into(),
        gpu_unlock_present: false,
        binary_and_weights_available: false,
    });

    // 8 executable negative controls
    let mut negative_controls = Vec::new();

    // NC1: Non-Linux OS (Windows) rejected cleanly as Unsupported
    {
        let d = qualify_maya_host(&MayaHostSpec {
            host_label: "win-rtx3090".into(),
            os: "windows".into(),
            gpu_arch: "sm_86".into(),
            vram_gb: 24,
            cuda_major: 12,
            cuda_minor: 8,
            gguf_quant: "UD-Q4_K_XL".into(),
            gpu_unlock_present: false,
            binary_and_weights_available: true,
        });
        negative_controls.push(NegativeControl {
            case_id: "nc1_windows_os_unsupported".into(),
            outcome: d.outcome,
            detail: d.reason,
        });
    }

    // NC2: Non-sm_86 GPU rejected cleanly as Unsupported
    {
        let d = qualify_maya_host(&MayaHostSpec {
            host_label: "linux-rtx4090-sm89".into(),
            os: "linux".into(),
            gpu_arch: "sm_89".into(),
            vram_gb: 24,
            cuda_major: 12,
            cuda_minor: 8,
            gguf_quant: "UD-Q4_K_XL".into(),
            gpu_unlock_present: false,
            binary_and_weights_available: true,
        });
        negative_controls.push(NegativeControl {
            case_id: "nc2_non_sm86_arch_unsupported".into(),
            outcome: d.outcome,
            detail: d.reason,
        });
    }

    // NC3: sm_86 GPU below 24 GB VRAM rejected cleanly as Unsupported
    {
        let d = qualify_maya_host(&MayaHostSpec {
            host_label: "linux-rtx3080-12gb".into(),
            os: "linux".into(),
            gpu_arch: "sm_86".into(),
            vram_gb: 12,
            cuda_major: 12,
            cuda_minor: 8,
            gguf_quant: "UD-Q4_K_XL".into(),
            gpu_unlock_present: false,
            binary_and_weights_available: true,
        });
        negative_controls.push(NegativeControl {
            case_id: "nc3_vram_below_24gb_unsupported".into(),
            outcome: d.outcome,
            detail: d.reason,
        });
    }

    // NC4: Wrong CUDA version (12.4 instead of 12.8) rejected cleanly as Unsupported
    {
        let d = qualify_maya_host(&MayaHostSpec {
            host_label: "linux-rtx3090-cuda124".into(),
            os: "linux".into(),
            gpu_arch: "sm_86".into(),
            vram_gb: 24,
            cuda_major: 12,
            cuda_minor: 4,
            gguf_quant: "UD-Q4_K_XL".into(),
            gpu_unlock_present: true,
            binary_and_weights_available: true,
        });
        negative_controls.push(NegativeControl {
            case_id: "nc4_cuda_not_12_8_unsupported".into(),
            outcome: d.outcome,
            detail: d.reason,
        });
    }

    // NC5: Non-UD-Q4_K_XL GGUF rejected cleanly as Unsupported
    {
        let d = qualify_maya_host(&MayaHostSpec {
            host_label: "linux-rtx3090-wrong-gguf".into(),
            os: "linux".into(),
            gpu_arch: "sm_86".into(),
            vram_gb: 24,
            cuda_major: 12,
            cuda_minor: 8,
            gguf_quant: "Q8_0".into(),
            gpu_unlock_present: false,
            binary_and_weights_available: true,
        });
        negative_controls.push(NegativeControl {
            case_id: "nc5_non_ud_q4_k_xl_gguf_unsupported".into(),
            outcome: d.outcome,
            detail: d.reason,
        });
    }

    // NC6: Swapped 4-bit nibble order in register unpack breaks parity vs scalar reference
    {
        let bad_l = run_mini_expert_to_logits(
            &w_expert,
            &w_lm_head,
            &sample_input_0,
            1,
            UnpackFaults {
                swap_nibble_order: true,
                corrupt_block_scale: false,
            },
        )?;
        let err = vec_max_abs_diff(&ref_logits_0, &bad_l)?;
        negative_controls.push(NegativeControl {
            case_id: "nc6_swapped_nibble_register_unpack_caught".into(),
            outcome: if err > 1.0e-6 {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!("swapped nibble order caused {:.3e} logit divergence", err),
        });
    }

    // NC7: Corrupted block scale spikes full-vocab KL divergence
    {
        let bad_l = run_mini_expert_to_logits(
            &w_expert,
            &w_lm_head,
            &sample_input_0,
            2,
            UnpackFaults {
                swap_nibble_order: false,
                corrupt_block_scale: true,
            },
        )?;
        let kl = full_vocab_kl_divergence(&ref_logits_0, &bad_l)?;
        negative_controls.push(NegativeControl {
            case_id: "nc7_corrupted_block_scale_kl_spike".into(),
            outcome: if kl > 1.0e-3 {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!("corrupted block scale raised full-vocab KL to {:.3e}", kl),
        });
    }

    // NC8: Non-finite activation rejected before unpack kernel
    {
        let mut nan_x = sample_input_0.clone();
        nan_x[3] = f64::NAN;
        let res = matvec_register_unpack_fp16(&w_expert, &nan_x, UnpackFaults::default());
        negative_controls.push(NegativeControl {
            case_id: "nc8_non_finite_activation_rejected".into(),
            outcome: if res.is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "NaN activation cleanly rejected in FP16 register-unpack path".into(),
        });
    }

    Ok(MayaIk2ServeReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "Linux host with 24 GB sm_86 GPU (RTX 3090 class), CUDA 12.8, and pinned Unsloth UD-Q4_K_XL GGUF",
            "Strict host gate, separated X-post vs README non-gating notes, packed 4-bit register-unpack parity, and FP16 full-vocab KL verified on CPU; maya-ik2 binary and UD-Q4_K_XL GGUF did not run locally",
        ),
        synthetic_contract_fixture: true,
        default_enabled: false,
        distinct_from_issue_57: true,
        installs_gpu_unlock: false,
        x_post_unverified_claims,
        readme_unverified_claims,
        exact_register_unpack_max_abs_err: exact_max_err,
        exact_register_unpack_outcome: exact_outcome,
        fp16_register_unpack_max_abs_err: fp16_max_err,
        fp16_full_vocab_kl_divergence: mean_kl,
        fp16_greedy_top1_agreement: top1_agreement,
        local_host_gate_decision,
        negative_controls,
    })
}

fn claim_note_json(note: &UnverifiedAuthorClaimNote) -> String {
    format!(
        "{{\"source_id\":{},\"source_ref\":{},\"claimed_launches_per_token\":{},\"claimed_plain_decode_tok_s\":{},\"claimed_http_or_drafter_note\":{},\"verified_by_lab\":{},\"is_acceptance_gate\":{}}}",
        quote(&note.source_id),
        quote(&note.source_ref),
        note.claimed_launches_per_token,
        finite_f64(note.claimed_plain_decode_tok_s),
        quote(&note.claimed_http_or_drafter_note),
        note.verified_by_lab,
        note.is_acceptance_gate,
    )
}

pub fn inputs_json(report: &MayaIk2ServeReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.maya-ik2-serve.inputs.v1\",\n",
            "  \"issue\": 59,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"distinct_from_issue_57\": {},\n",
            "  \"installs_gpu_unlock\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"pins\": {{\n",
            "    \"maya_inkling_mini\": {{\"repo\": {}, \"commit\": {}, \"license\": {}}},\n",
            "    \"llama_cpp_baseline\": {{\"repo\": {}, \"tag\": {}, \"commit\": {}}},\n",
            "    \"unsloth_gguf\": {{\"repo\": {}, \"commit\": {}, \"file\": {}, \"license\": {}}}\n",
            "  }},\n",
            "  \"x_post_unverified_claims\": {},\n",
            "  \"readme_unverified_claims\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.distinct_from_issue_57,
        report.installs_gpu_unlock,
        report.synthetic_contract_fixture,
        quote(MAYA_ENGINE_REPO),
        quote(MAYA_ENGINE_COMMIT),
        quote(MAYA_ENGINE_LICENSE),
        quote(LLAMA_CPP_REPO),
        quote(LLAMA_CPP_TAG),
        quote(LLAMA_CPP_COMMIT),
        quote(UNSLOTH_GGUF_REPO),
        quote(UNSLOTH_GGUF_COMMIT),
        quote(UNSLOTH_GGUF_FILE),
        quote(MODEL_WEIGHTS_LICENSE),
        claim_note_json(&report.x_post_unverified_claims),
        claim_note_json(&report.readme_unverified_claims),
    )
}

pub fn results_json(report: &MayaIk2ServeReport) -> String {
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
            "  \"schema\": \"inference-lab.maya-ik2-serve.results.v1\",\n",
            "  \"issue\": 59,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"distinct_from_issue_57\": {},\n",
            "  \"installs_gpu_unlock\": {},\n",
            "  \"exact_register_unpack_max_abs_err\": {},\n",
            "  \"exact_register_unpack_outcome\": {},\n",
            "  \"fp16_register_unpack_max_abs_err\": {},\n",
            "  \"fp16_full_vocab_kl_divergence\": {},\n",
            "  \"fp16_greedy_top1_agreement\": {},\n",
            "  \"local_host_gate_decision\": {{\"admitted_host_requirements\":{},\"gpu_unlock_present_recorded\":{},\"installs_gpu_unlock\":{},\"outcome\":{},\"reason\":{}}},\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.synthetic_contract_fixture,
        report.default_enabled,
        report.distinct_from_issue_57,
        report.installs_gpu_unlock,
        finite_f64(report.exact_register_unpack_max_abs_err),
        quote(report.exact_register_unpack_outcome.as_str()),
        finite_f64(report.fp16_register_unpack_max_abs_err),
        finite_f64(report.fp16_full_vocab_kl_divergence),
        opt_f64(report.fp16_greedy_top1_agreement),
        report.local_host_gate_decision.admitted_host_requirements,
        report.local_host_gate_decision.gpu_unlock_present_recorded,
        report.local_host_gate_decision.installs_gpu_unlock,
        quote(report.local_host_gate_decision.outcome.as_str()),
        quote(&report.local_host_gate_decision.reason),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &MayaIk2ServeReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #59: Default-Off `maya-ik2` Serve of Inkling 2 Mini on 24 GB `sm_86`"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Execution class: `{}` (`synthetic_contract_fixture: {}`, `default_enabled: {}`, `distinct_from_issue_57: {}`, `installs_gpu_unlock: {}`)",
        report.execution_class.as_str(),
        report.synthetic_contract_fixture,
        report.default_enabled,
        report.distinct_from_issue_57,
        report.installs_gpu_unlock
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
        "- Exact 4-bit register unpack vs scalar reference: `max_abs_err = {:.3e}` (`{}`)",
        report.exact_register_unpack_max_abs_err,
        report.exact_register_unpack_outcome.as_str()
    )
    .unwrap();
    writeln!(
        out,
        "- FP16 `sm_86` register-unpack emulation: `max_abs_err = {:.3e}`, `mean_full_vocab_kl = {:.3e}`, `top1_agreement = {:?}`",
        report.fp16_register_unpack_max_abs_err,
        report.fp16_full_vocab_kl_divergence,
        report.fp16_greedy_top1_agreement
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Separated Unverified Non-Gating Author Notes").unwrap();
    writeln!(
        out,
        "- X post (`{}`): `{} launches/tok`, `{} plain tok/s` — {}",
        report.x_post_unverified_claims.source_ref,
        report.x_post_unverified_claims.claimed_launches_per_token,
        report.x_post_unverified_claims.claimed_plain_decode_tok_s,
        report.x_post_unverified_claims.claimed_http_or_drafter_note
    )
    .unwrap();
    writeln!(
        out,
        "- Pinned README (`{}`): `{} launches/tok`, `{} plain tok/s` — {}",
        report.readme_unverified_claims.source_ref,
        report.readme_unverified_claims.claimed_launches_per_token,
        report.readme_unverified_claims.claimed_plain_decode_tok_s,
        report.readme_unverified_claims.claimed_http_or_drafter_note
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
    fn default_harness_passes_register_unpack_parity_and_eight_negative_controls() {
        let report = run_default_harness().expect("harness must succeed");
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
        assert!(report.distinct_from_issue_57);
        assert!(!report.installs_gpu_unlock);
        assert!(!report.x_post_unverified_claims.is_acceptance_gate);
        assert!(!report.readme_unverified_claims.is_acceptance_gate);
        assert_eq!(
            report.x_post_unverified_claims.claimed_launches_per_token,
            781
        );
        assert_eq!(
            report.readme_unverified_claims.claimed_launches_per_token,
            757
        );
        assert!(report.exact_register_unpack_max_abs_err <= 1.0e-12);
        assert_eq!(
            report.exact_register_unpack_outcome,
            OutcomeStatus::Accepted
        );
        assert!(report.fp16_register_unpack_max_abs_err.is_finite());
        assert!(report.fp16_full_vocab_kl_divergence >= 0.0);
        assert!(report.fp16_full_vocab_kl_divergence < 1.0e-2);
        assert_eq!(
            report.local_host_gate_decision.outcome,
            OutcomeStatus::Unsupported
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
        assert!(inputs_json(&report).ends_with('\n'));
        assert!(results_json(&report).ends_with('\n'));
        assert!(summary_markdown(&report).contains("Issue #59"));
    }

    #[test]
    fn host_gate_records_gpu_unlock_without_installing_it() {
        let ok_no_unlock = qualify_maya_host(&MayaHostSpec {
            host_label: "rtx3090-stock".into(),
            os: "linux".into(),
            gpu_arch: "sm_86".into(),
            vram_gb: 24,
            cuda_major: 12,
            cuda_minor: 8,
            gguf_quant: "UD-Q4_K_XL".into(),
            gpu_unlock_present: false,
            binary_and_weights_available: true,
        });
        assert_eq!(ok_no_unlock.outcome, OutcomeStatus::Accepted);
        assert!(!ok_no_unlock.gpu_unlock_present_recorded);
        assert!(!ok_no_unlock.installs_gpu_unlock);

        let ok_with_unlock = qualify_maya_host(&MayaHostSpec {
            host_label: "rtx3090-unlocked".into(),
            os: "linux".into(),
            gpu_arch: "sm_86".into(),
            vram_gb: 24,
            cuda_major: 12,
            cuda_minor: 8,
            gguf_quant: "UD-Q4_K_XL".into(),
            gpu_unlock_present: true,
            binary_and_weights_available: false,
        });
        assert_eq!(
            ok_with_unlock.outcome,
            OutcomeStatus::UnavailableHardwareRuntime
        );
        assert!(ok_with_unlock.gpu_unlock_present_recorded);
        assert!(!ok_with_unlock.installs_gpu_unlock);
    }

    #[test]
    fn fp16_cast_handles_zero_normals_subnormals_and_saturation() {
        assert_eq!(cast_fp16(0.0).unwrap(), 0.0);
        assert_eq!(cast_fp16(1.5).unwrap(), 1.5);
        assert_eq!(cast_fp16(-2.25).unwrap(), -2.25);
        assert_eq!(cast_fp16(100_000.0).unwrap(), 65504.0);
        assert!(cast_fp16(f64::NAN).is_err());
    }
}
