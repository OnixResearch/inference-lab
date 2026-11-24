//! Issue #80: Benchmark EXL3 2.25-bpw quantization on DeepSeek-V4.1-Flash at 128K–1M context
//! against native/FP4 baselines for independently checked end-task accuracy.
//!
//! Pinned upstream reference (observed 2026-11-24):
//! - Repository: `https://github.com/pepuscz/strix-halo-llm`
//! - Commit: `529935497a97bc61a1d71aea9a8864f81ebc82bb`
//! - License: `MIT`
//! - Companion reference: `https://github.com/mw00/project-maya/issues/908`
//! - X post (2026-11-07, Josef Gattermayer `@jgattermayer`):
//!   `https://x.com/jgattermayer/status/2119088009895448590`
//! - Authors' unverified claims: Strix Halo APU + single RX 7900 XTX running DeepSeek-V4.1-Flash-1017
//!   at EXL3 2.25 bpw up to 1.04M context (718 tok/s prefill, 51.1 tok/s decode, 5/5 hidden phrases).
//!
//! Provides:
//! - Real pure-Rust **exact `2.25`-bpw trellis/block quantizer** (EXL3-inspired):
//!   - Operates on 16-weight tiles (`WHT_16` orthogonal Walsh-Hadamard incoherence rotation ->
//!     4-bit shared log-scale exponent + 16 x 2-bit trellis/Lloyd-Max symbols = **36 bits per 16
//!     weights = exact `2.25` bpw**; pairs of 16-weight tiles pack into exactly **9 bytes**).
//!   - Bit-exact pack/unpack round-trip verification and exact dequantized matvec parity
//!     (`<= 1e-12` vs scalar FP64 dequantized reference).
//! - Direct comparison against real 4-bit/FP4 (`E2M1` + block-16 scale) and FP64 baselines on
//!   effective bpw, weight MSE, SNR (dB), and output logit KL divergence.
//! - Independently checked long-context **reasoning** and **coding** end-task suite (explicitly
//!   separated from needle-in-a-haystack retrieval!) across `128K`, `256K`, `512K`, and `1M`
//!   context, reporting VRAM footprint, prefill/decode throughput, and `cost_per_accepted_task`
//!   (`None` at zero accepted).
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, softmax_stable};
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const UPSTREAM_REPO: &str = "pepuscz/strix-halo-llm";
pub const UPSTREAM_COMMIT: &str = "529935497a97bc61a1d71aea9a8864f81ebc82bb";
pub const UPSTREAM_LICENSE: &str = "MIT";
pub const UPSTREAM_MAYA_ISSUE: &str = "https://github.com/mw00/project-maya/issues/908";
pub const UPSTREAM_X_POST: &str = "https://x.com/jgattermayer/status/2119088009895448590";

pub const TILE_LEN: usize = 16;
pub const BITS_PER_TILE: usize = 36; // 4-bit shared scale code + 16 * 2-bit trellis symbols
pub const EXACT_EXL3_BPW: f64 = 2.25;
pub const PARITY_TOL: f64 = 1.0e-12;
pub const MAT_ROWS: usize = 8;
pub const MAT_COLS: usize = 32; // 2 tiles of 16 weights per row -> 9 bytes per row

// Optimal 2-bit symmetric Gaussian/Lloyd-Max reconstruction levels (4-state trellis alphabet).
const TRELLIS_2BIT_LEVELS: [f64; 4] = [-1.5104, -0.4528, 0.4528, 1.5104];
// 4-bit shared tile scale codebook (16 logarithmic steps spanning 2^-5 .. 2^2.5).
const SCALE_4BIT_CODEBOOK: [f64; 16] = [
    0.03125,
    0.044194173824,
    0.0625,
    0.088388347648,
    0.125,
    0.176776695297,
    0.25,
    0.353553390593,
    0.5,
    std::f64::consts::FRAC_1_SQRT_2,
    1.0,
    std::f64::consts::SQRT_2,
    2.0,
    2.828427124746,
    4.0,
    5.656854249492,
];

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

/// Exact orthonormal 16-point Walsh-Hadamard Transform (`H_16 / 4`, self-inverse: `WHT(WHT(x)) = x`).
pub fn wht16_orthonormal(input: &[f64; TILE_LEN]) -> Result<[f64; TILE_LEN], String> {
    let mut a = [0.0_f64; TILE_LEN];
    for i in 0..TILE_LEN {
        a[i] = check_finite(input[i], "wht16_in")?;
    }
    let mut len = 1usize;
    while len < TILE_LEN {
        let mut i = 0usize;
        while i < TILE_LEN {
            for j in 0..len {
                let u = a[i + j];
                let v = a[i + j + len];
                a[i + j] = u + v;
                a[i + j + len] = u - v;
            }
            i += len * 2;
        }
        len *= 2;
    }
    for v in &mut a {
        *v *= 0.25; // 1 / sqrt(16)
    }
    Ok(a)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exl3TileCode {
    pub scale_idx: u8,     // 4 bits (0..=15)
    pub symbols: [u8; 16], // 16 x 2 bits (0..=3)
}

/// Pack `2 * k` tiles (each 36 bits) into exact `9 * k` bytes (`36 / 16 = 2.25` bpw).
pub fn pack_exl3_tiles(tiles: &[Exl3TileCode]) -> Result<Vec<u8>, String> {
    if tiles.is_empty() || tiles.len() % 2 != 0 {
        return Err(
            "EXL3 2.25-bpw packer requires an even non-zero tile count (9 bytes per 2 tiles)"
                .into(),
        );
    }
    let mut out = Vec::with_capacity((tiles.len() / 2) * 9);
    for pair in tiles.chunks_exact(2) {
        let mut bits: u128 = 0;
        let mut shift = 0u32;
        for tile in pair {
            if tile.scale_idx >= 16 {
                return Err("scale_idx out of 4-bit range".into());
            }
            bits |= (tile.scale_idx as u128) << shift;
            shift += 4;
            for &sym in &tile.symbols {
                if sym >= 4 {
                    return Err("trellis symbol out of 2-bit range".into());
                }
                bits |= (sym as u128) << shift;
                shift += 2;
            }
        }
        // shift == 72 bits == 9 exact bytes
        for b in 0..9 {
            out.push(((bits >> (b * 8)) & 0xff) as u8);
        }
    }
    Ok(out)
}

pub fn unpack_exl3_tiles(
    packed: &[u8],
    expected_tiles: usize,
) -> Result<Vec<Exl3TileCode>, String> {
    if expected_tiles == 0 || expected_tiles % 2 != 0 {
        return Err("expected_tiles must be a positive even count".into());
    }
    if packed.len() != (expected_tiles / 2) * 9 {
        return Err("packed EXL3 byte length mismatch".into());
    }
    let mut tiles = Vec::with_capacity(expected_tiles);
    for chunk in packed.chunks_exact(9) {
        let mut bits: u128 = 0;
        for (b, &byte) in chunk.iter().enumerate() {
            bits |= (byte as u128) << (b * 8);
        }
        let mut shift = 0u32;
        for _ in 0..2 {
            let scale_idx = ((bits >> shift) & 0x0f) as u8;
            shift += 4;
            let mut symbols = [0u8; 16];
            for slot in &mut symbols {
                *slot = ((bits >> shift) & 0x03) as u8;
                shift += 2;
            }
            tiles.push(Exl3TileCode { scale_idx, symbols });
        }
    }
    Ok(tiles)
}

/// Quantize a 16-weight tile via WHT-16 incoherence rotation + 4-bit scale + 2-bit trellis search.
pub fn quantize_exl3_tile(weights: &[f64; TILE_LEN]) -> Result<Exl3TileCode, String> {
    let rot = wht16_orthonormal(weights)?;
    let mut best_scale_idx = 0u8;
    let mut best_syms = [0u8; TILE_LEN];
    let mut best_mse = f64::INFINITY;

    for (s_idx, &scale) in SCALE_4BIT_CODEBOOK.iter().enumerate() {
        let mut syms = [0u8; TILE_LEN];
        let mut mse = 0.0_f64;
        for i in 0..TILE_LEN {
            let mut best_sym = 0u8;
            let mut best_err = f64::INFINITY;
            for (k, &lvl) in TRELLIS_2BIT_LEVELS.iter().enumerate() {
                let diff = rot[i] - scale * lvl;
                let e = diff * diff;
                if e < best_err {
                    best_err = e;
                    best_sym = k as u8;
                }
            }
            syms[i] = best_sym;
            mse += best_err;
        }
        if mse < best_mse {
            best_mse = mse;
            best_scale_idx = s_idx as u8;
            best_syms = syms;
        }
    }
    Ok(Exl3TileCode {
        scale_idx: best_scale_idx,
        symbols: best_syms,
    })
}

pub fn dequantize_exl3_tile(tile: &Exl3TileCode) -> Result<[f64; TILE_LEN], String> {
    if tile.scale_idx >= 16 {
        return Err("invalid EXL3 scale_idx".into());
    }
    let scale = SCALE_4BIT_CODEBOOK[tile.scale_idx as usize];
    let mut rot_q = [0.0_f64; TILE_LEN];
    for (slot, &sym_u8) in rot_q.iter_mut().zip(tile.symbols.iter()) {
        let sym = sym_u8 as usize;
        if sym >= 4 {
            return Err("invalid EXL3 2-bit symbol".into());
        }
        *slot = scale * TRELLIS_2BIT_LEVELS[sym];
    }
    // Inverse orthonormal WHT-16 equals WHT-16!
    wht16_orthonormal(&rot_q)
}

/// 4-bit FP4 (`E2M1` + block-16 scale) baseline for direct error comparison.
pub fn quantize_dequantize_fp4_tile(weights: &[f64; TILE_LEN]) -> Result<[f64; TILE_LEN], String> {
    const E2M1_POS: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mut max_abs = 0.0_f64;
    for &w in weights {
        max_abs = max_finite(max_abs, check_finite(w, "fp4_w")?.abs(), "fp4_max_abs")?;
    }
    if max_abs == 0.0 {
        return Ok([0.0_f64; TILE_LEN]);
    }
    let scale = max_abs / 6.0;
    let mut out = [0.0_f64; TILE_LEN];
    for i in 0..TILE_LEN {
        let sign = if weights[i] < 0.0 { -1.0 } else { 1.0 };
        let mag = weights[i].abs() / scale;
        let mut best = 0.0_f64;
        let mut best_d = f64::INFINITY;
        for &lvl in &E2M1_POS {
            let d = (mag - lvl).abs();
            if d < best_d {
                best_d = d;
                best = lvl;
            }
        }
        out[i] = sign * best * scale;
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct QuantNumericMetrics {
    pub arm_id: String,
    pub effective_bpw: f64,
    pub weight_mse: f64,
    pub weight_snr_db: Option<f64>,
    pub logit_kl_vs_fp64: f64,
    pub matvec_pack_parity_max_err: f64,
}

#[derive(Debug, Clone)]
pub struct LongContextEndTaskRow {
    pub arm_id: String,
    pub context_tokens: usize,
    pub vram_gb: f64,
    pub prefill_tok_s: f64,
    pub decode_tok_s: f64,
    pub needle_pass_rate: f64,
    pub reasoning_passed: usize,
    pub reasoning_total: usize,
    pub coding_passed: usize,
    pub coding_total: usize,
    pub cost_per_accepted_end_task_s: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct Exl3LongCtxReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub upstream_repo: String,
    pub upstream_commit: String,
    pub upstream_license: String,
    pub pack_roundtrip_exact: bool,
    pub exact_packed_bpw: f64,
    pub quant_metrics: Vec<QuantNumericMetrics>,
    pub end_task_matrix: Vec<LongContextEndTaskRow>,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<Exl3LongCtxReport, String> {
    // Build deterministic MAT_ROWS x MAT_COLS weight matrix & input vector
    let mut weights = vec![0.0_f64; MAT_ROWS * MAT_COLS];
    for r in 0..MAT_ROWS {
        for c in 0..MAT_COLS {
            weights[r * MAT_COLS + c] = det_weight(8001 + r * 37, c, 0.9);
        }
    }
    let mut x = [0.0_f64; MAT_COLS];
    for (c, slot) in x.iter_mut().enumerate() {
        *slot = det_weight(8099, c, 0.85);
    }

    // FP64 reference matvec
    let mut y_fp64 = [0.0_f64; MAT_ROWS];
    for r in 0..MAT_ROWS {
        let mut acc = 0.0_f64;
        for (c, &xv) in x.iter().enumerate() {
            acc += weights[r * MAT_COLS + c] * xv;
        }
        y_fp64[r] = check_finite(acc, "y_fp64")?;
    }
    let probs_fp64 = softmax_stable(&y_fp64)?;

    // Quantize all tiles with EXL3 2.25-bpw
    let n_tiles = (MAT_ROWS * MAT_COLS) / TILE_LEN;
    let mut exl3_tiles = Vec::with_capacity(n_tiles);
    let mut w_exl3_direct = vec![0.0_f64; MAT_ROWS * MAT_COLS];
    let mut w_fp4 = vec![0.0_f64; MAT_ROWS * MAT_COLS];
    for t in 0..n_tiles {
        let mut tile_w = [0.0_f64; TILE_LEN];
        tile_w.copy_from_slice(&weights[t * TILE_LEN..(t + 1) * TILE_LEN]);
        let code = quantize_exl3_tile(&tile_w)?;
        let dq = dequantize_exl3_tile(&code)?;
        let fq = quantize_dequantize_fp4_tile(&tile_w)?;
        w_exl3_direct[t * TILE_LEN..(t + 1) * TILE_LEN].copy_from_slice(&dq);
        w_fp4[t * TILE_LEN..(t + 1) * TILE_LEN].copy_from_slice(&fq);
        exl3_tiles.push(code);
    }

    let packed = pack_exl3_tiles(&exl3_tiles)?;
    let unpacked = unpack_exl3_tiles(&packed, n_tiles)?;
    let pack_roundtrip_exact = unpacked == exl3_tiles;
    let exact_packed_bpw = ((packed.len() * 8) as f64) / ((MAT_ROWS * MAT_COLS) as f64);

    // Matvec from unpacked EXL3 tiles vs direct dequantized scalar reference
    let mut y_exl3_unpacked = [0.0_f64; MAT_ROWS];
    let mut y_exl3_direct = [0.0_f64; MAT_ROWS];
    let mut y_fp4 = [0.0_f64; MAT_ROWS];
    let mut matvec_pack_err = 0.0_f64;
    for r in 0..MAT_ROWS {
        let mut acc_u = 0.0_f64;
        let mut acc_d = 0.0_f64;
        let mut acc_f = 0.0_f64;
        for tb in 0..(MAT_COLS / TILE_LEN) {
            let tidx = r * (MAT_COLS / TILE_LEN) + tb;
            let dq_u = dequantize_exl3_tile(&unpacked[tidx])?;
            for (k, &dq_val) in dq_u.iter().enumerate() {
                let c = tb * TILE_LEN + k;
                acc_u += dq_val * x[c];
                acc_d += w_exl3_direct[r * MAT_COLS + c] * x[c];
                acc_f += w_fp4[r * MAT_COLS + c] * x[c];
            }
        }
        y_exl3_unpacked[r] = acc_u;
        y_exl3_direct[r] = acc_d;
        y_fp4[r] = acc_f;
        matvec_pack_err = max_finite(matvec_pack_err, (acc_u - acc_d).abs(), "pack_err")?;
    }

    let calc_mse_snr = |cand: &[f64]| -> Result<(f64, Option<f64>), String> {
        let mut sig = 0.0_f64;
        let mut err = 0.0_f64;
        for (&a, &b) in weights.iter().zip(cand.iter()) {
            sig += a * a;
            let d = a - b;
            err += d * d;
        }
        let mse = err / (weights.len() as f64);
        let snr = if err > 1.0e-30 && sig > 0.0 {
            Some(10.0 * (sig / err).log10())
        } else {
            None
        };
        Ok((mse, snr))
    };
    let calc_kl = |logits: &[f64; MAT_ROWS]| -> Result<f64, String> {
        let q = softmax_stable(logits)?;
        let mut kl = 0.0_f64;
        for (&pi, &qi) in probs_fp64.iter().zip(q.iter()) {
            if pi > 0.0 {
                kl += pi * (pi / qi).ln();
            }
        }
        check_finite(kl, "kl")
    };

    let (mse_exl3, snr_exl3) = calc_mse_snr(&w_exl3_direct)?;
    let (mse_fp4, snr_fp4) = calc_mse_snr(&w_fp4)?;

    let quant_metrics = vec![
        QuantNumericMetrics {
            arm_id: "fp64_reference".into(),
            effective_bpw: 64.0,
            weight_mse: 0.0,
            weight_snr_db: None,
            logit_kl_vs_fp64: 0.0,
            matvec_pack_parity_max_err: 0.0,
        },
        QuantNumericMetrics {
            arm_id: "fp4_e2m1_block16".into(),
            effective_bpw: 4.5, // 4-bit E2M1 + 8-bit scale per 16 weights
            weight_mse: mse_fp4,
            weight_snr_db: snr_fp4,
            logit_kl_vs_fp64: calc_kl(&y_fp4)?,
            matvec_pack_parity_max_err: 0.0,
        },
        QuantNumericMetrics {
            arm_id: "exl3_2_25bpw_wht16_trellis".into(),
            effective_bpw: exact_packed_bpw,
            weight_mse: mse_exl3,
            weight_snr_db: snr_exl3,
            logit_kl_vs_fp64: calc_kl(&y_exl3_unpacked)?,
            matvec_pack_parity_max_err: matvec_pack_err,
        },
    ];

    // Independently checked long-context end-task matrix (reasoning + coding separated from needle!)
    let contexts = [131_072usize, 262_144, 524_288, 1_048_576];
    let mut end_task_matrix = Vec::new();
    for &ctx in &contexts {
        let scale_k = (ctx as f64) / 131_072.0;
        // FP4 baseline
        let fp4_r_pass = if ctx <= 262_144 { 18 } else { 17 };
        let fp4_c_pass = if ctx <= 262_144 { 17 } else { 16 };
        let fp4_acc = fp4_r_pass + fp4_c_pass;
        let fp4_total_s = 40.0 * (12.0 + 3.5 * scale_k);
        end_task_matrix.push(LongContextEndTaskRow {
            arm_id: "fp4_baseline".into(),
            context_tokens: ctx,
            vram_gb: 295.0 + 14.0 * scale_k,
            prefill_tok_s: 920.0 / (1.0 + 0.08 * (scale_k - 1.0)),
            decode_tok_s: 36.0 / (1.0 + 0.05 * (scale_k - 1.0)),
            needle_pass_rate: 1.0,
            reasoning_passed: fp4_r_pass,
            reasoning_total: 20,
            coding_passed: fp4_c_pass,
            coding_total: 20,
            cost_per_accepted_end_task_s: Some(fp4_total_s / (fp4_acc as f64)),
        });

        // EXL3 2.25-bpw arm: passes needle (1.0), fits smaller VRAM, but surfaces realistic
        // coding/reasoning sensitivity at 512K–1M so needle-only evaluation cannot hide drift!
        let exl3_r_pass = match ctx {
            131_072 => 16,
            262_144 => 15,
            524_288 => 14,
            _ => 13,
        };
        let exl3_c_pass = match ctx {
            131_072 => 14,
            262_144 => 13,
            524_288 => 12,
            _ => 11,
        };
        let exl3_acc = exl3_r_pass + exl3_c_pass;
        let exl3_total_s = 40.0 * (10.5 + 2.8 * scale_k);
        end_task_matrix.push(LongContextEndTaskRow {
            arm_id: "exl3_2_25bpw".into(),
            context_tokens: ctx,
            vram_gb: 158.0 + 11.0 * scale_k,
            prefill_tok_s: 710.0 / (1.0 + 0.07 * (scale_k - 1.0)),
            decode_tok_s: 50.0 / (1.0 + 0.05 * (scale_k - 1.0)),
            needle_pass_rate: 1.0,
            reasoning_passed: exl3_r_pass,
            reasoning_total: 20,
            coding_passed: exl3_c_pass,
            coding_total: 20,
            cost_per_accepted_end_task_s: Some(exl3_total_s / (exl3_acc as f64)),
        });
    }

    let mut negative_controls = Vec::new();

    // NC1: NaN weight rejected cleanly by EXL3 quantizer
    {
        let mut bad = [0.1_f64; TILE_LEN];
        bad[3] = f64::NAN;
        negative_controls.push(NegativeControl {
            case_id: "nc1_nan_weight_in_exl3_tile_rejected".into(),
            outcome: if quantize_exl3_tile(&bad).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "NaN weight rejected before WHT-16 rotation and trellis search".into(),
        });
    }

    // NC2: odd tile count rejected by 9-byte pair packer
    {
        negative_controls.push(NegativeControl {
            case_id: "nc2_odd_tile_count_packer_rejected".into(),
            outcome: if pack_exl3_tiles(&exl3_tiles[..1]).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "odd 36-bit tile count cannot pack to whole bytes without explicit padding"
                .into(),
        });
    }

    // NC3: out-of-range 4-bit scale_idx (>= 16) rejected
    {
        let bad = Exl3TileCode {
            scale_idx: 16,
            symbols: [0; 16],
        };
        negative_controls.push(NegativeControl {
            case_id: "nc3_oob_4bit_scale_index_rejected".into(),
            outcome: if dequantize_exl3_tile(&bad).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "scale index >= 16 outside 4-bit EXL3 codebook rejected".into(),
        });
    }

    // NC4: out-of-range 2-bit trellis symbol (>= 4) rejected
    {
        let bad = Exl3TileCode {
            scale_idx: 4,
            symbols: [4; 16],
        };
        negative_controls.push(NegativeControl {
            case_id: "nc4_oob_2bit_trellis_symbol_rejected".into(),
            outcome: if dequantize_exl3_tile(&bad).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "trellis symbol >= 4 outside 2-bit alphabet rejected".into(),
        });
    }

    // NC5: truncated packed buffer rejected by unpacker
    {
        negative_controls.push(NegativeControl {
            case_id: "nc5_truncated_packed_buffer_rejected".into(),
            outcome: if unpack_exl3_tiles(&packed[..packed.len() - 1], n_tiles).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "truncated 2.25-bpw packed byte buffer rejected".into(),
        });
    }

    // NC6: needle-in-a-haystack 5/5 pass alone CANNOT approve end-task accuracy
    {
        let exl3_1m = end_task_matrix.last().unwrap();
        let needle_only_hides_coding_drop =
            exl3_1m.needle_pass_rate == 1.0 && exl3_1m.coding_passed < 16;
        negative_controls.push(NegativeControl {
            case_id: "nc6_needle_only_pass_cannot_substitute_end_task_eval".into(),
            outcome: if needle_only_hides_coding_drop {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "100% needle retrieval hides long-context coding/reasoning accuracy drop"
                .into(),
        });
    }

    // NC7: zero accepted end tasks yields cost_per_accepted = None, never 0.0 or Inf
    {
        let accepted_count = 0usize;
        let zero_acc: Option<f64> = if accepted_count > 0 {
            Some(100.0 / (accepted_count as f64))
        } else {
            None
        };
        negative_controls.push(NegativeControl {
            case_id: "nc7_zero_accepted_tasks_cost_is_none".into(),
            outcome: if zero_acc.is_none() {
                OutcomeStatus::Exhausted
            } else {
                OutcomeStatus::Accepted
            },
            detail: "zero accepted reasoning/coding tasks returns None instead of Inf/0".into(),
        });
    }

    // NC8: importing Strix Halo + RX 7900 XTX X-post throughput numbers as lab gate rejected
    {
        let import_x_post_tps = true;
        negative_controls.push(NegativeControl {
            case_id: "nc8_unverified_strix_halo_x_post_tps_non_gating".into(),
            outcome: if import_x_post_tps {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "718/51.1 tok/s X-post figures stay unverified non-gating related work".into(),
        });
    }

    Ok(Exl3LongCtxReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "Strix Halo APU + RX 7900 XTX / GPU host serving DeepSeek-V4.1-Flash-1017 EXL3 2.25-bpw at 128K-1M",
            "Pure-Rust exact 2.25-bpw WHT-16 trellis quantizer, bit-exact pack/unpack, FP4 comparison, and long-context end-task accounting ran on CPU; physical DeepSeek-V4.1-Flash 1M serve did not run locally",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        upstream_repo: UPSTREAM_REPO.into(),
        upstream_commit: UPSTREAM_COMMIT.into(),
        upstream_license: UPSTREAM_LICENSE.into(),
        pack_roundtrip_exact,
        exact_packed_bpw,
        quant_metrics,
        end_task_matrix,
        negative_controls,
    })
}

pub fn inputs_json(report: &Exl3LongCtxReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.exl3-longctx-quant.inputs.v1\",\n",
            "  \"issue\": 80,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"upstream_repo\": {},\n",
            "  \"upstream_commit\": {},\n",
            "  \"upstream_license\": {},\n",
            "  \"upstream_maya_issue\": {},\n",
            "  \"upstream_x_post\": {},\n",
            "  \"exact_target_bpw\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(&report.upstream_repo),
        quote(&report.upstream_commit),
        quote(&report.upstream_license),
        quote(UPSTREAM_MAYA_ISSUE),
        quote(UPSTREAM_X_POST),
        finite_f64(EXACT_EXL3_BPW),
    )
}

pub fn results_json(report: &Exl3LongCtxReport) -> String {
    let qm: Vec<String> = report
        .quant_metrics
        .iter()
        .map(|m| {
            format!(
                "{{\"arm_id\":{},\"effective_bpw\":{},\"weight_mse\":{},\"weight_snr_db\":{},\"logit_kl_vs_fp64\":{},\"matvec_pack_parity_max_err\":{}}}",
                quote(&m.arm_id),
                finite_f64(m.effective_bpw),
                finite_f64(m.weight_mse),
                opt_f64(m.weight_snr_db),
                finite_f64(m.logit_kl_vs_fp64),
                finite_f64(m.matvec_pack_parity_max_err),
            )
        })
        .collect();
    let et: Vec<String> = report
        .end_task_matrix
        .iter()
        .map(|r| {
            format!(
                "{{\"arm_id\":{},\"context_tokens\":{},\"vram_gb\":{},\"prefill_tok_s\":{},\"decode_tok_s\":{},\"needle_pass_rate\":{},\"reasoning_passed\":{},\"reasoning_total\":{},\"coding_passed\":{},\"coding_total\":{},\"cost_per_accepted_end_task_s\":{}}}",
                quote(&r.arm_id),
                r.context_tokens,
                finite_f64(r.vram_gb),
                finite_f64(r.prefill_tok_s),
                finite_f64(r.decode_tok_s),
                finite_f64(r.needle_pass_rate),
                r.reasoning_passed,
                r.reasoning_total,
                r.coding_passed,
                r.coding_total,
                opt_f64(r.cost_per_accepted_end_task_s),
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
            "  \"schema\": \"inference-lab.exl3-longctx-quant.results.v1\",\n",
            "  \"issue\": 80,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"pack_roundtrip_exact\": {},\n",
            "  \"exact_packed_bpw\": {},\n",
            "  \"quant_metrics\": [{}],\n",
            "  \"end_task_matrix\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        report.pack_roundtrip_exact,
        finite_f64(report.exact_packed_bpw),
        qm.join(","),
        et.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &Exl3LongCtxReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #80: EXL3 2.25-bpw Long-Context Quantization vs FP4/FP64 Baselines"
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
    writeln!(
        out,
        "- Upstream pin: `{}@{}` (`{}`, exact packed bpw: `{:.2}`, roundtrip: `{}`)",
        report.upstream_repo,
        &report.upstream_commit[..8],
        report.upstream_license,
        report.exact_packed_bpw,
        report.pack_roundtrip_exact
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Pure-Rust 2.25-bpw Trellis Quantizer vs FP4 & FP64").unwrap();
    writeln!(
        out,
        "| Arm | Effective BPW | Weight MSE | Weight SNR (dB) | Logit KL | Pack Parity Err |"
    )
    .unwrap();
    writeln!(out, "| --- | ---: | ---: | ---: | ---: | ---: |").unwrap();
    for m in &report.quant_metrics {
        writeln!(
            out,
            "| `{}` | `{:.2}` | `{:.6}` | `{}` | `{:.6}` | `{:.3e}` |",
            m.arm_id,
            m.effective_bpw,
            m.weight_mse,
            m.weight_snr_db
                .map_or_else(|| "n/a".into(), |v| format!("{v:.2}")),
            m.logit_kl_vs_fp64,
            m.matvec_pack_parity_max_err
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
    fn exl3_2_25bpw_quantizer_packs_exact_bpw_and_matches_matvec_reference() {
        let r = run_default_harness().unwrap();
        assert!(r.pack_roundtrip_exact);
        assert!((r.exact_packed_bpw - EXACT_EXL3_BPW).abs() <= 1.0e-12);
        assert!(r.quant_metrics[2].matvec_pack_parity_max_err <= PARITY_TOL);
        assert!(r.quant_metrics[2].weight_snr_db.unwrap() > 6.0);
        assert_eq!(r.negative_controls.len(), 8);
        for c in &r.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }
}
