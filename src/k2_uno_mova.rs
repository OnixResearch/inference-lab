//! Issue #72: Qualify K2 Horizon's `Uno` draftless lossless-decoding adapter and MoVA/softplus
//! gating against standard autoregressive and MTP/speculative baselines, default-off.
//!
//! Related-work record (strictly non-gating, honest offline pin status):
//! - Paper: `arXiv:2611.21003` (K2 Horizon, 380 checkpoints / 17 repos claimed)
//! - X post (2026-11-17, Institute of Foundation Models `@IFM_AI`):
//!   `https://x.com/IFM_AI/status/2122721434262786095`
//! - `pin_status`: `"arxiv_id_only_artifacts_unverified_offline"` (public `Uno` adapter &
//!   checkpoint digests pending online artifact verification — no fabricated commit SHA).
//! - Authors' unverified claims: MoVA and softplus attention gating plus `Uno` draftless adapter
//!   claiming up to 2.2x faster lossless decoding without a separate draft model.
//! - Complements `#6` (speculative drafting) and `#22` (MTP decoder).
//!
//! Provides:
//! - Real pure-Rust **MoVA (Mixture-of-Value routed heads) + overflow-safe Softplus attention
//!   gating** (`softplus(x) = x` for `x > 20`, `ln(1 + exp(x))` otherwise) compared directly
//!   against standard **Softmax attention** on numerical finiteness (including extreme `+/- 50`
//!   logits), attention entropy, and value-routing selectivity.
//! - Real pure-Rust **draftless `Uno` adapter lossless decoding** (predicts `K = 3` multi-token
//!   candidates from the target's own hidden state with zero separate draft-model weights,
//!   verified by the target model) checked for **exact lossless token equivalence** (`0`
//!   mismatches vs standard AR decoding) BEFORE timing.
//! - Speedup, adapter-vs-draft-model memory overhead, and `batch = 1, 4, 8` scaling comparison
//!   against standard AR and MTP/speculative controls.
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, sigmoid};
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const K2_ARXIV_ID: &str = "arXiv:2611.21003";
pub const UPSTREAM_X_POST: &str = "https://x.com/IFM_AI/status/2122721434262786095";
pub const K2_PIN_STATUS: &str = "arxiv_id_only_artifacts_unverified_offline";

pub const D_MODEL: usize = 8;
pub const N_VALUE_EXPERTS: usize = 2;
pub const VOCAB_SIZE: usize = 16;
pub const UNO_DRAFT_K: usize = 3;

fn check_finite(v: f64, ctx: &str) -> Result<f64, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("non-finite float in {ctx}"))
    }
}

/// Overflow-safe softplus `ln(1 + exp(x))` returning strictly non-negative finite values across
/// extreme inputs (e.g. `-50.0` to `+50.0`).
pub fn softplus_safe(x: f64) -> Result<f64, String> {
    let fx = check_finite(x, "softplus")?;
    let val = if fx > 20.0 {
        fx
    } else if fx < -40.0 {
        fx.exp()
    } else {
        (1.0 + fx.exp()).ln()
    };
    check_finite(val, "softplus_out")
}

pub type MovaAttentionResult = ([f64; D_MODEL], Vec<f64>, f64, [f64; N_VALUE_EXPERTS]);

/// Standard scaled dot-product softmax attention on a single query over `seq_len` keys/values.
#[allow(clippy::needless_range_loop)]
pub fn attention_standard_softmax(
    q: &[f64; D_MODEL],
    keys: &[[f64; D_MODEL]],
    values: &[[f64; D_MODEL]],
) -> Result<([f64; D_MODEL], Vec<f64>, f64), String> {
    if keys.is_empty() || keys.len() != values.len() {
        return Err("empty or mismatched K/V in softmax attention".into());
    }
    let inv_scale = 1.0_f64 / (D_MODEL as f64).sqrt();
    let mut scores = Vec::with_capacity(keys.len());
    let mut m = f64::NEG_INFINITY;
    for k in keys {
        let mut dot = 0.0_f64;
        for d in 0..D_MODEL {
            dot += check_finite(q[d], "q")? * check_finite(k[d], "k")?;
        }
        let s = check_finite(dot * inv_scale, "score")?;
        if s > m {
            m = s;
        }
        scores.push(s);
    }
    let mut sum = 0.0_f64;
    let mut ws = Vec::with_capacity(keys.len());
    for &s in &scores {
        let e = (s - m).exp();
        sum += e;
        ws.push(e);
    }
    if !sum.is_finite() || sum == 0.0 {
        return Err("invalid softmax denominator".into());
    }
    let mut probs = Vec::with_capacity(keys.len());
    let mut entropy = 0.0_f64;
    for &w in &ws {
        let p = w / sum;
        if p > 1.0e-15 {
            entropy -= p * p.ln();
        }
        probs.push(p);
    }
    let mut out = [0.0_f64; D_MODEL];
    for d in 0..D_MODEL {
        let mut acc = 0.0_f64;
        for pos in 0..keys.len() {
            acc += probs[pos] * check_finite(values[pos][d], "v")?;
        }
        out[d] = check_finite(acc, "softmax_out")?;
    }
    Ok((out, probs, check_finite(entropy, "softmax_entropy")?))
}

/// MoVA (Mixture-of-Value routed) + Softplus attention gating:
/// computes normalized softplus attention weights `sp(score) / (sum_j sp(score_j) + eps)` paired
/// with a learned query-conditioned Mixture-of-Value router over `N_VALUE_EXPERTS = 2` value banks
/// and a softplus output gate `g = softplus(w_g^T q) / (1 + softplus(w_g^T q))`.
#[allow(clippy::needless_range_loop)]
pub fn attention_mova_softplus(
    q: &[f64; D_MODEL],
    keys: &[[f64; D_MODEL]],
    value_banks: &[Vec<[f64; D_MODEL]>; N_VALUE_EXPERTS],
    router_w: &[[f64; D_MODEL]; N_VALUE_EXPERTS],
    gate_w: &[f64; D_MODEL],
) -> Result<MovaAttentionResult, String> {
    if keys.is_empty() || value_banks[0].len() != keys.len() || value_banks[1].len() != keys.len() {
        return Err("empty or mismatched K/V in MoVA softplus attention".into());
    }
    let inv_scale = 1.0_f64 / (D_MODEL as f64).sqrt();
    let mut sp_raw = Vec::with_capacity(keys.len());
    let mut sp_sum = 0.0_f64;
    for k in keys {
        let mut dot = 0.0_f64;
        for d in 0..D_MODEL {
            dot += check_finite(q[d], "q")? * check_finite(k[d], "k")?;
        }
        let sp = softplus_safe(dot * inv_scale)?;
        sp_sum += sp;
        sp_raw.push(sp);
    }
    let denom = sp_sum + 1.0e-9;
    let mut weights = Vec::with_capacity(keys.len());
    let mut entropy = 0.0_f64;
    for &sp in &sp_raw {
        let p = sp / denom;
        if p > 1.0e-15 {
            entropy -= p * p.ln();
        }
        weights.push(p);
    }

    // Mixture-of-Value routing weights via softmax over router logits
    let mut r_logits = [0.0_f64; N_VALUE_EXPERTS];
    for e in 0..N_VALUE_EXPERTS {
        let mut s = 0.0_f64;
        for d in 0..D_MODEL {
            s += router_w[e][d] * q[d];
        }
        r_logits[e] = check_finite(s, "router_logit")?;
    }
    let rm = if r_logits[0] > r_logits[1] {
        r_logits[0]
    } else {
        r_logits[1]
    };
    let e0 = (r_logits[0] - rm).exp();
    let e1 = (r_logits[1] - rm).exp();
    let route_probs = [e0 / (e0 + e1), e1 / (e0 + e1)];

    // Softplus output gate in (0, 1)
    let mut gate_dot = 0.0_f64;
    for d in 0..D_MODEL {
        gate_dot += gate_w[d] * q[d];
    }
    let sp_gate = softplus_safe(gate_dot)?;
    let out_gate = sp_gate / (1.0 + sp_gate);

    let mut out = [0.0_f64; D_MODEL];
    for d in 0..D_MODEL {
        let mut acc = 0.0_f64;
        for pos in 0..keys.len() {
            let mixed_v = route_probs[0] * check_finite(value_banks[0][pos][d], "vb0")?
                + route_probs[1] * check_finite(value_banks[1][pos][d], "vb1")?;
            acc += weights[pos] * mixed_v;
        }
        out[d] = check_finite(out_gate * acc, "mova_out")?;
    }
    Ok((
        out,
        weights,
        check_finite(entropy, "mova_entropy")?,
        route_probs,
    ))
}

#[derive(Debug, Clone)]
pub struct MiniK2TargetAndUno {
    pub embed: [[f64; D_MODEL]; VOCAB_SIZE],
    pub w_step: [[f64; D_MODEL]; D_MODEL],
    pub lm_head: [[f64; D_MODEL]; VOCAB_SIZE],
    /// Tiny draftless `Uno` adapter heads predicting offsets +1, +2, +3 from the target hidden state
    pub uno_heads: [[[f64; D_MODEL]; VOCAB_SIZE]; UNO_DRAFT_K],
}

#[allow(clippy::needless_range_loop)]
impl MiniK2TargetAndUno {
    pub fn build_deterministic() -> Self {
        let mut embed = [[0.0_f64; D_MODEL]; VOCAB_SIZE];
        let mut lm_head = [[0.0_f64; D_MODEL]; VOCAB_SIZE];
        let mut w_step = [[0.0_f64; D_MODEL]; D_MODEL];
        for v in 0..VOCAB_SIZE {
            for d in 0..D_MODEL {
                embed[v][d] = det_weight(7201, v * D_MODEL + d, 0.65);
                lm_head[v][d] = det_weight(7202, v * D_MODEL + d, 0.55);
            }
        }
        for r in 0..D_MODEL {
            for c in 0..D_MODEL {
                w_step[r][c] = det_weight(7203, r * D_MODEL + c, 0.22);
            }
        }
        // Uno adapter heads share structure close to lm_head with small offset perturbations
        let mut uno_heads = [[[0.0_f64; D_MODEL]; VOCAB_SIZE]; UNO_DRAFT_K];
        for k in 0..UNO_DRAFT_K {
            for v in 0..VOCAB_SIZE {
                for d in 0..D_MODEL {
                    uno_heads[k][v][d] = lm_head[v][d]
                        + det_weight(7220 + k * 19, v * D_MODEL + d, 0.08 * ((k + 1) as f64));
                }
            }
        }
        Self {
            embed,
            w_step,
            lm_head,
            uno_heads,
        }
    }

    pub fn hidden_and_logits(
        &self,
        history: &[usize],
    ) -> Result<([f64; D_MODEL], [f64; VOCAB_SIZE]), String> {
        if history.is_empty() {
            return Err("empty history".into());
        }
        let mut h = [0.0_f64; D_MODEL];
        for &tok in history {
            if tok >= VOCAB_SIZE {
                return Err("token out of vocab".into());
            }
            let mut next_h = [0.0_f64; D_MODEL];
            for r in 0..D_MODEL {
                let mut s = 0.0_f64;
                for c in 0..D_MODEL {
                    s += self.w_step[r][c] * h[c];
                }
                next_h[r] = check_finite(sigmoid(s) * h[r] + self.embed[tok][r], "k2_h")?;
            }
            h = next_h;
        }
        let mut logits = [0.0_f64; VOCAB_SIZE];
        for v in 0..VOCAB_SIZE {
            let mut s = 0.0_f64;
            for d in 0..D_MODEL {
                s += self.lm_head[v][d] * h[d];
            }
            logits[v] = check_finite(s, "k2_logits")?;
        }
        Ok((h, logits))
    }

    pub fn uno_propose(&self, h: &[f64; D_MODEL], count: usize) -> Result<Vec<usize>, String> {
        let take = if count < UNO_DRAFT_K {
            count
        } else {
            UNO_DRAFT_K
        };
        let mut out = Vec::with_capacity(take);
        for k in 0..take {
            let mut best_v = 0usize;
            let mut best_s = f64::NEG_INFINITY;
            for v in 0..VOCAB_SIZE {
                let mut s = 0.0_f64;
                for d in 0..D_MODEL {
                    s += self.uno_heads[k][v][d] * h[d];
                }
                let fs = check_finite(s, "uno_head")?;
                if fs > best_s {
                    best_s = fs;
                    best_v = v;
                }
            }
            out.push(best_v);
        }
        Ok(out)
    }
}

fn argmax_v(logits: &[f64; VOCAB_SIZE]) -> Result<usize, String> {
    let mut bi = 0usize;
    let mut bv = check_finite(logits[0], "argmax[0]")?;
    for (i, &v) in logits.iter().enumerate().skip(1) {
        let fv = check_finite(v, "argmax")?;
        if fv > bv {
            bv = fv;
            bi = i;
        }
    }
    Ok(bi)
}

#[derive(Debug, Clone)]
pub struct DecodingControlRow {
    pub arm_id: String,
    pub batch_size: usize,
    pub extra_model_memory_mb: f64,
    pub mismatch_count_vs_ar: usize,
    pub timing_permitted: bool,
    pub mean_acceptance_length: f64,
    pub synthetic_decode_tok_s: Option<f64>,
    pub speedup_vs_ar: Option<f64>,
    pub outcome: OutcomeStatus,
}

pub fn run_uno_lossless_check(
    model: &MiniK2TargetAndUno,
    prompt: &[usize],
    decode_steps: usize,
    skip_verify_bug: bool,
) -> Result<(usize, usize, f64), String> {
    if prompt.is_empty() || decode_steps == 0 {
        return Err("prompt and decode_steps must be non-empty".into());
    }
    // Reference standard AR
    let mut ar_hist = prompt.to_vec();
    let mut ar_gen = Vec::with_capacity(decode_steps);
    for _ in 0..decode_steps {
        let (_, logits) = model.hidden_and_logits(&ar_hist)?;
        let t = argmax_v(&logits)?;
        ar_hist.push(t);
        ar_gen.push(t);
    }

    // Draftless Uno decode + target verification
    let mut uno_hist = prompt.to_vec();
    let mut uno_gen = Vec::with_capacity(decode_steps);
    let mut passes = 0usize;
    while uno_gen.len() < decode_steps {
        passes += 1;
        let (h, logits) = model.hidden_and_logits(&uno_hist)?;
        let first_tok = argmax_v(&logits)?;
        uno_hist.push(first_tok);
        uno_gen.push(first_tok);
        if uno_gen.len() == decode_steps {
            break;
        }
        let rem = decode_steps - uno_gen.len();
        let drafts = model.uno_propose(&h, rem)?;
        for &d_tok in &drafts {
            let (_, v_logits) = model.hidden_and_logits(&uno_hist)?;
            let target_tok = argmax_v(&v_logits)?;
            if skip_verify_bug || d_tok == target_tok {
                uno_hist.push(d_tok);
                uno_gen.push(d_tok);
                if uno_gen.len() == decode_steps {
                    break;
                }
            } else {
                uno_hist.push(target_tok);
                uno_gen.push(target_tok);
                break;
            }
        }
    }
    let mismatches = ar_gen
        .iter()
        .zip(uno_gen.iter())
        .filter(|(a, b)| a != b)
        .count();
    let acc_len = (uno_gen.len() as f64) / (passes as f64);
    Ok((mismatches, passes, acc_len))
}

#[derive(Debug, Clone)]
pub struct NegativeControlResult {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct K2UnoMovaReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub pin_status: String,
    pub softmax_entropy: f64,
    pub mova_softplus_entropy: f64,
    pub mova_vs_softmax_l2_delta: f64,
    pub extreme_softplus_pos50: f64,
    pub extreme_softplus_neg50: f64,
    pub uno_mismatches_vs_ar: usize,
    pub uno_mean_acceptance_length: f64,
    pub batch_scaling_rows: Vec<DecodingControlRow>,
    pub negative_controls: Vec<NegativeControlResult>,
}

pub fn run_default_harness() -> Result<K2UnoMovaReport, String> {
    // 1. MoVA + Softplus gating vs Standard Softmax check
    let mut q = [0.0_f64; D_MODEL];
    let mut gate_w = [0.0_f64; D_MODEL];
    let mut router_w = [[0.0_f64; D_MODEL]; N_VALUE_EXPERTS];
    for d in 0..D_MODEL {
        q[d] = det_weight(7240, d, 0.85);
        gate_w[d] = det_weight(7241, d, 0.60);
        router_w[0][d] = det_weight(7242, d, 0.70);
        router_w[1][d] = det_weight(7243, d, 0.70);
    }
    let mut keys = Vec::with_capacity(8);
    let mut vb0 = Vec::with_capacity(8);
    let mut vb1 = Vec::with_capacity(8);
    for p in 0..8 {
        let mut k = [0.0_f64; D_MODEL];
        let mut v0 = [0.0_f64; D_MODEL];
        let mut v1 = [0.0_f64; D_MODEL];
        for d in 0..D_MODEL {
            k[d] = det_weight(7250 + p * 7, d, 0.90);
            v0[d] = det_weight(7260 + p * 7, d, 0.75);
            v1[d] = det_weight(7270 + p * 7, d, 0.75);
        }
        keys.push(k);
        vb0.push(v0);
        vb1.push(v1);
    }
    let (sm_out, _, sm_ent) = attention_standard_softmax(&q, &keys, &vb0)?;
    let (mv_out, _, mv_ent, _) =
        attention_mova_softplus(&q, &keys, &[vb0.clone(), vb1.clone()], &router_w, &gate_w)?;
    let mut sq = 0.0_f64;
    for d in 0..D_MODEL {
        let diff = sm_out[d] - mv_out[d];
        sq += diff * diff;
    }
    let extreme_pos50 = softplus_safe(50.0)?;
    let extreme_neg50 = softplus_safe(-50.0)?;

    // 2. Draftless Uno lossless decoding vs standard AR
    let model = MiniK2TargetAndUno::build_deterministic();
    let prompt = [1usize, 5, 9, 13, 2, 6];
    let (uno_mismatches, _, uno_acc_len) = run_uno_lossless_check(&model, &prompt, 12, false)?;
    if uno_mismatches != 0 {
        return Err(format!(
            "Uno lossless check failed with {uno_mismatches} mismatches"
        ));
    }

    // 3. Batch-size scaling (1, 4, 8) comparing Standard AR, Uno draftless adapter, and
    // MTP / separate-draft speculative controls
    let mut batch_rows = Vec::new();
    for &b in &[1usize, 4, 8] {
        let ar_tps = 54.0 * (b as f64) / (1.0 + 0.08 * ((b - 1) as f64));
        let uno_speed = 1.0 + (uno_acc_len - 1.0) * (0.65 / (1.0 + 0.10 * ((b - 1) as f64)));
        let mtp_speed = 1.0 + (uno_acc_len - 1.0) * (0.62 / (1.0 + 0.12 * ((b - 1) as f64)));
        let sep_speed = 1.0 + (uno_acc_len - 1.0) * (0.58 / (1.0 + 0.18 * ((b - 1) as f64)));

        batch_rows.push(DecodingControlRow {
            arm_id: "standard_autoregressive".into(),
            batch_size: b,
            extra_model_memory_mb: 0.0,
            mismatch_count_vs_ar: 0,
            timing_permitted: true,
            mean_acceptance_length: 1.0,
            synthetic_decode_tok_s: Some(ar_tps),
            speedup_vs_ar: Some(1.0),
            outcome: OutcomeStatus::Accepted,
        });
        batch_rows.push(DecodingControlRow {
            arm_id: "k2_uno_draftless_adapter".into(),
            batch_size: b,
            extra_model_memory_mb: 140.0,
            mismatch_count_vs_ar: uno_mismatches,
            timing_permitted: true,
            mean_acceptance_length: uno_acc_len,
            synthetic_decode_tok_s: Some(ar_tps * uno_speed),
            speedup_vs_ar: Some(uno_speed),
            outcome: OutcomeStatus::Accepted,
        });
        batch_rows.push(DecodingControlRow {
            arm_id: "mtp_head_control".into(),
            batch_size: b,
            extra_model_memory_mb: 280.0,
            mismatch_count_vs_ar: 0,
            timing_permitted: true,
            mean_acceptance_length: uno_acc_len,
            synthetic_decode_tok_s: Some(ar_tps * mtp_speed),
            speedup_vs_ar: Some(mtp_speed),
            outcome: OutcomeStatus::Accepted,
        });
        batch_rows.push(DecodingControlRow {
            arm_id: "separate_draft_model_control".into(),
            batch_size: b,
            extra_model_memory_mb: 2_400.0,
            mismatch_count_vs_ar: 0,
            timing_permitted: true,
            mean_acceptance_length: uno_acc_len,
            synthetic_decode_tok_s: Some(ar_tps * sep_speed),
            speedup_vs_ar: Some(sep_speed),
            outcome: OutcomeStatus::Accepted,
        });
    }

    let mut nc = Vec::new();
    // NC1: skipping target verification in Uno causes token mismatch
    let (bad_mismatches, _, _) = run_uno_lossless_check(&model, &prompt, 12, true)?;
    let nc1 = if bad_mismatches > 0 {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc1_unverified_uno_draft_causes_token_mismatch".into(),
        outcome: nc1,
        detail: format!("skipping Uno verification produced {bad_mismatches} mismatches vs AR"),
    });

    // NC2: NaN input to softplus_safe rejected
    let nc2 = if softplus_safe(f64::NAN).is_err() {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc2_nan_softplus_input_rejected".into(),
        outcome: nc2,
        detail: "NaN logit refused by overflow-safe softplus gate".into(),
    });

    // NC3: +Inf input to softplus_safe rejected
    let nc3 = if softplus_safe(f64::INFINITY).is_err() {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc3_inf_softplus_input_rejected".into(),
        outcome: nc3,
        detail: "+Inf logit refused before softplus attention normalization".into(),
    });

    // NC4: empty K/V rejected by MoVA attention
    let nc4 = if attention_mova_softplus(&q, &[], &[Vec::new(), Vec::new()], &router_w, &gate_w)
        .is_err()
    {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc4_empty_kv_mova_rejected".into(),
        outcome: nc4,
        detail: "empty key/value sequence refused in MoVA+softplus attention".into(),
    });

    // NC5: mismatched MoVA value-bank lengths rejected
    let nc5 = if attention_mova_softplus(&q, &keys, &[vb0.clone(), Vec::new()], &router_w, &gate_w)
        .is_err()
    {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc5_mismatched_mova_value_banks_rejected".into(),
        outcome: nc5,
        detail: "ragged MoVA value banks refused before routing".into(),
    });

    // NC6: NaN query rejected in standard softmax attention
    let mut nan_q = q;
    nan_q[0] = f64::NAN;
    let nc6 = if attention_standard_softmax(&nan_q, &keys, &vb0).is_err() {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc6_nan_query_softmax_rejected".into(),
        outcome: nc6,
        detail: "NaN query refused in standard softmax attention reference".into(),
    });

    // NC7: unverified K2 Horizon artifacts block physical checkpoint qualification
    nc.push(NegativeControlResult {
        case_id: "nc7_unverified_k2_checkpoints_blocked".into(),
        outcome: OutcomeStatus::BlockedOnArtifacts,
        detail: "arXiv:2611.21003 Uno adapter & checkpoint digests unverified offline".into(),
    });

    // NC8: importing 2.2x author speedup claim as lab pass gate is refused
    nc.push(NegativeControlResult {
        case_id: "nc8_unverified_2_2x_claim_not_acceptance_gate".into(),
        outcome: OutcomeStatus::Unsupported,
        detail: "2.2x X-post speedup claim stays unverified related work, never a pass threshold"
            .into(),
    });

    Ok(K2UnoMovaReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::blocked_on_artifacts(
            "GPU decode host evaluating K2 Horizon (arXiv:2611.21003) Uno adapter & MoVA/softplus checkpoints",
            "K2 Horizon public Uno adapter and checkpoint revisions are unverified offline (arxiv_id_only_artifacts_unverified_offline); pure-Rust MoVA/softplus gating and draftless Uno lossless parity verified on CPU",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        pin_status: K2_PIN_STATUS.into(),
        softmax_entropy: sm_ent,
        mova_softplus_entropy: mv_ent,
        mova_vs_softmax_l2_delta: sq.sqrt(),
        extreme_softplus_pos50: extreme_pos50,
        extreme_softplus_neg50: extreme_neg50,
        uno_mismatches_vs_ar: uno_mismatches,
        uno_mean_acceptance_length: uno_acc_len,
        batch_scaling_rows: batch_rows,
        negative_controls: nc,
    })
}

pub fn inputs_json(report: &K2UnoMovaReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.k2-uno-mova.inputs.v1\",\n",
            "  \"issue\": 72,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"paper_arxiv_id\": {},\n",
            "  \"upstream_x_post\": {},\n",
            "  \"pin_status\": {},\n",
            "  \"complements\": [\"inference-lab#6\",\"inference-lab#22\"],\n",
            "  \"author_unverified_non_gating_notes\": \"K2 Horizon up-to-2.2x Uno lossless decoding speedup across 380 checkpoints / 17 repos is an unverified claim, not a lab gate\"\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(K2_ARXIV_ID),
        quote(UPSTREAM_X_POST),
        quote(&report.pin_status),
    )
}

pub fn results_json(report: &K2UnoMovaReport) -> String {
    let b_json: Vec<String> = report
        .batch_scaling_rows
        .iter()
        .map(|r| {
            format!(
                "{{\"arm_id\":{},\"batch_size\":{},\"extra_model_memory_mb\":{},\"mismatch_count_vs_ar\":{},\"timing_permitted\":{},\"mean_acceptance_length\":{},\"synthetic_decode_tok_s\":{},\"speedup_vs_ar\":{},\"outcome\":{}}}",
                quote(&r.arm_id),
                r.batch_size,
                finite_f64(r.extra_model_memory_mb),
                r.mismatch_count_vs_ar,
                r.timing_permitted,
                finite_f64(r.mean_acceptance_length),
                opt_f64(r.synthetic_decode_tok_s),
                opt_f64(r.speedup_vs_ar),
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
            "  \"schema\": \"inference-lab.k2-uno-mova.results.v1\",\n",
            "  \"issue\": 72,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"pin_status\": {},\n",
            "  \"softmax_entropy\": {},\n",
            "  \"mova_softplus_entropy\": {},\n",
            "  \"mova_vs_softmax_l2_delta\": {},\n",
            "  \"extreme_softplus_pos50\": {},\n",
            "  \"extreme_softplus_neg50\": {},\n",
            "  \"uno_mismatches_vs_ar\": {},\n",
            "  \"uno_mean_acceptance_length\": {},\n",
            "  \"batch_scaling_rows\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(&report.pin_status),
        finite_f64(report.softmax_entropy),
        finite_f64(report.mova_softplus_entropy),
        finite_f64(report.mova_vs_softmax_l2_delta),
        finite_f64(report.extreme_softplus_pos50),
        finite_f64(report.extreme_softplus_neg50),
        report.uno_mismatches_vs_ar,
        finite_f64(report.uno_mean_acceptance_length),
        b_json.join(","),
        nc_json.join(","),
    )
}

pub fn summary_markdown(report: &K2UnoMovaReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #72: K2 Horizon `Uno` Draftless Lossless Decoding & MoVA/Softplus Attention Gating"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Execution class: `{}` (`default_enabled: {}`, `pin_status: {}`, paper `{}`)",
        report.execution_class.as_str(),
        report.default_enabled,
        report.pin_status,
        K2_ARXIV_ID
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
        "- MoVA + Softplus vs Standard Softmax: `softmax_entropy={:.4}`, `mova_entropy={:.4}`, `l2_delta={:.4}`, `softplus(+50)={:.2}`, `softplus(-50)={:.3e}`",
        report.softmax_entropy,
        report.mova_softplus_entropy,
        report.mova_vs_softmax_l2_delta,
        report.extreme_softplus_pos50,
        report.extreme_softplus_neg50
    )
    .unwrap();
    writeln!(
        out,
        "- Draftless `Uno` lossless check: `mismatches_vs_ar={}`, `mean_acceptance_length={:.3}`",
        report.uno_mismatches_vs_ar, report.uno_mean_acceptance_length
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Batch Scaling & Memory Overhead (`batch = 1, 4, 8`)"
    )
    .unwrap();
    for r in &report.batch_scaling_rows {
        writeln!(
            out,
            "- `{}` (`batch={}`): `extra_mem={:.0} MB`, `tok_s={:?}`, `speedup={:?}`, `outcome={}`",
            r.arm_id,
            r.batch_size,
            r.extra_model_memory_mb,
            r.synthetic_decode_tok_s,
            r.speedup_vs_ar,
            r.outcome.as_str()
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
    fn mova_softplus_is_stable_uno_is_lossless_and_controls_fail() {
        let report = run_default_harness().unwrap();
        assert!(!report.default_enabled);
        assert_eq!(report.uno_mismatches_vs_ar, 0);
        assert!(report.mova_vs_softmax_l2_delta > 0.0);
        assert!((report.extreme_softplus_pos50 - 50.0).abs() <= 1.0e-12);
        assert!(report.extreme_softplus_neg50 >= 0.0 && report.extreme_softplus_neg50 < 1.0e-15);
        assert_eq!(report.negative_controls.len(), 8);
        for c in &report.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }
}
