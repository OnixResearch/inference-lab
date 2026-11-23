//! Issue #33: Qualify Inkling block attention residuals and dynamic sinks with matched ablations.
//!
//! Pinned source basis (strictly non-gating external claims):
//! - Primary target: `thinkingmachines/Inkling-2-Small` (vLLM overview 2026-11-12, NeMo AutoModel
//!   architecture docs, Liu Liu highlight).
//! - Related work ONLY (explicitly NOT in scope and NOT imported as gates):
//!   - `thinkingmachines/Inkling-2-Mini@6e7c8248f75265bdd4fbab5273304243b5d8c2cb` (tracked in `#57`).
//!   - Goodfire's 20 Nov 2026 mechanistic study (`d_2027`, writer/reader cosine survey, lens study)
//!     and NDIF Looped Lens Challenge (future looped-model probes belong in `#45`).
//!
//! Provides:
//! - Independent transparent FP64 reference and candidate execution path for a tractable 12-layer
//!   (two 6-layer blocks) miniature with RMSNorm, learned depth queries attending over completed
//!   6-layer block sums + active block accumulation/reset/final aggregation, input-dependent dynamic
//!   attention sinks, static sink control, and output gating (`sigmoid(W_g x)`).
//! - Tiny hand-computable fixtures covering 6-layer block boundaries, multi-token/multi-head causal
//!   and sliding/full attention masks, and extreme sink/gate values (`-50.0`, `+50.0`).
//! - Full-sequence prefill vs chunked prefill vs token-by-token decode parity with cache/conv state
//!   and reset under declared dtype tolerances.
//! - Clean 6-arm factorial ablation separating residual aggregation, dynamic sink, static sink, and
//!   output gate, plus explicit separation between weight-disabling sensitivity and matched-budget
//!   small-model training superiority, and 7 executable negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, sigmoid};
use crate::json::{finite_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const INKLING_BLOCK_LAYERS: usize = 6;
pub const INKLING_TOTAL_LAYERS: usize = 12; // Two 6-layer blocks
pub const INKLING_DIM: usize = 4;
pub const INKLING_HEADS: usize = 2;
pub const INKLING_HEAD_DIM: usize = 2;
pub const INKLING_SLIDING_WINDOW: usize = 3;
pub const RELATED_MINI_COMMIT: &str = "6e7c8248f75265bdd4fbab5273304243b5d8c2cb";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualMode {
    OrdinaryCumulative,
    BlockAttentionResidual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SinkMode {
    None,
    StaticSink,
    DynamicInputSink,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateMode {
    FixedEnabled,
    AblatedIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AblationConfig {
    pub arm_id: &'static str,
    pub residual: ResidualMode,
    pub sink: SinkMode,
    pub gate: GateMode,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct InklingFaults {
    pub omit_completed_block_candidate: bool,
    pub skip_block_reset_at_boundary: bool,
    pub leak_causal_future_token: bool,
    pub force_extreme_sink_override: Option<f64>,
    pub force_extreme_gate_override: Option<f64>,
}

fn checked_max_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err(format!("seq len mismatch: {} vs {}", a.len(), b.len()));
    }
    let mut max_err = 0.0_f64;
    for (ra, rb) in a.iter().zip(b.iter()) {
        if ra.len() != rb.len() {
            return Err(format!("dim mismatch: {} vs {}", ra.len(), rb.len()));
        }
        for (&x, &y) in ra.iter().zip(rb.iter()) {
            if !x.is_finite() || !y.is_finite() {
                return Err("non-finite value in Inkling output comparison".into());
            }
            let d = (x - y).abs();
            if d > max_err {
                max_err = d;
            }
        }
    }
    Ok(max_err)
}

fn rms_norm(x: &[f64; INKLING_DIM]) -> Result<[f64; INKLING_DIM], String> {
    let mut sq = 0.0_f64;
    for &v in x {
        if !v.is_finite() {
            return Err("non-finite input to RMSNorm".into());
        }
        sq += v * v;
    }
    let inv = 1.0 / (sq / (INKLING_DIM as f64) + 1.0e-6).sqrt();
    let mut out = [0.0_f64; INKLING_DIM];
    for i in 0..INKLING_DIM {
        out[i] = x[i] * inv;
    }
    Ok(out)
}

fn lin_proj(x: &[f64; INKLING_DIM], seed: usize) -> [f64; INKLING_DIM] {
    let mut out = [0.0_f64; INKLING_DIM];
    for (o, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for (i, &xv) in x.iter().enumerate() {
            acc += xv * det_weight(seed + o * 11, i, 0.5);
        }
        *slot = acc;
    }
    out
}

/// Depth-wise attention over candidate block sums (and active partial block sum).
fn depth_attention_mix(
    layer: usize,
    candidates: &[[f64; INKLING_DIM]],
) -> Result<[f64; INKLING_DIM], String> {
    if candidates.is_empty() {
        return Err("empty block candidates in depth attention".into());
    }
    let mut query = [0.0_f64; INKLING_DIM];
    for (i, q) in query.iter_mut().enumerate() {
        *q = det_weight(3300 + layer * 17, i, 0.75);
    }
    let mut logits = Vec::with_capacity(candidates.len());
    let mut max_l = f64::NEG_INFINITY;
    for c in candidates {
        let nc = rms_norm(c)?;
        let l: f64 = query.iter().zip(nc.iter()).map(|(a, b)| a * b).sum();
        if !l.is_finite() {
            return Err("non-finite depth-attention logit".into());
        }
        if l > max_l {
            max_l = l;
        }
        logits.push(l);
    }
    let mut exps = Vec::with_capacity(logits.len());
    let mut sum_e = 0.0_f64;
    for l in logits {
        let e = (l - max_l).exp();
        sum_e += e;
        exps.push(e);
    }
    let mut out = [0.0_f64; INKLING_DIM];
    for (e, c) in exps.iter().zip(candidates.iter()) {
        let w = *e / sum_e;
        for d in 0..INKLING_DIM {
            out[d] += w * c[d];
        }
    }
    Ok(out)
}

struct AttentionSinkArgs<'a> {
    layer: usize,
    head: usize,
    q_h: &'a [f64],
    k_hist: &'a [[f64; INKLING_HEAD_DIM]],
    v_hist: &'a [[f64; INKLING_HEAD_DIM]],
    token_input: &'a [f64; INKLING_DIM],
    sink_mode: SinkMode,
    sliding: bool,
    t_idx: usize,
    total_len: usize,
    faults: InklingFaults,
}

/// Softmax attention with optional static/dynamic sink logit (sink absorbs probability mass without
/// contributing to value mix, preserving exact numerical stability even at +/-50 sink logits).
fn attention_with_sink(args: &AttentionSinkArgs<'_>) -> Result<[f64; INKLING_HEAD_DIM], String> {
    let end_exclusive = if args.faults.leak_causal_future_token {
        args.total_len
    } else {
        args.t_idx + 1
    };
    let start = if args.sliding {
        (args.t_idx + 1).saturating_sub(INKLING_SLIDING_WINDOW)
    } else {
        0
    };
    let scale = 1.0 / (INKLING_HEAD_DIM as f64).sqrt();
    let mut logits = Vec::with_capacity(end_exclusive - start + 1);
    let mut max_l = f64::NEG_INFINITY;
    for s in start..end_exclusive {
        let dot = args.q_h[0] * args.k_hist[s][0] + args.q_h[1] * args.k_hist[s][1];
        let l = dot * scale;
        if !l.is_finite() {
            return Err("non-finite attention logit".into());
        }
        if l > max_l {
            max_l = l;
        }
        logits.push(l);
    }

    let sink_logit: Option<f64> = match args.sink_mode {
        SinkMode::None => None,
        SinkMode::StaticSink => {
            let s = args
                .faults
                .force_extreme_sink_override
                .unwrap_or_else(|| det_weight(3340 + args.layer * 5, args.head, 0.8));
            if !s.is_finite() {
                return Err("non-finite static sink logit".into());
            }
            if s > max_l {
                max_l = s;
            }
            Some(s)
        }
        SinkMode::DynamicInputSink => {
            let s = if let Some(ov) = args.faults.force_extreme_sink_override {
                ov
            } else {
                let mut acc = 0.0_f64;
                for (i, &xv) in args.token_input.iter().enumerate() {
                    acc += xv * det_weight(3360 + args.layer * 9 + args.head * 3, i, 0.9);
                }
                acc
            };
            if !s.is_finite() {
                return Err("non-finite dynamic sink logit".into());
            }
            if s > max_l {
                max_l = s;
            }
            Some(s)
        }
    };

    let mut denom = 0.0_f64;
    let mut exps = Vec::with_capacity(logits.len());
    for l in logits {
        let e = (l - max_l).exp();
        denom += e;
        exps.push(e);
    }
    if let Some(sl) = sink_logit {
        denom += (sl - max_l).exp();
    }
    if !denom.is_finite() || denom <= 0.0 {
        return Err("non-finite or zero softmax denominator".into());
    }

    let mut out = [0.0_f64; INKLING_HEAD_DIM];
    for (idx, e) in exps.iter().enumerate() {
        let w = *e / denom;
        let s = start + idx;
        out[0] += w * args.v_hist[s][0];
        out[1] += w * args.v_hist[s][1];
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct InklingCache {
    pub k: Vec<Vec<[[f64; INKLING_HEAD_DIM]; INKLING_HEADS]>>, // [layer][pos][head]
    pub v: Vec<Vec<[[f64; INKLING_HEAD_DIM]; INKLING_HEADS]>>,
    pub conv_prev: Vec<[f64; INKLING_DIM]>,
}

impl InklingCache {
    pub fn fresh() -> Self {
        Self {
            k: vec![Vec::new(); INKLING_TOTAL_LAYERS],
            v: vec![Vec::new(); INKLING_TOTAL_LAYERS],
            conv_prev: vec![[0.0; INKLING_DIM]; INKLING_TOTAL_LAYERS],
        }
    }
}

/// Independent transparent full-sequence reference forward pass.
pub fn forward_full_reference(
    tokens: &[u32],
    cfg: AblationConfig,
    faults: InklingFaults,
) -> Result<Vec<Vec<f64>>, String> {
    if tokens.is_empty() {
        return Err("empty token sequence".into());
    }
    let seq_len = tokens.len();
    let mut embeds: Vec<[f64; INKLING_DIM]> = (0..seq_len)
        .map(|t| {
            let mut e = [0.0_f64; INKLING_DIM];
            for (d, slot) in e.iter_mut().enumerate() {
                *slot = det_weight(3301 + (tokens[t] as usize) * 19 + t * 3, d, 0.6);
            }
            e
        })
        .collect();

    // Per-token completed 6-layer block sums + active block sum
    let mut completed_blocks: Vec<Vec<[f64; INKLING_DIM]>> =
        embeds.iter().map(|&e| vec![e]).collect();
    let mut active_block: Vec<[f64; INKLING_DIM]> = vec![[0.0; INKLING_DIM]; seq_len];
    let mut ordinary_stream = embeds.clone();

    for layer in 0..INKLING_TOTAL_LAYERS {
        let sliding = layer % 2 == 0;
        let mut layer_inputs = vec![[0.0_f64; INKLING_DIM]; seq_len];
        for t in 0..seq_len {
            layer_inputs[t] = match cfg.residual {
                ResidualMode::OrdinaryCumulative => ordinary_stream[t],
                ResidualMode::BlockAttentionResidual => {
                    let mut cands = completed_blocks[t].clone();
                    if faults.omit_completed_block_candidate && cands.len() > 1 {
                        cands.pop();
                    }
                    if layer % INKLING_BLOCK_LAYERS != 0 {
                        cands.push(active_block[t]);
                    }
                    depth_attention_mix(layer, &cands)?
                }
            };
        }

        // Precompute K/V across sequence (with causal depthwise conv mixing prev token)
        let mut k_seq = vec![[[0.0_f64; INKLING_HEAD_DIM]; INKLING_HEADS]; seq_len];
        let mut v_seq = vec![[[0.0_f64; INKLING_HEAD_DIM]; INKLING_HEADS]; seq_len];
        let mut q_seq = vec![[[0.0_f64; INKLING_HEAD_DIM]; INKLING_HEADS]; seq_len];
        let mut prev_conv = [0.0_f64; INKLING_DIM];
        for t in 0..seq_len {
            let norm_in = rms_norm(&layer_inputs[t])?;
            let mut conv_x = [0.0_f64; INKLING_DIM];
            for d in 0..INKLING_DIM {
                conv_x[d] = 0.75 * norm_in[d] + 0.25 * prev_conv[d];
            }
            prev_conv = norm_in;
            let qv = lin_proj(&conv_x, 3400 + layer * 40);
            let kv = lin_proj(&conv_x, 3410 + layer * 40);
            let vv = lin_proj(&conv_x, 3420 + layer * 40);
            for h in 0..INKLING_HEADS {
                for hd in 0..INKLING_HEAD_DIM {
                    q_seq[t][h][hd] = qv[h * INKLING_HEAD_DIM + hd];
                    k_seq[t][h][hd] = kv[h * INKLING_HEAD_DIM + hd];
                    v_seq[t][h][hd] = vv[h * INKLING_HEAD_DIM + hd];
                }
            }
        }

        for t in 0..seq_len {
            let mut concat = [0.0_f64; INKLING_DIM];
            for h in 0..INKLING_HEADS {
                let kh: Vec<[f64; INKLING_HEAD_DIM]> = k_seq.iter().map(|row| row[h]).collect();
                let vh: Vec<[f64; INKLING_HEAD_DIM]> = v_seq.iter().map(|row| row[h]).collect();
                let ah = attention_with_sink(&AttentionSinkArgs {
                    layer,
                    head: h,
                    q_h: &q_seq[t][h],
                    k_hist: &kh,
                    v_hist: &vh,
                    token_input: &layer_inputs[t],
                    sink_mode: cfg.sink,
                    sliding,
                    t_idx: t,
                    total_len: seq_len,
                    faults,
                })?;
                concat[h * INKLING_HEAD_DIM] = ah[0];
                concat[h * INKLING_HEAD_DIM + 1] = ah[1];
            }
            let gated = match cfg.gate {
                GateMode::FixedEnabled => {
                    let g_raw = match faults.force_extreme_gate_override {
                        Some(v) => {
                            if !v.is_finite() {
                                return Err("non-finite gate logit override".into());
                            }
                            [v; INKLING_DIM]
                        }
                        None => lin_proj(&layer_inputs[t], 3430 + layer * 40),
                    };
                    let mut g_out = [0.0_f64; INKLING_DIM];
                    for d in 0..INKLING_DIM {
                        g_out[d] = sigmoid(g_raw[d]) * concat[d];
                    }
                    g_out
                }
                GateMode::AblatedIdentity => concat,
            };
            let delta = lin_proj(&gated, 3435 + layer * 40);

            match cfg.residual {
                ResidualMode::OrdinaryCumulative => {
                    for (dst, &dv) in ordinary_stream[t].iter_mut().zip(delta.iter()) {
                        *dst += dv;
                    }
                }
                ResidualMode::BlockAttentionResidual => {
                    for (dst, &dv) in active_block[t].iter_mut().zip(delta.iter()) {
                        *dst += dv;
                    }
                    if (layer + 1) % INKLING_BLOCK_LAYERS == 0 {
                        completed_blocks[t].push(active_block[t]);
                        if !faults.skip_block_reset_at_boundary {
                            active_block[t] = [0.0; INKLING_DIM];
                        }
                    }
                }
            }
        }
    }

    // Final aggregation across completed block sums for BlockAttentionResidual
    let mut outputs = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let final_h = match cfg.residual {
            ResidualMode::OrdinaryCumulative => ordinary_stream[t],
            ResidualMode::BlockAttentionResidual => {
                depth_attention_mix(INKLING_TOTAL_LAYERS, &completed_blocks[t])?
            }
        };
        embeds[t] = final_h;
        outputs.push(final_h.to_vec());
    }
    Ok(outputs)
}

/// Cached step/chunk candidate path (supports chunked prefill and token-by-token decode).
pub fn forward_chunked_cached(
    tokens: &[u32],
    chunk_sizes: &[usize],
    cfg: AblationConfig,
    cache: &mut InklingCache,
    faults: InklingFaults,
) -> Result<Vec<Vec<f64>>, String> {
    let total: usize = chunk_sizes.iter().sum();
    if total != tokens.len() || tokens.is_empty() {
        return Err("chunk sizes must partition non-empty token slice".into());
    }
    let mut outputs = Vec::with_capacity(tokens.len());
    let mut offset = 0usize;
    for &cs in chunk_sizes {
        for &tok in &tokens[offset..offset + cs] {
            let t_pos = cache.k[0].len();
            let mut e = [0.0_f64; INKLING_DIM];
            for (d, slot) in e.iter_mut().enumerate() {
                *slot = det_weight(3301 + (tok as usize) * 19 + t_pos * 3, d, 0.6);
            }
            let mut completed_blocks = vec![e];
            let mut active_block = [0.0_f64; INKLING_DIM];
            let mut ordinary_h = e;

            for layer in 0..INKLING_TOTAL_LAYERS {
                let sliding = layer % 2 == 0;
                let layer_in = match cfg.residual {
                    ResidualMode::OrdinaryCumulative => ordinary_h,
                    ResidualMode::BlockAttentionResidual => {
                        let mut cands = completed_blocks.clone();
                        if faults.omit_completed_block_candidate && cands.len() > 1 {
                            cands.pop();
                        }
                        if layer % INKLING_BLOCK_LAYERS != 0 {
                            cands.push(active_block);
                        }
                        depth_attention_mix(layer, &cands)?
                    }
                };
                let norm_in = rms_norm(&layer_in)?;
                let mut conv_x = [0.0_f64; INKLING_DIM];
                for d in 0..INKLING_DIM {
                    conv_x[d] = 0.75 * norm_in[d] + 0.25 * cache.conv_prev[layer][d];
                }
                cache.conv_prev[layer] = norm_in;
                let qv = lin_proj(&conv_x, 3400 + layer * 40);
                let kv = lin_proj(&conv_x, 3410 + layer * 40);
                let vv = lin_proj(&conv_x, 3420 + layer * 40);
                let mut q_h = [[0.0_f64; INKLING_HEAD_DIM]; INKLING_HEADS];
                let mut k_h = [[0.0_f64; INKLING_HEAD_DIM]; INKLING_HEADS];
                let mut v_h = [[0.0_f64; INKLING_HEAD_DIM]; INKLING_HEADS];
                for h in 0..INKLING_HEADS {
                    for hd in 0..INKLING_HEAD_DIM {
                        q_h[h][hd] = qv[h * INKLING_HEAD_DIM + hd];
                        k_h[h][hd] = kv[h * INKLING_HEAD_DIM + hd];
                        v_h[h][hd] = vv[h * INKLING_HEAD_DIM + hd];
                    }
                }
                cache.k[layer].push(k_h);
                cache.v[layer].push(v_h);

                let cur_len = cache.k[layer].len();
                let mut concat = [0.0_f64; INKLING_DIM];
                for h in 0..INKLING_HEADS {
                    let kh: Vec<[f64; INKLING_HEAD_DIM]> =
                        cache.k[layer].iter().map(|row| row[h]).collect();
                    let vh: Vec<[f64; INKLING_HEAD_DIM]> =
                        cache.v[layer].iter().map(|row| row[h]).collect();
                    let ah = attention_with_sink(&AttentionSinkArgs {
                        layer,
                        head: h,
                        q_h: &q_h[h],
                        k_hist: &kh,
                        v_hist: &vh,
                        token_input: &layer_in,
                        sink_mode: cfg.sink,
                        sliding,
                        t_idx: cur_len - 1,
                        total_len: cur_len,
                        faults,
                    })?;
                    concat[h * INKLING_HEAD_DIM] = ah[0];
                    concat[h * INKLING_HEAD_DIM + 1] = ah[1];
                }
                let gated = match cfg.gate {
                    GateMode::FixedEnabled => {
                        let g_raw = match faults.force_extreme_gate_override {
                            Some(v) => {
                                if !v.is_finite() {
                                    return Err("non-finite gate logit override".into());
                                }
                                [v; INKLING_DIM]
                            }
                            None => lin_proj(&layer_in, 3430 + layer * 40),
                        };
                        let mut g_out = [0.0_f64; INKLING_DIM];
                        for d in 0..INKLING_DIM {
                            g_out[d] = sigmoid(g_raw[d]) * concat[d];
                        }
                        g_out
                    }
                    GateMode::AblatedIdentity => concat,
                };
                let delta = lin_proj(&gated, 3435 + layer * 40);
                match cfg.residual {
                    ResidualMode::OrdinaryCumulative => {
                        for d in 0..INKLING_DIM {
                            ordinary_h[d] += delta[d];
                        }
                    }
                    ResidualMode::BlockAttentionResidual => {
                        for d in 0..INKLING_DIM {
                            active_block[d] += delta[d];
                        }
                        if (layer + 1) % INKLING_BLOCK_LAYERS == 0 {
                            completed_blocks.push(active_block);
                            if !faults.skip_block_reset_at_boundary {
                                active_block = [0.0; INKLING_DIM];
                            }
                        }
                    }
                }
            }

            let final_h = match cfg.residual {
                ResidualMode::OrdinaryCumulative => ordinary_h,
                ResidualMode::BlockAttentionResidual => {
                    depth_attention_mix(INKLING_TOTAL_LAYERS, &completed_blocks)?
                }
            };
            outputs.push(final_h.to_vec());
        }
        offset += cs;
    }
    Ok(outputs)
}

#[derive(Debug, Clone)]
pub struct BehavioralClaimCheck {
    pub matched_small_models_trained: bool,
    pub equal_data_token_seed_budgets: bool,
    pub pretrained_component_disabling_only: bool,
}

pub fn validate_behavioral_claim(claim: &BehavioralClaimCheck) -> OutcomeStatus {
    if claim.pretrained_component_disabling_only
        || !claim.matched_small_models_trained
        || !claim.equal_data_token_seed_budgets
    {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    }
}

#[derive(Debug, Clone)]
pub struct AblationArmResult {
    pub arm_id: String,
    pub full_vs_chunked_err: f64,
    pub full_vs_decode_err: f64,
    pub output_l2_norm: f64,
    pub outcome: OutcomeStatus,
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct InklingResidualsReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub synthetic_contract_fixture: bool,
    pub equations_digest: DigestRef,
    pub ablation_arms: Vec<AblationArmResult>,
    pub extreme_sink_neg50_finite: bool,
    pub extreme_sink_pos50_finite: bool,
    pub extreme_gate_neg50_finite: bool,
    pub extreme_gate_pos50_finite: bool,
    pub training_superiority_established: bool,
    pub full_594b_claim_supported: bool,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<InklingResidualsReport, String> {
    let tokens = [2u32, 5, 1, 7, 3, 4];
    let arms = [
        AblationConfig {
            arm_id: "ordinary_residual_no_sink",
            residual: ResidualMode::OrdinaryCumulative,
            sink: SinkMode::None,
            gate: GateMode::FixedEnabled,
        },
        AblationConfig {
            arm_id: "ordinary_residual_static_sink",
            residual: ResidualMode::OrdinaryCumulative,
            sink: SinkMode::StaticSink,
            gate: GateMode::FixedEnabled,
        },
        AblationConfig {
            arm_id: "block_residual_only",
            residual: ResidualMode::BlockAttentionResidual,
            sink: SinkMode::None,
            gate: GateMode::FixedEnabled,
        },
        AblationConfig {
            arm_id: "dynamic_sink_only",
            residual: ResidualMode::OrdinaryCumulative,
            sink: SinkMode::DynamicInputSink,
            gate: GateMode::FixedEnabled,
        },
        AblationConfig {
            arm_id: "combined_block_res_dynamic_sink_gate_fixed",
            residual: ResidualMode::BlockAttentionResidual,
            sink: SinkMode::DynamicInputSink,
            gate: GateMode::FixedEnabled,
        },
        AblationConfig {
            arm_id: "combined_block_res_dynamic_sink_gate_ablated",
            residual: ResidualMode::BlockAttentionResidual,
            sink: SinkMode::DynamicInputSink,
            gate: GateMode::AblatedIdentity,
        },
    ];

    let mut ablation_arms = Vec::new();
    for cfg in arms {
        let full = forward_full_reference(&tokens, cfg, InklingFaults::default())?;
        let mut c_chunk = InklingCache::fresh();
        let chunked = forward_chunked_cached(
            &tokens,
            &[2, 3, 1],
            cfg,
            &mut c_chunk,
            InklingFaults::default(),
        )?;
        let mut c_dec = InklingCache::fresh();
        let decoded = forward_chunked_cached(
            &tokens,
            &[1, 1, 1, 1, 1, 1],
            cfg,
            &mut c_dec,
            InklingFaults::default(),
        )?;
        let e_chunk = checked_max_diff(&full, &chunked)?;
        let e_dec = checked_max_diff(&full, &decoded)?;
        let norm = full
            .iter()
            .flat_map(|r| r.iter())
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        ablation_arms.push(AblationArmResult {
            arm_id: cfg.arm_id.into(),
            full_vs_chunked_err: e_chunk,
            full_vs_decode_err: e_dec,
            output_l2_norm: norm,
            outcome: if e_chunk <= 1.0e-12 && e_dec <= 1.0e-12 && norm.is_finite() {
                OutcomeStatus::Accepted
            } else {
                OutcomeStatus::Invalid
            },
        });
    }

    let combined = arms[4];
    let full_combined = forward_full_reference(&tokens, combined, InklingFaults::default())?;
    let neg50 = forward_full_reference(
        &tokens,
        combined,
        InklingFaults {
            force_extreme_sink_override: Some(-50.0),
            ..InklingFaults::default()
        },
    )?;
    let pos50 = forward_full_reference(
        &tokens,
        combined,
        InklingFaults {
            force_extreme_sink_override: Some(50.0),
            ..InklingFaults::default()
        },
    )?;
    let extreme_sink_neg50_finite = neg50.iter().flatten().all(|x| x.is_finite());
    let extreme_sink_pos50_finite = pos50.iter().flatten().all(|x| x.is_finite());
    let gate_neg50 = forward_full_reference(
        &tokens,
        combined,
        InklingFaults {
            force_extreme_gate_override: Some(-50.0),
            ..InklingFaults::default()
        },
    )?;
    let gate_pos50 = forward_full_reference(
        &tokens,
        combined,
        InklingFaults {
            force_extreme_gate_override: Some(50.0),
            ..InklingFaults::default()
        },
    )?;
    let extreme_gate_neg50_finite = gate_neg50.iter().flatten().all(|x| x.is_finite());
    let extreme_gate_pos50_finite = gate_pos50.iter().flatten().all(|x| x.is_finite());

    let mut negative_controls = Vec::new();

    // NC1: stale cache state not cleared across reset
    {
        let mut stale = InklingCache::fresh();
        let _ = forward_chunked_cached(
            &[0, 1],
            &[2],
            combined,
            &mut stale,
            InklingFaults::default(),
        )?;
        let out_bad = forward_chunked_cached(
            &tokens,
            &[6],
            combined,
            &mut stale,
            InklingFaults::default(),
        )?;
        let err = checked_max_diff(&full_combined, &out_bad)?;
        negative_controls.push(NegativeControl {
            case_id: "nc1_stale_cache_across_reset".into(),
            outcome: if err > 1.0e-6 {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!("stale KV/conv cache caused {:.3e} divergence", err),
        });
    }

    // NC2: crossed sequence cache
    {
        let mut crossed = InklingCache::fresh();
        let _ = forward_chunked_cached(
            &[7, 6, 5],
            &[3],
            combined,
            &mut crossed,
            InklingFaults::default(),
        )?;
        let bad = forward_chunked_cached(
            &tokens[3..],
            &[3],
            combined,
            &mut crossed,
            InklingFaults::default(),
        )?;
        let err = checked_max_diff(&full_combined[3..], &bad)?;
        negative_controls.push(NegativeControl {
            case_id: "nc2_crossed_sequence_state".into(),
            outcome: if err > 1.0e-6 {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!("crossed sequence prefix caused {:.3e} divergence", err),
        });
    }

    // NC3: missing completed block candidate in depth attention
    {
        let bad = forward_full_reference(
            &tokens,
            combined,
            InklingFaults {
                omit_completed_block_candidate: true,
                ..InklingFaults::default()
            },
        )?;
        let err = checked_max_diff(&full_combined, &bad)?;
        negative_controls.push(NegativeControl {
            case_id: "nc3_missing_block_candidate".into(),
            outcome: if err > 1.0e-6 {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!(
                "omitting completed 6-layer block caused {:.3e} divergence",
                err
            ),
        });
    }

    // NC4: skipped 6-layer block reset at boundary
    {
        let bad = forward_full_reference(
            &tokens,
            combined,
            InklingFaults {
                skip_block_reset_at_boundary: true,
                ..InklingFaults::default()
            },
        )?;
        let err = checked_max_diff(&full_combined, &bad)?;
        negative_controls.push(NegativeControl {
            case_id: "nc4_skipped_block_reset".into(),
            outcome: if err > 1.0e-6 {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!("skipping 6-layer reset caused {:.3e} divergence", err),
        });
    }

    // NC5: non-finite NaN sink logit rejected
    {
        let res = forward_full_reference(
            &tokens,
            combined,
            InklingFaults {
                force_extreme_sink_override: Some(f64::NAN),
                ..InklingFaults::default()
            },
        );
        negative_controls.push(NegativeControl {
            case_id: "nc5_nonfinite_sink_logit_rejected".into(),
            outcome: if res.is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "NaN sink logit fails explicit finiteness check".into(),
        });
    }

    // NC6: causal mask leak exposes future token
    {
        let bad = forward_full_reference(
            &tokens,
            combined,
            InklingFaults {
                leak_causal_future_token: true,
                ..InklingFaults::default()
            },
        )?;
        let err = checked_max_diff(&full_combined, &bad)?;
        negative_controls.push(NegativeControl {
            case_id: "nc6_causal_mask_future_leak".into(),
            outcome: if err > 1.0e-6 {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!("future-token leak caused {:.3e} divergence", err),
        });
    }

    // NC7: disabling-only pretrained sensitivity mislabeled as training superiority
    {
        let st = validate_behavioral_claim(&BehavioralClaimCheck {
            matched_small_models_trained: false,
            equal_data_token_seed_budgets: false,
            pretrained_component_disabling_only: true,
        });
        negative_controls.push(NegativeControl {
            case_id: "nc7_disabling_only_not_training_superiority".into(),
            outcome: st,
            detail: "disabling pretrained components proves sensitivity, not training superiority"
                .into(),
        });
    }

    Ok(InklingResidualsReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::blocked_on_artifacts(
            "Inkling-2-Small 594B & matched-budget small-model training runs",
            "Miniature 12-layer (2x6 block) numerical conformance executed on CPU; full 594B weights and matched-budget small-model training runs are absent locally",
        ),
        synthetic_contract_fixture: true,
        equations_digest: DigestRef::sha256(
            b"inkling2-small-6layer-block-attnres-rmsnorm-dynamic-sink-sigmoid-gate-v1",
        ),
        ablation_arms,
        extreme_sink_neg50_finite,
        extreme_sink_pos50_finite,
        extreme_gate_neg50_finite,
        extreme_gate_pos50_finite,
        training_superiority_established: false,
        full_594b_claim_supported: false,
        negative_controls,
    })
}

pub fn inputs_json(report: &InklingResidualsReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.inkling-residuals.inputs.v1\",\n",
            "  \"issue\": 33,\n",
            "  \"execution_class\": {},\n",
            "  \"block_layers\": {},\n",
            "  \"total_layers\": {},\n",
            "  \"sliding_window\": {},\n",
            "  \"equations_digest\": {},\n",
            "  \"non_gating_related_work\": {{\"inkling_2_mini_commit\":{},\"mini_serving_issue\":57,\"goodfire_looped_probes_issue\":45}},\n",
            "  \"synthetic_contract_fixture\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        INKLING_BLOCK_LAYERS,
        INKLING_TOTAL_LAYERS,
        INKLING_SLIDING_WINDOW,
        report.equations_digest.to_json(),
        quote(RELATED_MINI_COMMIT),
        report.synthetic_contract_fixture,
    )
}

pub fn results_json(report: &InklingResidualsReport) -> String {
    let arms: Vec<String> = report
        .ablation_arms
        .iter()
        .map(|a| {
            format!(
                "{{\"arm_id\":{},\"full_vs_chunked_err\":{},\"full_vs_decode_err\":{},\"output_l2_norm\":{},\"outcome\":{}}}",
                quote(&a.arm_id),
                finite_f64(a.full_vs_chunked_err),
                finite_f64(a.full_vs_decode_err),
                finite_f64(a.output_l2_norm),
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
            "  \"schema\": \"inference-lab.inkling-residuals.results.v1\",\n",
            "  \"issue\": 33,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"extreme_sink_neg50_finite\": {},\n",
            "  \"extreme_sink_pos50_finite\": {},\n",
            "  \"extreme_gate_neg50_finite\": {},\n",
            "  \"extreme_gate_pos50_finite\": {},\n",
            "  \"training_superiority_established\": {},\n",
            "  \"full_594b_claim_supported\": {},\n",
            "  \"ablation_arms\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.synthetic_contract_fixture,
        report.extreme_sink_neg50_finite,
        report.extreme_sink_pos50_finite,
        report.extreme_gate_neg50_finite,
        report.extreme_gate_pos50_finite,
        report.training_superiority_established,
        report.full_594b_claim_supported,
        arms.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &InklingResidualsReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #33: Inkling Block Attention Residuals & Dynamic Sinks"
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
        "- Artifact/hardware gate: `{}` ({})",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(
        out,
        "- Extreme sink stability (`-50.0` / `+50.0`): `{}` / `{}`",
        report.extreme_sink_neg50_finite, report.extreme_sink_pos50_finite
    )
    .unwrap();
    writeln!(
        out,
        "- Behavioral bounds: `training_superiority_established = {}`, `full_594b_claim_supported = {}`",
        report.training_superiority_established, report.full_594b_claim_supported
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Matched Ablation Arms (12 layers = 2 x 6-layer blocks)"
    )
    .unwrap();
    for a in &report.ablation_arms {
        writeln!(
            out,
            "- `{}`: `chunked_err = {:.3e}`, `decode_err = {:.3e}`, `l2 = {:.6}`, `outcome = {}`",
            a.arm_id,
            a.full_vs_chunked_err,
            a.full_vs_decode_err,
            a.output_l2_norm,
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
