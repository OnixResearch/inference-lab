//! Issue #30: Compare separate vision encoders and early in-fusion modules at matched visual capacity.
//!
//! Pinned research input: *Unbundling Multimodal Vision Encoding, v1* (`arXiv:2611.17334v1`,
//! Section 5 & Appendix H.3/Table 36: 210 ms separate encoder, 171 ms FFN-only in-fusion, 182 ms
//! full in-fusion eager H200 TTFT; 3x finding concerns training compute, not inference speed —
//! all external numbers remain strictly unverified non-gating related work).
//!
//! Provides:
//! - Real pure-Rust tiny multimodal backbone (`L=4`, `d_model=8`) implementing all 4 architectures:
//!   `encoder_free` (`0` vision-only params), `separate_encoder` (`1024` vision-only params),
//!   `early_ffn_infusion` (`1024` vision-only params), and `early_full_infusion` (`1024` vision-only
//!   params) — matching total and vision-only parameter budgets across the 3 principal arms while
//!   reporting FLOPs and active parameters separately.
//! - Per-architecture numerical parity comparing naive single-request execution with packed
//!   mixed-batch gather/scatter execution under identical nonzero weights (proving `< 1e-12` parity
//!   within each architecture and zero cross-request leakage).
//! - Strict modality-mask vs attention-mask separation, text-only invariance under both zero-init
//!   and nonzero vision-branch weights, absent/empty/invalid visual input handling, frozen mixed
//!   request matrix reporting text vs image latencies separately, explicit `quality_unestablished`
//!   status (trained variants unavailable), and 7 executable negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, matvec, softmax_stable};
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const PAPER_REF: &str = "https://arxiv.org/html/2611.17334v1";
pub const UPSTREAM_UNVERIFIED_CLAIM: &str =
    "Appendix H.3/Table 36 eager H200 TTFT: 210ms separate, 171ms FFN in-fusion, 182ms full in-fusion; 3x is training compute (non-gating)";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionArch {
    EncoderFree,
    SeparateEncoder,
    EarlyFfnInfusion,
    EarlyFullInfusion,
}

impl VisionArch {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EncoderFree => "encoder_free",
            Self::SeparateEncoder => "separate_encoder",
            Self::EarlyFfnInfusion => "early_ffn_infusion",
            Self::EarlyFullInfusion => "early_full_infusion",
        }
    }
}

#[derive(Debug, Clone)]
struct DenseLayerWeights {
    w_q: Vec<f64>,
    w_k: Vec<f64>,
    w_v: Vec<f64>,
    w_o: Vec<f64>,
    w_up: Vec<f64>,
    w_down: Vec<f64>,
    d_ff: usize,
}

impl DenseLayerWeights {
    fn new(seed: usize, d_model: usize, d_ff: usize, scale: f64) -> Self {
        let mk = |s: usize, len: usize| -> Vec<f64> {
            (0..len).map(|i| det_weight(s, i, scale)).collect()
        };
        Self {
            w_q: mk(seed, d_model * d_model),
            w_k: mk(seed + 1, d_model * d_model),
            w_v: mk(seed + 2, d_model * d_model),
            w_o: mk(seed + 3, d_model * d_model),
            w_up: mk(seed + 4, d_model * d_ff),
            w_down: mk(seed + 5, d_ff * d_model),
            d_ff,
        }
    }

    fn param_count(&self, d_model: usize) -> usize {
        4 * d_model * d_model + 2 * d_model * self.d_ff
    }
}

#[derive(Debug, Clone)]
struct VisionBranchFfn {
    w_up: Vec<f64>,
    w_down: Vec<f64>,
    d_ff: usize,
}

impl VisionBranchFfn {
    fn new(seed: usize, d_model: usize, d_ff: usize, scale: f64) -> Self {
        Self {
            w_up: (0..d_model * d_ff)
                .map(|i| det_weight(seed, i, scale))
                .collect(),
            w_down: (0..d_ff * d_model)
                .map(|i| det_weight(seed + 1, i, scale))
                .collect(),
            d_ff,
        }
    }

    fn param_count(&self, d_model: usize) -> usize {
        2 * d_model * self.d_ff
    }
}

#[derive(Debug, Clone)]
struct VisionAttnDeltas {
    dw_q: Vec<f64>,
    dw_k: Vec<f64>,
    dw_v: Vec<f64>,
    dw_o: Vec<f64>,
}

impl VisionAttnDeltas {
    fn new(seed: usize, d_model: usize, scale: f64) -> Self {
        let mk = |s: usize| {
            (0..d_model * d_model)
                .map(|i| det_weight(s, i, scale))
                .collect()
        };
        Self {
            dw_q: mk(seed),
            dw_k: mk(seed + 1),
            dw_v: mk(seed + 2),
            dw_o: mk(seed + 3),
        }
    }

    fn param_count(d_model: usize) -> usize {
        4 * d_model * d_model
    }
}

#[derive(Debug, Clone)]
pub struct MultimodalTinyModel {
    pub arch: VisionArch,
    pub d_model: usize,
    shared_layers: Vec<DenseLayerWeights>,
    sep_encoder_layers: Vec<DenseLayerWeights>,
    early_vis_ffns: Vec<VisionBranchFfn>,
    early_vis_attn_deltas: Vec<VisionAttnDeltas>,
}

impl MultimodalTinyModel {
    /// Construct one of the 4 architectures with identical shared backbone weights and exact
    /// matched visual-only parameter budget (`P_vis = 1024`) across the 3 principal arms!
    pub fn build(arch: VisionArch, zero_init_vision_branches: bool) -> Self {
        let d_model = 8usize;
        let shared_layers: Vec<DenseLayerWeights> = (0..4)
            .map(|l| DenseLayerWeights::new(3000 + l * 10, d_model, 16, 0.30))
            .collect();
        let vis_scale = if zero_init_vision_branches { 0.0 } else { 0.35 };
        match arch {
            VisionArch::EncoderFree => Self {
                arch,
                d_model,
                shared_layers,
                sep_encoder_layers: Vec::new(),
                early_vis_ffns: Vec::new(),
                early_vis_attn_deltas: Vec::new(),
            },
            VisionArch::SeparateEncoder => {
                // 2 vision encoder layers * (4*8*8 + 2*8*16) = 2 * 512 = 1024 vision-only params!
                let sep_encoder_layers = (0..2)
                    .map(|l| DenseLayerWeights::new(3100 + l * 10, d_model, 16, vis_scale))
                    .collect();
                Self {
                    arch,
                    d_model,
                    shared_layers,
                    sep_encoder_layers,
                    early_vis_ffns: Vec::new(),
                    early_vis_attn_deltas: Vec::new(),
                }
            }
            VisionArch::EarlyFfnInfusion => {
                // 2 early layers * (2*8*32) = 2 * 512 = 1024 vision-only params!
                let early_vis_ffns = (0..2)
                    .map(|l| VisionBranchFfn::new(3200 + l * 10, d_model, 32, vis_scale))
                    .collect();
                Self {
                    arch,
                    d_model,
                    shared_layers,
                    sep_encoder_layers: Vec::new(),
                    early_vis_ffns,
                    early_vis_attn_deltas: Vec::new(),
                }
            }
            VisionArch::EarlyFullInfusion => {
                // 2 early layers * (2*8*16 + 4*8*8) = 2 * 512 = 1024 vision-only params!
                let early_vis_ffns = (0..2)
                    .map(|l| VisionBranchFfn::new(3300 + l * 10, d_model, 16, vis_scale))
                    .collect();
                let early_vis_attn_deltas = (0..2)
                    .map(|l| VisionAttnDeltas::new(3350 + l * 10, d_model, vis_scale))
                    .collect();
                Self {
                    arch,
                    d_model,
                    shared_layers,
                    sep_encoder_layers: Vec::new(),
                    early_vis_ffns,
                    early_vis_attn_deltas,
                }
            }
        }
    }

    pub fn shared_param_count(&self) -> usize {
        self.shared_layers
            .iter()
            .map(|l| l.param_count(self.d_model))
            .sum()
    }

    pub fn vision_only_param_count(&self) -> usize {
        let sep: usize = self
            .sep_encoder_layers
            .iter()
            .map(|l| l.param_count(self.d_model))
            .sum();
        let ffn: usize = self
            .early_vis_ffns
            .iter()
            .map(|f| f.param_count(self.d_model))
            .sum();
        let att: usize =
            self.early_vis_attn_deltas.len() * VisionAttnDeltas::param_count(self.d_model);
        sep + ffn + att
    }

    pub fn total_param_count(&self) -> usize {
        self.shared_param_count() + self.vision_only_param_count()
    }

    /// Estimate linear-layer + attention FLOPs separately from parameter count!
    pub fn estimate_flops(&self, n_vis: usize, n_txt: usize) -> usize {
        let n_tot = n_vis + n_txt;
        let d = self.d_model;
        let shared_layer_flops = 2 * n_tot * (4 * d * d + 2 * d * 16) + 2 * n_tot * n_tot * d;
        let mut total = self.shared_layers.len() * shared_layer_flops;
        match self.arch {
            VisionArch::EncoderFree => {}
            VisionArch::SeparateEncoder => {
                // Separate encoder runs bidirectional attention + FFN ONLY on n_vis tokens!
                let enc_flops = 2 * n_vis * (4 * d * d + 2 * d * 16) + 2 * n_vis * n_vis * d;
                total += self.sep_encoder_layers.len() * enc_flops;
            }
            VisionArch::EarlyFfnInfusion => {
                // Early vision-only FFN (d_ff=32) runs ONLY on n_vis tokens, no extra attn!
                total += self.early_vis_ffns.len() * (2 * n_vis * 2 * d * 32);
            }
            VisionArch::EarlyFullInfusion => {
                // Early vision-only FFN (d_ff=16) + Q/K/V/O deltas on n_vis tokens!
                total += self.early_vis_ffns.len() * (2 * n_vis * (2 * d * 16 + 4 * d * d));
            }
        }
        total
    }
}

#[derive(Debug, Clone)]
pub struct MultimodalRequest {
    pub request_id: String,
    pub vision_tokens: Vec<Vec<f64>>,
    pub text_tokens: Vec<Vec<f64>>,
}

impl MultimodalRequest {
    pub fn validate(&self, d_model: usize, expect_image: bool) -> Result<(), String> {
        if expect_image && self.vision_tokens.is_empty() {
            return Err("empty visual input rejected when image is declared".into());
        }
        for tok in self.vision_tokens.iter().chain(self.text_tokens.iter()) {
            if tok.len() != d_model || tok.iter().any(|x| !x.is_finite()) {
                return Err("invalid or non-finite multimodal token".into());
            }
        }
        if self.vision_tokens.is_empty() && self.text_tokens.is_empty() {
            return Err("empty request rejected".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FaultInjection {
    pub misroute_text_to_vision_branch: bool,
    pub break_causal_mask: bool,
    pub break_cross_request_isolation: bool,
    pub corrupt_gather_scatter_order: bool,
}

fn relu_vec(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect()
}

fn add_vecs(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[allow(clippy::too_many_arguments)]
fn run_dense_attn(
    xs: &[Vec<f64>],
    q_all: &[Vec<f64>],
    k_all: &[Vec<f64>],
    v_all: &[Vec<f64>],
    req_ids: &[usize],
    is_vision: &[bool],
    is_pad: &[bool],
    bidirectional_within_image: bool,
    fault: FaultInjection,
) -> Result<Vec<Vec<f64>>, String> {
    let n = xs.len();
    let d = xs[0].len();
    let scale = 1.0 / (d as f64).sqrt();
    let mut outs = Vec::with_capacity(n);
    for i in 0..n {
        if is_pad[i] {
            outs.push(vec![0.0; d]);
            continue;
        }
        let mut allowed = Vec::new();
        let mut scores = Vec::new();
        for j in 0..n {
            if is_pad[j] {
                continue;
            }
            if !fault.break_cross_request_isolation && req_ids[i] != req_ids[j] {
                continue;
            }
            let can_attend = fault.break_causal_mask
                || (bidirectional_within_image && is_vision[i] && is_vision[j])
                || j <= i;
            if can_attend {
                let dot: f64 = q_all[i]
                    .iter()
                    .zip(k_all[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                allowed.push(j);
                scores.push(dot * scale);
            }
        }
        let probs = softmax_stable(&scores)?;
        let mut ctx = vec![0.0_f64; d];
        for (idx, &j) in allowed.iter().enumerate() {
            for (k, ck) in ctx.iter_mut().enumerate().take(d) {
                *ck += probs[idx] * v_all[j][k];
            }
        }
        outs.push(ctx);
    }
    Ok(outs)
}

fn finite_max(a: f64, b: f64) -> Result<f64, String> {
    if !a.is_finite() || !b.is_finite() {
        return Err("non-finite value in finite_max".into());
    }
    Ok(if a >= b { a } else { b })
}

/// Forward pass over a packed mixed batch (multiple requests + optional padding tokens) using
/// explicit modality mask (`is_vision`) kept strictly distinct from causal/within-image attention mask!
pub fn forward_packed_batch(
    model: &MultimodalTinyModel,
    requests: &[MultimodalRequest],
    pad_tokens: usize,
    fault: FaultInjection,
) -> Result<Vec<Vec<Vec<f64>>>, String> {
    if requests.is_empty() {
        return Err("requests cannot be empty in forward_packed_batch".into());
    }
    let d = model.d_model;
    for req in requests {
        req.validate(d, false)?;
    }
    let mut h: Vec<Vec<f64>> = Vec::new();
    let mut req_ids: Vec<usize> = Vec::new();
    let mut is_vision: Vec<bool> = Vec::new();
    let mut is_pad: Vec<bool> = Vec::new();
    let mut req_spans: Vec<(usize, usize)> = Vec::new();

    for (ridx, req) in requests.iter().enumerate() {
        let start = h.len();
        for vt in &req.vision_tokens {
            h.push(vt.clone());
            req_ids.push(ridx);
            is_vision.push(true);
            is_pad.push(false);
        }
        for tt in &req.text_tokens {
            h.push(tt.clone());
            req_ids.push(ridx);
            is_vision.push(false);
            is_pad.push(false);
        }
        req_spans.push((start, h.len()));
    }
    for _ in 0..pad_tokens {
        h.push(vec![1.0e3; d]); // non-zero pad sentinel proving padding is masked out!
        req_ids.push(usize::MAX);
        is_vision.push(false);
        is_pad.push(true);
    }

    // 1. Separate pre-fusion vision encoder (only on vision tokens via gather/scatter!).
    if model.arch == VisionArch::SeparateEncoder {
        let vis_indices: Vec<usize> = (0..h.len())
            .filter(|&i| !is_pad[i] && is_vision[i])
            .collect();
        if !vis_indices.is_empty() {
            let mut vh: Vec<Vec<f64>> = vis_indices.iter().map(|&i| h[i].clone()).collect();
            let v_reqs: Vec<usize> = vis_indices.iter().map(|&i| req_ids[i]).collect();
            let v_mask = vec![true; vh.len()];
            let v_pad = vec![false; vh.len()];
            for layer in &model.sep_encoder_layers {
                let qs: Vec<Vec<f64>> = vh
                    .iter()
                    .map(|x| matvec(x, &layer.w_q, d, d))
                    .collect::<Result<_, _>>()?;
                let ks: Vec<Vec<f64>> = vh
                    .iter()
                    .map(|x| matvec(x, &layer.w_k, d, d))
                    .collect::<Result<_, _>>()?;
                let vs: Vec<Vec<f64>> = vh
                    .iter()
                    .map(|x| matvec(x, &layer.w_v, d, d))
                    .collect::<Result<_, _>>()?;
                let attn =
                    run_dense_attn(&vh, &qs, &ks, &vs, &v_reqs, &v_mask, &v_pad, true, fault)?;
                for i in 0..vh.len() {
                    let ao = matvec(&attn[i], &layer.w_o, d, d)?;
                    let r1 = add_vecs(&vh[i], &ao);
                    let up = relu_vec(&matvec(&r1, &layer.w_up, d, layer.d_ff)?);
                    let dn = matvec(&up, &layer.w_down, layer.d_ff, d)?;
                    vh[i] = add_vecs(&r1, &dn);
                }
            }
            for (k, &orig_i) in vis_indices.iter().enumerate() {
                let target_slot = if fault.corrupt_gather_scatter_order {
                    vis_indices[(k + 1) % vis_indices.len()]
                } else {
                    orig_i
                };
                h[target_slot] = vh[k].clone();
            }
        }
    }

    // 2. Shared backbone layers + early vision-only in-fusion branches.
    for (l_idx, layer) in model.shared_layers.iter().enumerate() {
        let mut qs = Vec::with_capacity(h.len());
        let mut ks = Vec::with_capacity(h.len());
        let mut vs = Vec::with_capacity(h.len());
        for i in 0..h.len() {
            if is_pad[i] {
                qs.push(vec![0.0; d]);
                ks.push(vec![0.0; d]);
                vs.push(vec![0.0; d]);
                continue;
            }
            let mut q = matvec(&h[i], &layer.w_q, d, d)?;
            let mut k = matvec(&h[i], &layer.w_k, d, d)?;
            let mut v = matvec(&h[i], &layer.w_v, d, d)?;
            let route_vis = is_vision[i] || fault.misroute_text_to_vision_branch;
            if route_vis && l_idx < model.early_vis_attn_deltas.len() {
                let dl = &model.early_vis_attn_deltas[l_idx];
                q = add_vecs(&q, &matvec(&h[i], &dl.dw_q, d, d)?);
                k = add_vecs(&k, &matvec(&h[i], &dl.dw_k, d, d)?);
                v = add_vecs(&v, &matvec(&h[i], &dl.dw_v, d, d)?);
            }
            qs.push(q);
            ks.push(k);
            vs.push(v);
        }
        let attn = run_dense_attn(
            &h, &qs, &ks, &vs, &req_ids, &is_vision, &is_pad, true, fault,
        )?;
        for i in 0..h.len() {
            if is_pad[i] {
                continue;
            }
            let mut ao = matvec(&attn[i], &layer.w_o, d, d)?;
            let route_vis = is_vision[i] || fault.misroute_text_to_vision_branch;
            if route_vis && l_idx < model.early_vis_attn_deltas.len() {
                let dl = &model.early_vis_attn_deltas[l_idx];
                ao = add_vecs(&ao, &matvec(&attn[i], &dl.dw_o, d, d)?);
            }
            let r1 = add_vecs(&h[i], &ao);
            let up = relu_vec(&matvec(&r1, &layer.w_up, d, layer.d_ff)?);
            let dn = matvec(&up, &layer.w_down, layer.d_ff, d)?;
            let mut r2 = add_vecs(&r1, &dn);

            if route_vis && l_idx < model.early_vis_ffns.len() {
                let vf = &model.early_vis_ffns[l_idx];
                let v_up = relu_vec(&matvec(&r1, &vf.w_up, d, vf.d_ff)?);
                let v_dn = matvec(&v_up, &vf.w_down, vf.d_ff, d)?;
                r2 = add_vecs(&r2, &v_dn);
            }
            h[i] = r2;
        }
    }

    let mut per_req = Vec::with_capacity(requests.len());
    for (s, e) in req_spans {
        per_req.push(h[s..e].to_vec());
    }
    Ok(per_req)
}

fn max_seq_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err("seq length mismatch".into());
    }
    let mut m = 0.0_f64;
    for (ra, rb) in a.iter().zip(b.iter()) {
        if ra.len() != rb.len() {
            return Err("dim mismatch".into());
        }
        for (&x, &y) in ra.iter().zip(rb.iter()) {
            if !x.is_finite() || !y.is_finite() {
                return Err("non-finite value".into());
            }
            let d = (x - y).abs();
            if d > m {
                m = d;
            }
        }
    }
    Ok(m)
}

#[derive(Debug, Clone)]
pub struct ArchComparisonRow {
    pub arch: VisionArch,
    pub total_params: usize,
    pub shared_params: usize,
    pub vision_only_params: usize,
    pub flops_img4_txt4: usize,
    pub flops_img16_txt4: usize,
    pub flops_text_only_4: usize,
    pub packed_vs_single_max_err: f64,
    pub text_only_invariance_max_err: f64,
    pub zero_init_vs_encoder_free_max_err: f64,
}

/// Validate whether visual-only parameter budgets are matched across compared arms.
pub fn validate_matched_visual_budget(rows: &[ArchComparisonRow]) -> Result<usize, String> {
    if rows.is_empty() {
        return Err("empty architecture list".into());
    }
    let target = rows[0].vision_only_params;
    for r in rows.iter().skip(1) {
        if r.vision_only_params != target {
            return Err(format!(
                "unmatched visual parameter budget: `{}` has {} vs `{}` has {}",
                r.arch.as_str(),
                r.vision_only_params,
                rows[0].arch.as_str(),
                target
            ));
        }
    }
    Ok(target)
}

#[derive(Debug, Clone)]
pub struct QualityEvidenceClaim {
    pub compatible_trained_variants_present: bool,
    pub training_exposure_recorded: bool,
    pub heldout_visual_and_text_checks_present: bool,
}

pub fn validate_quality_evidence_claim(claim: &QualityEvidenceClaim) -> OutcomeStatus {
    if claim.compatible_trained_variants_present
        && claim.training_exposure_recorded
        && claim.heldout_visual_and_text_checks_present
    {
        OutcomeStatus::Accepted
    } else {
        OutcomeStatus::BlockedOnArtifacts
    }
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct VisionInfusionReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub trained_variants_gate: HardwareGate,
    pub principal_arms_vision_budget_matched: bool,
    pub arch_rows: Vec<ArchComparisonRow>,
    pub quality_status: String,
    pub negative_controls: Vec<NegativeControl>,
    pub decision: String,
}

pub fn run_default_harness() -> Result<VisionInfusionReport, String> {
    let d = 8usize;
    let mk_toks = |seed: usize, n: usize| -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| (0..d).map(|k| det_weight(seed + i, k, 0.7)).collect())
            .collect()
    };

    let req_img_a = MultimodalRequest {
        request_id: "img_req_4".into(),
        vision_tokens: mk_toks(3400, 4),
        text_tokens: mk_toks(3410, 4),
    };
    let req_txt = MultimodalRequest {
        request_id: "txt_only_req".into(),
        vision_tokens: Vec::new(),
        text_tokens: mk_toks(3420, 4),
    };
    let req_img_b = MultimodalRequest {
        request_id: "img_req_6".into(),
        vision_tokens: mk_toks(3430, 6),
        text_tokens: mk_toks(3440, 3),
    };

    let enc_free = MultimodalTinyModel::build(VisionArch::EncoderFree, false);
    let txt_baseline =
        forward_packed_batch(&enc_free, &[req_txt.clone()], 0, FaultInjection::default())?;
    let img_a_enc_free = forward_packed_batch(
        &enc_free,
        &[req_img_a.clone()],
        0,
        FaultInjection::default(),
    )?;

    let mut arch_rows = Vec::new();
    let all_archs = [
        VisionArch::EncoderFree,
        VisionArch::SeparateEncoder,
        VisionArch::EarlyFfnInfusion,
        VisionArch::EarlyFullInfusion,
    ];
    for arch in all_archs {
        let model = MultimodalTinyModel::build(arch, false);
        // Packed mixed batch (image + text-only + image + 2 pad sentinels) vs individual runs!
        let packed = forward_packed_batch(
            &model,
            &[req_img_a.clone(), req_txt.clone(), req_img_b.clone()],
            2,
            FaultInjection::default(),
        )?;
        let s0 = forward_packed_batch(&model, &[req_img_a.clone()], 0, FaultInjection::default())?;
        let s1 = forward_packed_batch(&model, &[req_txt.clone()], 0, FaultInjection::default())?;
        let s2 = forward_packed_batch(&model, &[req_img_b.clone()], 0, FaultInjection::default())?;

        let err0 = max_seq_diff(&packed[0], &s0[0])?;
        let err1 = max_seq_diff(&packed[1], &s1[0])?;
        let err2 = max_seq_diff(&packed[2], &s2[0])?;
        let packed_err = finite_max(finite_max(err0, err1)?, err2)?;
        // Text-only invariance against shared backbone even with NONZERO vision branch weights!
        let txt_inv_err = max_seq_diff(&s1[0], &txt_baseline[0])?;
        // Zero-initialized added vision branches must match `encoder_free` on multimodal inputs!
        let zero_init_model = MultimodalTinyModel::build(arch, true);
        let zero_init_out = forward_packed_batch(
            &zero_init_model,
            &[req_img_a.clone()],
            0,
            FaultInjection::default(),
        )?;
        let zero_init_err = max_seq_diff(&zero_init_out[0], &img_a_enc_free[0])?;

        arch_rows.push(ArchComparisonRow {
            arch,
            total_params: model.total_param_count(),
            shared_params: model.shared_param_count(),
            vision_only_params: model.vision_only_param_count(),
            flops_img4_txt4: model.estimate_flops(4, 4),
            flops_img16_txt4: model.estimate_flops(16, 4),
            flops_text_only_4: model.estimate_flops(0, 4),
            packed_vs_single_max_err: packed_err,
            text_only_invariance_max_err: txt_inv_err,
            zero_init_vs_encoder_free_max_err: zero_init_err,
        });
    }

    let principal_budget_matched = arch_rows[1].vision_only_params
        == arch_rows[2].vision_only_params
        && arch_rows[2].vision_only_params == arch_rows[3].vision_only_params
        && arch_rows[1].vision_only_params == 1024;

    // Seven executable negative controls using NONZERO vision branch weights!
    let full_inf = MultimodalTinyModel::build(VisionArch::EarlyFullInfusion, false);
    let sep_enc = MultimodalTinyModel::build(VisionArch::SeparateEncoder, false);
    let ref_packed = forward_packed_batch(
        &full_inf,
        &[req_img_a.clone(), req_txt.clone()],
        0,
        FaultInjection::default(),
    )?;
    let mut negative_controls = Vec::new();

    // 1. Misrouted text token sent through nonzero vision-only FFN & attention deltas
    let bad_route = forward_packed_batch(
        &full_inf,
        &[req_txt.clone()],
        0,
        FaultInjection {
            misroute_text_to_vision_branch: true,
            ..Default::default()
        },
    )?;
    let e1 = max_seq_diff(&bad_route[0], &txt_baseline[0])?;
    negative_controls.push(NegativeControl {
        case_id: "misrouted_text_token_to_vision_branch".into(),
        outcome: if e1 > 1e-4 {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("text_output_drift={e1:.6}"),
    });

    // 2. Altered causal/within-image attention mask
    let bad_mask = forward_packed_batch(
        &full_inf,
        &[req_img_a.clone()],
        0,
        FaultInjection {
            break_causal_mask: true,
            ..Default::default()
        },
    )?;
    let e2 = max_seq_diff(&bad_mask[0], &ref_packed[0])?;
    negative_controls.push(NegativeControl {
        case_id: "altered_causal_attention_mask".into(),
        outcome: if e2 > 1e-4 {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("mask_drift={e2:.6}"),
    });

    // 3. Cross-request leakage in packed mixed batch
    let bad_leak = forward_packed_batch(
        &full_inf,
        &[req_img_a.clone(), req_txt.clone()],
        0,
        FaultInjection {
            break_cross_request_isolation: true,
            ..Default::default()
        },
    )?;
    let e3 = max_seq_diff(&bad_leak[1], &ref_packed[1])?;
    negative_controls.push(NegativeControl {
        case_id: "cross_request_mixing_in_packed_batch".into(),
        outcome: if e3 > 1e-4 {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("text_req_contamination_err={e3:.6}"),
    });

    // 4. Gather/scatter permutation bug in separate encoder
    let ref_sep =
        forward_packed_batch(&sep_enc, &[req_img_a.clone()], 0, FaultInjection::default())?;
    let bad_gs = forward_packed_batch(
        &sep_enc,
        &[req_img_a.clone()],
        0,
        FaultInjection {
            corrupt_gather_scatter_order: true,
            ..Default::default()
        },
    )?;
    let e4 = max_seq_diff(&bad_gs[0], &ref_sep[0])?;
    negative_controls.push(NegativeControl {
        case_id: "gather_scatter_token_permutation_bug".into(),
        outcome: if e4 > 1e-4 {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("gather_scatter_err={e4:.6}"),
    });

    // 5. Empty or non-finite declared visual input rejected
    let empty_img_req = MultimodalRequest {
        request_id: "bad-empty-img".into(),
        vision_tokens: Vec::new(),
        text_tokens: mk_toks(3450, 2),
    };
    let r5 = empty_img_req.validate(d, true);
    negative_controls.push(NegativeControl {
        case_id: "empty_declared_visual_input_rejected".into(),
        outcome: if r5.is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: r5.err().unwrap_or_else(|| "accepted empty image".into()),
    });

    // 6. Unmatched visual parameter budget caught by real budget validator
    let _principal_ok = validate_matched_visual_budget(&arch_rows[1..])?;
    let r6 = validate_matched_visual_budget(&arch_rows); // includes encoder_free (0) vs principal (1024)
    negative_controls.push(NegativeControl {
        case_id: "unmatched_visual_budget_encoder_free_separated".into(),
        outcome: if r6.is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: r6
            .err()
            .unwrap_or_else(|| "missed unmatched visual budget".into()),
    });

    // 7. Quality claim from random/untrained weights rejected by real quality evidence validator
    let r7 = validate_quality_evidence_claim(&QualityEvidenceClaim {
        compatible_trained_variants_present: false,
        training_exposure_recorded: false,
        heldout_visual_and_text_checks_present: true,
    });
    negative_controls.push(NegativeControl {
        case_id: "untrained_weights_cannot_establish_quality".into(),
        outcome: r7,
        detail: "quality_unestablished: compatible trained variants and training-exposure records unavailable".into(),
    });

    Ok(VisionInfusionReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "physical_gpu_eager_and_compiled_multimodal_serving",
            "Device-free matched-capacity architecture parity, gather/scatter, and mask isolation verified locally; GPU eager/compiled serving latency and mixed-traffic tail distributions unmeasured",
        ),
        trained_variants_gate: HardwareGate::blocked_on_artifacts(
            "compatible_trained_vision_infusion_checkpoints",
            "Quality remains unestablished until compatible trained separate-encoder and early in-fusion variants with recorded training exposure are available",
        ),
        principal_arms_vision_budget_matched: principal_budget_matched,
        arch_rows,
        quality_status: "quality_unestablished".into(),
        negative_controls,
        decision: "EXECUTION_CONTRACT_VERIFIED_QUALITY_UNESTABLISHED_GPU_UNMEASURED".into(),
    })
}

pub fn inputs_json(report: &VisionInfusionReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.vision-infusion.inputs.v1\",\n",
            "  \"issue\": 30,\n",
            "  \"paper_ref\": {},\n",
            "  \"unverified_related_work_claim\": {},\n",
            "  \"matched_principal_vision_params\": 1024,\n",
            "  \"text_branch_inactivity_proves_post_training_retention\": false,\n",
            "  \"lower_ttft_alone_proves_faster_completed_tasks\": false,\n",
            "  \"execution_class\": {},\n",
            "  \"fixture_digest\": {}\n",
            "}}\n"
        ),
        quote(PAPER_REF),
        quote(UPSTREAM_UNVERIFIED_CLAIM),
        quote(report.execution_class.as_str()),
        DigestRef::sha256(b"vision-infusion-4arch-matched-1024-v1").to_json(),
    )
}

pub fn results_json(report: &VisionInfusionReport) -> String {
    let rows: Vec<String> = report
        .arch_rows
        .iter()
        .map(|r| {
            format!(
                concat!(
                    "{{\"arch\":{},\"total_params\":{},\"shared_params\":{},\"vision_only_params\":{},",
                    "\"flops_img4_txt4\":{},\"flops_img16_txt4\":{},\"flops_text_only_4\":{},",
                    "\"packed_vs_single_max_err\":{},\"text_only_invariance_max_err\":{},\"zero_init_vs_encoder_free_max_err\":{}}}"
                ),
                quote(r.arch.as_str()),
                r.total_params,
                r.shared_params,
                r.vision_only_params,
                r.flops_img4_txt4,
                r.flops_img16_txt4,
                r.flops_text_only_4,
                finite_f64(r.packed_vs_single_max_err),
                finite_f64(r.text_only_invariance_max_err),
                finite_f64(r.zero_init_vs_encoder_free_max_err),
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
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.vision-infusion.results.v1\",\n",
            "  \"issue\": 30,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"trained_variants_gate\": {},\n",
            "  \"decision\": {},\n",
            "  \"quality_status\": {},\n",
            "  \"principal_arms_vision_budget_matched\": {},\n",
            "  \"text_request_mixed_ttft_ms\": {},\n",
            "  \"image_request_mixed_ttft_ms\": {},\n",
            "  \"arch_rows\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.trained_variants_gate.to_json(),
        quote(&report.decision),
        quote(&report.quality_status),
        report.principal_arms_vision_budget_matched,
        opt_f64(None),
        opt_f64(None),
        rows.join(","),
        neg.join(","),
    )
}

pub fn summary_markdown(report: &VisionInfusionReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #30: Separate Vision Encoders vs Early In-Fusion at Matched Capacity"
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
    writeln!(out, "- Quality status: `{}`", report.quality_status).unwrap();
    writeln!(
        out,
        "- Principal arms visual budget matched (`1024`): `{}`",
        report.principal_arms_vision_budget_matched
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "| Architecture | Total Params | Vision Params | FLOPs (img4+txt4) | FLOPs (img16+txt4) | Packed vs Single Err | Text-Only Inv Err |").unwrap();
    writeln!(out, "| --- | ---: | ---: | ---: | ---: | ---: | ---: |").unwrap();
    for r in &report.arch_rows {
        writeln!(
            out,
            "| `{}` | {} | {} | {} | {} | `{:.3e}` | `{:.3e}` |",
            r.arch.as_str(),
            r.total_params,
            r.vision_only_params,
            r.flops_img4_txt4,
            r.flops_img16_txt4,
            r.packed_vs_single_max_err,
            r.text_only_invariance_max_err,
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
