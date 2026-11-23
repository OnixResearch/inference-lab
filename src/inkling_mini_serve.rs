//! Issue #57: Measure a default-off local serve of pinned Inkling 2 Mini.
//!
//! Pinned upstream identities (observed 20 Nov 2026; Apache-2.0 + Model Acceptable Use Policy):
//! - `thinkingmachines/Inkling-2-Mini@6e7c8248f75265bdd4fbab5273304243b5d8c2cb`
//! - `thinkingmachines/TML-d24-varloop-research@767c9d0250aa2938e289e50ca6db5f1b39ebae74`
//! - `thinking-machines-lab/tinker-cookbook@37261cae3d3ff11f08573b2a84a4d44c809c38f3`
//!
//! Scope boundaries (explicitly kept distinct):
//! - `#33`: Inkling 2 Small block attention residuals and dynamic sinks miniature.
//! - `#26`: NVFP4/MXFP8 layer numerics qualification.
//! - `#35`: Spark serving profile for Nemotron 3.5.
//! - `#18`: Qwen3.8-27B quantization and serving.
//! - `rsi-lab#88`: Closed FP4-forward training comparison.
//! - `#59`: `maya-ik2` custom engine on 24 GB `sm_86` with Unsloth GGUF (`maya-ik2` is NOT on
//!   Thinking Machines' runtime list and is cleanly rejected for `#57`).
//!
//! Provides:
//! - Admitted host floor qualification (DGX Spark, >=32 GB Apple Silicon, or >=32 GB VRAM GPU)
//!   and authors' runtime list enforcement (`sglang`, `vllm`, `ollama`, `unsloth`,
//!   `nvidia-dynamo`, `tokenspeed`, `llama.cpp`).
//! - Miniature 24-stored-layer looped MoE transformer: 6 entry + 12 middle looped 3x + 6 exit =
//!   48 layer applications, exact 8-of-128 MoE routing, NVFP4 layers `0..20` and MXFP8 layers
//!   `20..24`.
//! - Genuinely independent `forward_unrolled_48_reference` vs `forward_looped_3x_shared`
//!   (`max_abs_err <= 1e-12`), plus optional research `varloop` sweep (`loop_count = 1, 2, 3, 4`).
//! - 8 executable negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::det_weight;
use crate::json::{finite_f64, quote, static_str_array};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const INKLING_MINI_REPO: &str = "thinkingmachines/Inkling-2-Mini";
pub const INKLING_MINI_COMMIT: &str = "6e7c8248f75265bdd4fbab5273304243b5d8c2cb";
pub const VARLOOP_RESEARCH_REPO: &str = "thinkingmachines/TML-d24-varloop-research";
pub const VARLOOP_RESEARCH_COMMIT: &str = "767c9d0250aa2938e289e50ca6db5f1b39ebae74";
pub const TINKER_COOKBOOK_REPO: &str = "thinking-machines-lab/tinker-cookbook";
pub const TINKER_COOKBOOK_COMMIT: &str = "37261cae3d3ff11f08573b2a84a4d44c809c38f3";
pub const MODEL_LICENSE: &str = "Apache-2.0 + Model Acceptable Use Policy";
pub const MODEL_AUP_URL: &str = "https://thinkingmachines.ai/model-acceptable-use-policy/";

pub const STORED_LAYERS: usize = 24;
pub const ENTRY_LAYERS: usize = 6;
pub const MIDDLE_LAYERS: usize = 12;
pub const EXIT_LAYERS: usize = 6;
pub const DEFAULT_LOOP_COUNT: usize = 3;
pub const UNROLLED_LAYER_APPLICATIONS: usize =
    ENTRY_LAYERS + MIDDLE_LAYERS * DEFAULT_LOOP_COUNT + EXIT_LAYERS;
pub const NVFP4_LAYER_END: usize = 20;
pub const NUM_EXPERTS: usize = 128;
pub const TOP_K_EXPERTS: usize = 8;
pub const MINI_DIM: usize = 4;

pub const AUTHORS_RUNTIME_LIST: [&str; 7] = [
    "sglang",
    "vllm",
    "ollama",
    "unsloth",
    "nvidia-dynamo",
    "tokenspeed",
    "llama.cpp",
];

pub const DISTINCT_RELATED_ISSUES: [&str; 6] = [
    "inference-lab#33",
    "inference-lab#26",
    "inference-lab#35",
    "inference-lab#18",
    "rsi-lab#88",
    "inference-lab#59",
];

/// Exact FP4 E2M1 positive magnitudes for deterministic layer quantization.
const FP4_E2M1_MAGS: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerQuantRegime {
    Nvfp4,
    Mxfp8,
}

impl LayerQuantRegime {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Nvfp4 => "nvfp4",
            Self::Mxfp8 => "mxfp8",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostFloorClass {
    DgxSpark,
    AppleSilicon { unified_memory_gb: u32 },
    DiscreteGpu { vram_gb: u32 },
    CpuOnly { ram_gb: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostServeRequest {
    pub host_label: String,
    pub platform: HostFloorClass,
    pub runtime_name: String,
    pub runtime_available_locally: bool,
    pub checkpoint_repo: String,
    pub checkpoint_commit: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostQualificationResult {
    pub host_floor_met: bool,
    pub runtime_in_authors_list: bool,
    pub pin_valid: bool,
    pub outcome: OutcomeStatus,
    pub reason: String,
}

/// Qualify host floor (DGX Spark / >=32GB Apple Silicon / >=32GB VRAM GPU), authors' runtime list
/// (explicitly rejecting `maya-ik2` as unlisted for #57), and exact commit pin.
pub fn qualify_inkling_host_and_runtime(req: &HostServeRequest) -> HostQualificationResult {
    let pin_valid = (req.checkpoint_repo == INKLING_MINI_REPO
        && req.checkpoint_commit == INKLING_MINI_COMMIT)
        || (req.checkpoint_repo == VARLOOP_RESEARCH_REPO
            && req.checkpoint_commit == VARLOOP_RESEARCH_COMMIT);

    let host_floor_met = match req.platform {
        HostFloorClass::DgxSpark => true,
        HostFloorClass::AppleSilicon { unified_memory_gb } => unified_memory_gb >= 32,
        HostFloorClass::DiscreteGpu { vram_gb } => vram_gb >= 32,
        HostFloorClass::CpuOnly { .. } => false,
    };

    let normalized_rt = req.runtime_name.to_ascii_lowercase();
    let runtime_in_authors_list = AUTHORS_RUNTIME_LIST.contains(&normalized_rt.as_str());

    if !pin_valid {
        return HostQualificationResult {
            host_floor_met,
            runtime_in_authors_list,
            pin_valid: false,
            outcome: OutcomeStatus::Invalid,
            reason: format!(
                "checkpoint {}@{} does not match pinned Inkling 2 Mini or varloop SHA",
                req.checkpoint_repo, req.checkpoint_commit
            ),
        };
    }

    if normalized_rt == "maya-ik2" || !runtime_in_authors_list {
        return HostQualificationResult {
            host_floor_met,
            runtime_in_authors_list: false,
            pin_valid: true,
            outcome: OutcomeStatus::Unsupported,
            reason: format!(
                "runtime {:?} is not on Thinking Machines' runtime list for #57 (maya-ik2 belongs strictly to #59)",
                req.runtime_name
            ),
        };
    }

    if !host_floor_met {
        return HostQualificationResult {
            host_floor_met: false,
            runtime_in_authors_list: true,
            pin_valid: true,
            outcome: OutcomeStatus::Unsupported,
            reason: format!(
                "host {} below #57 floor (requires DGX Spark, >=32GB Apple Silicon, or >=32GB VRAM GPU)",
                req.host_label
            ),
        };
    }

    if !req.runtime_available_locally {
        return HostQualificationResult {
            host_floor_met: true,
            runtime_in_authors_list: true,
            pin_valid: true,
            outcome: OutcomeStatus::UnavailableHardwareRuntime,
            reason: format!(
                "host {} meets floor and runtime {} is admitted, but local runtime/weights are not present",
                req.host_label, req.runtime_name
            ),
        };
    }

    HostQualificationResult {
        host_floor_met: true,
        runtime_in_authors_list: true,
        pin_valid: true,
        outcome: OutcomeStatus::Accepted,
        reason: "host floor, admitted runtime, and exact pin satisfied".into(),
    }
}

/// Deterministic NVFP4 E2M1 scalar quantize/dequantize with block scale (no `f64::min`/`max`).
fn quantize_dequantize_nvfp4(val: f64, block_scale: f64) -> Result<f64, String> {
    if !val.is_finite() || !block_scale.is_finite() || block_scale <= 0.0 {
        return Err("invalid value or block scale in NVFP4 quantization".into());
    }
    let scaled = val / block_scale;
    if scaled == 0.0 {
        return Ok(0.0);
    }
    let sign = if scaled < 0.0 { -1.0 } else { 1.0 };
    let abs_v = scaled.abs();
    let clamped = if abs_v > 6.0 { 6.0 } else { abs_v };
    let mut best_mag = FP4_E2M1_MAGS[0];
    let mut best_dist = (clamped - best_mag).abs();
    for &grid in FP4_E2M1_MAGS.iter().skip(1) {
        let d = (clamped - grid).abs();
        if d < best_dist {
            best_dist = d;
            best_mag = grid;
        }
    }
    Ok(sign * best_mag * block_scale)
}

/// Deterministic MXFP8-like 8-bit uniform-per-block-power-of-two quantize/dequantize.
fn quantize_dequantize_mxfp8(val: f64, block_scale: f64) -> Result<f64, String> {
    if !val.is_finite() || !block_scale.is_finite() || block_scale <= 0.0 {
        return Err("invalid value or block scale in MXFP8 quantization".into());
    }
    let scaled = val / block_scale;
    if !scaled.is_finite() {
        return Err("non-finite scaled value in MXFP8 quantization".into());
    }
    let clamped = scaled.clamp(-448.0, 448.0);
    let step = 1.0 / 16.0;
    let q = (clamped / step).round() * step;
    Ok(q * block_scale)
}

#[derive(Debug, Clone)]
pub struct MiniatureMoeLayer {
    pub layer_index: usize,
    pub quant_regime: LayerQuantRegime,
    pub router_weights: [[f64; MINI_DIM]; NUM_EXPERTS],
    pub expert_scales: [[f64; MINI_DIM]; NUM_EXPERTS],
    pub residual_alpha: f64,
}

#[derive(Debug, Clone)]
pub struct MiniatureInklingMini {
    pub layers: Vec<MiniatureMoeLayer>,
}

#[derive(Debug, Clone)]
pub struct ForwardTrace {
    pub output: [f64; MINI_DIM],
    pub layer_applications: usize,
    pub active_expert_evaluations: usize,
    pub nvfp4_applications: usize,
    pub mxfp8_applications: usize,
    pub first_layer_top_experts: Vec<usize>,
}

impl MiniatureInklingMini {
    pub fn build_deterministic() -> Result<Self, String> {
        let mut layers = Vec::with_capacity(STORED_LAYERS);
        for layer_idx in 0..STORED_LAYERS {
            let regime = if layer_idx < NVFP4_LAYER_END {
                LayerQuantRegime::Nvfp4
            } else {
                LayerQuantRegime::Mxfp8
            };
            let mut router_weights = [[0.0_f64; MINI_DIM]; NUM_EXPERTS];
            let mut expert_scales = [[0.0_f64; MINI_DIM]; NUM_EXPERTS];
            for e in 0..NUM_EXPERTS {
                for d in 0..MINI_DIM {
                    let raw_r = det_weight(5700 + layer_idx * 17, e * MINI_DIM + d, 0.45);
                    let raw_w = det_weight(5900 + layer_idx * 23, e * MINI_DIM + d, 0.35);
                    router_weights[e][d] = raw_r;
                    expert_scales[e][d] = match regime {
                        LayerQuantRegime::Nvfp4 => quantize_dequantize_nvfp4(raw_w, 0.125)?,
                        LayerQuantRegime::Mxfp8 => quantize_dequantize_mxfp8(raw_w, 0.0625)?,
                    };
                }
            }
            layers.push(MiniatureMoeLayer {
                layer_index: layer_idx,
                quant_regime: regime,
                router_weights,
                expert_scales,
                residual_alpha: 0.04,
            });
        }
        Ok(Self { layers })
    }
}

fn rms_norm_vec(x: &[f64; MINI_DIM]) -> Result<[f64; MINI_DIM], String> {
    let mut sum_sq = 0.0_f64;
    for &v in x {
        if !v.is_finite() {
            return Err("non-finite activation in RMSNorm".into());
        }
        sum_sq += v * v;
    }
    let denom = (sum_sq / (MINI_DIM as f64) + 1.0e-6).sqrt();
    if !denom.is_finite() || denom == 0.0 {
        return Err("invalid RMSNorm denominator".into());
    }
    let mut out = [0.0_f64; MINI_DIM];
    for (dst, &src) in out.iter_mut().zip(x.iter()) {
        *dst = src / denom;
    }
    Ok(out)
}

/// Apply one MoE layer with exact `top_k`-of-128 routing and softmax-normalized expert weights.
fn apply_moe_layer(
    layer: &MiniatureMoeLayer,
    state: &[f64; MINI_DIM],
    top_k: usize,
) -> Result<([f64; MINI_DIM], Vec<usize>), String> {
    if top_k == 0 || top_k > NUM_EXPERTS {
        return Err(format!("invalid MoE top_k: {top_k}"));
    }
    let normed = rms_norm_vec(state)?;
    let mut scores: Vec<(usize, f64)> = Vec::with_capacity(NUM_EXPERTS);
    for (e, rw) in layer.router_weights.iter().enumerate() {
        let mut dot = 0.0_f64;
        for d in 0..MINI_DIM {
            dot += normed[d] * rw[d];
        }
        if !dot.is_finite() {
            return Err("non-finite router score".into());
        }
        scores.push((e, dot));
    }

    // All router scores confirmed finite above; deterministic descending sort with index tie-break.
    scores.sort_by(|(ia, sa), (ib, sb)| sb.total_cmp(sa).then_with(|| ia.cmp(ib)));

    let selected = &scores[..top_k];
    let mut max_s = selected[0].1;
    for &(_, s) in selected.iter().skip(1) {
        if s > max_s {
            max_s = s;
        }
    }
    let mut exp_vals = Vec::with_capacity(top_k);
    let mut exp_sum = 0.0_f64;
    for &(_, s) in selected {
        let ex = (s - max_s).exp();
        if !ex.is_finite() {
            return Err("non-finite softmax numerator in MoE routing".into());
        }
        exp_sum += ex;
        exp_vals.push(ex);
    }
    if !exp_sum.is_finite() || exp_sum <= 0.0 {
        return Err("non-positive softmax denominator in MoE routing".into());
    }

    let mut delta = [0.0_f64; MINI_DIM];
    let mut chosen_ids = Vec::with_capacity(top_k);
    for (idx, &(expert_id, _)) in selected.iter().enumerate() {
        let gate = exp_vals[idx] / exp_sum;
        chosen_ids.push(expert_id);
        let ew = &layer.expert_scales[expert_id];
        for d in 0..MINI_DIM {
            // Bounded nonlinear expert transform (tanh) + residual scaling.
            let act = (normed[d] * ew[d] + 0.1 * normed[(d + 1) % MINI_DIM]).tanh();
            delta[d] += gate * act;
        }
    }

    let mut next = [0.0_f64; MINI_DIM];
    for d in 0..MINI_DIM {
        let v = state[d] + layer.residual_alpha * delta[d];
        if !v.is_finite() {
            return Err("non-finite residual output in MoE layer".into());
        }
        next[d] = v;
    }
    Ok((next, chosen_ids))
}

/// Independent unrolled 48-layer reference: materializes explicit 48-step layer list
/// `[0..6, 6..18, 6..18, 6..18, 18..24]` and evaluates sequentially without loop sharing control.
pub fn forward_unrolled_48_reference(
    model: &MiniatureInklingMini,
    input: &[f64; MINI_DIM],
) -> Result<ForwardTrace, String> {
    if model.layers.len() != STORED_LAYERS {
        return Err("expected 24 stored layers".into());
    }
    let mut unrolled_layers: Vec<MiniatureMoeLayer> =
        Vec::with_capacity(UNROLLED_LAYER_APPLICATIONS);
    for idx in 0..ENTRY_LAYERS {
        unrolled_layers.push(model.layers[idx].clone());
    }
    for _pass in 0..DEFAULT_LOOP_COUNT {
        for idx in ENTRY_LAYERS..(ENTRY_LAYERS + MIDDLE_LAYERS) {
            unrolled_layers.push(model.layers[idx].clone());
        }
    }
    for idx in (ENTRY_LAYERS + MIDDLE_LAYERS)..STORED_LAYERS {
        unrolled_layers.push(model.layers[idx].clone());
    }

    let mut state = *input;
    let mut nvfp4_apps = 0usize;
    let mut mxfp8_apps = 0usize;
    let mut first_top = Vec::new();
    for (step, layer) in unrolled_layers.iter().enumerate() {
        let (next, top) = apply_moe_layer(layer, &state, TOP_K_EXPERTS)?;
        if step == 0 {
            first_top = top;
        }
        match layer.quant_regime {
            LayerQuantRegime::Nvfp4 => nvfp4_apps += 1,
            LayerQuantRegime::Mxfp8 => mxfp8_apps += 1,
        }
        state = next;
    }

    Ok(ForwardTrace {
        output: state,
        layer_applications: unrolled_layers.len(),
        active_expert_evaluations: unrolled_layers.len() * TOP_K_EXPERTS,
        nvfp4_applications: nvfp4_apps,
        mxfp8_applications: mxfp8_apps,
        first_layer_top_experts: first_top,
    })
}

/// Looped execution sharing the 12 stored middle layers across `loop_count` passes.
pub fn forward_looped_shared(
    model: &MiniatureInklingMini,
    input: &[f64; MINI_DIM],
    loop_count: usize,
    top_k_override: Option<usize>,
) -> Result<ForwardTrace, String> {
    if model.layers.len() != STORED_LAYERS {
        return Err("expected 24 stored layers".into());
    }
    if loop_count == 0 {
        return Err("loop_count must be >= 1".into());
    }
    let k = top_k_override.unwrap_or(TOP_K_EXPERTS);
    let mut state = *input;
    let mut apps = 0usize;
    let mut nvfp4_apps = 0usize;
    let mut mxfp8_apps = 0usize;
    let mut first_top = Vec::new();

    // 6 entry layers
    for layer in &model.layers[0..ENTRY_LAYERS] {
        let (next, top) = apply_moe_layer(layer, &state, k)?;
        if apps == 0 {
            first_top = top;
        }
        apps += 1;
        match layer.quant_regime {
            LayerQuantRegime::Nvfp4 => nvfp4_apps += 1,
            LayerQuantRegime::Mxfp8 => mxfp8_apps += 1,
        }
        state = next;
    }

    // 12 shared middle layers looped `loop_count` times
    for _pass in 0..loop_count {
        for layer in &model.layers[ENTRY_LAYERS..(ENTRY_LAYERS + MIDDLE_LAYERS)] {
            let (next, _) = apply_moe_layer(layer, &state, k)?;
            apps += 1;
            match layer.quant_regime {
                LayerQuantRegime::Nvfp4 => nvfp4_apps += 1,
                LayerQuantRegime::Mxfp8 => mxfp8_apps += 1,
            }
            state = next;
        }
    }

    // 6 exit layers
    for layer in &model.layers[(ENTRY_LAYERS + MIDDLE_LAYERS)..STORED_LAYERS] {
        let (next, _) = apply_moe_layer(layer, &state, k)?;
        apps += 1;
        match layer.quant_regime {
            LayerQuantRegime::Nvfp4 => nvfp4_apps += 1,
            LayerQuantRegime::Mxfp8 => mxfp8_apps += 1,
        }
        state = next;
    }

    Ok(ForwardTrace {
        output: state,
        layer_applications: apps,
        active_expert_evaluations: apps * k,
        nvfp4_applications: nvfp4_apps,
        mxfp8_applications: mxfp8_apps,
        first_layer_top_experts: first_top,
    })
}

/// Convenience wrapper for the standard 3x looped Mini forward pass.
pub fn forward_looped_3x_shared(
    model: &MiniatureInklingMini,
    input: &[f64; MINI_DIM],
) -> Result<ForwardTrace, String> {
    forward_looped_shared(model, input, DEFAULT_LOOP_COUNT, None)
}

fn max_abs_diff(a: &[f64; MINI_DIM], b: &[f64; MINI_DIM]) -> Result<f64, String> {
    let mut m = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return Err("non-finite value in vector diff".into());
        }
        let d = (x - y).abs();
        if d > m {
            m = d;
        }
    }
    Ok(m)
}

fn l2_norm(v: &[f64; MINI_DIM]) -> Result<f64, String> {
    let mut s = 0.0_f64;
    for &x in v {
        if !x.is_finite() {
            return Err("non-finite value in l2_norm".into());
        }
        s += x * x;
    }
    Ok(s.sqrt())
}

#[derive(Debug, Clone)]
pub struct VarloopSliceRow {
    pub loop_count: usize,
    pub layer_applications: usize,
    pub active_expert_evaluations: usize,
    pub output_l2_norm: f64,
    pub max_diff_vs_3x: f64,
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct InklingMiniServeReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub synthetic_contract_fixture: bool,
    pub default_enabled: bool,
    pub stored_layers: usize,
    pub unrolled_layer_applications: usize,
    pub looped_layer_applications: usize,
    pub moe_routed_experts: usize,
    pub moe_total_experts: usize,
    pub nvfp4_stored_layers: usize,
    pub mxfp8_stored_layers: usize,
    pub unrolled_vs_looped_max_abs_err: f64,
    pub parity_outcome: OutcomeStatus,
    pub varloop_optional_research_only: bool,
    pub varloop_sweep: Vec<VarloopSliceRow>,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<InklingMiniServeReport, String> {
    let model = MiniatureInklingMini::build_deterministic()?;
    let nvfp4_stored_layers = model
        .layers
        .iter()
        .filter(|l| l.quant_regime == LayerQuantRegime::Nvfp4)
        .count();
    let mxfp8_stored_layers = model
        .layers
        .iter()
        .filter(|l| l.quant_regime == LayerQuantRegime::Mxfp8)
        .count();

    let frozen_input = [0.65_f64, -0.42, 0.88, -0.21];
    let unrolled = forward_unrolled_48_reference(&model, &frozen_input)?;
    let looped = forward_looped_3x_shared(&model, &frozen_input)?;
    let unrolled_vs_looped_max_abs_err = max_abs_diff(&unrolled.output, &looped.output)?;
    let parity_ok = unrolled.layer_applications == UNROLLED_LAYER_APPLICATIONS
        && looped.layer_applications == UNROLLED_LAYER_APPLICATIONS
        && unrolled.active_expert_evaluations == looped.active_expert_evaluations
        && unrolled_vs_looped_max_abs_err <= 1.0e-12;

    // Optional research-only TML-d24-varloop-research sweep across loop_count = 1, 2, 3, 4
    let mut varloop_sweep = Vec::new();
    for lc in [1usize, 2, 3, 4] {
        let trace = forward_looped_shared(&model, &frozen_input, lc, None)?;
        let norm = l2_norm(&trace.output)?;
        let diff_vs_3x = max_abs_diff(&trace.output, &looped.output)?;
        varloop_sweep.push(VarloopSliceRow {
            loop_count: lc,
            layer_applications: trace.layer_applications,
            active_expert_evaluations: trace.active_expert_evaluations,
            output_l2_norm: norm,
            max_diff_vs_3x: diff_vs_3x,
        });
    }

    // 8 executable negative controls
    let mut negative_controls = Vec::new();

    // NC1: Discrete GPU below 32 GB floor rejected cleanly as Unsupported
    {
        let q = qualify_inkling_host_and_runtime(&HostServeRequest {
            host_label: "rtx-3090-24gb".into(),
            platform: HostFloorClass::DiscreteGpu { vram_gb: 24 },
            runtime_name: "vllm".into(),
            runtime_available_locally: true,
            checkpoint_repo: INKLING_MINI_REPO.into(),
            checkpoint_commit: INKLING_MINI_COMMIT.into(),
        });
        negative_controls.push(NegativeControl {
            case_id: "nc1_gpu_below_32gb_floor_unsupported".into(),
            outcome: q.outcome,
            detail: q.reason,
        });
    }

    // NC2: Apple Silicon below 32 GB unified memory rejected cleanly as Unsupported
    {
        let q = qualify_inkling_host_and_runtime(&HostServeRequest {
            host_label: "mac-16gb".into(),
            platform: HostFloorClass::AppleSilicon {
                unified_memory_gb: 16,
            },
            runtime_name: "ollama".into(),
            runtime_available_locally: true,
            checkpoint_repo: INKLING_MINI_REPO.into(),
            checkpoint_commit: INKLING_MINI_COMMIT.into(),
        });
        negative_controls.push(NegativeControl {
            case_id: "nc2_apple_silicon_below_32gb_unsupported".into(),
            outcome: q.outcome,
            detail: q.reason,
        });
    }

    // NC3: Reject `maya-ik2` as unlisted for #57 even on an admitted DGX Spark host
    {
        let q = qualify_inkling_host_and_runtime(&HostServeRequest {
            host_label: "dgx-spark".into(),
            platform: HostFloorClass::DgxSpark,
            runtime_name: "maya-ik2".into(),
            runtime_available_locally: true,
            checkpoint_repo: INKLING_MINI_REPO.into(),
            checkpoint_commit: INKLING_MINI_COMMIT.into(),
        });
        negative_controls.push(NegativeControl {
            case_id: "nc3_maya_ik2_unlisted_for_issue_57".into(),
            outcome: q.outcome,
            detail: q.reason,
        });
    }

    // NC4: Unpinned Hugging Face tip rejected
    {
        let q = qualify_inkling_host_and_runtime(&HostServeRequest {
            host_label: "dgx-spark".into(),
            platform: HostFloorClass::DgxSpark,
            runtime_name: "sglang".into(),
            runtime_available_locally: true,
            checkpoint_repo: INKLING_MINI_REPO.into(),
            checkpoint_commit: "main".into(),
        });
        negative_controls.push(NegativeControl {
            case_id: "nc4_unpinned_hf_tip_rejected".into(),
            outcome: q.outcome,
            detail: q.reason,
        });
    }

    // NC5: Admitted host floor + admitted runtime without local runtime/weights returns unavailable_hardware_runtime
    {
        let q = qualify_inkling_host_and_runtime(&HostServeRequest {
            host_label: "dgx-spark-offline".into(),
            platform: HostFloorClass::DgxSpark,
            runtime_name: "sglang".into(),
            runtime_available_locally: false,
            checkpoint_repo: INKLING_MINI_REPO.into(),
            checkpoint_commit: INKLING_MINI_COMMIT.into(),
        });
        negative_controls.push(NegativeControl {
            case_id: "nc5_missing_local_runtime_reports_unavailable".into(),
            outcome: q.outcome,
            detail: q.reason,
        });
    }

    // NC6: Loop count mismatch (2x = 36 apps vs unrolled 48 apps) diverges beyond tolerance
    {
        let two_pass = forward_looped_shared(&model, &frozen_input, 2, None)?;
        let err = max_abs_diff(&unrolled.output, &two_pass.output)?;
        negative_controls.push(NegativeControl {
            case_id: "nc6_loop_count_mismatch_detected".into(),
            outcome: if err > 1.0e-12 || two_pass.layer_applications != UNROLLED_LAYER_APPLICATIONS
            {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!(
                "loop_count=2 ({} apps) diverged {:.3e} from 48-app reference",
                two_pass.layer_applications, err
            ),
        });
    }

    // NC7: Wrong MoE routing top-k (4-of-128 instead of 8-of-128) breaks parity
    {
        let bad_k = forward_looped_shared(&model, &frozen_input, DEFAULT_LOOP_COUNT, Some(4))?;
        let err = max_abs_diff(&unrolled.output, &bad_k.output)?;
        negative_controls.push(NegativeControl {
            case_id: "nc7_wrong_moe_topk_routing_detected".into(),
            outcome: if err > 1.0e-12 {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!(
                "4-of-128 MoE routing diverged {:.3e} from 8-of-128 reference",
                err
            ),
        });
    }

    // NC8: Non-finite input activation rejected cleanly
    {
        let nan_in = [0.5, f64::NAN, 0.2, -0.1];
        let res = forward_looped_3x_shared(&model, &nan_in);
        negative_controls.push(NegativeControl {
            case_id: "nc8_non_finite_activation_rejected".into(),
            outcome: if res.is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "NaN activation caught before MoE routing".into(),
        });
    }

    Ok(InklingMiniServeReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "Admitted host (DGX Spark / >=32GB Apple Silicon / >=32GB VRAM GPU) with pinned Inkling 2 Mini checkpoint and an authors-listed runtime",
            "Host/runtime gate, 24-stored/48-applied looped 8-of-128 MoE parity, NVFP4/MXFP8 layer split, and optional varloop sweep verified on CPU; 26.4B Inkling 2 Mini weights were not downloaded or served locally",
        ),
        synthetic_contract_fixture: true,
        default_enabled: false,
        stored_layers: STORED_LAYERS,
        unrolled_layer_applications: unrolled.layer_applications,
        looped_layer_applications: looped.layer_applications,
        moe_routed_experts: TOP_K_EXPERTS,
        moe_total_experts: NUM_EXPERTS,
        nvfp4_stored_layers,
        mxfp8_stored_layers,
        unrolled_vs_looped_max_abs_err,
        parity_outcome: if parity_ok {
            OutcomeStatus::Accepted
        } else {
            OutcomeStatus::Invalid
        },
        varloop_optional_research_only: true,
        varloop_sweep,
        negative_controls,
    })
}

pub fn inputs_json(report: &InklingMiniServeReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.inkling-mini-serve.inputs.v1\",\n",
            "  \"issue\": 57,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"license\": {},\n",
            "  \"model_aup_url\": {},\n",
            "  \"pins\": {{\n",
            "    \"inkling_2_mini\": {{\"repo\": {}, \"commit\": {}}},\n",
            "    \"tml_d24_varloop_research\": {{\"repo\": {}, \"commit\": {}}},\n",
            "    \"tinker_cookbook\": {{\"repo\": {}, \"commit\": {}}}\n",
            "  }},\n",
            "  \"authors_runtime_list\": {},\n",
            "  \"distinct_related_issues\": {},\n",
            "  \"author_unverified_non_gating_notes\": \"Thinking Machines DeepSWE/Vals/SWE-bench/AA Index tables and 614 GB/s BF16 theoretical ceiling are unverified author claims, not acceptance gates\"\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(MODEL_LICENSE),
        quote(MODEL_AUP_URL),
        quote(INKLING_MINI_REPO),
        quote(INKLING_MINI_COMMIT),
        quote(VARLOOP_RESEARCH_REPO),
        quote(VARLOOP_RESEARCH_COMMIT),
        quote(TINKER_COOKBOOK_REPO),
        quote(TINKER_COOKBOOK_COMMIT),
        static_str_array(&AUTHORS_RUNTIME_LIST),
        static_str_array(&DISTINCT_RELATED_ISSUES),
    )
}

pub fn results_json(report: &InklingMiniServeReport) -> String {
    let sweep: Vec<String> = report
        .varloop_sweep
        .iter()
        .map(|row| {
            format!(
                "{{\"loop_count\":{},\"layer_applications\":{},\"active_expert_evaluations\":{},\"output_l2_norm\":{},\"max_diff_vs_3x\":{}}}",
                row.loop_count,
                row.layer_applications,
                row.active_expert_evaluations,
                finite_f64(row.output_l2_norm),
                finite_f64(row.max_diff_vs_3x),
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
            "  \"schema\": \"inference-lab.inkling-mini-serve.results.v1\",\n",
            "  \"issue\": 57,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"stored_layers\": {},\n",
            "  \"unrolled_layer_applications\": {},\n",
            "  \"looped_layer_applications\": {},\n",
            "  \"moe_routed_experts\": {},\n",
            "  \"moe_total_experts\": {},\n",
            "  \"nvfp4_stored_layers\": {},\n",
            "  \"mxfp8_stored_layers\": {},\n",
            "  \"unrolled_vs_looped_max_abs_err\": {},\n",
            "  \"parity_outcome\": {},\n",
            "  \"varloop_optional_research_only\": {},\n",
            "  \"varloop_sweep\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.synthetic_contract_fixture,
        report.default_enabled,
        report.stored_layers,
        report.unrolled_layer_applications,
        report.looped_layer_applications,
        report.moe_routed_experts,
        report.moe_total_experts,
        report.nvfp4_stored_layers,
        report.mxfp8_stored_layers,
        finite_f64(report.unrolled_vs_looped_max_abs_err),
        quote(report.parity_outcome.as_str()),
        report.varloop_optional_research_only,
        sweep.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &InklingMiniServeReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #57: Default-Off Local Serve Qualification for Pinned Inkling 2 Mini"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Execution class: `{}` (`synthetic_contract_fixture: {}`, `default_enabled: {}`)",
        report.execution_class.as_str(),
        report.synthetic_contract_fixture,
        report.default_enabled
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
        "- Topology: `{} stored layers` (`{} NVFP4`, `{} MXFP8`), `{} -> {} applications` (6 entry + 12x3 middle + 6 exit), `{}-of-{} MoE`",
        report.stored_layers,
        report.nvfp4_stored_layers,
        report.mxfp8_stored_layers,
        report.unrolled_layer_applications,
        report.looped_layer_applications,
        report.moe_routed_experts,
        report.moe_total_experts
    )
    .unwrap();
    writeln!(
        out,
        "- Unrolled 48-layer reference vs 3x shared-loop parity: `max_abs_err = {:.3e}` (`{}`)",
        report.unrolled_vs_looped_max_abs_err,
        report.parity_outcome.as_str()
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Optional Research Varloop Sweep (`TML-d24-varloop-research`, research-only)"
    )
    .unwrap();
    for row in &report.varloop_sweep {
        writeln!(
            out,
            "- `loop_count={}`: `layer_apps={}`, `active_experts={}`, `l2_norm={:.6}`, `max_diff_vs_3x={:.3e}`",
            row.loop_count,
            row.layer_applications,
            row.active_expert_evaluations,
            row.output_l2_norm,
            row.max_diff_vs_3x
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
    fn default_harness_passes_looped_parity_and_eight_negative_controls() {
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
        assert_eq!(report.stored_layers, 24);
        assert_eq!(report.unrolled_layer_applications, 48);
        assert_eq!(report.looped_layer_applications, 48);
        assert_eq!(report.nvfp4_stored_layers, 20);
        assert_eq!(report.mxfp8_stored_layers, 4);
        assert_eq!(report.moe_routed_experts, 8);
        assert_eq!(report.moe_total_experts, 128);
        assert!(report.unrolled_vs_looped_max_abs_err <= 1.0e-12);
        assert_eq!(report.parity_outcome, OutcomeStatus::Accepted);
        assert_eq!(report.varloop_sweep.len(), 4);
        assert_eq!(report.varloop_sweep[0].layer_applications, 24);
        assert_eq!(report.varloop_sweep[1].layer_applications, 36);
        assert_eq!(report.varloop_sweep[2].layer_applications, 48);
        assert_eq!(report.varloop_sweep[3].layer_applications, 60);
        assert!(report.varloop_sweep[2].max_diff_vs_3x <= 1.0e-12);
        assert!(report.varloop_sweep[0].max_diff_vs_3x > 1.0e-6);
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
        assert!(summary_markdown(&report).contains("Issue #57"));
    }

    #[test]
    fn admits_all_listed_runtimes_on_valid_hosts_and_rejects_maya_ik2() {
        for rt in AUTHORS_RUNTIME_LIST {
            let q = qualify_inkling_host_and_runtime(&HostServeRequest {
                host_label: "apple-64gb".into(),
                platform: HostFloorClass::AppleSilicon {
                    unified_memory_gb: 64,
                },
                runtime_name: rt.into(),
                runtime_available_locally: true,
                checkpoint_repo: INKLING_MINI_REPO.into(),
                checkpoint_commit: INKLING_MINI_COMMIT.into(),
            });
            assert_eq!(
                q.outcome,
                OutcomeStatus::Accepted,
                "runtime {rt} must be admitted"
            );
        }
        let maya = qualify_inkling_host_and_runtime(&HostServeRequest {
            host_label: "gpu-48gb".into(),
            platform: HostFloorClass::DiscreteGpu { vram_gb: 48 },
            runtime_name: "maya-ik2".into(),
            runtime_available_locally: true,
            checkpoint_repo: INKLING_MINI_REPO.into(),
            checkpoint_commit: INKLING_MINI_COMMIT.into(),
        });
        assert_eq!(maya.outcome, OutcomeStatus::Unsupported);
    }
}
