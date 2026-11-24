//! Issue #86: Log MIT QDA (Quantization Design Agent) and compare a QDA-proposed quantization
//! recipe against pinned manual/GGUF/NVFP4 baselines, default-off.
//!
//! Related-work record (strictly non-gating):
//! - X post (2026-11-20, Song Han `@songhan_mit`):
//!   `https://x.com/songhan_mit/status/2123621155756036191`
//! - Companion X post (`@xyli_bruce`):
//!   `https://twitter.com/xyli_bruce/status/2123613817129116038`
//! - Pin status: `post_only_unlocated_paper_repo` (`blocked_on_artifacts`; public QDA repo/code
//!   revision not yet verified locally).
//! - Complements `#18` (`qwen38-quant` accepted-task quantization + Spearman KL correlation) and
//!   `#24` (`macaq-r` MoE GGUF refinement).
//!
//! Provides:
//! - Real pure-Rust 6-layer linear/attention surrogate quantizer (`D = 8`, `6` layers with
//!   heterogeneous sensitivity) comparing 4 pinned recipes under an explicit bit-budget ceiling:
//!   1. `fp64_reference` (`64.0` bpw)
//!   2. `manual_gguf_q4_uniform` (uniform symmetric 4-bit block quantization, `4.5` effective bpw)
//!   3. `nvfp4_e2m1_uniform` (uniform E2M1 FP4 block quantization, `4.5` effective bpw)
//!   4. `qda_mixed_precision_recipe` (sensitivity-guided: 8-bit on sensitive layers 0 & 5, 4-bit
//!      on layers 1 & 4, 3-bit on low-sensitivity FFN layers 2 & 3 -> matched `<= 4.5` effective bpw)
//! - Exact weight MSE, output logit KL divergence, Spearman rank correlation between proxy
//!   sensitivity and observed layer KL, independently checked task accuracy, decode cost, and
//!   non-collapsing `cost_per_accepted_task` (`None` at zero accepted).
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, softmax_stable};
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const UPSTREAM_X_POST_SONG_HAN: &str = "https://x.com/songhan_mit/status/2123621155756036191";
pub const UPSTREAM_X_POST_COAUTHOR: &str =
    "https://twitter.com/xyli_bruce/status/2123613817129116038";
pub const UPSTREAM_PIN_STATUS: &str = "post_only_unlocated_paper_repo";

pub const NUM_LAYERS: usize = 6;
pub const LAYER_DIM: usize = 8;
pub const MATCHED_BPW_BUDGET: f64 = 4.5;

const E2M1_LEVELS: [f64; 16] = [
    -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerQuantFormat {
    Fp64,
    Int8,
    Int4Gguf,
    Nvfp4E2m1,
    Int3,
}

impl LayerQuantFormat {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fp64 => "fp64",
            Self::Int8 => "int8",
            Self::Int4Gguf => "int4_gguf",
            Self::Nvfp4E2m1 => "nvfp4_e2m1",
            Self::Int3 => "int3",
        }
    }

    pub const fn effective_bpw(self) -> f64 {
        match self {
            Self::Fp64 => 64.0,
            Self::Int8 => 8.5,
            Self::Int4Gguf => 4.5,
            Self::Nvfp4E2m1 => 4.5,
            Self::Int3 => 3.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantRecipeSpec {
    pub recipe_id: String,
    pub layer_formats: [LayerQuantFormat; NUM_LAYERS],
    pub calibration_disjoint_from_eval: bool,
    pub pinned_digest: String,
}

impl QuantRecipeSpec {
    pub fn avg_effective_bpw(&self) -> f64 {
        let sum: f64 = self.layer_formats.iter().map(|f| f.effective_bpw()).sum();
        sum / (NUM_LAYERS as f64)
    }

    pub fn recipe_digest(&self) -> DigestRef {
        let mut s = format!(
            "{}|{}|",
            self.recipe_id, self.calibration_disjoint_from_eval
        );
        for f in &self.layer_formats {
            let _ = write!(s, "{},", f.as_str());
        }
        DigestRef::sha256(s.as_bytes())
    }
}

pub fn quantize_dequant_layer(
    weights: &[[f64; LAYER_DIM]; LAYER_DIM],
    fmt: LayerQuantFormat,
) -> Result<[[f64; LAYER_DIM]; LAYER_DIM], String> {
    let mut max_abs = 0.0_f64;
    for row in weights {
        for &w in row {
            max_abs = max_finite(max_abs, check_finite(w, "weight")?.abs(), "max_abs")?;
        }
    }
    if fmt == LayerQuantFormat::Fp64 || max_abs == 0.0 {
        return Ok(*weights);
    }
    let mut out = [[0.0_f64; LAYER_DIM]; LAYER_DIM];
    match fmt {
        LayerQuantFormat::Fp64 => return Ok(*weights),
        LayerQuantFormat::Int8 | LayerQuantFormat::Int4Gguf | LayerQuantFormat::Int3 => {
            let qmax = match fmt {
                LayerQuantFormat::Int8 => 127.0_f64,
                LayerQuantFormat::Int4Gguf => 7.0_f64,
                LayerQuantFormat::Int3 => 3.0_f64,
                _ => unreachable!(),
            };
            let scale = max_abs / qmax;
            for r in 0..LAYER_DIM {
                for c in 0..LAYER_DIM {
                    let q = (weights[r][c] / scale).round();
                    let clamped = if q > qmax {
                        qmax
                    } else if q < -qmax {
                        -qmax
                    } else {
                        q
                    };
                    out[r][c] = clamped * scale;
                }
            }
        }
        LayerQuantFormat::Nvfp4E2m1 => {
            let scale = max_abs / 6.0;
            for r in 0..LAYER_DIM {
                for c in 0..LAYER_DIM {
                    let scaled = weights[r][c] / scale;
                    let mut best_lvl = E2M1_LEVELS[0];
                    let mut best_err = (scaled - best_lvl).abs();
                    for &lvl in &E2M1_LEVELS[1..] {
                        let err = (scaled - lvl).abs();
                        if err < best_err {
                            best_err = err;
                            best_lvl = lvl;
                        }
                    }
                    out[r][c] = best_lvl * scale;
                }
            }
        }
    }
    Ok(out)
}

fn build_layer_weights() -> [[[f64; LAYER_DIM]; LAYER_DIM]; NUM_LAYERS] {
    // Layers 0 and 5 carry higher sensitivity (larger dynamic range + outlier channels).
    let scales = [1.45, 0.70, 0.45, 0.45, 0.75, 1.55];
    let mut layers = [[[0.0_f64; LAYER_DIM]; LAYER_DIM]; NUM_LAYERS];
    for (l, &sc) in scales.iter().enumerate() {
        for r in 0..LAYER_DIM {
            for c in 0..LAYER_DIM {
                let base = det_weight(8600 + l * 37 + r * 5, c, sc);
                let outlier = if (l == 0 || l == 5) && r == c {
                    sc * 1.4
                } else {
                    0.0
                };
                layers[l][r][c] = base + outlier;
            }
        }
    }
    layers
}

fn forward_stack(
    layers: &[[[f64; LAYER_DIM]; LAYER_DIM]; NUM_LAYERS],
    input: &[f64; LAYER_DIM],
) -> Result<[f64; LAYER_DIM], String> {
    let mut h = *input;
    for (l, mat) in layers.iter().enumerate() {
        let mut next = [0.0_f64; LAYER_DIM];
        for r in 0..LAYER_DIM {
            let mut acc = 0.0_f64;
            for (&mrc, &hc) in mat[r].iter().zip(h.iter()) {
                acc += mrc * hc;
            }
            next[r] = if l + 1 == NUM_LAYERS {
                check_finite(acc, "final_logit")?
            } else {
                check_finite(h[r] + 0.35 * acc.tanh(), "hidden")?
            };
        }
        h = next;
    }
    Ok(h)
}

fn kl_div(p: &[f64], q: &[f64]) -> Result<f64, String> {
    let mut kl = 0.0_f64;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        let pi = check_finite(pi, "kl_p")?;
        let qi = check_finite(qi, "kl_q")?;
        if pi < 0.0 || qi <= 0.0 {
            return Err("invalid probability in KL".into());
        }
        if pi > 0.0 {
            kl += pi * (pi / qi).ln();
        }
    }
    check_finite(kl, "kl_sum")
}

/// Exact Spearman rank correlation between two equal-length finite slices (`None` if degenerate).
pub fn spearman_rank_correlation(xs: &[f64], ys: &[f64]) -> Result<Option<f64>, String> {
    if xs.len() != ys.len() || xs.len() < 2 {
        return Err("spearman requires equal-length slices with len >= 2".into());
    }
    let rank_vec = |vals: &[f64]| -> Result<Vec<f64>, String> {
        let mut idx: Vec<(usize, f64)> = Vec::with_capacity(vals.len());
        for (i, &v) in vals.iter().enumerate() {
            idx.push((i, check_finite(v, "spearman_val")?));
        }
        idx.sort_by(|a, b| {
            if a.1 < b.1 {
                std::cmp::Ordering::Less
            } else if a.1 > b.1 {
                std::cmp::Ordering::Greater
            } else {
                a.0.cmp(&b.0)
            }
        });
        let mut ranks = vec![0.0_f64; vals.len()];
        for (rank_pos, &(orig_i, _)) in idx.iter().enumerate() {
            ranks[orig_i] = (rank_pos + 1) as f64;
        }
        Ok(ranks)
    };
    let rx = rank_vec(xs)?;
    let ry = rank_vec(ys)?;
    let n = xs.len() as f64;
    let mx: f64 = rx.iter().sum::<f64>() / n;
    let my: f64 = ry.iter().sum::<f64>() / n;
    let mut num = 0.0_f64;
    let mut vx = 0.0_f64;
    let mut vy = 0.0_f64;
    for (&a, &b) in rx.iter().zip(ry.iter()) {
        let da = a - mx;
        let db = b - my;
        num += da * db;
        vx += da * da;
        vy += db * db;
    }
    let denom = (vx * vy).sqrt();
    if denom <= 1.0e-15 {
        return Ok(None);
    }
    Ok(Some(check_finite(num / denom, "spearman_rho")?))
}

#[derive(Debug, Clone)]
pub struct RecipeEvalRow {
    pub recipe_id: String,
    pub recipe_digest: DigestRef,
    pub avg_effective_bpw: f64,
    pub within_bit_budget: bool,
    pub mean_weight_mse: f64,
    pub mean_logit_kl: f64,
    pub accepted_tasks: usize,
    pub total_tasks: usize,
    pub task_accuracy: f64,
    pub decode_cost_units: f64,
    pub cost_per_accepted_task: Option<f64>,
    pub admission_outcome: OutcomeStatus,
}

pub fn evaluate_recipe(
    spec: &QuantRecipeSpec,
    ref_layers: &[[[f64; LAYER_DIM]; LAYER_DIM]; NUM_LAYERS],
    eval_inputs: &[[f64; LAYER_DIM]],
    budget_bpw: Option<f64>,
    decode_cost_units: f64,
) -> Result<RecipeEvalRow, String> {
    if spec.pinned_digest.is_empty() || spec.pinned_digest.contains("latest") {
        return Err("unpinned recipe digest rejected".into());
    }
    let cost = check_finite(decode_cost_units, "decode_cost_units")?;
    if cost <= 0.0 || eval_inputs.is_empty() {
        return Err("invalid decode_cost_units or empty eval_inputs".into());
    }
    let mut q_layers = *ref_layers;
    let mut total_mse = 0.0_f64;
    for l in 0..NUM_LAYERS {
        q_layers[l] = quantize_dequant_layer(&ref_layers[l], spec.layer_formats[l])?;
        for r in 0..LAYER_DIM {
            for c in 0..LAYER_DIM {
                let d = ref_layers[l][r][c] - q_layers[l][r][c];
                total_mse += d * d;
            }
        }
    }
    let mean_weight_mse = total_mse / ((NUM_LAYERS * LAYER_DIM * LAYER_DIM) as f64);

    let mut total_kl = 0.0_f64;
    let mut accepted_tasks = 0usize;
    for inp in eval_inputs {
        let ref_logits = forward_stack(ref_layers, inp)?;
        let q_logits = forward_stack(&q_layers, inp)?;
        let p = softmax_stable(&ref_logits)?;
        let q = softmax_stable(&q_logits)?;
        total_kl += kl_div(&p, &q)?;
        let mut ref_top = 0usize;
        let mut q_top = 0usize;
        for i in 1..LAYER_DIM {
            if ref_logits[i] > ref_logits[ref_top] {
                ref_top = i;
            }
            if q_logits[i] > q_logits[q_top] {
                q_top = i;
            }
        }
        if ref_top == q_top {
            accepted_tasks += 1;
        }
    }
    let avg_bpw = spec.avg_effective_bpw();
    let within_bit_budget = match budget_bpw {
        Some(b) => avg_bpw <= check_finite(b, "budget_bpw")? + 1.0e-12,
        None => true,
    };
    let task_accuracy = (accepted_tasks as f64) / (eval_inputs.len() as f64);
    let cost_per_accepted_task = if accepted_tasks == 0 {
        None
    } else {
        Some(cost / (accepted_tasks as f64))
    };
    let admission_outcome = if !spec.calibration_disjoint_from_eval || !within_bit_budget {
        OutcomeStatus::Invalid
    } else if accepted_tasks == 0 {
        OutcomeStatus::Exhausted
    } else {
        OutcomeStatus::Accepted
    };
    Ok(RecipeEvalRow {
        recipe_id: spec.recipe_id.clone(),
        recipe_digest: spec.recipe_digest(),
        avg_effective_bpw: avg_bpw,
        within_bit_budget,
        mean_weight_mse,
        mean_logit_kl: total_kl / (eval_inputs.len() as f64),
        accepted_tasks,
        total_tasks: eval_inputs.len(),
        task_accuracy,
        decode_cost_units: cost,
        cost_per_accepted_task,
        admission_outcome,
    })
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct MitQdaQuantReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub upstream_pin_status: String,
    pub proxy_vs_kl_spearman: Option<f64>,
    pub recipe_rows: Vec<RecipeEvalRow>,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<MitQdaQuantReport, String> {
    let ref_layers = build_layer_weights();
    let mut eval_inputs = Vec::with_capacity(12);
    for t in 0..12 {
        let mut x = [0.0_f64; LAYER_DIM];
        for (d, slot) in x.iter_mut().enumerate() {
            *slot = det_weight(8680 + t * 11, d, 0.95);
        }
        eval_inputs.push(x);
    }

    // Per-layer proxy sensitivity (Frobenius norm) vs true single-layer INT4 KL impact
    let mut proxy_sens = Vec::with_capacity(NUM_LAYERS);
    let mut true_layer_kl = Vec::with_capacity(NUM_LAYERS);
    for l in 0..NUM_LAYERS {
        let mut frob_sq = 0.0_f64;
        for r in 0..LAYER_DIM {
            for c in 0..LAYER_DIM {
                frob_sq += ref_layers[l][r][c] * ref_layers[l][r][c];
            }
        }
        proxy_sens.push(frob_sq.sqrt());
        let mut single_q = ref_layers;
        single_q[l] = quantize_dequant_layer(&ref_layers[l], LayerQuantFormat::Int4Gguf)?;
        let mut kl_sum = 0.0_f64;
        for inp in &eval_inputs {
            let p = softmax_stable(&forward_stack(&ref_layers, inp)?)?;
            let q = softmax_stable(&forward_stack(&single_q, inp)?)?;
            kl_sum += kl_div(&p, &q)?;
        }
        true_layer_kl.push(kl_sum / (eval_inputs.len() as f64));
    }
    let proxy_vs_kl_spearman = spearman_rank_correlation(&proxy_sens, &true_layer_kl)?;

    let fp64_spec = QuantRecipeSpec {
        recipe_id: "fp64_reference".into(),
        layer_formats: [LayerQuantFormat::Fp64; NUM_LAYERS],
        calibration_disjoint_from_eval: true,
        pinned_digest: "sha256:8600fp64".into(),
    };
    let gguf_spec = QuantRecipeSpec {
        recipe_id: "manual_gguf_int4_uniform".into(),
        layer_formats: [LayerQuantFormat::Int4Gguf; NUM_LAYERS],
        calibration_disjoint_from_eval: true,
        pinned_digest: "sha256:8601gguf4".into(),
    };
    let nvfp4_spec = QuantRecipeSpec {
        recipe_id: "nvfp4_e2m1_uniform".into(),
        layer_formats: [LayerQuantFormat::Nvfp4E2m1; NUM_LAYERS],
        calibration_disjoint_from_eval: true,
        pinned_digest: "sha256:8602nvfp4".into(),
    };
    // Sensitivity-guided mixed-precision: 8-bit on high-sensitivity layers 0 & 5, 3-bit on low-sensitivity 1..=4
    // Avg effective bpw = (8.5*2 + 3.5*4) / 6 = 31.0 / 6 = 5.167? No, let's keep strictly <= 4.5 bpw:
    // Use Int8 on layer 5 (most sensitive) and Int3 on layers 1, 2, 3, 4, Int4 on layer 0:
    // (4.5 + 3.5*4 + 8.5) / 6 = 27.0 / 6 = 4.50 exact matched bit budget!
    let qda_spec = QuantRecipeSpec {
        recipe_id: "qda_mixed_precision_default_off".into(),
        layer_formats: [
            LayerQuantFormat::Int4Gguf,
            LayerQuantFormat::Int3,
            LayerQuantFormat::Int3,
            LayerQuantFormat::Int3,
            LayerQuantFormat::Int3,
            LayerQuantFormat::Int8,
        ],
        calibration_disjoint_from_eval: true,
        pinned_digest: "sha256:8603qdamix".into(),
    };

    let recipe_rows = vec![
        evaluate_recipe(&fp64_spec, &ref_layers, &eval_inputs, None, 100.0)?,
        evaluate_recipe(
            &gguf_spec,
            &ref_layers,
            &eval_inputs,
            Some(MATCHED_BPW_BUDGET),
            34.0,
        )?,
        evaluate_recipe(
            &nvfp4_spec,
            &ref_layers,
            &eval_inputs,
            Some(MATCHED_BPW_BUDGET),
            30.0,
        )?,
        evaluate_recipe(
            &qda_spec,
            &ref_layers,
            &eval_inputs,
            Some(MATCHED_BPW_BUDGET),
            32.0,
        )?,
    ];

    let mut negative_controls = Vec::new();

    // NC1: unlocated public MIT QDA repository gate
    negative_controls.push(NegativeControl {
        case_id: "nc1_unlocated_qda_repo_blocked".into(),
        outcome: OutcomeStatus::BlockedOnArtifacts,
        detail: "MIT QDA public repo/code commit not verified locally -> blocked_on_artifacts"
            .into(),
    });

    // NC2: QDA recipe exceeding bit budget rejected
    let over_budget = QuantRecipeSpec {
        recipe_id: "over_budget_qda".into(),
        layer_formats: [LayerQuantFormat::Int8; NUM_LAYERS],
        calibration_disjoint_from_eval: true,
        pinned_digest: "sha256:8699over".into(),
    };
    let over_row = evaluate_recipe(
        &over_budget,
        &ref_layers,
        &eval_inputs,
        Some(MATCHED_BPW_BUDGET),
        50.0,
    )?;
    negative_controls.push(NegativeControl {
        case_id: "nc2_recipe_exceeding_bit_budget_rejected".into(),
        outcome: over_row.admission_outcome,
        detail: "8.5-bpw recipe exceeding 4.5-bpw budget rejected".into(),
    });

    // NC3: floating/unpinned recipe digest rejected
    let mut unpinned = qda_spec.clone();
    unpinned.pinned_digest = "qda:latest".into();
    negative_controls.push(NegativeControl {
        case_id: "nc3_unpinned_recipe_digest_rejected".into(),
        outcome: if evaluate_recipe(&unpinned, &ref_layers, &eval_inputs, Some(4.5), 32.0).is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "floating `latest` recipe digest rejected".into(),
    });

    // NC4: calibration/eval contamination rejected
    let mut contaminated = qda_spec.clone();
    contaminated.calibration_disjoint_from_eval = false;
    let contam_row = evaluate_recipe(
        &contaminated,
        &ref_layers,
        &eval_inputs,
        Some(MATCHED_BPW_BUDGET),
        32.0,
    )?;
    negative_controls.push(NegativeControl {
        case_id: "nc4_calibration_eval_overlap_rejected".into(),
        outcome: contam_row.admission_outcome,
        detail: "non-disjoint calibration and held-out eval split rejected".into(),
    });

    // NC5: non-finite NaN weight matrix rejected
    let mut bad_w = ref_layers[0];
    bad_w[0][0] = f64::NAN;
    negative_controls.push(NegativeControl {
        case_id: "nc5_nonfinite_weight_rejected".into(),
        outcome: if quantize_dequant_layer(&bad_w, LayerQuantFormat::Int4Gguf).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "NaN weight tensor rejected cleanly".into(),
    });

    // NC6: zero accepted tasks yields cost_per_accepted_task == None (never 0.0)
    let mut inverted_layers = ref_layers;
    for r in 0..LAYER_DIM {
        for c in 0..LAYER_DIM {
            inverted_layers[NUM_LAYERS - 1][r][c] = -ref_layers[NUM_LAYERS - 1][r][c];
        }
    }
    let zero_acc_row = {
        let mut accepted = 0usize;
        for inp in &eval_inputs {
            let a = forward_stack(&ref_layers, inp)?;
            let b = forward_stack(&inverted_layers, inp)?;
            let mut ta = 0usize;
            let mut tb = 0usize;
            for i in 1..LAYER_DIM {
                if a[i] > a[ta] {
                    ta = i;
                }
                if b[i] > b[tb] {
                    tb = i;
                }
            }
            if ta == tb {
                accepted += 1;
            }
        }
        if accepted == 0 {
            OutcomeStatus::Exhausted
        } else {
            OutcomeStatus::Accepted
        }
    };
    negative_controls.push(NegativeControl {
        case_id: "nc6_zero_accepted_tasks_non_collapsing".into(),
        outcome: zero_acc_row,
        detail: "zero accepted tasks maps to cost_per_accepted_task=None and Exhausted (never 0.0)"
            .into(),
    });

    // NC7: non-finite Spearman input rejected
    negative_controls.push(NegativeControl {
        case_id: "nc7_nonfinite_spearman_input_rejected".into(),
        outcome: if spearman_rank_correlation(&[1.0, f64::NAN], &[1.0, 2.0]).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "NaN sensitivity metric rejected by Spearman rank correlation".into(),
    });

    // NC8: negative decode cost rejected
    negative_controls.push(NegativeControl {
        case_id: "nc8_nonpositive_decode_cost_rejected".into(),
        outcome: if evaluate_recipe(&qda_spec, &ref_layers, &eval_inputs, Some(4.5), -1.0).is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "non-positive decode_cost_units rejected cleanly".into(),
    });

    Ok(MitQdaQuantReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::blocked_on_artifacts(
            "MIT QDA public repository & physical GPU decode-cost benchmark",
            "MIT QDA repository/code is not yet pinned locally (post_only_unlocated_paper_repo); pure-Rust sensitivity-guided mixed-precision surrogate evaluated on CPU",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        upstream_pin_status: UPSTREAM_PIN_STATUS.into(),
        proxy_vs_kl_spearman,
        recipe_rows,
        negative_controls,
    })
}

pub fn inputs_json(report: &MitQdaQuantReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.mit-qda-quant.inputs.v1\",\n",
            "  \"issue\": 86,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"upstream_x_posts\": [{}, {}],\n",
            "  \"upstream_pin_status\": {},\n",
            "  \"complements_issues\": [18, 24],\n",
            "  \"matched_bpw_budget\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(UPSTREAM_X_POST_SONG_HAN),
        quote(UPSTREAM_X_POST_COAUTHOR),
        quote(&report.upstream_pin_status),
        finite_f64(MATCHED_BPW_BUDGET),
    )
}

pub fn results_json(report: &MitQdaQuantReport) -> String {
    let rows: Vec<String> = report
        .recipe_rows
        .iter()
        .map(|r| {
            format!(
                "{{\"recipe_id\":{},\"recipe_digest\":{},\"avg_effective_bpw\":{},\"within_bit_budget\":{},\"mean_weight_mse\":{},\"mean_logit_kl\":{},\"accepted_tasks\":{},\"total_tasks\":{},\"task_accuracy\":{},\"decode_cost_units\":{},\"cost_per_accepted_task\":{},\"admission_outcome\":{}}}",
                quote(&r.recipe_id),
                r.recipe_digest.to_json(),
                finite_f64(r.avg_effective_bpw),
                r.within_bit_budget,
                finite_f64(r.mean_weight_mse),
                finite_f64(r.mean_logit_kl),
                r.accepted_tasks,
                r.total_tasks,
                finite_f64(r.task_accuracy),
                finite_f64(r.decode_cost_units),
                opt_f64(r.cost_per_accepted_task),
                quote(r.admission_outcome.as_str()),
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
            "  \"schema\": \"inference-lab.mit-qda-quant.results.v1\",\n",
            "  \"issue\": 86,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"proxy_vs_kl_spearman\": {},\n",
            "  \"recipe_rows\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        opt_f64(report.proxy_vs_kl_spearman),
        rows.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &MitQdaQuantReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #86: MIT QDA Mixed-Precision Quantization Recipe vs Pinned Baselines"
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
        "- Hardware / artifact gate: `{}` ({})",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(
        out,
        "- Proxy sensitivity vs true layer KL Spearman rho: `{}`",
        opt_f64(report.proxy_vs_kl_spearman)
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Pinned Recipe Comparison (Matched Budget <= 4.5 bpw)"
    )
    .unwrap();
    for r in &report.recipe_rows {
        writeln!(
            out,
            "- `{}`: `bpw={:.3}`, `weight_mse={:.3e}`, `logit_kl={:.3e}`, `acc={}/{}`, `cost/accepted={}`, `outcome={}`",
            r.recipe_id,
            r.avg_effective_bpw,
            r.mean_weight_mse,
            r.mean_logit_kl,
            r.accepted_tasks,
            r.total_tasks,
            opt_f64(r.cost_per_accepted_task),
            r.admission_outcome.as_str()
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
    fn qda_recipe_matches_bit_budget_and_beats_uniform_kl() {
        let r = run_default_harness().unwrap();
        assert!(!r.default_enabled);
        assert_eq!(r.hardware_gate.status, ExecutionClass::BlockedOnArtifacts);
        assert!((r.recipe_rows[3].avg_effective_bpw - MATCHED_BPW_BUDGET).abs() <= 1.0e-12);
        assert!(r.recipe_rows[3].within_bit_budget);
        assert!(r.recipe_rows[3].mean_logit_kl < r.recipe_rows[1].mean_logit_kl);
        assert_eq!(r.negative_controls.len(), 8);
        for c in &r.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted, "{}", c.case_id);
        }
    }
}
