//! Issue #77: Add the Qwen3.5-4B two-neuron `down_proj` steering recipe as a default-off
//! reproducible intervention regression fixture.
//!
//! Pinned related work (strictly non-gating external notes):
//! - X post (2026-10-24, lyra `@_lyraaaa_`):
//!   `https://x.com/_lyraaaa_/status/2113833330429083662`
//! - Exact recipe recorded: two-neuron steering on Qwen3.5-4B:
//!   - `L14 N8949` with sign `+1`
//!   - `L13 N7956` with sign `-1`
//!   - Scaled at `down_proj` input by `delta = sign * dose * median(||resid||) / ||W_down[:, n]||`.
//! - Directly extends `#45` (`src/intervention_replay.rs`) with immutable manifests, exact
//!   KL/L1/top-1 comparison, matched-L2-norm random-direction controls, and deterministic replay
//!   receipt verification.
//!
//! Provides:
//! - Pure-Rust miniature multi-layer SwiGLU/MLP Qwen3.5-style fixture preserving the exact symbolic
//!   recipe provenance (`L14 N8949 +1`, `L13 N7956 -1`) mapped deterministically onto the tiny
//!   fixture's layers & intermediate `down_proj` columns.
//! - Exact implementation of `compute_median_resid_norm`, `down_proj_column_l2_norm`, and the
//!   scaling equation `sign * dose * median(||resid||) / ||W_down[:, n]||`.
//! - Dose-response sweep (`dose in {0.0, 0.25, 0.5, 1.0, 1.5, 2.0}`) comparing two-neuron steering
//!   against unsteered control (`dose = 0.0`) and matched-L2-norm random-direction control, with
//!   bit-identical replay verification on every dose point.
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, softmax_stable};
use crate::intervention_replay::{kl_divergence, l1_distance, ModelBackendArch, INTERP_VOCAB};
use crate::json::{finite_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const UPSTREAM_X_POST: &str = "https://x.com/_lyraaaa_/status/2113833330429083662";
pub const TARGET_MODEL_ID: &str = "Qwen3.5-4B";
pub const FIXTURE_REVISION: &str = "qwen3.5-4b-tiny-down-proj-fixture@sha256:7701a2b3";

pub const FIXTURE_LAYERS: usize = 4;
pub const D_MODEL: usize = 8;
pub const D_INTERMEDIATE: usize = 12;

fn check_finite(v: f64, ctx: &str) -> Result<f64, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("non-finite float in {ctx}"))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DownProjNeuronTarget {
    pub qwen35_symbolic_layer: usize,
    pub qwen35_symbolic_neuron: usize,
    pub fixture_layer: usize,
    pub fixture_neuron: usize,
    pub sign: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TwoNeuronSteeringSpec {
    pub experiment_id: String,
    pub target_arch: ModelBackendArch,
    pub model_revision_digest: String,
    pub seed: u64,
    pub prompt_tokens: Vec<u32>,
    pub dose: f64,
    pub targets: [DownProjNeuronTarget; 2],
    pub random_direction_control: bool,
}

impl TwoNeuronSteeringSpec {
    pub fn default_recipe(dose: f64, random_direction_control: bool) -> Self {
        Self {
            experiment_id: if random_direction_control {
                format!("qwen35-two-neuron-random-control-dose-{dose:.2}")
            } else {
                format!("qwen35-two-neuron-down-proj-dose-{dose:.2}")
            },
            target_arch: ModelBackendArch::QwenTransformer,
            model_revision_digest: FIXTURE_REVISION.into(),
            seed: 20261124,
            prompt_tokens: vec![1, 5, 2, 7, 3, 6],
            dose,
            targets: [
                DownProjNeuronTarget {
                    qwen35_symbolic_layer: 13,
                    qwen35_symbolic_neuron: 7956,
                    fixture_layer: 1,
                    fixture_neuron: 4,
                    sign: -1.0,
                },
                DownProjNeuronTarget {
                    qwen35_symbolic_layer: 14,
                    qwen35_symbolic_neuron: 8949,
                    fixture_layer: 2,
                    fixture_neuron: 9,
                    sign: 1.0,
                },
            ],
            random_direction_control,
        }
    }

    pub fn manifest_digest(&self) -> DigestRef {
        let repr = format!(
            "{}|{}|{}|{}|{:?}|{:.12}|L{}N{}({:+.0})|L{}N{}({:+.0})|rand={}",
            self.experiment_id,
            self.target_arch.as_str(),
            self.model_revision_digest,
            self.seed,
            self.prompt_tokens,
            self.dose,
            self.targets[0].qwen35_symbolic_layer,
            self.targets[0].qwen35_symbolic_neuron,
            self.targets[0].sign,
            self.targets[1].qwen35_symbolic_layer,
            self.targets[1].qwen35_symbolic_neuron,
            self.targets[1].sign,
            self.random_direction_control
        );
        DigestRef::sha256(repr.as_bytes())
    }
}

pub fn validate_steering_spec(spec: &TwoNeuronSteeringSpec) -> Result<(), String> {
    if spec.model_revision_digest.is_empty() || spec.model_revision_digest.contains("latest") {
        return Err("unpinned or mutable model revision rejected".into());
    }
    if spec.target_arch != ModelBackendArch::QwenTransformer {
        return Err(format!(
            "Qwen3.5 two-neuron IDs cannot transfer to `{}`",
            spec.target_arch.as_str()
        ));
    }
    if spec.prompt_tokens.is_empty() {
        return Err("empty prompt_tokens".into());
    }
    let d = check_finite(spec.dose, "dose")?;
    if d < 0.0 {
        return Err("negative steering dose rejected".into());
    }
    for t in &spec.targets {
        if t.fixture_layer >= FIXTURE_LAYERS || t.fixture_neuron >= D_INTERMEDIATE {
            return Err("target fixture layer or neuron out of bounds".into());
        }
        let s = check_finite(t.sign, "sign")?;
        if (s - 1.0).abs() > 1.0e-12 && (s + 1.0).abs() > 1.0e-12 {
            return Err("steering sign must be exact +1 or -1".into());
        }
    }
    Ok(())
}

pub fn w_down_entry(layer: usize, row_d: usize, col_n: usize) -> f64 {
    det_weight(7730 + layer * 97 + row_d * 13, col_n, 0.45)
}

pub fn down_proj_column_l2_norm(layer: usize, col_n: usize) -> Result<f64, String> {
    if layer >= FIXTURE_LAYERS || col_n >= D_INTERMEDIATE {
        return Err("down_proj column index out of bounds".into());
    }
    let mut sq = 0.0_f64;
    for row_d in 0..D_MODEL {
        let w = w_down_entry(layer, row_d, col_n);
        sq += w * w;
    }
    let norm = sq.sqrt();
    if !norm.is_finite() || norm <= 1.0e-12 {
        return Err("degenerate zero down_proj column norm".into());
    }
    Ok(norm)
}

pub fn compute_median_l2_norm(vectors: &[[f64; D_MODEL]]) -> Result<f64, String> {
    if vectors.is_empty() {
        return Err("empty residual stream for median norm".into());
    }
    let mut norms = Vec::with_capacity(vectors.len());
    for v in vectors {
        let mut sq = 0.0_f64;
        for &x in v {
            let xf = check_finite(x, "resid_x")?;
            sq += xf * xf;
        }
        norms.push(sq.sqrt());
    }
    norms.sort_by(|a, b| a.total_cmp(b));
    let mid = norms.len() / 2;
    let med = if norms.len() % 2 == 1 {
        norms[mid]
    } else {
        0.5 * (norms[mid - 1] + norms[mid])
    };
    check_finite(med, "median_resid_norm")
}

/// Exact `#77` formula: `sign * dose * median(||resid||) / ||W_down[:, n]||`.
pub fn exact_two_neuron_delta(
    sign: f64,
    dose: f64,
    median_resid_norm: f64,
    w_down_col_norm: f64,
) -> Result<f64, String> {
    let s = check_finite(sign, "sign")?;
    let d = check_finite(dose, "dose")?;
    let m = check_finite(median_resid_norm, "median_resid_norm")?;
    let w = check_finite(w_down_col_norm, "w_down_col_norm")?;
    if w <= 1.0e-12 {
        return Err("zero down_proj column norm divisor".into());
    }
    if m < 0.0 || d < 0.0 {
        return Err("negative dose or median residual norm".into());
    }
    check_finite(s * d * m / w, "steering_delta")
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

#[derive(Debug, Clone)]
pub struct SteeringRunTrace {
    pub manifest_digest: DigestRef,
    pub logits: [f64; INTERP_VOCAB],
    pub probs: [f64; INTERP_VOCAB],
    pub top1_token: u32,
    pub applied_deltas: [f64; 2],
    pub applied_resid_shift_norms: [f64; 2],
    pub trace_digest: DigestRef,
}

pub fn run_steering_fixture(spec: &TwoNeuronSteeringSpec) -> Result<SteeringRunTrace, String> {
    validate_steering_spec(spec)?;
    let seq_len = spec.prompt_tokens.len();
    let mut h: Vec<[f64; D_MODEL]> = spec
        .prompt_tokens
        .iter()
        .enumerate()
        .map(|(pos, &tok)| {
            let mut v = [0.0_f64; D_MODEL];
            for (d, slot) in v.iter_mut().enumerate() {
                *slot = det_weight(7701 + (tok as usize) * 17 + pos * 5, d, 0.6);
            }
            v
        })
        .collect();

    let mut applied_deltas = [0.0_f64; 2];
    let mut applied_resid_shift_norms = [0.0_f64; 2];

    for layer in 0..FIXTURE_LAYERS {
        // Compute residual stream median L2 norm BEFORE down_proj steering at this layer.
        let med_resid = compute_median_l2_norm(&h)?;
        let mut next_h = vec![[0.0_f64; D_MODEL]; seq_len];

        // Check if one of the two recipe targets hits this layer.
        let mut layer_target_idx: Option<usize> = None;
        for (ti, t) in spec.targets.iter().enumerate() {
            if t.fixture_layer == layer {
                layer_target_idx = Some(ti);
            }
        }

        let mut delta_scalar = 0.0_f64;
        let mut rand_dir = [0.0_f64; D_MODEL];
        if let Some(ti) = layer_target_idx {
            let tgt = &spec.targets[ti];
            let col_norm = down_proj_column_l2_norm(layer, tgt.fixture_neuron)?;
            delta_scalar = exact_two_neuron_delta(tgt.sign, spec.dose, med_resid, col_norm)?;
            applied_deltas[ti] = delta_scalar;
            // Notice ||delta * W_down[:, n]|| == |dose * median(||resid||)| by construction!
            let target_resid_shift_norm = (delta_scalar * col_norm).abs();
            applied_resid_shift_norms[ti] = target_resid_shift_norm;

            if spec.random_direction_control && target_resid_shift_norm > 0.0 {
                let mut sq = 0.0_f64;
                for (d, slot) in rand_dir.iter_mut().enumerate() {
                    *slot = det_weight(7790 + (spec.seed as usize % 97) * 7 + layer * 19, d, 1.0);
                    sq += *slot * *slot;
                }
                let rnorm = sq.sqrt();
                for slot in &mut rand_dir {
                    *slot = *slot / rnorm * target_resid_shift_norm;
                }
            }
        }

        for t in 0..seq_len {
            // Causal self-attention
            let mut scores = Vec::with_capacity(t + 1);
            for hs in h.iter().take(t + 1) {
                let dot: f64 = h[t].iter().zip(hs.iter()).map(|(a, b)| a * b).sum();
                scores.push(dot / (D_MODEL as f64).sqrt());
            }
            let attn_w = softmax_stable(&scores)?;
            let mut ctx = [0.0_f64; D_MODEL];
            for (s, &w) in attn_w.iter().enumerate() {
                for (c, &hs) in ctx.iter_mut().zip(h[s].iter()) {
                    *c += w * hs;
                }
            }

            // SwiGLU intermediate activation
            let mut inter = [0.0_f64; D_INTERMEDIATE];
            for (n, slot) in inter.iter_mut().enumerate() {
                let mut gate_acc = 0.0_f64;
                let mut up_acc = 0.0_f64;
                for (i, &cv) in ctx.iter().enumerate() {
                    gate_acc += cv * det_weight(7710 + layer * 41 + n * 5, i, 0.5);
                    up_acc += cv * det_weight(7720 + layer * 43 + n * 7, i, 0.5);
                }
                *slot = silu(gate_acc) * up_acc;
            }

            // Apply targeted two-neuron steering directly to the down_proj neuron input!
            if let Some(ti) = layer_target_idx {
                if !spec.random_direction_control {
                    let n = spec.targets[ti].fixture_neuron;
                    inter[n] += delta_scalar;
                }
            }

            // down_proj: D_INTERMEDIATE -> D_MODEL
            let mut mlp_out = [0.0_f64; D_MODEL];
            for (row_d, slot) in mlp_out.iter_mut().enumerate() {
                let mut acc = 0.0_f64;
                for (col_n, &iv) in inter.iter().enumerate() {
                    acc += w_down_entry(layer, row_d, col_n) * iv;
                }
                if layer_target_idx.is_some() && spec.random_direction_control {
                    acc += rand_dir[row_d];
                }
                *slot = acc;
            }

            for ((dst, &ht), &mo) in next_h[t].iter_mut().zip(h[t].iter()).zip(mlp_out.iter()) {
                *dst = check_finite(ht + 0.5 * mo, "next_h")?;
            }
        }
        h = next_h;
    }

    let last = h[seq_len - 1];
    let mut logits = [0.0_f64; INTERP_VOCAB];
    for (v, slot) in logits.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for (d, &hv) in last.iter().enumerate() {
            acc += hv * det_weight(7780 + v * 11, d, 0.9);
        }
        *slot = check_finite(acc, "steering_logit")?;
    }
    let probs_v = softmax_stable(&logits)?;
    let mut probs = [0.0_f64; INTERP_VOCAB];
    let mut best_i = 0usize;
    let mut best_p = f64::NEG_INFINITY;
    for i in 0..INTERP_VOCAB {
        probs[i] = check_finite(probs_v[i], "steering_prob")?;
        if probs[i] > best_p {
            best_p = probs[i];
            best_i = i;
        }
    }

    let manifest_digest = spec.manifest_digest();
    let mut raw = String::new();
    let _ = write!(raw, "{}|{}|", manifest_digest.hex, spec.seed);
    for l in &logits {
        let _ = write!(raw, "{l:.15e},");
    }
    Ok(SteeringRunTrace {
        manifest_digest,
        logits,
        probs,
        top1_token: best_i as u32,
        applied_deltas,
        applied_resid_shift_norms,
        trace_digest: DigestRef::sha256(raw.as_bytes()),
    })
}

pub fn verify_steering_replay(
    spec: &TwoNeuronSteeringSpec,
    expected_digest: &DigestRef,
) -> Result<OutcomeStatus, String> {
    let replayed = run_steering_fixture(spec)?;
    if &replayed.trace_digest == expected_digest {
        Ok(OutcomeStatus::Accepted)
    } else {
        Ok(OutcomeStatus::Invalid)
    }
}

#[derive(Debug, Clone)]
pub struct DoseResponseRow {
    pub dose: f64,
    pub targeted_kl_vs_unsteered: f64,
    pub targeted_l1_vs_unsteered: f64,
    pub targeted_top1: u32,
    pub targeted_replay_exact: bool,
    pub random_ctrl_kl_vs_unsteered: f64,
    pub random_ctrl_l1_vs_unsteered: f64,
    pub random_ctrl_top1: u32,
    pub random_ctrl_replay_exact: bool,
    pub resid_shift_norm_l13: f64,
    pub resid_shift_norm_l14: f64,
    pub trace_digest: DigestRef,
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct Qwen35SteeringReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub target_model_id: String,
    pub unsteered_top1: u32,
    pub dose_response: Vec<DoseResponseRow>,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<Qwen35SteeringReport, String> {
    let base_spec = TwoNeuronSteeringSpec::default_recipe(0.0, false);
    let base = run_steering_fixture(&base_spec)?;

    let doses = [0.0_f64, 0.25, 0.5, 1.0, 1.5, 2.0];
    let mut dose_response = Vec::with_capacity(doses.len());
    for dose in doses {
        let tgt_spec = TwoNeuronSteeringSpec::default_recipe(dose, false);
        let rand_spec = TwoNeuronSteeringSpec::default_recipe(dose, true);
        let tgt = run_steering_fixture(&tgt_spec)?;
        let rand = run_steering_fixture(&rand_spec)?;
        let tgt_replay_ok =
            verify_steering_replay(&tgt_spec, &tgt.trace_digest)? == OutcomeStatus::Accepted;
        let rand_replay_ok =
            verify_steering_replay(&rand_spec, &rand.trace_digest)? == OutcomeStatus::Accepted;
        dose_response.push(DoseResponseRow {
            dose,
            targeted_kl_vs_unsteered: kl_divergence(&base.probs, &tgt.probs)?,
            targeted_l1_vs_unsteered: l1_distance(&base.probs, &tgt.probs)?,
            targeted_top1: tgt.top1_token,
            targeted_replay_exact: tgt_replay_ok,
            random_ctrl_kl_vs_unsteered: kl_divergence(&base.probs, &rand.probs)?,
            random_ctrl_l1_vs_unsteered: l1_distance(&base.probs, &rand.probs)?,
            random_ctrl_top1: rand.top1_token,
            random_ctrl_replay_exact: rand_replay_ok,
            resid_shift_norm_l13: tgt.applied_resid_shift_norms[0],
            resid_shift_norm_l14: tgt.applied_resid_shift_norms[1],
            trace_digest: tgt.trace_digest,
        });
    }

    let mut negative_controls = Vec::new();

    // NC1: negative steering dose rejected
    {
        let bad = TwoNeuronSteeringSpec::default_recipe(-0.5, false);
        negative_controls.push(NegativeControl {
            case_id: "nc1_negative_steering_dose_rejected".into(),
            outcome: if validate_steering_spec(&bad).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "negative steering dose refused; sign lives on explicit +1/-1 target".into(),
        });
    }

    // NC2: NaN steering dose rejected
    {
        let bad = TwoNeuronSteeringSpec::default_recipe(f64::NAN, false);
        negative_controls.push(NegativeControl {
            case_id: "nc2_nan_steering_dose_rejected".into(),
            outcome: if run_steering_fixture(&bad).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "non-finite NaN dose rejected cleanly".into(),
        });
    }

    // NC3: cross-architecture neuron-ID transfer to recurrent model rejected
    {
        let mut bad = TwoNeuronSteeringSpec::default_recipe(1.0, false);
        bad.target_arch = ModelBackendArch::RecurrentStateful;
        negative_controls.push(NegativeControl {
            case_id: "nc3_cross_arch_qwen35_neuron_transfer_rejected".into(),
            outcome: if validate_steering_spec(&bad).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "Qwen3.5-4B neuron IDs (L14 N8949, L13 N7956) refused on non-Qwen arch".into(),
        });
    }

    // NC4: zero W_down column norm divisor rejected cleanly
    {
        negative_controls.push(NegativeControl {
            case_id: "nc4_zero_w_down_col_norm_divisor_rejected".into(),
            outcome: if exact_two_neuron_delta(1.0, 1.0, 2.0, 0.0).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "zero ||W_down[:, n]|| divisor refused instead of producing Inf/NaN".into(),
        });
    }

    // NC5: mutable `latest` checkpoint tag rejected
    {
        let mut bad = TwoNeuronSteeringSpec::default_recipe(1.0, false);
        bad.model_revision_digest = "Qwen3.5-4B:latest".into();
        negative_controls.push(NegativeControl {
            case_id: "nc5_mutable_latest_model_revision_rejected".into(),
            outcome: if validate_steering_spec(&bad).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "floating `latest` model revision fails immutable provenance check".into(),
        });
    }

    // NC6: tampered replay receipt detected by real re-execution
    {
        let spec = TwoNeuronSteeringSpec::default_recipe(1.0, false);
        let tampered = DigestRef::sha256(b"tampered-qwen35-steering-trace");
        let st = verify_steering_replay(&spec, &tampered)?;
        negative_controls.push(NegativeControl {
            case_id: "nc6_tampered_replay_receipt_detected".into(),
            outcome: st,
            detail: "re-execution against tampered SHA-256 receipt returns Invalid".into(),
        });
    }

    // NC7: non-unit sign (+2.0) rejected
    {
        let mut bad = TwoNeuronSteeringSpec::default_recipe(1.0, false);
        bad.targets[0].sign = 2.0;
        negative_controls.push(NegativeControl {
            case_id: "nc7_non_unit_steering_sign_rejected".into(),
            outcome: if validate_steering_spec(&bad).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "steering sign outside {-1, +1} rejected to preserve exact recipe formula"
                .into(),
        });
    }

    // NC8: empty residual stream for median norm rejected
    {
        negative_controls.push(NegativeControl {
            case_id: "nc8_empty_residual_stream_median_rejected".into(),
            outcome: if compute_median_l2_norm(&[]).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "empty residual stream cannot produce a silent 0.0 median norm".into(),
        });
    }

    Ok(Qwen35SteeringReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "Local Qwen3.5-4B pinned checkpoint serving host (reusing #45 intervention-replay)",
            "Pure-Rust tiny SwiGLU/MLP fixture verified exact two-neuron down_proj scaling formula, deterministic replay, and dose-response vs random-direction control on CPU; physical Qwen3.5-4B weights did not execute locally",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        target_model_id: TARGET_MODEL_ID.into(),
        unsteered_top1: base.top1_token,
        dose_response,
        negative_controls,
    })
}

pub fn inputs_json(report: &Qwen35SteeringReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.qwen35-neuron-steering.inputs.v1\",\n",
            "  \"issue\": 77,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"target_model_id\": {},\n",
            "  \"upstream_x_post\": {},\n",
            "  \"recipe_targets\": [\"L14 N8949 +1\", \"L13 N7956 -1\"],\n",
            "  \"scaling_formula\": \"sign * dose * median(||resid||) / ||W_down[:, n]||\",\n",
            "  \"related_issues\": [45]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(&report.target_model_id),
        quote(UPSTREAM_X_POST),
    )
}

pub fn results_json(report: &Qwen35SteeringReport) -> String {
    let rows: Vec<String> = report
        .dose_response
        .iter()
        .map(|r| {
            format!(
                "{{\"dose\":{},\"targeted_kl_vs_unsteered\":{},\"targeted_l1_vs_unsteered\":{},\"targeted_top1\":{},\"targeted_replay_exact\":{},\"random_ctrl_kl_vs_unsteered\":{},\"random_ctrl_l1_vs_unsteered\":{},\"random_ctrl_top1\":{},\"random_ctrl_replay_exact\":{},\"resid_shift_norm_l13\":{},\"resid_shift_norm_l14\":{},\"trace_digest\":{}}}",
                finite_f64(r.dose),
                finite_f64(r.targeted_kl_vs_unsteered),
                finite_f64(r.targeted_l1_vs_unsteered),
                r.targeted_top1,
                r.targeted_replay_exact,
                finite_f64(r.random_ctrl_kl_vs_unsteered),
                finite_f64(r.random_ctrl_l1_vs_unsteered),
                r.random_ctrl_top1,
                r.random_ctrl_replay_exact,
                finite_f64(r.resid_shift_norm_l13),
                finite_f64(r.resid_shift_norm_l14),
                r.trace_digest.to_json(),
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
            "  \"schema\": \"inference-lab.qwen35-neuron-steering.results.v1\",\n",
            "  \"issue\": 77,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"unsteered_top1\": {},\n",
            "  \"dose_response\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        report.unsteered_top1,
        rows.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &Qwen35SteeringReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #77: Qwen3.5-4B Two-Neuron `down_proj` Steering Regression Fixture"
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
        "- Recipe: `L14 N8949 (+1)`, `L13 N7956 (-1)`, `delta = sign * dose * median(||resid||) / ||W_down[:, n]||`"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Dose-Response vs Unsteered & Matched-Norm Random Control"
    )
    .unwrap();
    writeln!(
        out,
        "| Dose | Targeted KL | Targeted L1 | Top1 | Random Ctrl KL | Random Ctrl L1 | Exact Replay |"
    )
    .unwrap();
    writeln!(out, "| ---: | ---: | ---: | ---: | ---: | ---: | --- |").unwrap();
    for r in &report.dose_response {
        writeln!(
            out,
            "| `{:.2}` | `{:.6}` | `{:.6}` | {} | `{:.6}` | `{:.6}` | `{}` |",
            r.dose,
            r.targeted_kl_vs_unsteered,
            r.targeted_l1_vs_unsteered,
            r.targeted_top1,
            r.random_ctrl_kl_vs_unsteered,
            r.random_ctrl_l1_vs_unsteered,
            r.targeted_replay_exact && r.random_ctrl_replay_exact
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
    fn dose_zero_matches_unsteered_and_higher_doses_induce_deterministic_drift() {
        let r = run_default_harness().unwrap();
        assert!(!r.default_enabled);
        assert!(r.dose_response[0].targeted_kl_vs_unsteered <= 1.0e-12);
        assert!(r.dose_response[0].targeted_l1_vs_unsteered <= 1.0e-12);
        assert!(r.dose_response[5].targeted_kl_vs_unsteered > 0.01);
        for row in &r.dose_response {
            assert!(row.targeted_replay_exact);
            assert!(row.random_ctrl_replay_exact);
        }
        assert_eq!(r.negative_controls.len(), 8);
        for c in &r.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }
}
