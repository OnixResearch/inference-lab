//! Issue #92: Benchmark sparse-MoE speculative verification roofline and temporal expert-routing
//! overlap (`arXiv:2610.38981`, `QuantSpec` `arXiv:2502.10424`, `MoE-SpeQ`), default-off.
//!
//! Related-work references (strictly non-gating external notes):
//! - `arXiv:2610.38981`: shows speculative decoding on 30B–1T sparse MoEs beats naive dense
//!   roofline predictions via temporal expert-routing overlap across consecutive draft tokens.
//! - `QuantSpec` (`arXiv:2502.10424`) & `MoE-SpeQ`: analytical roofline governors paired with
//!   quantized KV caches and proactive expert prefetching.
//! - Directly informs `#6` (`adaptive-spec`), `#21` (`sparse-moe`), and `#65` (`spark-copydrafter`).
//!
//! Provides:
//! - Pure-Rust sparse-MoE top-`k` router & dispatch engine (`NUM_EXPERTS = 16`, `TOP_K = 2`,
//!   `D_MODEL = 8`) over consecutive draft blocks (`gamma = 1, 2, 4, 8`) measuring:
//!   - Exact **unique expert union** `|U(gamma)|` vs naive non-overlapping dense-style bound
//!     `min(E, gamma * top_k)` and **temporal expert-routing overlap ratio**
//!     `1 - |U(gamma)| / (gamma * top_k)`.
//!   - Exact single-token vs batched speculative-verification MoE output parity (`<= 1e-12`).
//! - Analytical **memory-bandwidth + compute + KV-cache roofline governor** comparing naive dense
//!   roofline predictions against overlap-aware sparse-MoE + QuantSpec FP8/INT4 KV rooflines to
//!   select optimal draft length `gamma*` under hard KV memory and HBM bandwidth bounds.
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, softmax_stable};
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const MOE_SPEC_ARXIV: &str = "arXiv:2610.38981";
pub const QUANTSPEC_ARXIV: &str = "arXiv:2502.10424";
pub const NUM_EXPERTS: usize = 16;
pub const TOP_K: usize = 2;
pub const D_MODEL: usize = 8;
pub const PARITY_TOL: f64 = 1.0e-12;

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

/// Deterministic top-k sparse MoE routing for one hidden token vector.
pub fn route_token_top_k(
    hidden: &[f64; D_MODEL],
    router_weights: &[[f64; D_MODEL]; NUM_EXPERTS],
    top_k: usize,
) -> Result<Vec<(usize, f64)>, String> {
    if top_k == 0 || top_k > NUM_EXPERTS {
        return Err(format!(
            "invalid top_k={top_k} for NUM_EXPERTS={NUM_EXPERTS}"
        ));
    }
    let mut logits = [0.0_f64; NUM_EXPERTS];
    for (e, slot) in logits.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for (&hd, &rwd) in hidden.iter().zip(router_weights[e].iter()) {
            acc += check_finite(hd, "h")? * check_finite(rwd, "rw")?;
        }
        *slot = check_finite(acc, "router_logit")?;
    }
    let mut scored: Vec<(usize, f64)> = logits.iter().copied().enumerate().collect();
    scored.sort_by(|a, b| {
        if (b.1 - a.1).abs() <= 1.0e-15 {
            a.0.cmp(&b.0)
        } else if b.1 > a.1 {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        }
    });
    let chosen: Vec<(usize, f64)> = scored.into_iter().take(top_k).collect();
    let top_logits: Vec<f64> = chosen.iter().map(|(_, s)| *s).collect();
    let probs = softmax_stable(&top_logits)?;
    Ok(chosen
        .into_iter()
        .zip(probs)
        .map(|((e, _), p)| (e, p))
        .collect())
}

fn apply_expert(expert_id: usize, x: &[f64; D_MODEL]) -> Result<[f64; D_MODEL], String> {
    let mut out = [0.0_f64; D_MODEL];
    for (o, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for (i, &xv) in x.iter().enumerate() {
            acc += xv * det_weight(9240 + expert_id * 29 + o * 5, i, 0.50);
        }
        *slot = check_finite(acc.tanh(), "expert_out")?;
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct DraftOverlapObservation {
    pub draft_len_gamma: usize,
    pub total_expert_slots: usize,
    pub naive_no_overlap_experts: usize,
    pub observed_unique_experts: usize,
    pub temporal_overlap_ratio: f64,
    pub batched_vs_sequential_max_err: f64,
}

pub fn measure_draft_block_overlap(
    draft_tokens: &[[f64; D_MODEL]],
    router_weights: &[[f64; D_MODEL]; NUM_EXPERTS],
    top_k: usize,
) -> Result<DraftOverlapObservation, String> {
    if draft_tokens.is_empty() {
        return Err("draft_tokens must be non-empty".into());
    }
    let gamma = draft_tokens.len();
    let mut active_mask = [false; NUM_EXPERTS];
    let mut seq_outputs = Vec::with_capacity(gamma);
    let mut routes = Vec::with_capacity(gamma);

    for tok in draft_tokens {
        let r = route_token_top_k(tok, router_weights, top_k)?;
        let mut y = [0.0_f64; D_MODEL];
        for &(e, w) in &r {
            active_mask[e] = true;
            let ey = apply_expert(e, tok)?;
            for d in 0..D_MODEL {
                y[d] += w * ey[d];
            }
        }
        seq_outputs.push(y);
        routes.push(r);
    }

    // Batched expert-grouped verification dispatch (reads each unique expert in U(gamma) once!)
    let mut batched_outputs = vec![[0.0_f64; D_MODEL]; gamma];
    for (e, &is_active) in active_mask.iter().enumerate() {
        if !is_active {
            continue;
        }
        for (t, r) in routes.iter().enumerate() {
            for &(re, w) in r {
                if re == e {
                    let ey = apply_expert(e, &draft_tokens[t])?;
                    for (dst, &val) in batched_outputs[t].iter_mut().zip(ey.iter()) {
                        *dst += w * val;
                    }
                }
            }
        }
    }

    let mut max_err = 0.0_f64;
    for (a, b) in seq_outputs.iter().zip(batched_outputs.iter()) {
        for (&x, &y) in a.iter().zip(b.iter()) {
            max_err = max_finite(max_err, (x - y).abs(), "moe_verify_err")?;
        }
    }

    let unique = active_mask.iter().filter(|&&b| b).count();
    let total_slots = gamma * top_k;
    let naive_bound = total_slots.min(NUM_EXPERTS);
    let overlap_ratio = 1.0 - (unique as f64) / (total_slots as f64);

    Ok(DraftOverlapObservation {
        draft_len_gamma: gamma,
        total_expert_slots: total_slots,
        naive_no_overlap_experts: naive_bound,
        observed_unique_experts: unique,
        temporal_overlap_ratio: overlap_ratio,
        batched_vs_sequential_max_err: max_err,
    })
}

#[derive(Debug, Clone)]
pub struct RooflineGovernorConfig {
    pub profile_id: String,
    pub hbm_bandwidth_gb_s: f64,
    pub peak_tflops: f64,
    pub vram_budget_gib: f64,
    pub resident_weights_gib: f64,
    pub bytes_per_expert_mb: f64,
    pub kv_bytes_per_token_kb: f64,
    pub context_tokens: usize,
    pub token_accept_prob_alpha: f64,
    pub kv_parity_passed: bool,
}

#[derive(Debug, Clone)]
pub struct RooflineGovernorPoint {
    pub profile_id: String,
    pub draft_len_gamma: usize,
    pub expected_accepted_tokens: f64,
    pub naive_dense_verify_ms: f64,
    pub overlap_aware_verify_ms: f64,
    pub naive_tok_per_s: f64,
    pub overlap_aware_tok_per_s: f64,
    pub total_vram_gib: f64,
    pub outcome: OutcomeStatus,
}

/// Analytical geometric expected accepted tokens: `E[L] = (1 - alpha^(gamma + 1)) / (1 - alpha)`.
pub fn expected_accepted_tokens(gamma: usize, alpha: f64) -> Result<f64, String> {
    if gamma == 0 {
        return Err("gamma must be >= 1".into());
    }
    let a = check_finite(alpha, "alpha")?;
    if !(0.0..=1.0).contains(&a) {
        return Err("acceptance probability alpha must be in [0, 1]".into());
    }
    if (1.0 - a).abs() <= 1.0e-12 {
        return Ok((gamma + 1) as f64);
    }
    Ok((1.0 - a.powi((gamma + 1) as i32)) / (1.0 - a))
}

pub fn evaluate_roofline_point(
    cfg: &RooflineGovernorConfig,
    obs: &DraftOverlapObservation,
) -> Result<RooflineGovernorPoint, String> {
    let bw = check_finite(cfg.hbm_bandwidth_gb_s, "hbm_bandwidth_gb_s")?;
    let tflops = check_finite(cfg.peak_tflops, "peak_tflops")?;
    let vram_lim = check_finite(cfg.vram_budget_gib, "vram_budget_gib")?;
    let res_w = check_finite(cfg.resident_weights_gib, "resident_weights_gib")?;
    let exp_mb = check_finite(cfg.bytes_per_expert_mb, "bytes_per_expert_mb")?;
    let kv_kb = check_finite(cfg.kv_bytes_per_token_kb, "kv_bytes_per_token_kb")?;
    if bw <= 0.0 || tflops <= 0.0 || vram_lim <= 0.0 || res_w < 0.0 || exp_mb <= 0.0 || kv_kb <= 0.0
    {
        return Err("invalid non-positive roofline parameter".into());
    }
    let expected_acc = expected_accepted_tokens(obs.draft_len_gamma, cfg.token_accept_prob_alpha)?;
    let kv_total_mb = (cfg.context_tokens as f64) * kv_kb / 1024.0;
    let total_vram_gib = res_w + kv_total_mb / 1024.0;

    let naive_bytes_mb = (obs.naive_no_overlap_experts as f64) * exp_mb + kv_total_mb;
    let overlap_bytes_mb = (obs.observed_unique_experts as f64) * exp_mb + kv_total_mb;
    // Bandwidth time (ms) = MB / (GB/s)
    let naive_bw_ms = naive_bytes_mb / bw;
    let overlap_bw_ms = overlap_bytes_mb / bw;
    // Compute time (ms): 2 * flops_per_active_slot
    let compute_ms = ((obs.total_expert_slots as f64) * exp_mb * 2.0e-3) / tflops;
    let naive_ms = max_finite(naive_bw_ms, compute_ms, "naive_ms")?;
    let overlap_ms = max_finite(overlap_bw_ms, compute_ms, "overlap_ms")?;

    let outcome = if !cfg.kv_parity_passed || obs.batched_vs_sequential_max_err > PARITY_TOL {
        OutcomeStatus::Invalid
    } else if total_vram_gib > vram_lim {
        OutcomeStatus::Oom
    } else {
        OutcomeStatus::Accepted
    };

    Ok(RooflineGovernorPoint {
        profile_id: cfg.profile_id.clone(),
        draft_len_gamma: obs.draft_len_gamma,
        expected_accepted_tokens: expected_acc,
        naive_dense_verify_ms: naive_ms,
        overlap_aware_verify_ms: overlap_ms,
        naive_tok_per_s: expected_acc * 1000.0 / naive_ms,
        overlap_aware_tok_per_s: expected_acc * 1000.0 / overlap_ms,
        total_vram_gib,
        outcome,
    })
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct MoeSpecRooflineReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub overlap_observations: Vec<DraftOverlapObservation>,
    pub roofline_points: Vec<RooflineGovernorPoint>,
    pub recommended_gamma_fp16_kv: Option<usize>,
    pub recommended_gamma_quantspec_fp8_kv: Option<usize>,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<MoeSpecRooflineReport, String> {
    let mut router = [[0.0_f64; D_MODEL]; NUM_EXPERTS];
    for (e, row) in router.iter_mut().enumerate() {
        for (d, slot) in row.iter_mut().enumerate() {
            *slot = det_weight(9201 + e * 13, d, 0.70);
        }
    }
    // Consecutive draft tokens exhibit realistic temporal coherence (small drift between tokens)
    let mut base_h = [0.0_f64; D_MODEL];
    for (d, slot) in base_h.iter_mut().enumerate() {
        *slot = det_weight(9299, d, 1.0);
    }
    let mut draft_seq = Vec::with_capacity(8);
    for t in 0..8 {
        let mut h = base_h;
        for (d, slot) in h.iter_mut().enumerate() {
            *slot += det_weight(9260 + t * 11, d, 0.22);
        }
        draft_seq.push(h);
    }

    let mut overlap_observations = Vec::new();
    for &gamma in &[1usize, 2, 4, 8] {
        overlap_observations.push(measure_draft_block_overlap(
            &draft_seq[..gamma],
            &router,
            TOP_K,
        )?);
    }

    let cfg_fp16 = RooflineGovernorConfig {
        profile_id: "moe_30b_a3b_fp16_kv".into(),
        hbm_bandwidth_gb_s: 273.0, // DGX Spark GB10 LPDDR5X nominal reference envelope
        peak_tflops: 120.0,
        vram_budget_gib: 112.0,
        resident_weights_gib: 24.0,
        bytes_per_expert_mb: 160.0,
        kv_bytes_per_token_kb: 64.0,
        context_tokens: 32_768,
        token_accept_prob_alpha: 0.78,
        kv_parity_passed: true,
    };
    let cfg_fp8 = RooflineGovernorConfig {
        profile_id: "moe_30b_a3b_quantspec_fp8_kv".into(),
        kv_bytes_per_token_kb: 32.0,
        ..cfg_fp16.clone()
    };

    let mut roofline_points = Vec::new();
    let mut best_fp16: Option<(usize, f64)> = None;
    let mut best_fp8: Option<(usize, f64)> = None;

    for obs in &overlap_observations {
        let p16 = evaluate_roofline_point(&cfg_fp16, obs)?;
        if p16.outcome == OutcomeStatus::Accepted
            && best_fp16.is_none_or(|(_, tps)| p16.overlap_aware_tok_per_s > tps)
        {
            best_fp16 = Some((p16.draft_len_gamma, p16.overlap_aware_tok_per_s));
        }
        roofline_points.push(p16);

        let p8 = evaluate_roofline_point(&cfg_fp8, obs)?;
        if p8.outcome == OutcomeStatus::Accepted
            && best_fp8.is_none_or(|(_, tps)| p8.overlap_aware_tok_per_s > tps)
        {
            best_fp8 = Some((p8.draft_len_gamma, p8.overlap_aware_tok_per_s));
        }
        roofline_points.push(p8);
    }

    let mut negative_controls = Vec::new();

    // NC1: top_k == 0 rejected
    negative_controls.push(NegativeControl {
        case_id: "nc1_zero_top_k_rejected".into(),
        outcome: if route_token_top_k(&draft_seq[0], &router, 0).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "top_k = 0 rejected cleanly".into(),
    });

    // NC2: top_k > NUM_EXPERTS rejected
    negative_controls.push(NegativeControl {
        case_id: "nc2_oob_top_k_rejected".into(),
        outcome: if route_token_top_k(&draft_seq[0], &router, NUM_EXPERTS + 1).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "top_k > NUM_EXPERTS rejected".into(),
    });

    // NC3: gamma == 0 rejected
    negative_controls.push(NegativeControl {
        case_id: "nc3_zero_draft_length_rejected".into(),
        outcome: if expected_accepted_tokens(0, 0.8).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "draft_len_gamma = 0 rejected".into(),
    });

    // NC4: acceptance probability outside [0, 1] rejected
    negative_controls.push(NegativeControl {
        case_id: "nc4_invalid_acceptance_alpha_rejected".into(),
        outcome: if expected_accepted_tokens(4, 1.25).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "alpha > 1.0 rejected cleanly".into(),
    });

    // NC5: KV memory exceeding VRAM budget returns Oom (never silently ignored)
    let mut oom_cfg = cfg_fp16.clone();
    oom_cfg.context_tokens = 2_000_000;
    let oom_pt = evaluate_roofline_point(&oom_cfg, &overlap_observations[2])?;
    negative_controls.push(NegativeControl {
        case_id: "nc5_kv_exceeding_vram_returns_oom".into(),
        outcome: oom_pt.outcome,
        detail: "2M-context FP16 KV exceeding 112 GiB budget returns Oom".into(),
    });

    // NC6: lossy KV skipping parity check disqualified
    let mut bad_kv_cfg = cfg_fp8.clone();
    bad_kv_cfg.kv_parity_passed = false;
    let bad_kv_pt = evaluate_roofline_point(&bad_kv_cfg, &overlap_observations[2])?;
    negative_controls.push(NegativeControl {
        case_id: "nc6_failed_kv_parity_disqualified".into(),
        outcome: bad_kv_pt.outcome,
        detail: "QuantSpec quantized KV failing parity gate is disqualified".into(),
    });

    // NC7: non-finite HBM bandwidth rejected
    let mut nan_bw = cfg_fp16.clone();
    nan_bw.hbm_bandwidth_gb_s = f64::NAN;
    negative_controls.push(NegativeControl {
        case_id: "nc7_nonfinite_bandwidth_rejected".into(),
        outcome: if evaluate_roofline_point(&nan_bw, &overlap_observations[0]).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "NaN memory bandwidth rejected".into(),
    });

    // NC8: empty draft block rejected
    negative_controls.push(NegativeControl {
        case_id: "nc8_empty_draft_block_rejected".into(),
        outcome: if measure_draft_block_overlap(&[], &router, TOP_K).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "empty draft token slice rejected".into(),
    });

    Ok(MoeSpecRooflineReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "30B–1T sparse MoE GPU speculative verification (arXiv:2610.38981, QuantSpec, MoE-SpeQ)",
            "Pure-Rust sparse-MoE temporal expert-routing overlap, batched verification parity, and analytical roofline governor executed on CPU; physical GPU MoE serving did not execute locally",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        overlap_observations,
        roofline_points,
        recommended_gamma_fp16_kv: best_fp16.map(|(g, _)| g),
        recommended_gamma_quantspec_fp8_kv: best_fp8.map(|(g, _)| g),
        negative_controls,
    })
}

pub fn inputs_json(report: &MoeSpecRooflineReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.moe-spec-roofline.inputs.v1\",\n",
            "  \"issue\": 92,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"references\": [{}, {}, \"MoE-SpeQ\"],\n",
            "  \"informs_issues\": [6, 21, 65],\n",
            "  \"num_experts\": {},\n",
            "  \"top_k\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(MOE_SPEC_ARXIV),
        quote(QUANTSPEC_ARXIV),
        NUM_EXPERTS,
        TOP_K,
    )
}

pub fn results_json(report: &MoeSpecRooflineReport) -> String {
    let overlaps: Vec<String> = report
        .overlap_observations
        .iter()
        .map(|o| {
            format!(
                "{{\"draft_len_gamma\":{},\"total_expert_slots\":{},\"naive_no_overlap_experts\":{},\"observed_unique_experts\":{},\"temporal_overlap_ratio\":{},\"batched_vs_sequential_max_err\":{}}}",
                o.draft_len_gamma,
                o.total_expert_slots,
                o.naive_no_overlap_experts,
                o.observed_unique_experts,
                finite_f64(o.temporal_overlap_ratio),
                finite_f64(o.batched_vs_sequential_max_err),
            )
        })
        .collect();
    let points: Vec<String> = report
        .roofline_points
        .iter()
        .map(|p| {
            format!(
                "{{\"profile_id\":{},\"draft_len_gamma\":{},\"expected_accepted_tokens\":{},\"naive_dense_verify_ms\":{},\"overlap_aware_verify_ms\":{},\"naive_tok_per_s\":{},\"overlap_aware_tok_per_s\":{},\"total_vram_gib\":{},\"outcome\":{}}}",
                quote(&p.profile_id),
                p.draft_len_gamma,
                finite_f64(p.expected_accepted_tokens),
                finite_f64(p.naive_dense_verify_ms),
                finite_f64(p.overlap_aware_verify_ms),
                finite_f64(p.naive_tok_per_s),
                finite_f64(p.overlap_aware_tok_per_s),
                opt_f64(Some(p.total_vram_gib)),
                quote(p.outcome.as_str()),
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
            "  \"schema\": \"inference-lab.moe-spec-roofline.results.v1\",\n",
            "  \"issue\": 92,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"recommended_gamma_fp16_kv\": {},\n",
            "  \"recommended_gamma_quantspec_fp8_kv\": {},\n",
            "  \"overlap_observations\": [{}],\n",
            "  \"roofline_points\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        report
            .recommended_gamma_fp16_kv
            .map_or_else(|| "null".into(), |g| g.to_string()),
        report
            .recommended_gamma_quantspec_fp8_kv
            .map_or_else(|| "null".into(), |g| g.to_string()),
        overlaps.join(","),
        points.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &MoeSpecRooflineReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #92: Sparse-MoE Speculative Verification Roofline & Temporal Expert Overlap"
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
        "- Governor recommended draft length: FP16 KV `{:?}`, QuantSpec FP8 KV `{:?}`",
        report.recommended_gamma_fp16_kv, report.recommended_gamma_quantspec_fp8_kv
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Temporal Expert-Routing Overlap & Batched Parity").unwrap();
    for o in &report.overlap_observations {
        writeln!(
            out,
            "- `gamma={}`: `unique_experts={}` vs `naive_bound={}` (`overlap={:.1}%`), `parity_err={:.3e}`",
            o.draft_len_gamma,
            o.observed_unique_experts,
            o.naive_no_overlap_experts,
            o.temporal_overlap_ratio * 100.0,
            o.batched_vs_sequential_max_err
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
    fn moe_temporal_overlap_beats_naive_roofline_with_exact_parity() {
        let r = run_default_harness().unwrap();
        assert!(!r.default_enabled);
        for o in &r.overlap_observations {
            assert!(o.batched_vs_sequential_max_err <= PARITY_TOL);
        }
        // At gamma=8, coherent tokens exhibit strictly positive temporal expert overlap
        assert!(r.overlap_observations[3].temporal_overlap_ratio > 0.25);
        assert!(
            r.overlap_observations[3].observed_unique_experts
                < r.overlap_observations[3].naive_no_overlap_experts
        );
        assert_eq!(r.negative_controls.len(), 8);
        for c in &r.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted, "{}", c.case_id);
        }
    }
}
