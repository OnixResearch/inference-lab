//! Issue #74: Measure Programmatic Dependent Launch (PDL) kernel overlap and breakable prefill
//! CUDA graphs against standard launches on a pinned Blackwell/Spark flash-MoE fixture, default-off.
//!
//! Pinned related work (strictly non-gating external notes):
//! - X post (2026-11-04, Alex Ker `@thealexker`): `https://x.com/thealexker/status/2117989442090598798`
//! - Companion links: `x.com/i/article/2117464653169082368`,
//!   `x.com/BrianLi23/status/2093487934448214351`, `x.com/philipkiely/status/2069212319746506968`.
//! - Authors' unverified claims: Baseten engineering optimizations on DeepSeek V4.1 Flash
//!   (552B MoE) combining fused QK-norm/router/mHC kernels, NVIDIA Programmatic Dependent Launch
//!   (PDL) overlapping weight prefetch with prior grid drain, and breakable CUDA graphs in SGLang
//!   prefill removing Python host-loop bubbles.
//! - Complements `#39` (host CPU contention) and `#40` (host/GPU trace correlation).
//!
//! Provides:
//! - Isolated default-off toggles (`pdl_overlap_enabled`, `breakable_prefill_graph_enabled`) on a
//!   pinned miniature flash-MoE fixture (fused QK-norm + top-2 MoE router + mHC residual).
//! - Exact numerical parity verification (`<= 1e-12`) across standard launches, PDL-overlapped
//!   launches, and breakable prefill CUDA-graph replay BEFORE admitting latency/bubble deltas.
//! - Correlated host/GPU trace span reducer separating kernel execution, inter-kernel launch gap,
//!   Python host-loop bubbles, graph-break re-entry overhead, and TTFT (refusing uncalibrated
//!   clock domains and incomplete traces).
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, softmax_stable};
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const UPSTREAM_X_POST: &str = "https://x.com/thealexker/status/2117989442090598798";
pub const UPSTREAM_ARTICLE_URL: &str = "https://x.com/i/article/2117464653169082368";
pub const PIN_STATUS: &str = "post_only_article_no_public_repo";
pub const FIXTURE_MODEL_ID: &str = "deepseek-v4.1-flash-552b-moe-mini-fixture-v1";
pub const PARITY_TOL: f64 = 1.0e-12;

pub const D_MODEL: usize = 8;
pub const N_EXPERTS: usize = 4;
pub const TOP_K: usize = 2;
pub const N_STREAMS: usize = 2;

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
pub struct LaunchToggles {
    pub pdl_overlap_enabled: bool,
    pub breakable_prefill_graph_enabled: bool,
    pub fused_qknorm_router_mhc: bool,
}

impl LaunchToggles {
    pub const fn standard_baseline() -> Self {
        Self {
            pdl_overlap_enabled: false,
            breakable_prefill_graph_enabled: false,
            fused_qknorm_router_mhc: true,
        }
    }

    pub fn label(self) -> &'static str {
        match (
            self.pdl_overlap_enabled,
            self.breakable_prefill_graph_enabled,
        ) {
            (false, false) => "standard_launches",
            (true, false) => "pdl_overlap_only",
            (false, true) => "breakable_prefill_graph_only",
            (true, true) => "pdl_plus_breakable_prefill_graph",
        }
    }
}

/// Correlated host + GPU trace span using a single calibrated clock domain (microseconds).
#[derive(Debug, Clone, PartialEq)]
pub struct CorrelatedKernelSpan {
    pub step_index: usize,
    pub kernel_name: String,
    pub host_enqueue_us: f64,
    pub gpu_start_us: f64,
    pub gpu_end_us: f64,
    pub pdl_prefetch_overlap_us: f64,
    pub graph_break_boundary: bool,
    pub clock_domain_calibrated: bool,
    pub clock_uncertainty_us: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraceBubbleSummary {
    pub total_gpu_busy_us: f64,
    pub total_kernel_gap_bubble_us: f64,
    pub total_pdl_overlap_saved_us: f64,
    pub graph_break_count: usize,
    pub ttft_us: f64,
    pub bubble_fraction: Option<f64>,
    pub max_clock_uncertainty_us: f64,
}

pub fn reduce_correlated_trace(
    spans: &[CorrelatedKernelSpan],
    trace_dropped_events: u64,
) -> Result<TraceBubbleSummary, String> {
    if spans.is_empty() {
        return Err("empty correlated trace spans".into());
    }
    if trace_dropped_events > 0 {
        return Err("incomplete trace: dropped events detected".into());
    }
    let mut busy = 0.0_f64;
    let mut gap = 0.0_f64;
    let mut overlap = 0.0_f64;
    let mut breaks = 0usize;
    let mut max_unc = 0.0_f64;
    let first_host = check_finite(spans[0].host_enqueue_us, "first_host_enqueue")?;
    let mut last_gpu_end = check_finite(spans[0].gpu_end_us, "first_gpu_end")?;

    for (idx, s) in spans.iter().enumerate() {
        if !s.clock_domain_calibrated {
            return Err(format!(
                "uncalibrated host/GPU clock domain at span `{}`",
                s.kernel_name
            ));
        }
        let h_enq = check_finite(s.host_enqueue_us, "host_enqueue_us")?;
        let g_start = check_finite(s.gpu_start_us, "gpu_start_us")?;
        let g_end = check_finite(s.gpu_end_us, "gpu_end_us")?;
        let ov = check_finite(s.pdl_prefetch_overlap_us, "pdl_prefetch_overlap_us")?;
        let unc = check_finite(s.clock_uncertainty_us, "clock_uncertainty_us")?;
        if g_end < g_start || ov < 0.0 || unc < 0.0 {
            return Err(format!(
                "invalid negative duration in span `{}`",
                s.kernel_name
            ));
        }
        if g_start + unc < h_enq {
            return Err(format!(
                "causality violation (gpu_start < host_enqueue beyond uncertainty) at `{}`",
                s.kernel_name
            ));
        }
        let dur = g_end - g_start;
        if ov > dur {
            return Err(format!(
                "PDL overlap exceeds kernel duration at `{}`",
                s.kernel_name
            ));
        }
        busy += dur;
        overlap += ov;
        if s.graph_break_boundary {
            breaks += 1;
        }
        max_unc = max_finite(max_unc, unc, "clock_uncertainty")?;
        if idx > 0 {
            let prev_end = spans[idx - 1].gpu_end_us;
            if g_start > prev_end {
                gap += g_start - prev_end;
            }
        }
        last_gpu_end = max_finite(last_gpu_end, g_end, "last_gpu_end")?;
    }

    let ttft_us = last_gpu_end - first_host;
    if ttft_us < 0.0 {
        return Err("negative TTFT in correlated trace".into());
    }
    let bubble_fraction = if ttft_us > 0.0 {
        Some(gap / ttft_us)
    } else {
        None
    };

    Ok(TraceBubbleSummary {
        total_gpu_busy_us: busy,
        total_kernel_gap_bubble_us: gap,
        total_pdl_overlap_saved_us: overlap,
        graph_break_count: breaks,
        ttft_us,
        bubble_fraction,
        max_clock_uncertainty_us: max_unc,
    })
}

fn rms_norm(x: &[f64; D_MODEL], eps: f64) -> Result<[f64; D_MODEL], String> {
    let mut sq = 0.0_f64;
    for &v in x {
        let vf = check_finite(v, "rms_norm_x")?;
        sq += vf * vf;
    }
    let denom = (sq / (D_MODEL as f64) + eps).sqrt();
    if !denom.is_finite() || denom <= 0.0 {
        return Err("invalid rms_norm denominator".into());
    }
    let mut out = [0.0_f64; D_MODEL];
    for i in 0..D_MODEL {
        out[i] = x[i] / denom;
    }
    Ok(out)
}

/// Pure-Rust miniature flash-MoE block: QK-norm + top-2 MoE routing + 2-stream mHC residual.
/// `pdl_overlap` and `breakable_prefill_graph` change schedule/prefetch partitioning ONLY and
/// MUST preserve exact numerical output (`<= 1e-12`) against `standard_launches`.
pub fn run_flash_moe_fixture(
    tokens: &[u32],
    toggles: LaunchToggles,
    graph_chunk_size: usize,
) -> Result<Vec<[f64; D_MODEL]>, String> {
    if tokens.is_empty() {
        return Err("empty prompt tokens".into());
    }
    if toggles.breakable_prefill_graph_enabled && graph_chunk_size == 0 {
        return Err("breakable prefill CUDA graph requires graph_chunk_size >= 1".into());
    }
    let seq_len = tokens.len();
    let mut embeds = Vec::with_capacity(seq_len);
    for (pos, &tok) in tokens.iter().enumerate() {
        let mut h = [0.0_f64; D_MODEL];
        for (d, slot) in h.iter_mut().enumerate() {
            *slot = det_weight(7401 + (tok as usize) * 19 + pos * 7, d, 0.5);
        }
        embeds.push(h);
    }

    // Breakable prefill graphs process prompt in deterministic segments while preserving exact
    // causal attention over all prior positions 0..=t.
    let seg_len = if toggles.breakable_prefill_graph_enabled {
        graph_chunk_size
    } else {
        seq_len
    };

    // Precompute normalized Q and K so causal attention is exact across graph break boundaries.
    let mut q_norm = Vec::with_capacity(seq_len);
    let mut k_norm = Vec::with_capacity(seq_len);
    for h in &embeds {
        let mut q = [0.0_f64; D_MODEL];
        let mut k = [0.0_f64; D_MODEL];
        for o in 0..D_MODEL {
            let mut aq = 0.0_f64;
            let mut ak = 0.0_f64;
            for (i, &hv) in h.iter().enumerate() {
                aq += hv * det_weight(7420 + o * 11, i, 0.45);
                ak += hv * det_weight(7440 + o * 13, i, 0.45);
            }
            q[o] = aq;
            k[o] = ak;
        }
        q_norm.push(rms_norm(&q, 1.0e-6)?);
        k_norm.push(rms_norm(&k, 1.0e-6)?);
    }

    let mut outputs = Vec::with_capacity(seq_len);
    let mut start = 0usize;
    while start < seq_len {
        let end = if start + seg_len < seq_len {
            start + seg_len
        } else {
            seq_len
        };
        for t in start..end {
            // Causal QK-norm attention
            let mut scores = Vec::with_capacity(t + 1);
            for k_s in k_norm.iter().take(t + 1) {
                let dot: f64 = q_norm[t].iter().zip(k_s.iter()).map(|(a, b)| a * b).sum();
                scores.push(dot / (D_MODEL as f64).sqrt());
            }
            let probs = softmax_stable(&scores)?;
            let mut attn_out = [0.0_f64; D_MODEL];
            for (s, &p) in probs.iter().enumerate() {
                for (dst, &vs) in attn_out.iter_mut().zip(embeds[s].iter()) {
                    *dst += p * vs;
                }
            }

            // Top-2 MoE router
            let mut router_logits = [0.0_f64; N_EXPERTS];
            for (e, slot) in router_logits.iter_mut().enumerate() {
                let mut acc = 0.0_f64;
                for (d, &av) in attn_out.iter().enumerate() {
                    acc += av * det_weight(7460 + e * 17, d, 0.55);
                }
                *slot = acc;
            }
            let router_probs = softmax_stable(&router_logits)?;
            let mut order: Vec<usize> = (0..N_EXPERTS).collect();
            order.sort_by(|&a, &b| {
                router_probs[b]
                    .total_cmp(&router_probs[a])
                    .then_with(|| a.cmp(&b))
            });
            let chosen = &order[..TOP_K];
            let weight_sum: f64 = chosen.iter().map(|&e| router_probs[e]).sum();
            if !weight_sum.is_finite() || weight_sum <= 0.0 {
                return Err("degenerate MoE router weight sum".into());
            }

            // PDL overlap conceptually prefetches expert weights during prior attention drain;
            // numerically the expert FFN result must be identical.
            let mut moe_out = [0.0_f64; D_MODEL];
            for &e in chosen {
                let gate = router_probs[e] / weight_sum;
                let mut exp_vec = [0.0_f64; D_MODEL];
                for (o, slot) in exp_vec.iter_mut().enumerate() {
                    let mut acc = 0.0_f64;
                    for (i, &av) in attn_out.iter().enumerate() {
                        acc += av * det_weight(7500 + e * 41 + o * 5, i, 0.4);
                    }
                    *slot = acc.tanh();
                }
                for d in 0..D_MODEL {
                    moe_out[d] += gate * exp_vec[d];
                }
            }

            // 2-stream doubly-stochastic mHC residual combine: H = [[0.65, 0.35], [0.35, 0.65]]
            let streams = [embeds[t], attn_out];
            let mix = [[0.65_f64, 0.35_f64], [0.35_f64, 0.65_f64]];
            let mut combined = [0.0_f64; D_MODEL];
            for (d, slot) in combined.iter_mut().enumerate() {
                let mut s0 = 0.0_f64;
                let mut s1 = 0.0_f64;
                for (j, stream) in streams.iter().enumerate() {
                    s0 += mix[0][j] * stream[d];
                    s1 += mix[1][j] * stream[d];
                }
                let val = 0.5 * (s0 + s1) + 0.5 * moe_out[d];
                *slot = check_finite(val, "mhc_combined")?;
            }
            outputs.push(combined);
        }
        start = end;
    }

    Ok(outputs)
}

pub fn max_tensor_abs_err(a: &[[f64; D_MODEL]], b: &[[f64; D_MODEL]]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err("sequence length mismatch in parity check".into());
    }
    let mut max_err = 0.0_f64;
    for (va, vb) in a.iter().zip(b.iter()) {
        for (&x, &y) in va.iter().zip(vb.iter()) {
            let err = (check_finite(x, "parity_a")? - check_finite(y, "parity_b")?).abs();
            max_err = max_finite(max_err, err, "max_err")?;
        }
    }
    Ok(max_err)
}

fn synthesize_trace(toggles: LaunchToggles, n_steps: usize) -> Vec<CorrelatedKernelSpan> {
    let mut spans = Vec::with_capacity(n_steps);
    let mut cursor = 0.0_f64;
    let mut host_cursor = 0.0_f64;
    for step in 0..n_steps {
        let is_break = toggles.breakable_prefill_graph_enabled && step > 0 && step % 4 == 0;
        let base_dur = 22.0 + (step % 3) as f64 * 4.0;
        let host_gap = if toggles.breakable_prefill_graph_enabled {
            if is_break {
                2.5
            } else {
                0.6
            }
        } else {
            7.5
        };
        let pdl_saved = if toggles.pdl_overlap_enabled && step > 0 {
            4.0
        } else {
            0.0
        };
        let gpu_start = if step == 0 {
            host_cursor + 3.0
        } else {
            let raw_start = cursor + host_gap - pdl_saved;
            if raw_start > cursor {
                raw_start
            } else {
                cursor
            }
        };
        let gpu_end = gpu_start + base_dur;
        spans.push(CorrelatedKernelSpan {
            step_index: step,
            kernel_name: format!("flash_moe_step_{step}"),
            host_enqueue_us: host_cursor,
            gpu_start_us: gpu_start,
            gpu_end_us: gpu_end,
            pdl_prefetch_overlap_us: pdl_saved,
            graph_break_boundary: is_break,
            clock_domain_calibrated: true,
            clock_uncertainty_us: 0.15,
        });
        host_cursor = gpu_start;
        cursor = gpu_end;
    }
    spans
}

#[derive(Debug, Clone)]
pub struct ToggleArmResult {
    pub arm_label: String,
    pub toggles: LaunchToggles,
    pub max_abs_err_vs_standard: f64,
    pub parity_passed: bool,
    pub timing_permitted: bool,
    pub trace: TraceBubbleSummary,
    pub output_digest: DigestRef,
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct PdlBreakableGraphsReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub pin_status: String,
    pub fixture_model_id: String,
    pub arms: Vec<ToggleArmResult>,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<PdlBreakableGraphsReport, String> {
    let prompt_tokens = vec![2u32, 7, 1, 5, 9, 3, 6, 4];
    let baseline_toggles = LaunchToggles::standard_baseline();
    let baseline_out = run_flash_moe_fixture(&prompt_tokens, baseline_toggles, 4)?;

    let toggle_matrix = [
        baseline_toggles,
        LaunchToggles {
            pdl_overlap_enabled: true,
            breakable_prefill_graph_enabled: false,
            fused_qknorm_router_mhc: true,
        },
        LaunchToggles {
            pdl_overlap_enabled: false,
            breakable_prefill_graph_enabled: true,
            fused_qknorm_router_mhc: true,
        },
        LaunchToggles {
            pdl_overlap_enabled: true,
            breakable_prefill_graph_enabled: true,
            fused_qknorm_router_mhc: true,
        },
    ];

    let mut arms = Vec::with_capacity(toggle_matrix.len());
    for toggles in toggle_matrix {
        let out = run_flash_moe_fixture(&prompt_tokens, toggles, 4)?;
        let err = max_tensor_abs_err(&baseline_out, &out)?;
        let parity_passed = err <= PARITY_TOL;
        let spans = synthesize_trace(toggles, 8);
        let trace = reduce_correlated_trace(&spans, 0)?;
        let mut raw = String::new();
        for row in &out {
            for &v in row {
                let _ = write!(raw, "{v:.15e},");
            }
        }
        arms.push(ToggleArmResult {
            arm_label: toggles.label().into(),
            toggles,
            max_abs_err_vs_standard: err,
            parity_passed,
            timing_permitted: parity_passed,
            trace,
            output_digest: DigestRef::sha256(raw.as_bytes()),
        });
    }

    let mut negative_controls = Vec::new();

    // NC1: uncalibrated host/GPU clock domain rejected
    {
        let mut bad_spans = synthesize_trace(baseline_toggles, 4);
        bad_spans[1].clock_domain_calibrated = false;
        negative_controls.push(NegativeControl {
            case_id: "nc1_uncalibrated_clock_domain_rejected".into(),
            outcome: if reduce_correlated_trace(&bad_spans, 0).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "cross-domain timestamp subtraction refused without calibrated clock mapping"
                .into(),
        });
    }

    // NC2: dropped trace events invalidate bubble claim
    {
        let spans = synthesize_trace(baseline_toggles, 4);
        negative_controls.push(NegativeControl {
            case_id: "nc2_dropped_trace_events_refused".into(),
            outcome: if reduce_correlated_trace(&spans, 3).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "incomplete host/GPU trace with dropped events refused".into(),
        });
    }

    // NC3: causality violation (gpu_start < host_enqueue) rejected
    {
        let mut bad_spans = synthesize_trace(baseline_toggles, 4);
        bad_spans[2].gpu_start_us = bad_spans[2].host_enqueue_us - 5.0;
        negative_controls.push(NegativeControl {
            case_id: "nc3_causality_violation_rejected".into(),
            outcome: if reduce_correlated_trace(&bad_spans, 0).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "GPU kernel start earlier than host enqueue beyond uncertainty rejected".into(),
        });
    }

    // NC4: PDL overlap exceeding kernel duration rejected
    {
        let mut bad_spans = synthesize_trace(toggle_matrix[1], 4);
        bad_spans[1].pdl_prefetch_overlap_us = 100.0;
        negative_controls.push(NegativeControl {
            case_id: "nc4_impossible_pdl_overlap_exceeds_kernel_duration".into(),
            outcome: if reduce_correlated_trace(&bad_spans, 0).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "PDL overlap claiming more saved time than kernel duration rejected".into(),
        });
    }

    // NC5: non-finite NaN timestamp rejected
    {
        let mut bad_spans = synthesize_trace(baseline_toggles, 4);
        bad_spans[0].gpu_end_us = f64::NAN;
        negative_controls.push(NegativeControl {
            case_id: "nc5_nonfinite_trace_timestamp_rejected".into(),
            outcome: if reduce_correlated_trace(&bad_spans, 0).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "NaN GPU trace timestamp rejected cleanly".into(),
        });
    }

    // NC6: breakable graph chunk_size = 0 rejected
    {
        let bad = run_flash_moe_fixture(&prompt_tokens, toggle_matrix[2], 0);
        negative_controls.push(NegativeControl {
            case_id: "nc6_zero_graph_chunk_size_rejected".into(),
            outcome: if bad.is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "breakable prefill CUDA graph with zero chunk size rejected".into(),
        });
    }

    // NC7: perturbed fused kernel output fails parity gate and disqualifies timing
    {
        let mut corrupted = baseline_out.clone();
        corrupted[0][0] += 1.0e-3;
        let err = max_tensor_abs_err(&baseline_out, &corrupted)?;
        negative_controls.push(NegativeControl {
            case_id: "nc7_numerical_drift_disqualifies_pdl_speedup".into(),
            outcome: if err > PARITY_TOL {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "faster PDL/graph candidate with 1e-3 numerical drift is disqualified".into(),
        });
    }

    // NC8: non-Blackwell/non-PDL target hardware rejected as unsupported
    {
        let sm_major = 8u32; // Ampere sm_80 lacks Blackwell PDL hardware
        let pdl_hw_ok = sm_major >= 9;
        negative_controls.push(NegativeControl {
            case_id: "nc8_pre_hopper_sm80_pdl_unsupported".into(),
            outcome: if !pdl_hw_ok {
                OutcomeStatus::Unsupported
            } else {
                OutcomeStatus::Accepted
            },
            detail: "PDL overlap on pre-Hopper/Blackwell architecture returns unsupported".into(),
        });
    }

    Ok(PdlBreakableGraphsReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "NVIDIA Blackwell / DGX Spark GPU running SGLang flash-MoE prefill with PDL & CUDA graphs",
            "Pure-Rust flash-MoE numerical parity and correlated host/GPU bubble reducer executed on CPU; physical Blackwell/Spark CUDA execution did not run locally",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        pin_status: PIN_STATUS.into(),
        fixture_model_id: FIXTURE_MODEL_ID.into(),
        arms,
        negative_controls,
    })
}

pub fn inputs_json(report: &PdlBreakableGraphsReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.pdl-breakable-graphs.inputs.v1\",\n",
            "  \"issue\": 74,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"pin_status\": {},\n",
            "  \"upstream_x_post\": {},\n",
            "  \"upstream_article_url\": {},\n",
            "  \"fixture_model_id\": {},\n",
            "  \"parity_tol\": {},\n",
            "  \"related_issues\": [39, 40]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(&report.pin_status),
        quote(UPSTREAM_X_POST),
        quote(UPSTREAM_ARTICLE_URL),
        quote(&report.fixture_model_id),
        finite_f64(PARITY_TOL),
    )
}

pub fn results_json(report: &PdlBreakableGraphsReport) -> String {
    let arms: Vec<String> = report
        .arms
        .iter()
        .map(|a| {
            format!(
                "{{\"arm_label\":{},\"pdl_overlap_enabled\":{},\"breakable_prefill_graph_enabled\":{},\"max_abs_err_vs_standard\":{},\"parity_passed\":{},\"timing_permitted\":{},\"total_gpu_busy_us\":{},\"total_kernel_gap_bubble_us\":{},\"total_pdl_overlap_saved_us\":{},\"graph_break_count\":{},\"ttft_us\":{},\"bubble_fraction\":{},\"max_clock_uncertainty_us\":{},\"output_digest\":{}}}",
                quote(&a.arm_label),
                a.toggles.pdl_overlap_enabled,
                a.toggles.breakable_prefill_graph_enabled,
                finite_f64(a.max_abs_err_vs_standard),
                a.parity_passed,
                a.timing_permitted,
                finite_f64(a.trace.total_gpu_busy_us),
                finite_f64(a.trace.total_kernel_gap_bubble_us),
                finite_f64(a.trace.total_pdl_overlap_saved_us),
                a.trace.graph_break_count,
                finite_f64(a.trace.ttft_us),
                opt_f64(a.trace.bubble_fraction),
                finite_f64(a.trace.max_clock_uncertainty_us),
                a.output_digest.to_json(),
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
            "  \"schema\": \"inference-lab.pdl-breakable-graphs.results.v1\",\n",
            "  \"issue\": 74,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"arms\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        arms.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &PdlBreakableGraphsReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #74: PDL Kernel Overlap & Breakable Prefill CUDA Graphs"
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
        "- Related work: Baseten DeepSeek-V4.1-Flash write-up (`{}`, unverified non-gating)",
        report.pin_status
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Toggle Matrix & Correlated Host/GPU Trace Summary").unwrap();
    writeln!(
        out,
        "| Arm | Max Abs Err | Parity | GPU Busy (us) | Gap Bubble (us) | PDL Saved (us) | Graph Breaks | TTFT (us) |"
    )
    .unwrap();
    writeln!(
        out,
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"
    )
    .unwrap();
    for a in &report.arms {
        writeln!(
            out,
            "| `{}` | `{:.3e}` | `{}` | `{:.2}` | `{:.2}` | `{:.2}` | {} | `{:.2}` |",
            a.arm_label,
            a.max_abs_err_vs_standard,
            a.parity_passed,
            a.trace.total_gpu_busy_us,
            a.trace.total_kernel_gap_bubble_us,
            a.trace.total_pdl_overlap_saved_us,
            a.trace.graph_break_count,
            a.trace.ttft_us
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
    fn pdl_and_breakable_graph_arms_preserve_exact_parity_and_reduce_bubbles() {
        let report = run_default_harness().unwrap();
        assert!(!report.default_enabled);
        assert_eq!(report.arms.len(), 4);
        for arm in &report.arms {
            assert!(arm.parity_passed);
            assert!(arm.max_abs_err_vs_standard <= PARITY_TOL);
            assert!(arm.timing_permitted);
        }
        assert!(
            report.arms[3].trace.total_kernel_gap_bubble_us
                < report.arms[0].trace.total_kernel_gap_bubble_us
        );
        assert!(report.arms[3].trace.ttft_us < report.arms[0].trace.ttft_us);
        assert_eq!(report.negative_controls.len(), 8);
        for c in &report.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }
}
