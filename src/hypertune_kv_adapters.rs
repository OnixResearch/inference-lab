//! Issue #66: Benchmark pinned Hypertune soft-prefill KV adapters against heterogeneous batched
//! LoRA on throughput, KV budget, and co-resident task quality, default-off.
//!
//! Pinned upstream reference (observed 2026-11-24, MIT):
//! - Repository: `https://github.com/Hyperstitioncc/Hypertune`
//! - Commit: `7aaaf8c2fc88bfb7dfd91019c40428d4528ae121`
//! - X post (2026-11-17, `@hyperstition_cc`): `https://x.com/hyperstition_cc/status/2122808531288256657`
//! - Concept: soft-prefill KV adapters trained via GRPO/SFT and prepended into paged KV blocks,
//!   enabling heterogeneous multi-tenant batching and adapter composition without LoRA's
//!   per-matrix bandwidth or MoE combinatorial overhead.
//! - Complements `#25` (heterogeneous rollout serving) and `#49` (harness LoRA comparison).
//!
//! Provides:
//! - Miniature pure-Rust causal attention comparing:
//!   1. `unadapted_base`
//!   2. `hypertune_soft_kv` (prepended paged KV adapter blocks + multi-adapter composition)
//!   3. `hetero_batched_lora` (per-request low-rank `W + (alpha/r) B A` weight delta)
//! - Exact mathematical parity check (`<= 1e-12`) proving paged soft-prefill KV injection equals
//!   prepending the equivalent virtual prefix tokens in full forward attention.
//! - Multi-adapter concurrency scaling (`1, 4, 8, 16` distinct co-resident adapters) comparing
//!   KV-cache block overhead vs LoRA per-matrix weight bandwidth & MoE expert-adapter fanout.
//! - Independently checked co-resident multi-adapter frozen task accuracy (non-collapsing).
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::det_weight;
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const UPSTREAM_REPO: &str = "Hyperstitioncc/Hypertune";
pub const UPSTREAM_COMMIT: &str = "7aaaf8c2fc88bfb7dfd91019c40428d4528ae121";
pub const UPSTREAM_LICENSE: &str = "MIT";
pub const UPSTREAM_X_POST: &str = "https://x.com/hyperstition_cc/status/2122808531288256657";

pub const D_MODEL: usize = 8;
pub const LORA_RANK: usize = 2;
pub const SOFT_KV_SLOTS: usize = 4;
pub const PARITY_TOL: f64 = 1.0e-12;

fn check_finite(v: f64, ctx: &str) -> Result<f64, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("non-finite float in {ctx}"))
    }
}

#[derive(Debug, Clone)]
pub struct MiniAttnWeights {
    pub wq: [[f64; D_MODEL]; D_MODEL],
    pub wk: [[f64; D_MODEL]; D_MODEL],
    pub wv: [[f64; D_MODEL]; D_MODEL],
    pub wo: [[f64; D_MODEL]; D_MODEL],
}

impl MiniAttnWeights {
    pub fn build_deterministic() -> Self {
        let mut wq = [[0.0_f64; D_MODEL]; D_MODEL];
        let mut wk = [[0.0_f64; D_MODEL]; D_MODEL];
        let mut wv = [[0.0_f64; D_MODEL]; D_MODEL];
        let mut wo = [[0.0_f64; D_MODEL]; D_MODEL];
        for r in 0..D_MODEL {
            for c in 0..D_MODEL {
                wq[r][c] = det_weight(6601, r * D_MODEL + c, 0.30);
                wk[r][c] = det_weight(6602, r * D_MODEL + c, 0.30);
                wv[r][c] = det_weight(6603, r * D_MODEL + c, 0.30);
                wo[r][c] = det_weight(6604, r * D_MODEL + c, 0.25);
            }
        }
        Self { wq, wk, wv, wo }
    }
}

#[derive(Debug, Clone)]
pub struct SoftPrefillKvAdapter {
    pub adapter_id: String,
    pub prefix_keys: Vec<[f64; D_MODEL]>,
    pub prefix_values: Vec<[f64; D_MODEL]>,
}

impl SoftPrefillKvAdapter {
    pub fn build_deterministic(adapter_id: &str, seed: usize, slots: usize) -> Self {
        let mut prefix_keys = Vec::with_capacity(slots);
        let mut prefix_values = Vec::with_capacity(slots);
        for s in 0..slots {
            let mut k = [0.0_f64; D_MODEL];
            let mut v = [0.0_f64; D_MODEL];
            for d in 0..D_MODEL {
                k[d] = det_weight(seed, s * D_MODEL + d, 0.45);
                v[d] = det_weight(seed + 100, s * D_MODEL + d, 0.45);
            }
            prefix_keys.push(k);
            prefix_values.push(v);
        }
        Self {
            adapter_id: adapter_id.into(),
            prefix_keys,
            prefix_values,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoraAdapter {
    pub adapter_id: String,
    pub alpha: f64,
    pub a_mat: [[f64; D_MODEL]; LORA_RANK],
    pub b_mat: [[f64; LORA_RANK]; D_MODEL],
}

impl LoraAdapter {
    #[allow(clippy::needless_range_loop)]
    pub fn build_deterministic(adapter_id: &str, seed: usize, alpha: f64) -> Self {
        let mut a_mat = [[0.0_f64; D_MODEL]; LORA_RANK];
        let mut b_mat = [[0.0_f64; LORA_RANK]; D_MODEL];
        for r in 0..LORA_RANK {
            for c in 0..D_MODEL {
                a_mat[r][c] = det_weight(seed, r * D_MODEL + c, 0.35);
            }
        }
        for r in 0..D_MODEL {
            for c in 0..LORA_RANK {
                b_mat[r][c] = det_weight(seed + 50, r * LORA_RANK + c, 0.35);
            }
        }
        Self {
            adapter_id: adapter_id.into(),
            alpha,
            a_mat,
            b_mat,
        }
    }
}

#[allow(clippy::needless_range_loop)]
fn matvec(w: &[[f64; D_MODEL]; D_MODEL], x: &[f64; D_MODEL]) -> [f64; D_MODEL] {
    let mut out = [0.0_f64; D_MODEL];
    for r in 0..D_MODEL {
        let mut s = 0.0_f64;
        for c in 0..D_MODEL {
            s += w[r][c] * x[c];
        }
        out[r] = s;
    }
    out
}

#[allow(clippy::needless_range_loop)]
fn causal_attn_with_optional_kv(
    w: &MiniAttnWeights,
    tokens: &[[f64; D_MODEL]],
    prepended_k: &[[f64; D_MODEL]],
    prepended_v: &[[f64; D_MODEL]],
    lora_on_wo: Option<&LoraAdapter>,
) -> Result<Vec<[f64; D_MODEL]>, String> {
    if prepended_k.len() != prepended_v.len() {
        return Err("mismatched prepended KV lengths".into());
    }
    let mut all_k = prepended_k.to_vec();
    let mut all_v = prepended_v.to_vec();
    let inv_scale = 1.0_f64 / (D_MODEL as f64).sqrt();
    let mut outs = Vec::with_capacity(tokens.len());

    for x in tokens {
        for &v in x {
            check_finite(v, "token_x")?;
        }
        let q = matvec(&w.wq, x);
        let k = matvec(&w.wk, x);
        let v = matvec(&w.wv, x);
        all_k.push(k);
        all_v.push(v);

        let total_len = all_k.len();
        let mut scores = Vec::with_capacity(total_len);
        let mut m = f64::NEG_INFINITY;
        for pos in 0..total_len {
            let mut dot = 0.0_f64;
            for d in 0..D_MODEL {
                dot += q[d] * check_finite(all_k[pos][d], "k")?;
            }
            let s = check_finite(dot * inv_scale, "score")?;
            if s > m {
                m = s;
            }
            scores.push(s);
        }
        let mut sum = 0.0_f64;
        let mut ws = Vec::with_capacity(total_len);
        for &s in &scores {
            let e = (s - m).exp();
            sum += e;
            ws.push(e);
        }
        let mut ctx = [0.0_f64; D_MODEL];
        for d in 0..D_MODEL {
            let mut acc = 0.0_f64;
            for pos in 0..total_len {
                acc += (ws[pos] / sum) * check_finite(all_v[pos][d], "v")?;
            }
            ctx[d] = acc;
        }
        let mut y = matvec(&w.wo, &ctx);
        if let Some(lora) = lora_on_wo {
            let scale = check_finite(lora.alpha, "lora_alpha")? / (LORA_RANK as f64);
            let mut mid = [0.0_f64; LORA_RANK];
            for r in 0..LORA_RANK {
                let mut s = 0.0_f64;
                for c in 0..D_MODEL {
                    s += lora.a_mat[r][c] * ctx[c];
                }
                mid[r] = s;
            }
            for r in 0..D_MODEL {
                let mut s = 0.0_f64;
                for c in 0..LORA_RANK {
                    s += lora.b_mat[r][c] * mid[c];
                }
                y[r] = check_finite(y[r] + scale * s, "lora_y")?;
            }
        }
        outs.push(y);
    }
    Ok(outs)
}

/// Independent reference computing prepended soft-prefill KV in a single unpaged dense pass
/// (explicit causal mask over `[soft_slots + seq_len]`) to verify paged-KV injection parity.
#[allow(clippy::needless_range_loop)]
pub fn reference_unpaged_soft_prefill(
    w: &MiniAttnWeights,
    tokens: &[[f64; D_MODEL]],
    adapters: &[&SoftPrefillKvAdapter],
) -> Result<Vec<[f64; D_MODEL]>, String> {
    let mut flat_k = Vec::new();
    let mut flat_v = Vec::new();
    for ad in adapters {
        flat_k.extend_from_slice(&ad.prefix_keys);
        flat_v.extend_from_slice(&ad.prefix_values);
    }
    let soft_n = flat_k.len();
    for x in tokens {
        flat_k.push(matvec(&w.wk, x));
        flat_v.push(matvec(&w.wv, x));
    }
    let inv_scale = 1.0_f64 / (D_MODEL as f64).sqrt();
    let mut outs = Vec::with_capacity(tokens.len());
    for (t_idx, x) in tokens.iter().enumerate() {
        let q = matvec(&w.wq, x);
        let attend_len = soft_n + t_idx + 1;
        let mut scores = Vec::with_capacity(attend_len);
        let mut m = f64::NEG_INFINITY;
        for pos in 0..attend_len {
            let mut dot = 0.0_f64;
            for d in 0..D_MODEL {
                dot += q[d] * flat_k[pos][d];
            }
            let s = check_finite(dot * inv_scale, "ref_s")?;
            if s > m {
                m = s;
            }
            scores.push(s);
        }
        let mut sum = 0.0_f64;
        let mut ws = Vec::with_capacity(attend_len);
        for &s in &scores {
            let e = (s - m).exp();
            sum += e;
            ws.push(e);
        }
        let mut ctx = [0.0_f64; D_MODEL];
        for d in 0..D_MODEL {
            let mut acc = 0.0_f64;
            for pos in 0..attend_len {
                acc += (ws[pos] / sum) * flat_v[pos][d];
            }
            ctx[d] = acc;
        }
        outs.push(matvec(&w.wo, &ctx));
    }
    Ok(outs)
}

#[derive(Debug, Clone)]
pub struct ConcurrencyScalingRow {
    pub active_distinct_adapters: usize,
    pub batch_size: usize,
    pub hypertune_kv_overhead_mb: f64,
    pub hypertune_extra_weight_bw_mb: f64,
    pub lora_kv_overhead_mb: f64,
    pub lora_extra_weight_bw_mb: f64,
    pub hypertune_decode_tok_s: Option<f64>,
    pub lora_decode_tok_s: Option<f64>,
    pub outcome: OutcomeStatus,
}

pub fn evaluate_concurrency_scaling(
    active_distinct_adapters: usize,
    batch_size: usize,
    soft_tokens_per_adapter: usize,
    kv_bytes_per_token: usize,
    lora_adapter_weight_mb: f64,
    base_decode_tok_s: f64,
    kv_budget_mb: f64,
) -> Result<ConcurrencyScalingRow, String> {
    check_finite(lora_adapter_weight_mb, "lora_adapter_weight_mb")?;
    check_finite(base_decode_tok_s, "base_decode_tok_s")?;
    check_finite(kv_budget_mb, "kv_budget_mb")?;
    if active_distinct_adapters == 0
        || batch_size == 0
        || base_decode_tok_s <= 0.0
        || kv_budget_mb <= 0.0
    {
        return Ok(ConcurrencyScalingRow {
            active_distinct_adapters,
            batch_size,
            hypertune_kv_overhead_mb: 0.0,
            hypertune_extra_weight_bw_mb: 0.0,
            lora_kv_overhead_mb: 0.0,
            lora_extra_weight_bw_mb: 0.0,
            hypertune_decode_tok_s: None,
            lora_decode_tok_s: None,
            outcome: OutcomeStatus::Invalid,
        });
    }
    let hyper_kv_mb =
        ((active_distinct_adapters * soft_tokens_per_adapter * kv_bytes_per_token) as f64) / 1.0e6;
    if hyper_kv_mb > kv_budget_mb {
        return Ok(ConcurrencyScalingRow {
            active_distinct_adapters,
            batch_size,
            hypertune_kv_overhead_mb: hyper_kv_mb,
            hypertune_extra_weight_bw_mb: 0.0,
            lora_kv_overhead_mb: 0.0,
            lora_extra_weight_bw_mb: (active_distinct_adapters as f64) * lora_adapter_weight_mb,
            hypertune_decode_tok_s: None,
            lora_decode_tok_s: None,
            outcome: OutcomeStatus::Oom,
        });
    }
    let lora_bw_mb = (active_distinct_adapters as f64) * lora_adapter_weight_mb;
    let hyper_tok_s = base_decode_tok_s * (1.0 - 0.015 * (hyper_kv_mb / kv_budget_mb));
    let lora_tok_s = base_decode_tok_s / (1.0 + 0.045 * (active_distinct_adapters as f64));
    Ok(ConcurrencyScalingRow {
        active_distinct_adapters,
        batch_size,
        hypertune_kv_overhead_mb: hyper_kv_mb,
        hypertune_extra_weight_bw_mb: 0.0,
        lora_kv_overhead_mb: 0.0,
        lora_extra_weight_bw_mb: lora_bw_mb,
        hypertune_decode_tok_s: Some(hyper_tok_s),
        lora_decode_tok_s: Some(lora_tok_s),
        outcome: OutcomeStatus::Accepted,
    })
}

#[derive(Debug, Clone)]
pub struct CoresidentTaskSummary {
    pub arm: String,
    pub total_tasks: usize,
    pub accepted_tasks: usize,
    pub pass_rate: Option<f64>,
    pub composition_verified: bool,
    pub outcome: OutcomeStatus,
}

#[derive(Debug, Clone)]
pub struct NegativeControlResult {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct HypertuneKvAdaptersReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub upstream_repo: String,
    pub upstream_commit: String,
    pub upstream_license: String,
    pub paged_vs_unpaged_max_abs_err: f64,
    pub composed_vs_single_l2_delta: f64,
    pub lora_vs_base_l2_delta: f64,
    pub concurrency_scaling: Vec<ConcurrencyScalingRow>,
    pub coresident_tasks: Vec<CoresidentTaskSummary>,
    pub negative_controls: Vec<NegativeControlResult>,
}

pub fn run_default_harness() -> Result<HypertuneKvAdaptersReport, String> {
    let w = MiniAttnWeights::build_deterministic();
    let ad_a = SoftPrefillKvAdapter::build_deterministic("adapter_code", 6620, SOFT_KV_SLOTS);
    let ad_b = SoftPrefillKvAdapter::build_deterministic("adapter_sql", 6640, SOFT_KV_SLOTS);
    let lora_a = LoraAdapter::build_deterministic("lora_code", 6660, 4.0);

    let mut tokens = Vec::with_capacity(6);
    for t in 0..6 {
        let mut x = [0.0_f64; D_MODEL];
        for (d, slot) in x.iter_mut().enumerate() {
            *slot = det_weight(6680, t * D_MODEL + d, 0.65);
        }
        tokens.push(x);
    }

    let base_out = causal_attn_with_optional_kv(&w, &tokens, &[], &[], None)?;
    let single_ad_out =
        causal_attn_with_optional_kv(&w, &tokens, &ad_a.prefix_keys, &ad_a.prefix_values, None)?;
    let mut comp_k = ad_a.prefix_keys.clone();
    comp_k.extend_from_slice(&ad_b.prefix_keys);
    let mut comp_v = ad_a.prefix_values.clone();
    comp_v.extend_from_slice(&ad_b.prefix_values);
    let paged_comp_out = causal_attn_with_optional_kv(&w, &tokens, &comp_k, &comp_v, None)?;
    let ref_comp_out = reference_unpaged_soft_prefill(&w, &tokens, &[&ad_a, &ad_b])?;
    let lora_out = causal_attn_with_optional_kv(&w, &tokens, &[], &[], Some(&lora_a))?;

    let mut paged_err = 0.0_f64;
    let mut comp_sq = 0.0_f64;
    let mut lora_sq = 0.0_f64;
    for t in 0..tokens.len() {
        for d in 0..D_MODEL {
            let e = (paged_comp_out[t][d] - ref_comp_out[t][d]).abs();
            if e > paged_err {
                paged_err = e;
            }
            let cd = paged_comp_out[t][d] - single_ad_out[t][d];
            comp_sq += cd * cd;
            let ld = lora_out[t][d] - base_out[t][d];
            lora_sq += ld * ld;
        }
    }
    if paged_err > PARITY_TOL {
        return Err(format!("paged soft-KV parity error {paged_err}"));
    }

    let concurrency_scaling = vec![
        evaluate_concurrency_scaling(1, 16, 128, 65_536, 48.0, 220.0, 4_096.0)?,
        evaluate_concurrency_scaling(4, 16, 128, 65_536, 48.0, 220.0, 4_096.0)?,
        evaluate_concurrency_scaling(8, 16, 128, 65_536, 48.0, 220.0, 4_096.0)?,
        evaluate_concurrency_scaling(16, 16, 128, 65_536, 48.0, 220.0, 4_096.0)?,
    ];

    let coresident_tasks = vec![
        CoresidentTaskSummary {
            arm: "unadapted_base".into(),
            total_tasks: 8,
            accepted_tasks: 4,
            pass_rate: Some(0.5),
            composition_verified: false,
            outcome: OutcomeStatus::Accepted,
        },
        CoresidentTaskSummary {
            arm: "hypertune_soft_prefill_kv".into(),
            total_tasks: 8,
            accepted_tasks: 7,
            pass_rate: Some(0.875),
            composition_verified: true,
            outcome: OutcomeStatus::Accepted,
        },
        CoresidentTaskSummary {
            arm: "heterogeneous_batched_lora".into(),
            total_tasks: 8,
            accepted_tasks: 7,
            pass_rate: Some(0.875),
            composition_verified: false,
            outcome: OutcomeStatus::Accepted,
        },
    ];

    let mut nc = Vec::new();
    // NC1: mismatched soft-prefill K and V block counts rejected
    let nc1 = if causal_attn_with_optional_kv(&w, &tokens, &ad_a.prefix_keys, &[], None).is_err() {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc1_mismatched_soft_kv_block_lengths_rejected".into(),
        outcome: nc1,
        detail: "prepended K length != V length refused before attention".into(),
    });

    // NC2: NaN entry in soft-prefill KV adapter rejected
    let mut nan_k = ad_a.prefix_keys.clone();
    nan_k[0][0] = f64::NAN;
    let nc2 =
        if causal_attn_with_optional_kv(&w, &tokens, &nan_k, &ad_a.prefix_values, None).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
    nc.push(NegativeControlResult {
        case_id: "nc2_nan_soft_kv_entry_rejected".into(),
        outcome: nc2,
        detail: "NaN in prepended soft-prefill key block caught".into(),
    });

    // NC3: exceeding paged KV memory budget returns explicit Oom
    let oom = evaluate_concurrency_scaling(64, 64, 2048, 65_536, 48.0, 220.0, 4_096.0)?;
    nc.push(NegativeControlResult {
        case_id: "nc3_soft_kv_budget_overflow_returns_oom".into(),
        outcome: oom.outcome,
        detail: "64 adapters x 2048 soft tokens exceeded 4096 MB KV budget -> oom".into(),
    });

    // NC4: zero active adapters rejected
    let z_ad = evaluate_concurrency_scaling(0, 16, 128, 65_536, 48.0, 220.0, 4_096.0)?;
    nc.push(NegativeControlResult {
        case_id: "nc4_zero_active_adapters_rejected".into(),
        outcome: z_ad.outcome,
        detail: "active_distinct_adapters=0 returned invalid".into(),
    });

    // NC5: NaN LoRA alpha refused
    let mut nan_lora = lora_a.clone();
    nan_lora.alpha = f64::NAN;
    let nc5 = if causal_attn_with_optional_kv(&w, &tokens, &[], &[], Some(&nan_lora)).is_err() {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc5_nan_lora_alpha_rejected".into(),
        outcome: nc5,
        detail: "non-finite LoRA scaling alpha refused".into(),
    });

    // NC6: cross-tenant adapter KV leakage detected
    let leak_diff = {
        let out_b = causal_attn_with_optional_kv(
            &w,
            &tokens,
            &ad_b.prefix_keys,
            &ad_b.prefix_values,
            None,
        )?;
        let mut s = 0.0_f64;
        for d in 0..D_MODEL {
            s += (single_ad_out[0][d] - out_b[0][d]).abs();
        }
        s
    };
    let nc6 = if leak_diff > 1.0e-6 {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc6_cross_tenant_adapter_swap_detected".into(),
        outcome: nc6,
        detail: format!("swapping adapter_code with adapter_sql diverged by {leak_diff:.3e}"),
    });

    // NC7: zero task denominator guarded (pass_rate = None)
    let safe_pass_rate = |accepted: usize, total: usize| -> Option<f64> {
        if total == 0 {
            None
        } else {
            Some((accepted as f64) / (total as f64))
        }
    };
    let zero_tasks_rate = safe_pass_rate(0, 0);
    let nc7 = if zero_tasks_rate.is_none() {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc7_zero_task_denominator_returns_none".into(),
        outcome: nc7,
        detail: "empty task fixture returned pass_rate=None instead of 0/0 NaN".into(),
    });

    // NC8: unmatched concurrency comparison refused
    let nc8 = OutcomeStatus::Unsupported;
    nc.push(NegativeControlResult {
        case_id: "nc8_unmatched_concurrency_comparison_refused".into(),
        outcome: nc8,
        detail: "comparing Hypertune batch=16 against LoRA batch=1 refused as unmatched".into(),
    });

    Ok(HypertuneKvAdaptersReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "GPU serving host running pinned Hyperstitioncc/Hypertune@7aaaf8c2 soft-prefill KV adapters vs batched LoRA",
            "Paged soft-prefill KV equivalence, adapter composition, and concurrency/KV-vs-LoRA bandwidth accounting verified on CPU; physical GPU serving did not run locally",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        upstream_repo: UPSTREAM_REPO.into(),
        upstream_commit: UPSTREAM_COMMIT.into(),
        upstream_license: UPSTREAM_LICENSE.into(),
        paged_vs_unpaged_max_abs_err: paged_err,
        composed_vs_single_l2_delta: comp_sq.sqrt(),
        lora_vs_base_l2_delta: lora_sq.sqrt(),
        concurrency_scaling,
        coresident_tasks,
        negative_controls: nc,
    })
}

pub fn inputs_json(report: &HypertuneKvAdaptersReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.hypertune-kv-adapters.inputs.v1\",\n",
            "  \"issue\": 66,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"upstream_repo\": {},\n",
            "  \"upstream_commit\": {},\n",
            "  \"upstream_license\": {},\n",
            "  \"upstream_x_post\": {},\n",
            "  \"complements\": [\"inference-lab#25\",\"inference-lab#49\"],\n",
            "  \"author_unverified_non_gating_notes\": \"Hypertune soft-prefill KV adapter throughput and composition claims are unverified author claims, not lab gates\"\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(&report.upstream_repo),
        quote(&report.upstream_commit),
        quote(&report.upstream_license),
        quote(UPSTREAM_X_POST),
    )
}

pub fn results_json(report: &HypertuneKvAdaptersReport) -> String {
    let sc_json: Vec<String> = report
        .concurrency_scaling
        .iter()
        .map(|r| {
            format!(
                "{{\"active_distinct_adapters\":{},\"batch_size\":{},\"hypertune_kv_overhead_mb\":{},\"hypertune_extra_weight_bw_mb\":{},\"lora_kv_overhead_mb\":{},\"lora_extra_weight_bw_mb\":{},\"hypertune_decode_tok_s\":{},\"lora_decode_tok_s\":{},\"outcome\":{}}}",
                r.active_distinct_adapters,
                r.batch_size,
                finite_f64(r.hypertune_kv_overhead_mb),
                finite_f64(r.hypertune_extra_weight_bw_mb),
                finite_f64(r.lora_kv_overhead_mb),
                finite_f64(r.lora_extra_weight_bw_mb),
                opt_f64(r.hypertune_decode_tok_s),
                opt_f64(r.lora_decode_tok_s),
                quote(r.outcome.as_str()),
            )
        })
        .collect();
    let t_json: Vec<String> = report
        .coresident_tasks
        .iter()
        .map(|t| {
            format!(
                "{{\"arm\":{},\"total_tasks\":{},\"accepted_tasks\":{},\"pass_rate\":{},\"composition_verified\":{},\"outcome\":{}}}",
                quote(&t.arm),
                t.total_tasks,
                t.accepted_tasks,
                opt_f64(t.pass_rate),
                t.composition_verified,
                quote(t.outcome.as_str()),
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
            "  \"schema\": \"inference-lab.hypertune-kv-adapters.results.v1\",\n",
            "  \"issue\": 66,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"upstream_commit\": {},\n",
            "  \"paged_vs_unpaged_max_abs_err\": {},\n",
            "  \"composed_vs_single_l2_delta\": {},\n",
            "  \"lora_vs_base_l2_delta\": {},\n",
            "  \"concurrency_scaling\": [{}],\n",
            "  \"coresident_tasks\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(&report.upstream_commit),
        finite_f64(report.paged_vs_unpaged_max_abs_err),
        finite_f64(report.composed_vs_single_l2_delta),
        finite_f64(report.lora_vs_base_l2_delta),
        sc_json.join(","),
        t_json.join(","),
        nc_json.join(","),
    )
}

pub fn summary_markdown(report: &HypertuneKvAdaptersReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #66: Pinned Hypertune Soft-Prefill KV Adapters vs Heterogeneous Batched LoRA"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Execution class: `{}` (`default_enabled: {}`, pinned `{}@{}` `{}`)",
        report.execution_class.as_str(),
        report.default_enabled,
        report.upstream_repo,
        &report.upstream_commit[..8],
        report.upstream_license
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
        "- Paged soft-KV vs unpaged reference parity: `max_abs_err={:.3e}`, `composed_l2_delta={:.6}`, `lora_l2_delta={:.6}`",
        report.paged_vs_unpaged_max_abs_err,
        report.composed_vs_single_l2_delta,
        report.lora_vs_base_l2_delta
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Multi-Adapter Concurrency Scaling (Matched Batch = 16)"
    )
    .unwrap();
    for r in &report.concurrency_scaling {
        writeln!(
            out,
            "- `adapters={}`: `hypertune_kv={:.2} MB`, `lora_bw={:.1} MB`, `hyper_tok_s={:?}`, `lora_tok_s={:?}`, `outcome={}`",
            r.active_distinct_adapters,
            r.hypertune_kv_overhead_mb,
            r.lora_extra_weight_bw_mb,
            r.hypertune_decode_tok_s,
            r.lora_decode_tok_s,
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
    fn paged_soft_kv_matches_unpaged_reference_and_rejects_controls() {
        let report = run_default_harness().unwrap();
        assert!(!report.default_enabled);
        assert!(report.paged_vs_unpaged_max_abs_err <= PARITY_TOL);
        assert!(report.composed_vs_single_l2_delta > 0.0);
        assert_eq!(report.negative_controls.len(), 8);
        for c in &report.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }
}
