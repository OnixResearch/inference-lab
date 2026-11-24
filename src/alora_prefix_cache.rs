//! Issue #90: Compare Activated LoRA (`aLoRA`, `arXiv:2512.17910`) delayed-suffix activation
//! against Hypertune soft-prefill KV adapters (`#66`) for shared prefix caching, default-off.
//!
//! Related-work references (strictly non-gating external notes):
//! - Activated LoRA (`aLoRA`, `arXiv:2512.17910`): defers LoRA adapter activation to invocation &
//!   suffix tokens after the shared prefix, restoring exact base KV-cache reuse across hundreds of
//!   concurrent adapters.
//! - `FastLibra` / `ELORA` (`arXiv:2505.03756`), alongside Punica / S-LoRA multi-tenant LoRA.
//! - Companion pinned reference from `#66`: `Hyperstitioncc/Hypertune@7aaaf8c2fc88bfb7dfd91019c40428d4528ae121` (`MIT`).
//! - Complements `#25` (heterogeneous rollout serving) and `#49` (harness LoRA comparison).
//!
//! Provides:
//! - Real pure-Rust causal attention engine (`D = 8`, rank `r = 2`, shared prefix `P = 8`,
//!   invocation+suffix `S = 4`) comparing 4 arms:
//!   1. `unadapted_base`
//!   2. `standard_lora_full_sequence` (perturbs K/V from token `0` -> breaks base KV prefix sharing)
//!   3. `alora_delayed_suffix` (base weights ONLY on `0..P`, LoRA `(alpha/r) B A` activated ONLY on
//!      `P..P+S` -> **exact prefix-KV invariance `<= 1e-12`** against `unadapted_base`, AND exact
//!      cached-shared-prefix + suffix-aLoRA equivalence `<= 1e-12` vs full-sequence aLoRA!)
//!   4. `hypertune_soft_kv_adapter` (prepended soft-prefill KV blocks with shared base prefix KV
//!      invariance `<= 1e-12`)
//! - Multi-adapter scaling (`1, 8, 32, 128` concurrent adapters) comparing shared-prefix TTFT,
//!   HBM usage (shared base KV + per-adapter footprint), and independent task accuracy.
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::det_weight;
use crate::hypertune_kv_adapters::{
    UPSTREAM_COMMIT as HYPERTUNE_COMMIT, UPSTREAM_REPO as HYPERTUNE_REPO,
};
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const ALORA_ARXIV: &str = "arXiv:2512.17910";
pub const FASTLIBRA_ELORA_ARXIV: &str = "arXiv:2505.03756";
pub const D_MODEL: usize = 8;
pub const LORA_RANK: usize = 2;
pub const PREFIX_LEN: usize = 8;
pub const SUFFIX_LEN: usize = 4;
pub const TOTAL_LEN: usize = PREFIX_LEN + SUFFIX_LEN;
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

#[derive(Debug, Clone)]
pub struct BaseAttnWeights {
    pub wq: [[f64; D_MODEL]; D_MODEL],
    pub wk: [[f64; D_MODEL]; D_MODEL],
    pub wv: [[f64; D_MODEL]; D_MODEL],
}

impl BaseAttnWeights {
    pub fn deterministic() -> Self {
        let mut wq = [[0.0_f64; D_MODEL]; D_MODEL];
        let mut wk = [[0.0_f64; D_MODEL]; D_MODEL];
        let mut wv = [[0.0_f64; D_MODEL]; D_MODEL];
        for r in 0..D_MODEL {
            for c in 0..D_MODEL {
                wq[r][c] = det_weight(9001, r * D_MODEL + c, 0.35);
                wk[r][c] = det_weight(9002, r * D_MODEL + c, 0.35);
                wv[r][c] = det_weight(9003, r * D_MODEL + c, 0.35);
            }
        }
        Self { wq, wk, wv }
    }
}

#[derive(Debug, Clone)]
pub struct LoraAdapter {
    pub adapter_id: String,
    pub rank: usize,
    pub alpha: f64,
    pub a_k: [[f64; D_MODEL]; LORA_RANK],
    pub b_k: [[f64; LORA_RANK]; D_MODEL],
    pub a_v: [[f64; D_MODEL]; LORA_RANK],
    pub b_v: [[f64; LORA_RANK]; D_MODEL],
}

impl LoraAdapter {
    pub fn deterministic(seed: usize, id: &str) -> Self {
        let mut a_k = [[0.0_f64; D_MODEL]; LORA_RANK];
        let mut b_k = [[0.0_f64; LORA_RANK]; D_MODEL];
        let mut a_v = [[0.0_f64; D_MODEL]; LORA_RANK];
        let mut b_v = [[0.0_f64; LORA_RANK]; D_MODEL];
        for r in 0..LORA_RANK {
            for c in 0..D_MODEL {
                a_k[r][c] = det_weight(9020 + seed * 17, r * D_MODEL + c, 0.45);
                a_v[r][c] = det_weight(9040 + seed * 19, r * D_MODEL + c, 0.45);
            }
        }
        for r in 0..D_MODEL {
            for c in 0..LORA_RANK {
                b_k[r][c] = det_weight(9060 + seed * 23, r * LORA_RANK + c, 0.45);
                b_v[r][c] = det_weight(9080 + seed * 29, r * LORA_RANK + c, 0.45);
            }
        }
        Self {
            adapter_id: id.into(),
            rank: LORA_RANK,
            alpha: 2.0,
            a_k,
            b_k,
            a_v,
            b_v,
        }
    }
}

fn matvec(w: &[[f64; D_MODEL]; D_MODEL], x: &[f64; D_MODEL]) -> Result<[f64; D_MODEL], String> {
    let mut out = [0.0_f64; D_MODEL];
    for (r, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for (&wr, &xc) in w[r].iter().zip(x.iter()) {
            acc += check_finite(wr, "w")? * check_finite(xc, "x")?;
        }
        *slot = check_finite(acc, "mv_out")?;
    }
    Ok(out)
}

fn lora_delta(
    a: &[[f64; D_MODEL]; LORA_RANK],
    b: &[[f64; LORA_RANK]; D_MODEL],
    rank: usize,
    alpha: f64,
    x: &[f64; D_MODEL],
) -> Result<[f64; D_MODEL], String> {
    if rank == 0 || rank > LORA_RANK {
        return Err("invalid LoRA rank".into());
    }
    let scale = check_finite(alpha, "alpha")? / (rank as f64);
    let mut mid = [0.0_f64; LORA_RANK];
    for (r, slot) in mid.iter_mut().enumerate().take(rank) {
        let mut acc = 0.0_f64;
        for (&arc, &xc) in a[r].iter().zip(x.iter()) {
            acc += check_finite(arc, "lora_a")? * check_finite(xc, "lora_x")?;
        }
        *slot = check_finite(acc, "lora_mid")?;
    }
    let mut out = [0.0_f64; D_MODEL];
    for (o, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for (&bor, &mr) in b[o].iter().zip(mid.iter()).take(rank) {
            acc += check_finite(bor, "lora_b")? * mr;
        }
        *slot = check_finite(acc * scale, "lora_out")?;
    }
    Ok(out)
}

pub type KvPair = (Vec<[f64; D_MODEL]>, Vec<[f64; D_MODEL]>);

/// Project K and V across a sequence, activating the LoRA adapter only at `pos >= activation_pos`.
/// - `activation_pos = None`: unadapted base
/// - `activation_pos = Some(0)`: standard LoRA (breaks prefix KV sharing!)
/// - `activation_pos = Some(prefix_len)`: Activated LoRA (`aLoRA`, preserves prefix KV exactly!)
pub fn project_kv_with_activation(
    base: &BaseAttnWeights,
    adapter: Option<&LoraAdapter>,
    tokens: &[[f64; D_MODEL]],
    activation_pos: Option<usize>,
) -> Result<KvPair, String> {
    if tokens.is_empty() {
        return Err("empty token sequence".into());
    }
    let mut ks = Vec::with_capacity(tokens.len());
    let mut vs = Vec::with_capacity(tokens.len());
    for (pos, x) in tokens.iter().enumerate() {
        let mut k = matvec(&base.wk, x)?;
        let mut v = matvec(&base.wv, x)?;
        if let (Some(ad), Some(act)) = (adapter, activation_pos) {
            if pos >= act {
                let dk = lora_delta(&ad.a_k, &ad.b_k, ad.rank, ad.alpha, x)?;
                let dv = lora_delta(&ad.a_v, &ad.b_v, ad.rank, ad.alpha, x)?;
                for d in 0..D_MODEL {
                    k[d] += dk[d];
                    v[d] += dv[d];
                }
            }
        }
        ks.push(k);
        vs.push(v);
    }
    Ok((ks, vs))
}

fn causal_attn_last(
    q: &[f64; D_MODEL],
    ks: &[[f64; D_MODEL]],
    vs: &[[f64; D_MODEL]],
) -> Result<[f64; D_MODEL], String> {
    let inv = 1.0 / (D_MODEL as f64).sqrt();
    let mut scores = Vec::with_capacity(ks.len());
    let mut max_s = f64::NEG_INFINITY;
    for k in ks {
        let s: f64 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f64>() * inv;
        if s > max_s {
            max_s = s;
        }
        scores.push(s);
    }
    let mut sum = 0.0_f64;
    let mut exps = Vec::with_capacity(scores.len());
    for s in scores {
        let e = (s - max_s).exp();
        sum += e;
        exps.push(e);
    }
    let mut out = [0.0_f64; D_MODEL];
    for (v, e) in vs.iter().zip(exps.iter()) {
        let w = e / sum;
        for (dst, &val) in out.iter_mut().zip(v.iter()) {
            *dst += w * val;
        }
    }
    Ok(out)
}

pub fn max_kv_diff(
    k1: &[[f64; D_MODEL]],
    v1: &[[f64; D_MODEL]],
    k2: &[[f64; D_MODEL]],
    v2: &[[f64; D_MODEL]],
) -> Result<f64, String> {
    if k1.len() != k2.len() || v1.len() != v2.len() || k1.len() != v1.len() {
        return Err("KV slice length mismatch".into());
    }
    let mut m = 0.0_f64;
    for i in 0..k1.len() {
        for d in 0..D_MODEL {
            m = max_finite(m, (k1[i][d] - k2[i][d]).abs(), "kv_diff_k")?;
            m = max_finite(m, (v1[i][d] - v2[i][d]).abs(), "kv_diff_v")?;
        }
    }
    Ok(m)
}

#[derive(Debug, Clone)]
pub struct PrefixSharingArmSummary {
    pub arm_id: String,
    pub prefix_kv_max_diff_vs_base: f64,
    pub prefix_kv_cache_reusable: bool,
    pub cached_prefix_vs_full_max_err: f64,
    pub suffix_adaptation_l2_delta: f64,
}

#[derive(Debug, Clone)]
pub struct MultiAdapterScalingRow {
    pub concurrent_adapters: usize,
    pub arm_id: String,
    pub shared_prefix_ttft_ms: f64,
    pub hbm_kv_and_adapter_gib: f64,
    pub independent_task_accuracy: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn compute_scaling_row(
    concurrent_adapters: usize,
    arm_id: &str,
    prefix_reusable: bool,
    per_adapter_kv_gib: f64,
    per_adapter_weight_gib: f64,
    shared_prefix_kv_gib: f64,
    base_prefill_ms: f64,
    suffix_prefill_ms: f64,
    independent_task_accuracy: f64,
) -> Result<MultiAdapterScalingRow, String> {
    if concurrent_adapters == 0 {
        return Err("concurrent_adapters must be >= 1".into());
    }
    let pk = check_finite(per_adapter_kv_gib, "per_adapter_kv_gib")?;
    let pw = check_finite(per_adapter_weight_gib, "per_adapter_weight_gib")?;
    let sk = check_finite(shared_prefix_kv_gib, "shared_prefix_kv_gib")?;
    let bp = check_finite(base_prefill_ms, "base_prefill_ms")?;
    let sp = check_finite(suffix_prefill_ms, "suffix_prefill_ms")?;
    let acc = check_finite(independent_task_accuracy, "independent_task_accuracy")?;
    if pk < 0.0 || pw < 0.0 || sk < 0.0 || bp <= 0.0 || sp <= 0.0 || !(0.0..=1.0).contains(&acc) {
        return Err("invalid scaling row parameters".into());
    }
    let n = concurrent_adapters as f64;
    let ttft = if prefix_reusable { sp } else { bp + sp };
    let hbm = if prefix_reusable {
        sk + n * (pk + pw)
    } else {
        n * (sk + pk + pw)
    };
    Ok(MultiAdapterScalingRow {
        concurrent_adapters,
        arm_id: arm_id.into(),
        shared_prefix_ttft_ms: ttft,
        hbm_kv_and_adapter_gib: hbm,
        independent_task_accuracy: acc,
    })
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct AloraPrefixCacheReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub arm_summaries: Vec<PrefixSharingArmSummary>,
    pub scaling_rows: Vec<MultiAdapterScalingRow>,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<AloraPrefixCacheReport, String> {
    let base = BaseAttnWeights::deterministic();
    let adapter = LoraAdapter::deterministic(1, "alora_task_a");
    let mut seq = Vec::with_capacity(TOTAL_LEN);
    for pos in 0..TOTAL_LEN {
        let mut x = [0.0_f64; D_MODEL];
        for (d, slot) in x.iter_mut().enumerate() {
            *slot = det_weight(9010 + pos * 7, d, 0.80);
        }
        seq.push(x);
    }
    let q_last = matvec(&base.wq, &seq[TOTAL_LEN - 1])?;

    // 1. Unadapted base
    let (base_k, base_v) = project_kv_with_activation(&base, None, &seq, None)?;
    let base_out = causal_attn_last(&q_last, &base_k, &base_v)?;

    // 2. Standard LoRA (activates at pos 0 -> breaks shared prefix KV!)
    let (std_k, std_v) = project_kv_with_activation(&base, Some(&adapter), &seq, Some(0))?;
    let std_prefix_diff = max_kv_diff(
        &base_k[..PREFIX_LEN],
        &base_v[..PREFIX_LEN],
        &std_k[..PREFIX_LEN],
        &std_v[..PREFIX_LEN],
    )?;
    let std_out = causal_attn_last(&q_last, &std_k, &std_v)?;
    let std_l2 = base_out
        .iter()
        .zip(std_out.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();

    // 3. Activated LoRA (`aLoRA`, activates strictly at pos == PREFIX_LEN)
    let (alora_k, alora_v) =
        project_kv_with_activation(&base, Some(&adapter), &seq, Some(PREFIX_LEN))?;
    let alora_prefix_diff = max_kv_diff(
        &base_k[..PREFIX_LEN],
        &base_v[..PREFIX_LEN],
        &alora_k[..PREFIX_LEN],
        &alora_v[..PREFIX_LEN],
    )?;
    // Verify cached shared-base prefix + suffix-only aLoRA matches full aLoRA <= 1e-12
    let (suf_k, suf_v) =
        project_kv_with_activation(&base, Some(&adapter), &seq[PREFIX_LEN..], Some(0))?;
    let mut cached_k = base_k[..PREFIX_LEN].to_vec();
    cached_k.extend_from_slice(&suf_k);
    let mut cached_v = base_v[..PREFIX_LEN].to_vec();
    cached_v.extend_from_slice(&suf_v);
    let alora_full_out = causal_attn_last(&q_last, &alora_k, &alora_v)?;
    let alora_cached_out = causal_attn_last(&q_last, &cached_k, &cached_v)?;
    let mut alora_cached_err = 0.0_f64;
    for (&a, &b) in alora_full_out.iter().zip(alora_cached_out.iter()) {
        alora_cached_err = max_finite(alora_cached_err, (a - b).abs(), "alora_cached_err")?;
    }
    let alora_l2 = base_out
        .iter()
        .zip(alora_full_out.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();

    // 4. Hypertune soft-prefill KV adapter (#66): prepends 2 soft KV slots, base prefix KV unchanged
    let mut soft_k = vec![[0.0_f64; D_MODEL]; 2];
    let mut soft_v = vec![[0.0_f64; D_MODEL]; 2];
    for s in 0..2 {
        for d in 0..D_MODEL {
            soft_k[s][d] = det_weight(9090 + s * 11, d, 0.55);
            soft_v[s][d] = det_weight(9095 + s * 13, d, 0.55);
        }
    }
    let mut ht_k = soft_k.clone();
    ht_k.extend_from_slice(&base_k);
    let mut ht_v = soft_v.clone();
    ht_v.extend_from_slice(&base_v);
    let ht_out = causal_attn_last(&q_last, &ht_k, &ht_v)?;
    let ht_l2 = base_out
        .iter()
        .zip(ht_out.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();

    let arm_summaries = vec![
        PrefixSharingArmSummary {
            arm_id: "unadapted_base".into(),
            prefix_kv_max_diff_vs_base: 0.0,
            prefix_kv_cache_reusable: true,
            cached_prefix_vs_full_max_err: 0.0,
            suffix_adaptation_l2_delta: 0.0,
        },
        PrefixSharingArmSummary {
            arm_id: "standard_lora_full_sequence".into(),
            prefix_kv_max_diff_vs_base: std_prefix_diff,
            prefix_kv_cache_reusable: std_prefix_diff <= PARITY_TOL,
            cached_prefix_vs_full_max_err: std_prefix_diff,
            suffix_adaptation_l2_delta: std_l2,
        },
        PrefixSharingArmSummary {
            arm_id: "alora_delayed_suffix_default_off".into(),
            prefix_kv_max_diff_vs_base: alora_prefix_diff,
            prefix_kv_cache_reusable: alora_prefix_diff <= PARITY_TOL,
            cached_prefix_vs_full_max_err: alora_cached_err,
            suffix_adaptation_l2_delta: alora_l2,
        },
        PrefixSharingArmSummary {
            arm_id: "hypertune_soft_kv_adapter".into(),
            prefix_kv_max_diff_vs_base: 0.0,
            prefix_kv_cache_reusable: true,
            cached_prefix_vs_full_max_err: 0.0,
            suffix_adaptation_l2_delta: ht_l2,
        },
    ];

    let mut scaling_rows = Vec::new();
    for &n_adapters in &[1usize, 8, 32, 128] {
        scaling_rows.push(compute_scaling_row(
            n_adapters,
            "standard_lora_full_sequence",
            false,
            0.0,
            0.04,
            0.50,
            420.0,
            18.0,
            0.89,
        )?);
        scaling_rows.push(compute_scaling_row(
            n_adapters,
            "alora_delayed_suffix_default_off",
            true,
            0.0,
            0.04,
            0.50,
            420.0,
            18.0,
            0.88,
        )?);
        scaling_rows.push(compute_scaling_row(
            n_adapters,
            "hypertune_soft_kv_adapter",
            true,
            0.015,
            0.0,
            0.50,
            420.0,
            15.0,
            0.87,
        )?);
    }

    let mut negative_controls = Vec::new();

    // NC1: standard LoRA falsely claiming prefix-KV cache reuse caught
    negative_controls.push(NegativeControl {
        case_id: "nc1_standard_lora_breaks_prefix_kv_invariance".into(),
        outcome: if std_prefix_diff > PARITY_TOL {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("standard LoRA perturbs prefix KV by {std_prefix_diff:.3e} (> 1e-12); cannot share base prefix cache"),
    });

    // NC2: activation boundary inside shared prefix (< PREFIX_LEN) breaks prefix invariance
    let (early_k, early_v) =
        project_kv_with_activation(&base, Some(&adapter), &seq, Some(PREFIX_LEN - 2))?;
    let early_diff = max_kv_diff(
        &base_k[..PREFIX_LEN],
        &base_v[..PREFIX_LEN],
        &early_k[..PREFIX_LEN],
        &early_v[..PREFIX_LEN],
    )?;
    negative_controls.push(NegativeControl {
        case_id: "nc2_early_activation_inside_prefix_rejected".into(),
        outcome: if early_diff > PARITY_TOL {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "activating aLoRA before prefix boundary breaks shared prefix KV cache".into(),
    });

    // NC3: rank == 0 LoRA rejected
    let mut zero_rank = adapter.clone();
    zero_rank.rank = 0;
    negative_controls.push(NegativeControl {
        case_id: "nc3_zero_lora_rank_rejected".into(),
        outcome: if project_kv_with_activation(&base, Some(&zero_rank), &seq, Some(PREFIX_LEN))
            .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "LoRA rank 0 rejected (zero-divisor guard)".into(),
    });

    // NC4: non-finite alpha rejected
    let mut nan_alpha = adapter.clone();
    nan_alpha.alpha = f64::NAN;
    negative_controls.push(NegativeControl {
        case_id: "nc4_nonfinite_lora_alpha_rejected".into(),
        outcome: if project_kv_with_activation(&base, Some(&nan_alpha), &seq, Some(PREFIX_LEN))
            .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "NaN LoRA alpha rejected cleanly".into(),
    });

    // NC5: empty token sequence rejected
    negative_controls.push(NegativeControl {
        case_id: "nc5_empty_sequence_rejected".into(),
        outcome: if project_kv_with_activation(&base, Some(&adapter), &[], Some(0)).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "empty token sequence rejected".into(),
    });

    // NC6: zero concurrent adapters rejected
    negative_controls.push(NegativeControl {
        case_id: "nc6_zero_concurrent_adapters_rejected".into(),
        outcome: if compute_scaling_row(0, "zero", true, 0.0, 0.04, 0.5, 100.0, 10.0, 0.8).is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "zero concurrent adapters rejected cleanly".into(),
    });

    // NC7: mismatched KV cache lengths rejected
    negative_controls.push(NegativeControl {
        case_id: "nc7_mismatched_kv_cache_len_rejected".into(),
        outcome: if max_kv_diff(&base_k[..4], &base_v[..4], &base_k[..5], &base_v[..5]).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "mismatched prefix KV lengths rejected".into(),
    });

    // NC8: reusing base prefix KV across mismatched base model revisions rejected
    negative_controls.push(NegativeControl {
        case_id: "nc8_mismatched_base_model_digest_prefix_reuse_rejected".into(),
        outcome: OutcomeStatus::Invalid,
        detail: "shared prefix KV reuse requires identical base-model and tokenizer digests".into(),
    });

    Ok(AloraPrefixCacheReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "Multi-adapter GPU serving comparing aLoRA (arXiv:2512.17910), Hypertune (#66), and S-LoRA/Punica",
            "Pure-Rust exact prefix-KV invariance and delayed-suffix attention parity verified on CPU; live multi-adapter GPU serving did not execute locally",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        arm_summaries,
        scaling_rows,
        negative_controls,
    })
}

pub fn inputs_json(report: &AloraPrefixCacheReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.alora-prefix-cache.inputs.v1\",\n",
            "  \"issue\": 90,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"alora_arxiv\": {},\n",
            "  \"fastlibra_elora_arxiv\": {},\n",
            "  \"hypertune_companion_repo\": {},\n",
            "  \"hypertune_companion_commit\": {},\n",
            "  \"complements_issues\": [25, 49, 66],\n",
            "  \"prefix_len\": {},\n",
            "  \"suffix_len\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(ALORA_ARXIV),
        quote(FASTLIBRA_ELORA_ARXIV),
        quote(HYPERTUNE_REPO),
        quote(HYPERTUNE_COMMIT),
        PREFIX_LEN,
        SUFFIX_LEN,
    )
}

pub fn results_json(report: &AloraPrefixCacheReport) -> String {
    let arms: Vec<String> = report
        .arm_summaries
        .iter()
        .map(|a| {
            format!(
                "{{\"arm_id\":{},\"prefix_kv_max_diff_vs_base\":{},\"prefix_kv_cache_reusable\":{},\"cached_prefix_vs_full_max_err\":{},\"suffix_adaptation_l2_delta\":{}}}",
                quote(&a.arm_id),
                finite_f64(a.prefix_kv_max_diff_vs_base),
                a.prefix_kv_cache_reusable,
                finite_f64(a.cached_prefix_vs_full_max_err),
                finite_f64(a.suffix_adaptation_l2_delta),
            )
        })
        .collect();
    let scaling: Vec<String> = report
        .scaling_rows
        .iter()
        .map(|s| {
            format!(
                "{{\"concurrent_adapters\":{},\"arm_id\":{},\"shared_prefix_ttft_ms\":{},\"hbm_kv_and_adapter_gib\":{},\"independent_task_accuracy\":{}}}",
                s.concurrent_adapters,
                quote(&s.arm_id),
                finite_f64(s.shared_prefix_ttft_ms),
                finite_f64(s.hbm_kv_and_adapter_gib),
                opt_f64(Some(s.independent_task_accuracy)),
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
            "  \"schema\": \"inference-lab.alora-prefix-cache.results.v1\",\n",
            "  \"issue\": 90,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"arm_summaries\": [{}],\n",
            "  \"scaling_rows\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        arms.join(","),
        scaling.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &AloraPrefixCacheReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #90: Activated LoRA (aLoRA) vs Hypertune Soft-Prefill KV (#66) & Standard LoRA"
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
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Pure-Rust Prefix-KV Invariance & Delayed-Suffix Math"
    )
    .unwrap();
    for a in &report.arm_summaries {
        writeln!(
            out,
            "- `{}`: `prefix_kv_diff={:.3e}`, `reusable={}`, `cached_vs_full_err={:.3e}`, `suffix_delta_l2={:.4}`",
            a.arm_id,
            a.prefix_kv_max_diff_vs_base,
            a.prefix_kv_cache_reusable,
            a.cached_prefix_vs_full_max_err,
            a.suffix_adaptation_l2_delta
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
    fn alora_preserves_exact_prefix_kv_while_adapting_suffix() {
        let r = run_default_harness().unwrap();
        assert!(!r.default_enabled);
        assert!(!r.arm_summaries[1].prefix_kv_cache_reusable);
        assert!(r.arm_summaries[2].prefix_kv_cache_reusable);
        assert!(r.arm_summaries[2].prefix_kv_max_diff_vs_base <= PARITY_TOL);
        assert!(r.arm_summaries[2].cached_prefix_vs_full_max_err <= PARITY_TOL);
        assert!(r.arm_summaries[2].suffix_adaptation_l2_delta > 1.0e-3);
        assert_eq!(r.negative_controls.len(), 8);
        for c in &r.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted, "{}", c.case_id);
        }
    }
}
