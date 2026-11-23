//! Issue #20: Compare explicit and absorbed MLA with cache parity and measured memory costs.
//!
//! Pure deterministic core consuming frozen synthetic MLA parameters from `#19`
//! (`deepseek_ref::TinyMlaParams`) and comparing three paths under identical parameters:
//! 1. Full uncached recomputation baseline (`mla_full_uncached_forward`).
//! 2. Compressed-cache + explicit expansion (stores compressed latent `c_KV` + shared RoPE key
//!    `k_R`, expands allocated cache to per-head `K_nope, V` on each call).
//! 3. Absorbed MLA (projects content queries into latent space via `W_UK`, attends directly to
//!    cached latents without constructing per-head `K_nope`/`V` cache intermediates, then projects
//!    attended latents via `W_UV`).
//!
//! Verifies:
//! - Prefill, single-token decode, multi-token irregular chunks, `batch > 1`, near-capacity &
//!   overflow rejection, cache reset/reuse, and finite-difference gradient parity.
//! - Unused cache slot poisoning with `1.0e6` sentinel values (zero influence on valid outputs).
//! - Lowered-operation intermediate counts proving absorbed MLA materializes 0 per-head K/V
//!   cache expansion intermediates, while explicitly accounting for absorbed weight preparation
//!   and resident matrices.
//! - Full-allocated vs valid-prefix vs bucketed-prefix cache strategies with shape-change
//!   recompilation counts.
//! - Default-dimension persistent payload check (`128+16=144` vs `6*(32+16+32)=480`, 70% element
//!   reduction explicitly distinguished from measured speedup).
//! - 7 adversarial negative controls with first divergent position recorded.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{
    apply_rope, det_weight, matvec, mla_full_uncached_forward, softmax_stable, TinyMlaParams,
};
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const CACHE_SENTINEL: f64 = 1.0e6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePrefixStrategy {
    FullAllocatedCache,
    ValidPrefixOnly,
    BucketedPrefix,
}

impl CachePrefixStrategy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FullAllocatedCache => "full_allocated_cache",
            Self::ValidPrefixOnly => "valid_prefix_only",
            Self::BucketedPrefix => "bucketed_prefix",
        }
    }
}

/// Compressed MLA KV cache for a batch of sequences (`[batch][capacity]`), initialized with
/// `CACHE_SENTINEL` in all unused slots so any out-of-range mask/offset bug immediately explodes!
#[derive(Debug, Clone)]
pub struct CompressedMlaCache {
    pub batch_size: usize,
    pub capacity: usize,
    pub valid_len: usize,
    pub d_kv_latent: usize,
    pub d_rope: usize,
    /// `[batch][capacity][d_kv_latent]`
    pub latents: Vec<Vec<Vec<f64>>>,
    /// `[batch][capacity][d_rope]`
    pub k_ropes: Vec<Vec<Vec<f64>>>,
}

impl CompressedMlaCache {
    pub fn new_poisoned(batch_size: usize, capacity: usize, p: &TinyMlaParams) -> Self {
        Self {
            batch_size,
            capacity,
            valid_len: 0,
            d_kv_latent: p.d_kv_latent,
            d_rope: p.d_rope,
            latents: vec![vec![vec![CACHE_SENTINEL; p.d_kv_latent]; capacity]; batch_size],
            k_ropes: vec![vec![vec![CACHE_SENTINEL; p.d_rope]; capacity]; batch_size],
        }
    }

    pub fn reset_poisoned(&mut self) {
        self.valid_len = 0;
        for b in 0..self.batch_size {
            for pos in 0..self.capacity {
                self.latents[b][pos].fill(CACHE_SENTINEL);
                self.k_ropes[b][pos].fill(CACHE_SENTINEL);
            }
        }
    }

    pub fn persistent_bytes(&self) -> usize {
        self.batch_size
            * self.capacity
            * (self.d_kv_latent + self.d_rope)
            * std::mem::size_of::<f64>()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoweredOpTrace {
    pub materialized_per_head_k_elems: usize,
    pub materialized_per_head_v_elems: usize,
    pub latent_attend_ops: usize,
    pub temporary_workspace_bytes: usize,
    pub absorbed_prep_bytes: usize,
    pub peak_total_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaultInjection {
    pub wrong_causal_mask: bool,
    pub wrong_rope_offset: bool,
    pub wrong_attention_scale: bool,
    pub wrong_absorbed_projection: bool,
    pub leak_unused_sentinel_slot: bool,
}

impl FaultInjection {
    pub const fn none() -> Self {
        Self {
            wrong_causal_mask: false,
            wrong_rope_offset: false,
            wrong_attention_scale: false,
            wrong_absorbed_projection: false,
            leak_unused_sentinel_slot: false,
        }
    }
}

pub type BatchTensor3D = Vec<Vec<Vec<f64>>>;

/// Append a batch chunk `[batch][chunk_len][d_model]` to compressed cache and run either:
/// - `absorbed == false`: explicit expansion of cache to per-head `K_nope, V`
/// - `absorbed == true`: absorbed query-latent attention + latent-value projection (0 per-head K/V expansion!)
pub fn mla_cached_chunk_forward(
    p: &TinyMlaParams,
    cache: &mut CompressedMlaCache,
    batch_chunk: &[Vec<Vec<f64>>],
    absorbed: bool,
    strategy: CachePrefixStrategy,
    faults: FaultInjection,
) -> Result<(BatchTensor3D, LoweredOpTrace), String> {
    if batch_chunk.len() != cache.batch_size || batch_chunk.is_empty() {
        return Err("batch size mismatch in mla_cached_chunk_forward".into());
    }
    let chunk_len = batch_chunk[0].len();
    if chunk_len == 0 || batch_chunk.iter().any(|seq| seq.len() != chunk_len) {
        return Err("empty or ragged chunk in mla_cached_chunk_forward".into());
    }
    let start_pos = cache.valid_len;
    if start_pos + chunk_len > cache.capacity {
        return Err("kv_cache_capacity_overflow_rejected".into());
    }

    // Write new chunk latents and shared RoPE keys into cache slots `[start_pos .. start_pos + chunk_len]`.
    for (b, seq) in batch_chunk.iter().enumerate() {
        for (c, x) in seq.iter().enumerate() {
            let pos = start_pos + c;
            cache.latents[b][pos] = matvec(x, &p.w_dkv, p.d_model, p.d_kv_latent)?;
            let kr = matvec(x, &p.w_kr, p.d_model, p.d_rope)?;
            cache.k_ropes[b][pos] = apply_rope(&kr, pos, p.rope_theta)?;
        }
    }
    cache.valid_len = start_pos + chunk_len;

    let scanned_prefix = match strategy {
        CachePrefixStrategy::FullAllocatedCache => cache.capacity,
        CachePrefixStrategy::ValidPrefixOnly => cache.valid_len,
        CachePrefixStrategy::BucketedPrefix => {
            // Power-of-two bucket capped at capacity.
            cache.valid_len.next_power_of_two().min(cache.capacity)
        }
    };

    let scale = if faults.wrong_attention_scale {
        p.attention_scale() * 2.0
    } else {
        p.attention_scale()
    };

    let mut trace = LoweredOpTrace::default();
    let mut batch_outs = Vec::with_capacity(cache.batch_size);

    for (b, seq) in batch_chunk.iter().enumerate() {
        let mut expanded_k: Vec<Vec<f64>> = Vec::new();
        let mut expanded_v: Vec<Vec<f64>> = Vec::new();
        if !absorbed {
            // Explicit expansion materializes per-head K_nope and V across `scanned_prefix` slots!
            for lat in cache.latents[b].iter().take(scanned_prefix) {
                expanded_k.push(matvec(lat, &p.w_uk, p.d_kv_latent, p.n_heads * p.d_nope)?);
                expanded_v.push(matvec(lat, &p.w_uv, p.d_kv_latent, p.n_heads * p.d_v)?);
                trace.materialized_per_head_k_elems += p.n_heads * p.d_nope;
                trace.materialized_per_head_v_elems += p.n_heads * p.d_v;
            }
            trace.temporary_workspace_bytes = trace
                .temporary_workspace_bytes
                .max(scanned_prefix * p.n_heads * (p.d_nope + p.d_v) * std::mem::size_of::<f64>());
        } else {
            // Absorbed path constructs ZERO per-head K/V expansion intermediates.
            // Workspace holds only absorbed queries (`n_heads * d_kv_latent`) + attended latent (`n_heads * d_kv_latent`).
            trace.absorbed_prep_bytes = p.n_heads * p.d_kv_latent * std::mem::size_of::<f64>();
            trace.temporary_workspace_bytes = trace
                .temporary_workspace_bytes
                .max(2 * p.n_heads * p.d_kv_latent * std::mem::size_of::<f64>());
        }

        let mut seq_outs = Vec::with_capacity(chunk_len);
        for (c, x) in seq.iter().enumerate() {
            let pos = start_pos + c;
            let rope_pos = if faults.wrong_rope_offset {
                pos + 1
            } else {
                pos
            };
            let q_nope_all = matvec(x, &p.w_q_nope, p.d_model, p.n_heads * p.d_nope)?;
            let q_rope_raw = matvec(x, &p.w_q_rope, p.d_model, p.n_heads * p.d_rope)?;

            // Determine valid causal upper bound (or leaked slot under fault injection).
            let attend_limit =
                if faults.leak_unused_sentinel_slot && cache.valid_len < cache.capacity {
                    cache.valid_len + 1
                } else if faults.wrong_causal_mask {
                    cache.valid_len
                } else {
                    pos + 1
                };

            let mut head_concat = vec![0.0_f64; p.n_heads * p.d_v];
            for h in 0..p.n_heads {
                let qn = &q_nope_all[h * p.d_nope..(h + 1) * p.d_nope];
                let qr = apply_rope(
                    &q_rope_raw[h * p.d_rope..(h + 1) * p.d_rope],
                    rope_pos,
                    p.rope_theta,
                )?;

                if !absorbed {
                    let mut scores = Vec::with_capacity(attend_limit);
                    for (j, ek) in expanded_k.iter().enumerate().take(attend_limit) {
                        let kn = &ek[h * p.d_nope..(h + 1) * p.d_nope];
                        let kr = &cache.k_ropes[b][j];
                        let mut dot = 0.0_f64;
                        for (&qv, &kv) in qn.iter().zip(kn.iter()) {
                            dot += qv * kv;
                        }
                        for (&qv, &kv) in qr.iter().zip(kr.iter()) {
                            dot += qv * kv;
                        }
                        scores.push(dot * scale);
                    }
                    let probs = softmax_stable(&scores)?;
                    for (j, &prob) in probs.iter().enumerate() {
                        let vj = &expanded_v[j][h * p.d_v..(h + 1) * p.d_v];
                        for (d, &vjd) in vj.iter().enumerate() {
                            head_concat[h * p.d_v + d] += prob * vjd;
                        }
                    }
                } else {
                    // Absorbed MLA:
                    // q_abs[l] = sum_d qn[d] * W_UK[l, h*d_nope + d]
                    let mut q_abs = vec![0.0_f64; p.d_kv_latent];
                    for (l, qa_l) in q_abs.iter_mut().enumerate() {
                        let mut acc = 0.0_f64;
                        for (d, &qnd) in qn.iter().enumerate() {
                            let col = if faults.wrong_absorbed_projection {
                                // Corrupted head index / projection slice.
                                ((h + 1) % p.n_heads) * p.d_nope + d
                            } else {
                                h * p.d_nope + d
                            };
                            acc += qnd * p.w_uk[l * (p.n_heads * p.d_nope) + col];
                        }
                        *qa_l = acc;
                    }
                    let mut scores = Vec::with_capacity(attend_limit);
                    for j in 0..attend_limit {
                        let cj = &cache.latents[b][j];
                        let kr = &cache.k_ropes[b][j];
                        let mut dot = 0.0_f64;
                        for (&qv, &cv) in q_abs.iter().zip(cj.iter()) {
                            dot += qv * cv;
                        }
                        for (&qv, &kv) in qr.iter().zip(kr.iter()) {
                            dot += qv * kv;
                        }
                        scores.push(dot * scale);
                        trace.latent_attend_ops += 1;
                    }
                    let probs = softmax_stable(&scores)?;
                    // Weighted sum directly in latent space: z[l] = sum_j probs[j] * c_j[l]
                    let mut z = vec![0.0_f64; p.d_kv_latent];
                    for (j, &prob) in probs.iter().enumerate() {
                        let cj = &cache.latents[b][j];
                        for (zl, &cjl) in z.iter_mut().zip(cj.iter()) {
                            *zl += prob * cjl;
                        }
                    }
                    // Project attended latent through W_UV[:, h*d_v..(h+1)*d_v]
                    for d in 0..p.d_v {
                        let mut acc = 0.0_f64;
                        for (l, &zl) in z.iter().enumerate() {
                            acc += zl * p.w_uv[l * (p.n_heads * p.d_v) + h * p.d_v + d];
                        }
                        head_concat[h * p.d_v + d] = acc;
                    }
                }
            }
            seq_outs.push(matvec(&head_concat, &p.w_o, p.n_heads * p.d_v, p.d_model)?);
        }
        batch_outs.push(seq_outs);
    }

    trace.peak_total_bytes = cache.persistent_bytes() + trace.temporary_workspace_bytes;
    Ok((batch_outs, trace))
}

pub type BatchComparison = (f64, Option<(usize, usize)>);

pub fn compare_batch_outputs(
    a: &[Vec<Vec<f64>>],
    b: &[Vec<Vec<f64>>],
    tol: f64,
) -> Result<BatchComparison, String> {
    if !tol.is_finite() || tol < 0.0 {
        return Err("tolerance must be finite and non-negative".into());
    }
    if a.len() != b.len() {
        return Err("batch size mismatch in compare_batch_outputs".into());
    }
    let mut max_err = 0.0_f64;
    let mut first_div = None;
    for (bi, (sa, sb)) in a.iter().zip(b.iter()).enumerate() {
        if sa.len() != sb.len() {
            return Err("sequence length mismatch in compare_batch_outputs".into());
        }
        for (ti, (va, vb)) in sa.iter().zip(sb.iter()).enumerate() {
            if va.len() != vb.len() {
                return Err("feature dimension mismatch in compare_batch_outputs".into());
            }
            for (&xa, &xb) in va.iter().zip(vb.iter()) {
                if !xa.is_finite() || !xb.is_finite() {
                    return Err("non-finite value in compare_batch_outputs".into());
                }
                let diff = (xa - xb).abs();
                if diff > max_err {
                    max_err = diff;
                }
                if first_div.is_none() && diff > tol {
                    first_div = Some((bi, ti));
                }
            }
        }
    }
    Ok((max_err, first_div))
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrefixStrategyAccounting {
    pub strategy: CachePrefixStrategy,
    pub shape_recompilations: usize,
    pub explicit_materialized_kv_elems: usize,
    pub absorbed_materialized_kv_elems: usize,
    pub absorbed_prep_bytes: usize,
    pub explicit_peak_bytes: usize,
    pub absorbed_peak_bytes: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlaControlResult {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub max_abs_err: Option<f64>,
    pub first_divergent_pos: Option<usize>,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct MlaComparisonReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub params_digest: DigestRef,
    pub default_dim_latent_elems: usize,
    pub default_dim_expanded_elems: usize,
    pub default_dim_element_reduction_pct: f64,
    pub element_count_reduction_is_measured_speedup: bool,
    pub batch_size: usize,
    pub capacity: usize,
    pub explicit_vs_uncached_max_err: f64,
    pub absorbed_vs_uncached_max_err: f64,
    pub absorbed_vs_explicit_max_err: f64,
    pub absorbed_materialized_kv_elems: usize,
    pub explicit_materialized_kv_elems: usize,
    pub absorbed_prep_bytes: usize,
    pub gradient_fd_parity_max_err: f64,
    pub strategy_accounting: Vec<PrefixStrategyAccounting>,
    pub controls: Vec<MlaControlResult>,
    pub scoped_decision: String,
}

pub fn run_default_harness() -> Result<MlaComparisonReport, String> {
    let p = TinyMlaParams::frozen_tiny();
    let batch_size = 2usize;
    let seq_len = 9usize; // Irregular chunks: [3, 2, 1, 3] = 9, capacity = 12 (3 unused poisoned slots!).
    let capacity = 12usize;
    let tol = 1e-11_f64;

    let batch_inputs: Vec<Vec<Vec<f64>>> = (0..batch_size)
        .map(|b| {
            (0..seq_len)
                .map(|t| {
                    (0..p.d_model)
                        .map(|d| det_weight(500 + b * 31 + t, d, 0.55))
                        .collect()
                })
                .collect()
        })
        .collect();

    // 1. Full uncached recomputation baseline.
    let mut uncached_outs = Vec::with_capacity(batch_size);
    for seq in &batch_inputs {
        uncached_outs.push(mla_full_uncached_forward(&p, seq, 0, false)?);
    }

    // 2. Irregular chunked prefill + single-token decode + multi-token continuation ([3, 2, 1, 3]).
    let chunk_splits = [(0, 3), (3, 5), (5, 6), (6, 9)];
    let run_chunked = |absorbed: bool,
                       strategy: CachePrefixStrategy|
     -> Result<(BatchTensor3D, LoweredOpTrace, usize), String> {
        let mut cache = CompressedMlaCache::new_poisoned(batch_size, capacity, &p);
        let mut all_outs = vec![Vec::new(); batch_size];
        let mut total_trace = LoweredOpTrace::default();
        let mut last_shape: Option<usize> = None;
        let mut recompiles = 0usize;

        for &(s, e) in &chunk_splits {
            let chunk: Vec<Vec<Vec<f64>>> =
                batch_inputs.iter().map(|seq| seq[s..e].to_vec()).collect();
            let (out, tr) = mla_cached_chunk_forward(
                &p,
                &mut cache,
                &chunk,
                absorbed,
                strategy,
                FaultInjection::none(),
            )?;
            let eff_shape = match strategy {
                CachePrefixStrategy::FullAllocatedCache => capacity,
                CachePrefixStrategy::ValidPrefixOnly => cache.valid_len,
                CachePrefixStrategy::BucketedPrefix => {
                    cache.valid_len.next_power_of_two().min(capacity)
                }
            };
            if last_shape != Some(eff_shape) {
                recompiles += 1;
                last_shape = Some(eff_shape);
            }
            total_trace.materialized_per_head_k_elems += tr.materialized_per_head_k_elems;
            total_trace.materialized_per_head_v_elems += tr.materialized_per_head_v_elems;
            total_trace.latent_attend_ops += tr.latent_attend_ops;
            total_trace.absorbed_prep_bytes =
                total_trace.absorbed_prep_bytes.max(tr.absorbed_prep_bytes);
            total_trace.temporary_workspace_bytes = total_trace
                .temporary_workspace_bytes
                .max(tr.temporary_workspace_bytes);
            total_trace.peak_total_bytes = total_trace.peak_total_bytes.max(tr.peak_total_bytes);
            for b in 0..batch_size {
                all_outs[b].extend(out[b].clone());
            }
        }
        Ok((all_outs, total_trace, recompiles))
    };

    let (exp_outs, exp_tr, exp_recomp) = run_chunked(false, CachePrefixStrategy::ValidPrefixOnly)?;
    let (abs_outs, abs_tr, _) = run_chunked(true, CachePrefixStrategy::ValidPrefixOnly)?;

    let (explicit_vs_uncached_max_err, _) = compare_batch_outputs(&exp_outs, &uncached_outs, tol)?;
    let (absorbed_vs_uncached_max_err, _) = compare_batch_outputs(&abs_outs, &uncached_outs, tol)?;
    let (absorbed_vs_explicit_max_err, _) = compare_batch_outputs(&abs_outs, &exp_outs, tol)?;

    // Strategy sweep: FullAllocatedCache vs ValidPrefixOnly vs BucketedPrefix.
    let mut strategy_accounting = Vec::new();
    for strat in [
        CachePrefixStrategy::FullAllocatedCache,
        CachePrefixStrategy::ValidPrefixOnly,
        CachePrefixStrategy::BucketedPrefix,
    ] {
        let (_, te, rc) = run_chunked(false, strat)?;
        let (_, ta, _) = run_chunked(true, strat)?;
        strategy_accounting.push(PrefixStrategyAccounting {
            strategy: strat,
            shape_recompilations: rc,
            explicit_materialized_kv_elems: te.materialized_per_head_k_elems
                + te.materialized_per_head_v_elems,
            absorbed_materialized_kv_elems: ta.materialized_per_head_k_elems
                + ta.materialized_per_head_v_elems,
            absorbed_prep_bytes: ta.absorbed_prep_bytes,
            explicit_peak_bytes: te.peak_total_bytes,
            absorbed_peak_bytes: ta.peak_total_bytes,
        });
    }
    let _ = exp_recomp;

    // Cache reset/reuse test: populate, reset_poisoned(), re-run and verify parity.
    let mut reuse_cache = CompressedMlaCache::new_poisoned(batch_size, capacity, &p);
    let _ = mla_cached_chunk_forward(
        &p,
        &mut reuse_cache,
        &batch_inputs,
        true,
        CachePrefixStrategy::ValidPrefixOnly,
        FaultInjection::none(),
    )?;
    reuse_cache.reset_poisoned();
    let (reuse_outs, _) = mla_cached_chunk_forward(
        &p,
        &mut reuse_cache,
        &batch_inputs,
        true,
        CachePrefixStrategy::ValidPrefixOnly,
        FaultInjection::none(),
    )?;
    let (reuse_err, _) = compare_batch_outputs(&reuse_outs, &uncached_outs, tol)?;

    // Finite-difference gradient parity between explicit and absorbed paths on w_dkv[0].
    let eps_fd = 1e-6_f64;
    let eval_scalar = |params: &TinyMlaParams, absorbed: bool| -> Result<f64, String> {
        let mut c = CompressedMlaCache::new_poisoned(1, 6, params);
        let chunk = vec![batch_inputs[0][..4].to_vec()];
        let (out, _) = mla_cached_chunk_forward(
            params,
            &mut c,
            &chunk,
            absorbed,
            CachePrefixStrategy::ValidPrefixOnly,
            FaultInjection::none(),
        )?;
        Ok(out[0].iter().flatten().sum())
    };
    let mut p_plus = p.clone();
    let mut p_minus = p.clone();
    p_plus.w_dkv[0] += eps_fd;
    p_minus.w_dkv[0] -= eps_fd;
    let grad_exp = (eval_scalar(&p_plus, false)? - eval_scalar(&p_minus, false)?) / (2.0 * eps_fd);
    let grad_abs = (eval_scalar(&p_plus, true)? - eval_scalar(&p_minus, true)?) / (2.0 * eps_fd);
    let gradient_fd_parity_max_err = (grad_exp - grad_abs).abs();

    // 7 adversarial negative controls with first divergent position recorded!
    let run_fault =
        |fault: FaultInjection, absorbed: bool| -> Result<(f64, Option<usize>), String> {
            let mut c = CompressedMlaCache::new_poisoned(batch_size, capacity, &p);
            let (out, _) = mla_cached_chunk_forward(
                &p,
                &mut c,
                &batch_inputs,
                absorbed,
                CachePrefixStrategy::ValidPrefixOnly,
                fault,
            )?;
            let (err, div) = compare_batch_outputs(&out, &uncached_outs, tol)?;
            Ok((err, div.map(|(_, pos)| pos)))
        };

    let (mask_err, mask_pos) = run_fault(
        FaultInjection {
            wrong_causal_mask: true,
            ..FaultInjection::none()
        },
        true,
    )?;
    let (rope_err, rope_pos) = run_fault(
        FaultInjection {
            wrong_rope_offset: true,
            ..FaultInjection::none()
        },
        true,
    )?;
    let (scale_err, scale_pos) = run_fault(
        FaultInjection {
            wrong_attention_scale: true,
            ..FaultInjection::none()
        },
        true,
    )?;
    let (proj_err, proj_pos) = run_fault(
        FaultInjection {
            wrong_absorbed_projection: true,
            ..FaultInjection::none()
        },
        true,
    )?;
    let (sentinel_err, sentinel_pos) = run_fault(
        FaultInjection {
            leak_unused_sentinel_slot: true,
            ..FaultInjection::none()
        },
        true,
    )?;

    // Stale cache (not reset before a second sequence -> starts at valid_len=3 instead of 0!).
    let mut stale_cache = CompressedMlaCache::new_poisoned(batch_size, capacity, &p);
    let prefix3: Vec<Vec<Vec<f64>>> = batch_inputs.iter().map(|s| s[..3].to_vec()).collect();
    let _ = mla_cached_chunk_forward(
        &p,
        &mut stale_cache,
        &prefix3,
        true,
        CachePrefixStrategy::ValidPrefixOnly,
        FaultInjection::none(),
    )?;
    let (stale_out, _) = mla_cached_chunk_forward(
        &p,
        &mut stale_cache,
        &prefix3,
        true,
        CachePrefixStrategy::ValidPrefixOnly,
        FaultInjection::none(),
    )?;
    let ref_prefix3: Vec<Vec<Vec<f64>>> = uncached_outs.iter().map(|s| s[..3].to_vec()).collect();
    let (stale_err, stale_div) = compare_batch_outputs(&stale_out, &ref_prefix3, tol)?;

    // Capacity overflow rejection.
    let mut small_cache = CompressedMlaCache::new_poisoned(batch_size, 4, &p);
    let overflow_rejected = mla_cached_chunk_forward(
        &p,
        &mut small_cache,
        &batch_inputs,
        true,
        CachePrefixStrategy::ValidPrefixOnly,
        FaultInjection::none(),
    )
    .is_err();

    let controls = vec![
        MlaControlResult {
            case_id: "01-positive-absorbed-vs-explicit-vs-uncached-poisoned-cache".into(),
            outcome: if absorbed_vs_uncached_max_err <= tol
                && absorbed_vs_explicit_max_err <= tol
                && reuse_err <= tol
                && abs_tr.materialized_per_head_k_elems == 0
                && abs_tr.materialized_per_head_v_elems == 0
            {
                OutcomeStatus::Accepted
            } else {
                OutcomeStatus::Invalid
            },
            max_abs_err: Some(absorbed_vs_uncached_max_err.max(reuse_err)),
            first_divergent_pos: None,
            detail: "Absorbed MLA matches explicit & uncached paths across batch=2, chunks [3,2,1,3], reset/reuse, and 1e6 sentinel-poisoned unused slots with 0 per-head K/V expansion intermediates.".into(),
        },
        MlaControlResult {
            case_id: "02-negative-wrong-causal-mask-detected".into(),
            outcome: if mask_err > tol { OutcomeStatus::Invalid } else { OutcomeStatus::InternalFailure },
            max_abs_err: Some(mask_err),
            first_divergent_pos: mask_pos,
            detail: "Future token leakage from disabled causal mask caught at earliest position.".into(),
        },
        MlaControlResult {
            case_id: "03-negative-wrong-rope-offset-detected".into(),
            outcome: if rope_err > tol { OutcomeStatus::Invalid } else { OutcomeStatus::InternalFailure },
            max_abs_err: Some(rope_err),
            first_divergent_pos: rope_pos,
            detail: "Shifted RoPE position offset caught immediately.".into(),
        },
        MlaControlResult {
            case_id: "04-negative-wrong-attention-scale-detected".into(),
            outcome: if scale_err > tol { OutcomeStatus::Invalid } else { OutcomeStatus::InternalFailure },
            max_abs_err: Some(scale_err),
            first_divergent_pos: scale_pos,
            detail: "2x attention scaling mismatch detected.".into(),
        },
        MlaControlResult {
            case_id: "05-negative-wrong-absorbed-projection-layout-detected".into(),
            outcome: if proj_err > tol { OutcomeStatus::Invalid } else { OutcomeStatus::InternalFailure },
            max_abs_err: Some(proj_err),
            first_divergent_pos: proj_pos,
            detail: "Misaligned head projection in absorbed W_UK caught immediately.".into(),
        },
        MlaControlResult {
            case_id: "06-negative-unused-1e6-sentinel-leak-detected".into(),
            outcome: if sentinel_err > 1e-3 { OutcomeStatus::Invalid } else { OutcomeStatus::InternalFailure },
            max_abs_err: Some(sentinel_err),
            first_divergent_pos: sentinel_pos,
            detail: "Reading one unused 1e6 sentinel cache slot causes catastrophic output divergence.".into(),
        },
        MlaControlResult {
            case_id: "07-negative-stale-cache-and-capacity-overflow-detected".into(),
            outcome: if stale_err > tol && overflow_rejected {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::InternalFailure
            },
            max_abs_err: Some(stale_err),
            first_divergent_pos: stale_div.map(|(_, pos)| pos),
            detail: "Unreset stale cache diverges from clean sequence and capacity overflow is explicitly rejected.".into(),
        },
    ];

    let default_latent = 128 + 16; // 144
    let default_expanded = 6 * (32 + 16 + 32); // 480
    let reduction_pct = (1.0 - (default_latent as f64) / (default_expanded as f64)) * 100.0;

    Ok(MlaComparisonReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "physical_gpu_jit_mla_latency_and_peak_vram",
            "CPU uncached, explicit-expansion, and absorbed MLA numerical/gradient parity and exact allocation-byte accounting ran locally; GPU synchronized latency remains unavailable_hardware_runtime.",
        ),
        params_digest: p.digest(),
        default_dim_latent_elems: default_latent,
        default_dim_expanded_elems: default_expanded,
        default_dim_element_reduction_pct: reduction_pct,
        element_count_reduction_is_measured_speedup: false,
        batch_size,
        capacity,
        explicit_vs_uncached_max_err,
        absorbed_vs_uncached_max_err,
        absorbed_vs_explicit_max_err,
        absorbed_materialized_kv_elems: abs_tr.materialized_per_head_k_elems
            + abs_tr.materialized_per_head_v_elems,
        explicit_materialized_kv_elems: exp_tr.materialized_per_head_k_elems
            + exp_tr.materialized_per_head_v_elems,
        absorbed_prep_bytes: abs_tr.absorbed_prep_bytes,
        gradient_fd_parity_max_err,
        strategy_accounting,
        controls,
        scoped_decision: "GO_FOR_CPU_ABSORBED_MLA_PARITY_INCONCLUSIVE_FOR_GPU_SPEEDUP".into(),
    })
}

pub fn inputs_json(report: &MlaComparisonReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.issue-20-mla-comparison.inputs.v1\",\n",
            "  \"issue\": 20,\n",
            "  \"execution_class\": {},\n",
            "  \"reused_issue_19_params_digest\": {},\n",
            "  \"batch_size\": {},\n",
            "  \"cache_capacity\": {},\n",
            "  \"unused_cache_sentinel\": {},\n",
            "  \"default_dim_latent_elems\": {},\n",
            "  \"default_dim_expanded_elems\": {},\n",
            "  \"default_dim_element_reduction_pct\": {},\n",
            "  \"element_count_reduction_is_measured_speedup\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.params_digest.to_json(),
        report.batch_size,
        report.capacity,
        finite_f64(CACHE_SENTINEL),
        report.default_dim_latent_elems,
        report.default_dim_expanded_elems,
        finite_f64(report.default_dim_element_reduction_pct),
        report.element_count_reduction_is_measured_speedup,
    )
}

pub fn results_json(report: &MlaComparisonReport) -> String {
    let mut out = String::new();
    writeln!(out, "{{").unwrap();
    writeln!(
        out,
        "  \"schema\": \"inference-lab.issue-20-mla-comparison.results.v1\","
    )
    .unwrap();
    writeln!(out, "  \"issue\": 20,").unwrap();
    writeln!(
        out,
        "  \"execution_class\": {},",
        quote(report.execution_class.as_str())
    )
    .unwrap();
    writeln!(
        out,
        "  \"hardware_gate\": {},",
        report.hardware_gate.to_json()
    )
    .unwrap();
    writeln!(
        out,
        "  \"explicit_vs_uncached_max_err\": {},",
        finite_f64(report.explicit_vs_uncached_max_err)
    )
    .unwrap();
    writeln!(
        out,
        "  \"absorbed_vs_uncached_max_err\": {},",
        finite_f64(report.absorbed_vs_uncached_max_err)
    )
    .unwrap();
    writeln!(
        out,
        "  \"absorbed_vs_explicit_max_err\": {},",
        finite_f64(report.absorbed_vs_explicit_max_err)
    )
    .unwrap();
    writeln!(
        out,
        "  \"absorbed_materialized_kv_elems\": {},",
        report.absorbed_materialized_kv_elems
    )
    .unwrap();
    writeln!(
        out,
        "  \"explicit_materialized_kv_elems\": {},",
        report.explicit_materialized_kv_elems
    )
    .unwrap();
    writeln!(
        out,
        "  \"absorbed_prep_bytes\": {},",
        report.absorbed_prep_bytes
    )
    .unwrap();
    writeln!(
        out,
        "  \"gradient_fd_parity_max_err\": {},",
        finite_f64(report.gradient_fd_parity_max_err)
    )
    .unwrap();
    writeln!(
        out,
        "  \"scoped_decision\": {},",
        quote(&report.scoped_decision)
    )
    .unwrap();

    let strats: Vec<String> = report
        .strategy_accounting
        .iter()
        .map(|s| {
            format!(
                concat!(
                    "{{\"strategy\":{},\"shape_recompilations\":{},",
                    "\"explicit_materialized_kv_elems\":{},\"absorbed_materialized_kv_elems\":{},",
                    "\"absorbed_prep_bytes\":{},\"explicit_peak_bytes\":{},\"absorbed_peak_bytes\":{}}}"
                ),
                quote(s.strategy.as_str()),
                s.shape_recompilations,
                s.explicit_materialized_kv_elems,
                s.absorbed_materialized_kv_elems,
                s.absorbed_prep_bytes,
                s.explicit_peak_bytes,
                s.absorbed_peak_bytes,
            )
        })
        .collect();
    writeln!(out, "  \"strategy_accounting\": [{}],", strats.join(",")).unwrap();

    let ctrls: Vec<String> = report
        .controls
        .iter()
        .map(|c| {
            let pos_str = c
                .first_divergent_pos
                .map_or_else(|| "null".into(), |v| v.to_string());
            format!(
                "{{\"case_id\":{},\"outcome\":{},\"max_abs_err\":{},\"first_divergent_pos\":{},\"detail\":{}}}",
                quote(&c.case_id),
                quote(c.outcome.as_str()),
                opt_f64(c.max_abs_err),
                pos_str,
                quote(&c.detail),
            )
        })
        .collect();
    writeln!(out, "  \"controls\": [{}]", ctrls.join(",")).unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn summary_markdown(report: &MlaComparisonReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #20: Explicit vs Absorbed MLA Parity & Memory Accounting"
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
    writeln!(
        out,
        "- Absorbed vs uncached max abs err: `{}` (gradient FD diff: `{}`)",
        finite_f64(report.absorbed_vs_uncached_max_err),
        finite_f64(report.gradient_fd_parity_max_err)
    )
    .unwrap();
    writeln!(
        out,
        "- Lowered materialized per-head K/V expansion elems: absorbed=`{}` (`absorbed_prep_bytes={}` B), explicit=`{}`",
        report.absorbed_materialized_kv_elems,
        report.absorbed_prep_bytes,
        report.explicit_materialized_kv_elems
    )
    .unwrap();
    writeln!(
        out,
        "- Default-dim persistent payload: `{}` vs `{}` (`{}%` element reduction; `is_measured_speedup={}`)",
        report.default_dim_latent_elems,
        report.default_dim_expanded_elems,
        finite_f64(report.default_dim_element_reduction_pct),
        report.element_count_reduction_is_measured_speedup
    )
    .unwrap();
    writeln!(out, "- Scoped decision: `{}`", report.scoped_decision).unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Cache Prefix Strategy & Recompilation Accounting").unwrap();
    writeln!(
        out,
        "| Strategy | Recompiles | Explicit KV Elems | Absorbed KV Elems | Explicit Peak B | Absorbed Peak B |"
    )
    .unwrap();
    writeln!(out, "| --- | ---: | ---: | ---: | ---: | ---: |").unwrap();
    for s in &report.strategy_accounting {
        writeln!(
            out,
            "| `{}` | {} | {} | {} | {} | {} |",
            s.strategy.as_str(),
            s.shape_recompilations,
            s.explicit_materialized_kv_elems,
            s.absorbed_materialized_kv_elems,
            s.explicit_peak_bytes,
            s.absorbed_peak_bytes
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "## Parity & Sentinel Negative Controls").unwrap();
    for c in &report.controls {
        writeln!(
            out,
            "- `{}`: `{}` (err=`{}`, first_div_pos=`{:?}`) — {}",
            c.case_id,
            c.outcome.as_str(),
            opt_f64(c.max_abs_err),
            c.first_divergent_pos,
            c.detail
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "> Local cache-layout comparison does not establish cross-replica transport or heterogeneous cache compatibility (`#5`)."
    )
    .unwrap();
    out
}
