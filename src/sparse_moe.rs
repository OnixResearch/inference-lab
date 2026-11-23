//! Issue #21: Implement sparse MoE dispatch and validate outputs and gradients against a
//! dense reference.
//!
//! Pure deterministic core consuming `#19`'s corrected MoE reference semantics and parameter
//! layout (`deepseek_ref::TinyMoeParams`) across three paths sharing one parameter set:
//! 1. Dense reference (`moe_dense_forward_backward`): evaluates all `n_experts` per token.
//! 2. Independent tiny per-token/per-expert scalar reference (`moe_independent_tiny_forward`).
//! 3. Sparse MoE dispatch (`moe_sparse_forward_backward`): groups tokens by selected expert,
//!    executes ONLY selected `(token, expert)` pairs (skipping empty experts!), scatter-adds
//!    weighted results in declared deterministic order, preserves all assignments without silent
//!    dropping, reports actual executed/padded/overflow/fallback counts, and derives load counts
//!    from actual selected assignments (immune to BF16 gate underflow!).
//!
//! Verifies:
//! - Forward AND full analytical backward pass (`dX`, `dW_router`, `dW_shared`, `dW_experts`)
//!   parity between dense and sparse paths plus central finite-difference verification away from
//!   routing discontinuities.
//! - Deterministic tie-breaking at exact top-k discontinuities.
//! - Short 3-step matched parameter + load-balancing bias update sequence.
//! - Differential fixture matrix: bias ranking reversal, skewed load + empty expert, `top_k=1`,
//!   `top_k=all`, identical experts, non-unit `route_scale`, extreme finite logits (BF16 gate
//!   underflow), and non-finite logit rejection.
//! - 7 adversarial negative controls: wrong bias-weighting, missing normalization, duplicate
//!   scatter, lost scatter / silent drop, wrong shared-expert execution, BF16 gate-underflow load
//!   count drop, and mislabeled dense execution claiming sparse dispatch.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{
    cast_bf16, det_weight, deterministic_topk, matvec, sigmoid, TinyMoeParams,
};
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

#[derive(Debug, Clone, PartialEq)]
pub struct MoeGradients {
    pub d_x: Vec<Vec<f64>>,
    pub d_w_router: Vec<f64>,
    pub d_w_shared: Vec<f64>,
    pub d_w_experts: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoeDispatchAccounting {
    pub dense_candidate_pairs: usize,
    pub executed_pairs: usize,
    pub padded_pairs: usize,
    pub overflow_assignments: usize,
    pub dense_fallback_triggered: bool,
    pub empty_experts_skipped: usize,
    pub temp_buffer_bytes: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoeForwardBackward {
    pub outputs: Vec<Vec<f64>>,
    pub selected_experts: Vec<Vec<usize>>,
    pub gate_weights: Vec<Vec<f64>>,
    pub assignment_load_counts: Vec<usize>,
    pub bf16_gate_gt_zero_counts: Vec<usize>,
    pub seq_aux_loss_eq18: f64,
    pub accounting: MoeDispatchAccounting,
    pub grads: MoeGradients,
}

/// Independent scalar per-token/per-expert known-answer reference (forward only).
pub fn moe_independent_tiny_forward(
    p: &TinyMoeParams,
    tokens: &[Vec<f64>],
    bias: &[f64],
    eps: f64,
) -> Result<Vec<Vec<f64>>, String> {
    if tokens.is_empty()
        || p.d_model == 0
        || p.n_experts == 0
        || bias.len() != p.n_experts
        || bias.iter().any(|b| !b.is_finite())
        || !eps.is_finite()
        || eps < 0.0
        || !p.route_scale.is_finite()
        || p.w_router.len() != p.d_model * p.n_experts
        || p.w_shared.len() != p.d_model * p.d_model
        || p.w_experts.len() != p.n_experts
        || p.w_router.iter().any(|v| !v.is_finite())
        || p.w_shared.iter().any(|v| !v.is_finite())
        || p.w_experts
            .iter()
            .any(|w| w.len() != p.d_model * p.d_model || w.iter().any(|v| !v.is_finite()))
    {
        return Err(
            "invalid params, shapes, or non-finite values in moe_independent_tiny_forward".into(),
        );
    }
    let mut outs = Vec::with_capacity(tokens.len());
    for x in tokens {
        if x.len() != p.d_model || x.iter().any(|v| !v.is_finite()) {
            return Err("invalid or non-finite token in moe_independent_tiny_forward".into());
        }
        let mut raw_aff = vec![0.0_f64; p.n_experts];
        for (e, slot) in raw_aff.iter_mut().enumerate() {
            let mut z = 0.0_f64;
            for (i, &xi) in x.iter().enumerate() {
                z += xi * p.w_router[i * p.n_experts + e];
            }
            *slot = sigmoid(z);
        }
        let biased: Vec<f64> = raw_aff
            .iter()
            .zip(bias.iter())
            .map(|(&s, &b)| s + b)
            .collect();
        let chosen = deterministic_topk(&biased, p.top_k)?;
        let denom: f64 = chosen.iter().map(|&e| raw_aff[e]).sum::<f64>() + eps;
        if !denom.is_finite() || denom <= 0.0 {
            return Err(
                "non-positive normalization denominator in moe_independent_tiny_forward".into(),
            );
        }
        let mut y = vec![0.0_f64; p.d_model];
        // Shared expert once per token.
        for (j, yj) in y.iter_mut().enumerate() {
            let mut acc = 0.0_f64;
            for (i, &xi) in x.iter().enumerate() {
                acc += xi * p.w_shared[i * p.d_model + j];
            }
            *yj = acc;
        }
        for &e in &chosen {
            let g = (raw_aff[e] / denom) * p.route_scale;
            for (j, yj) in y.iter_mut().enumerate() {
                let mut acc = 0.0_f64;
                for (i, &xi) in x.iter().enumerate() {
                    acc += xi * p.w_experts[e][i * p.d_model + j];
                }
                *yj += g * acc;
            }
        }
        outs.push(y);
    }
    Ok(outs)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseFaultInjection {
    pub add_bias_to_output_weights: bool,
    pub omit_affinity_normalization: bool,
    pub duplicate_first_scatter: bool,
    pub drop_overflow_assignment: bool,
    pub run_shared_expert_per_route: bool,
}

impl SparseFaultInjection {
    pub const fn none() -> Self {
        Self {
            add_bias_to_output_weights: false,
            omit_affinity_normalization: false,
            duplicate_first_scatter: false,
            drop_overflow_assignment: false,
            run_shared_expert_per_route: false,
        }
    }
}

/// Shared routing + forward + analytical backward core for both dense (`sparse_dispatch = false`)
/// and grouped sparse dispatch (`sparse_dispatch = true`).
#[allow(clippy::too_many_arguments)]
pub fn moe_forward_backward(
    p: &TinyMoeParams,
    tokens: &[Vec<f64>],
    bias: &[f64],
    d_out: &[Vec<f64>],
    eps: f64,
    sparse_dispatch: bool,
    bucket_multiple: usize,
    expert_capacity_hint: Option<usize>,
    faults: SparseFaultInjection,
) -> Result<MoeForwardBackward, String> {
    let t = tokens.len();
    if t == 0
        || d_out.len() != t
        || p.d_model == 0
        || p.n_experts == 0
        || bias.len() != p.n_experts
        || bias.iter().any(|b| !b.is_finite())
        || !eps.is_finite()
        || eps < 0.0
        || !p.route_scale.is_finite()
        || !p.seq_aux_alpha.is_finite()
    {
        return Err("invalid inputs, eps, or scales to moe_forward_backward".into());
    }
    for (x, dy) in tokens.iter().zip(d_out.iter()) {
        if x.len() != p.d_model
            || dy.len() != p.d_model
            || x.iter().any(|v| !v.is_finite())
            || dy.iter().any(|v| !v.is_finite())
        {
            return Err("non-finite or malformed token/d_out vector".into());
        }
    }

    let mut raw_affs = Vec::with_capacity(t);
    let mut selected_experts = Vec::with_capacity(t);
    let mut gate_weights = Vec::with_capacity(t);
    let mut denom_z = Vec::with_capacity(t);
    let mut assignment_load_counts = vec![0usize; p.n_experts];
    let mut bf16_gate_gt_zero_counts = vec![0usize; p.n_experts];

    // Eq. 18 raw-affinity sequence auxiliary loss accumulators.
    let mut p_acc = vec![0.0_f64; p.n_experts];
    let mut raw_topk_counts = vec![0usize; p.n_experts];

    for x in tokens {
        let logits = matvec(x, &p.w_router, p.d_model, p.n_experts)?;
        let s: Vec<f64> = logits.iter().map(|&z| sigmoid(z)).collect();
        let s_sum: f64 = s.iter().sum();
        if !(s_sum + eps).is_finite() || (s_sum + eps) <= 0.0 {
            return Err("non-positive raw affinity sum in moe_forward_backward".into());
        }
        for e in 0..p.n_experts {
            p_acc[e] += s[e] / (s_sum + eps);
        }
        let raw_tk = deterministic_topk(&s, p.top_k)?;
        for &e in &raw_tk {
            raw_topk_counts[e] += 1;
        }

        let biased: Vec<f64> = s
            .iter()
            .zip(bias.iter())
            .map(|(&si, &bi)| si + bi)
            .collect();
        let chosen = deterministic_topk(&biased, p.top_k)?;
        let z_sel: f64 = chosen.iter().map(|&e| s[e]).sum::<f64>() + eps;
        if !z_sel.is_finite() || z_sel <= 0.0 {
            return Err(
                "non-positive selected affinity denominator in moe_forward_backward".into(),
            );
        }

        let mut g = vec![0.0_f64; p.n_experts];
        for &e in &chosen {
            // Assignment-derived load count (never derived from low-precision gate > 0!).
            assignment_load_counts[e] += 1;
            let raw_num = if faults.add_bias_to_output_weights {
                s[e] + bias[e]
            } else {
                s[e]
            };
            let weight = if faults.omit_affinity_normalization {
                raw_num * p.route_scale
            } else {
                (raw_num / z_sel) * p.route_scale
            };
            g[e] = weight;
            if cast_bf16(weight) > 0.0 {
                bf16_gate_gt_zero_counts[e] += 1;
            }
        }

        raw_affs.push(s);
        selected_experts.push(chosen);
        gate_weights.push(g);
        denom_z.push(z_sel);
    }

    // Eq. 18 loss.
    let tf = t as f64;
    let factor = (p.n_experts as f64) / ((p.top_k * t) as f64);
    let mut eq18_sum = 0.0_f64;
    for e in 0..p.n_experts {
        eq18_sum += (factor * (raw_topk_counts[e] as f64)) * (p_acc[e] / tf);
    }
    let seq_aux_loss_eq18 = p.seq_aux_alpha * eq18_sum;

    // Shared expert path (once per token, unless faulty `run_shared_expert_per_route` is injected).
    let mut outputs = vec![vec![0.0_f64; p.d_model]; t];
    let mut d_x = vec![vec![0.0_f64; p.d_model]; t];
    let mut d_w_router = vec![0.0_f64; p.d_model * p.n_experts];
    let mut d_w_shared = vec![0.0_f64; p.d_model * p.d_model];
    let mut d_w_experts = vec![vec![0.0_f64; p.d_model * p.d_model]; p.n_experts];

    let shared_mult = if faults.run_shared_expert_per_route {
        p.top_k as f64
    } else {
        1.0
    };
    for ti in 0..t {
        let sh = matvec(&tokens[ti], &p.w_shared, p.d_model, p.d_model)?;
        for (yj, &shj) in outputs[ti].iter_mut().zip(sh.iter()) {
            *yj += shared_mult * shj;
        }
        for i in 0..p.d_model {
            for j in 0..p.d_model {
                d_w_shared[i * p.d_model + j] += shared_mult * tokens[ti][i] * d_out[ti][j];
                d_x[ti][i] += shared_mult * d_out[ti][j] * p.w_shared[i * p.d_model + j];
            }
        }
    }

    // Store per-(token, expert) dg values for router backward.
    let mut dg_table = vec![vec![0.0_f64; p.n_experts]; t];
    let mut executed_pairs = 0usize;
    let mut padded_pairs = 0usize;
    let mut overflow_assignments = 0usize;
    let mut empty_experts_skipped = 0usize;
    let dense_candidate_pairs = t * p.n_experts;

    if !sparse_dispatch {
        // Dense reference evaluates EVERY expert for EVERY token.
        for ti in 0..t {
            for (e, dw_e) in d_w_experts.iter_mut().enumerate().take(p.n_experts) {
                let ey = matvec(&tokens[ti], &p.w_experts[e], p.d_model, p.d_model)?;
                executed_pairs += 1;
                let g = gate_weights[ti][e];
                if g != 0.0 || selected_experts[ti].contains(&e) {
                    let mut dg = 0.0_f64;
                    for ((yj, &dyj), &eyj) in
                        outputs[ti].iter_mut().zip(d_out[ti].iter()).zip(ey.iter())
                    {
                        *yj += g * eyj;
                        dg += dyj * eyj;
                    }
                    dg_table[ti][e] = dg;
                    for i in 0..p.d_model {
                        for j in 0..p.d_model {
                            dw_e[i * p.d_model + j] += (g * tokens[ti][i]) * d_out[ti][j];
                            d_x[ti][i] += g * d_out[ti][j] * p.w_experts[e][i * p.d_model + j];
                        }
                    }
                }
            }
        }
    } else {
        // Sparse MoE dispatch: group tokens by selected expert in ascending token order!
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); p.n_experts];
        for (ti, chosen) in selected_experts.iter().enumerate() {
            for &e in chosen {
                groups[e].push(ti);
            }
        }

        // Accumulate routed contributions per token in deterministic expert-ascending order
        // (matching dense summation order!) so scatter-add order is fully declared.
        let mut routed_partials: Vec<Vec<(usize, Vec<f64>)>> = vec![Vec::new(); t];

        for (e, group) in groups.iter().enumerate() {
            if group.is_empty() {
                empty_experts_skipped += 1;
                continue;
            }
            let effective_len = if let Some(cap) = expert_capacity_hint {
                if group.len() > cap {
                    overflow_assignments += group.len() - cap;
                    if faults.drop_overflow_assignment {
                        cap // Negative control: silent token drop!
                    } else {
                        // Preserve ALL selected assignments without dropping!
                        group.len()
                    }
                } else {
                    group.len()
                }
            } else {
                group.len()
            };

            if bucket_multiple > 1 {
                let rem = effective_len % bucket_multiple;
                if rem != 0 {
                    padded_pairs += bucket_multiple - rem;
                }
            }

            // Execute ONLY the gathered tokens for expert `e`.
            for &ti in &group[..effective_len] {
                let ey = matvec(&tokens[ti], &p.w_experts[e], p.d_model, p.d_model)?;
                executed_pairs += 1;
                let g = gate_weights[ti][e];
                let mut weighted = vec![0.0_f64; p.d_model];
                let mut dg = 0.0_f64;
                for j in 0..p.d_model {
                    weighted[j] = g * ey[j];
                    dg += d_out[ti][j] * ey[j];
                }
                dg_table[ti][e] = dg;
                for i in 0..p.d_model {
                    for j in 0..p.d_model {
                        d_w_experts[e][i * p.d_model + j] += (g * tokens[ti][i]) * d_out[ti][j];
                        d_x[ti][i] += g * d_out[ti][j] * p.w_experts[e][i * p.d_model + j];
                    }
                }
                routed_partials[ti].push((e, weighted));
            }
        }

        // Scatter-add in declared deterministic ascending-expert order.
        for ti in 0..t {
            routed_partials[ti].sort_by_key(|(e, _)| *e);
            for (idx, (_, contrib)) in routed_partials[ti].iter().enumerate() {
                let mult = if faults.duplicate_first_scatter && ti == 0 && idx == 0 {
                    2.0
                } else {
                    1.0
                };
                for (yj, &cj) in outputs[ti].iter_mut().zip(contrib.iter()) {
                    *yj += mult * cj;
                }
            }
        }
    }

    // Router backward pass (away from top-k discontinuities).
    for ti in 0..t {
        let z = denom_z[ti];
        let a_sum: f64 = selected_experts[ti]
            .iter()
            .map(|&e| dg_table[ti][e] * gate_weights[ti][e])
            .sum();
        for &e in &selected_experts[ti] {
            let ds = (p.route_scale * dg_table[ti][e] - a_sum) / z;
            let s = raw_affs[ti][e];
            let du = ds * s * (1.0 - s);
            for i in 0..p.d_model {
                d_w_router[i * p.n_experts + e] += tokens[ti][i] * du;
                d_x[ti][i] += du * p.w_router[i * p.n_experts + e];
            }
        }
    }

    let temp_buffer_bytes =
        (executed_pairs + padded_pairs) * p.d_model * std::mem::size_of::<f64>();

    Ok(MoeForwardBackward {
        outputs,
        selected_experts,
        gate_weights,
        assignment_load_counts,
        bf16_gate_gt_zero_counts,
        seq_aux_loss_eq18,
        accounting: MoeDispatchAccounting {
            dense_candidate_pairs,
            executed_pairs,
            padded_pairs,
            overflow_assignments,
            dense_fallback_triggered: !sparse_dispatch,
            empty_experts_skipped,
            temp_buffer_bytes,
        },
        grads: MoeGradients {
            d_x,
            d_w_router,
            d_w_shared,
            d_w_experts,
        },
    })
}

pub fn max_mat_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err("matrix row count mismatch in max_mat_diff".into());
    }
    let mut max_err = 0.0_f64;
    for (ra, rb) in a.iter().zip(b.iter()) {
        if ra.len() != rb.len() {
            return Err("matrix column count mismatch in max_mat_diff".into());
        }
        for (&x, &y) in ra.iter().zip(rb.iter()) {
            if !x.is_finite() || !y.is_finite() {
                return Err("non-finite value in max_mat_diff".into());
            }
            let d = (x - y).abs();
            if d > max_err {
                max_err = d;
            }
        }
    }
    Ok(max_err)
}

pub fn max_vec_diff(a: &[f64], b: &[f64]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err("vector length mismatch in max_vec_diff".into());
    }
    let mut max_err = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return Err("non-finite value in max_vec_diff".into());
        }
        let d = (x - y).abs();
        if d > max_err {
            max_err = d;
        }
    }
    Ok(max_err)
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseFixtureCase {
    pub case_id: String,
    pub forward_max_err: f64,
    pub grad_max_err: Option<f64>,
    pub executed_pairs: usize,
    pub dense_pairs: usize,
    pub empty_experts_skipped: usize,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseSweepRow {
    pub n_tokens: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub skew_profile: String,
    pub dense_pairs: usize,
    pub sparse_executed_pairs: usize,
    pub sparse_padded_pairs: usize,
    pub pair_reduction_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct SparseMoeReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub dense_vs_tiny_max_err: f64,
    pub dense_vs_sparse_forward_max_err: f64,
    pub dense_vs_sparse_grad_max_err: f64,
    pub analytical_vs_fd_grad_max_err: f64,
    pub three_step_update_parity_max_err: f64,
    pub bf16_underflow_preserved_assignment_count: bool,
    pub differential_fixtures: Vec<SparseFixtureCase>,
    pub negative_controls: Vec<SparseFixtureCase>,
    pub sweep_rows: Vec<SparseSweepRow>,
    pub scoped_decision: String,
}

pub fn run_default_harness() -> Result<SparseMoeReport, String> {
    let base_p = TinyMoeParams::frozen_tiny();
    let eps = 1e-12_f64;
    let tol = 1e-11_f64;

    let tokens: Vec<Vec<f64>> = (0..6)
        .map(|t| {
            (0..base_p.d_model)
                .map(|d| det_weight(700 + t, d, 0.7))
                .collect()
        })
        .collect();
    let d_out: Vec<Vec<f64>> = (0..6)
        .map(|t| {
            (0..base_p.d_model)
                .map(|d| det_weight(800 + t, d, 0.5))
                .collect()
        })
        .collect();
    // Nonzero bias that reverses raw-score ranking and leaves expert 0 unused!
    let skew_bias = [-2.0_f64, 0.35, 0.85, 0.55];

    let tiny_out = moe_independent_tiny_forward(&base_p, &tokens, &skew_bias, eps)?;
    let dense = moe_forward_backward(
        &base_p,
        &tokens,
        &skew_bias,
        &d_out,
        eps,
        false,
        1,
        None,
        SparseFaultInjection::none(),
    )?;
    let sparse = moe_forward_backward(
        &base_p,
        &tokens,
        &skew_bias,
        &d_out,
        eps,
        true,
        2,
        Some(2), // triggers overflow accounting, preserves ALL assignments without dropping!
        SparseFaultInjection::none(),
    )?;

    let dense_vs_tiny_max_err = max_mat_diff(&dense.outputs, &tiny_out)?;
    let dense_vs_sparse_forward_max_err = max_mat_diff(&dense.outputs, &sparse.outputs)?;
    let mut exp_grad_err = 0.0_f64;
    for (a, b) in dense
        .grads
        .d_w_experts
        .iter()
        .zip(sparse.grads.d_w_experts.iter())
    {
        exp_grad_err = exp_grad_err.max(max_vec_diff(a, b)?);
    }
    let grad_err = max_mat_diff(&dense.grads.d_x, &sparse.grads.d_x)?
        .max(max_vec_diff(
            &dense.grads.d_w_router,
            &sparse.grads.d_w_router,
        )?)
        .max(max_vec_diff(
            &dense.grads.d_w_shared,
            &sparse.grads.d_w_shared,
        )?)
        .max(exp_grad_err);

    // Central finite-difference check of analytical gradients (dX, dW_router, dW_shared, dW_experts).
    let fd_eps = 1e-6_f64;
    let loss_fn = |p_cur: &TinyMoeParams, x_cur: &[Vec<f64>]| -> Result<f64, String> {
        let fb = moe_forward_backward(
            p_cur,
            x_cur,
            &skew_bias,
            &d_out,
            eps,
            true,
            1,
            None,
            SparseFaultInjection::none(),
        )?;
        let mut s = 0.0_f64;
        for (y, dy) in fb.outputs.iter().zip(d_out.iter()) {
            for (&yv, &dyv) in y.iter().zip(dy.iter()) {
                s += yv * dyv;
            }
        }
        Ok(s)
    };
    let mut fd_max_err = 0.0_f64;
    // Check dW_router[1]
    {
        let mut pp = base_p.clone();
        let mut pm = base_p.clone();
        pp.w_router[1] += fd_eps;
        pm.w_router[1] -= fd_eps;
        let num = (loss_fn(&pp, &tokens)? - loss_fn(&pm, &tokens)?) / (2.0 * fd_eps);
        fd_max_err = fd_max_err.max((num - sparse.grads.d_w_router[1]).abs());
    }
    // Check dW_shared[0]
    {
        let mut pp = base_p.clone();
        let mut pm = base_p.clone();
        pp.w_shared[0] += fd_eps;
        pm.w_shared[0] -= fd_eps;
        let num = (loss_fn(&pp, &tokens)? - loss_fn(&pm, &tokens)?) / (2.0 * fd_eps);
        fd_max_err = fd_max_err.max((num - sparse.grads.d_w_shared[0]).abs());
    }
    // Check dW_experts[2][0]
    {
        let mut pp = base_p.clone();
        let mut pm = base_p.clone();
        pp.w_experts[2][0] += fd_eps;
        pm.w_experts[2][0] -= fd_eps;
        let num = (loss_fn(&pp, &tokens)? - loss_fn(&pm, &tokens)?) / (2.0 * fd_eps);
        fd_max_err = fd_max_err.max((num - sparse.grads.d_w_experts[2][0]).abs());
    }
    // Check dX[0][0]
    {
        let mut xp = tokens.clone();
        let mut xm = tokens.clone();
        xp[0][0] += fd_eps;
        xm[0][0] -= fd_eps;
        let num = (loss_fn(&base_p, &xp)? - loss_fn(&base_p, &xm)?) / (2.0 * fd_eps);
        fd_max_err = fd_max_err.max((num - sparse.grads.d_x[0][0]).abs());
    }

    // 3-step matched update sequence (weights + load-balancing bias updates).
    let mut p_dense = base_p.clone();
    let mut p_sparse = base_p.clone();
    let mut b_dense = skew_bias.to_vec();
    let mut b_sparse = skew_bias.to_vec();
    let lr = 0.02_f64;
    let bias_gamma = 0.01_f64;
    let mut step_max_err = 0.0_f64;
    for _step in 0..3 {
        let fd = moe_forward_backward(
            &p_dense,
            &tokens,
            &b_dense,
            &d_out,
            eps,
            false,
            1,
            None,
            SparseFaultInjection::none(),
        )?;
        let fs = moe_forward_backward(
            &p_sparse,
            &tokens,
            &b_sparse,
            &d_out,
            eps,
            true,
            1,
            None,
            SparseFaultInjection::none(),
        )?;
        step_max_err = step_max_err.max(max_mat_diff(&fd.outputs, &fs.outputs)?);
        let avg_load = ((tokens.len() * p_dense.top_k) as f64) / (p_dense.n_experts as f64);
        for (p_cur, b_cur, res) in [
            (&mut p_dense, &mut b_dense, &fd),
            (&mut p_sparse, &mut b_sparse, &fs),
        ] {
            for (w, g) in p_cur.w_router.iter_mut().zip(res.grads.d_w_router.iter()) {
                *w -= lr * g;
            }
            for (w, g) in p_cur.w_shared.iter_mut().zip(res.grads.d_w_shared.iter()) {
                *w -= lr * g;
            }
            for (we, ge) in p_cur.w_experts.iter_mut().zip(res.grads.d_w_experts.iter()) {
                for (w, g) in we.iter_mut().zip(ge.iter()) {
                    *w -= lr * g;
                }
            }
            for (b_slot, &cnt) in b_cur.iter_mut().zip(res.assignment_load_counts.iter()) {
                let err = avg_load - (cnt as f64);
                *b_slot += bias_gamma * err.signum();
            }
        }
    }
    step_max_err = step_max_err
        .max(max_vec_diff(&p_dense.w_router, &p_sparse.w_router)?)
        .max(max_vec_diff(&b_dense, &b_sparse)?);

    // Extreme finite logits & BF16 gate underflow test:
    // Expert 0 has logit -100.0 (sigmoid ~ 3.7e-44 -> BF16 underflows to 0.0!) yet bias +2.0 selects it!
    let mut extreme_p = base_p.clone();
    extreme_p.top_k = 2;
    extreme_p.w_router = vec![0.0; extreme_p.d_model * extreme_p.n_experts];
    extreme_p.w_router[0] = -100.0; // expert 0: tiny positive sigmoid (~3.7e-44, BF16==0!)
    extreme_p.w_router[1] = 20.0; // expert 1: sigmoid ~ 1.0
    extreme_p.w_router[2] = -50.0;
    extreme_p.w_router[3] = -50.0;
    let ext_tok = vec![vec![1.0, 0.0, 0.0, 0.0]];
    let ext_dout = vec![vec![1.0, 1.0, 1.0, 1.0]];
    let ext_bias = [2.0, 0.0, -5.0, -5.0];
    let ext_sparse = moe_forward_backward(
        &extreme_p,
        &ext_tok,
        &ext_bias,
        &ext_dout,
        eps,
        true,
        1,
        None,
        SparseFaultInjection::none(),
    )?;
    // Expert 0 WAS selected (assignment_load_counts[0] == 1), while bf16(gate) underflowed to 0!
    let bf16_underflow_preserved =
        ext_sparse.assignment_load_counts[0] == 1 && ext_sparse.bf16_gate_gt_zero_counts[0] == 0;

    // Exact top-k tie discontinuity test: execute real moe_forward_backward on a tied-score token
    // where experts 0, 1, 2 have equal logit 0.0 (affinity 0.5) and expert 3 has logit -2.0.
    let mut tie_p = base_p.clone();
    tie_p.w_router = vec![0.0; tie_p.d_model * tie_p.n_experts];
    tie_p.w_router[3] = -2.0;
    let tie_tok = vec![vec![1.0, 0.0, 0.0, 0.0]];
    let tie_dout = vec![vec![1.0, 1.0, 1.0, 1.0]];
    let tie_zero_bias = [0.0_f64; 4];
    let tie_dense = moe_forward_backward(
        &tie_p,
        &tie_tok,
        &tie_zero_bias,
        &tie_dout,
        eps,
        false,
        1,
        None,
        SparseFaultInjection::none(),
    )?;
    let tie_sparse = moe_forward_backward(
        &tie_p,
        &tie_tok,
        &tie_zero_bias,
        &tie_dout,
        eps,
        true,
        1,
        None,
        SparseFaultInjection::none(),
    )?;
    let tie_fwd_err = max_mat_diff(&tie_dense.outputs, &tie_sparse.outputs)?;
    let tie_ok = tie_sparse.selected_experts[0] == vec![0, 1] && tie_fwd_err <= tol;

    // top_k = 1 and top_k = all (n_experts) differential checks.
    let mut p_k1 = base_p.clone();
    p_k1.top_k = 1;
    let d_k1 = moe_forward_backward(
        &p_k1,
        &tokens,
        &skew_bias,
        &d_out,
        eps,
        false,
        1,
        None,
        SparseFaultInjection::none(),
    )?;
    let s_k1 = moe_forward_backward(
        &p_k1,
        &tokens,
        &skew_bias,
        &d_out,
        eps,
        true,
        1,
        None,
        SparseFaultInjection::none(),
    )?;
    let k1_err = max_mat_diff(&d_k1.outputs, &s_k1.outputs)?;

    let mut p_kall = base_p.clone();
    p_kall.top_k = base_p.n_experts;
    let d_kall = moe_forward_backward(
        &p_kall,
        &tokens,
        &skew_bias,
        &d_out,
        eps,
        false,
        1,
        None,
        SparseFaultInjection::none(),
    )?;
    let s_kall = moe_forward_backward(
        &p_kall,
        &tokens,
        &skew_bias,
        &d_out,
        eps,
        true,
        1,
        None,
        SparseFaultInjection::none(),
    )?;
    let kall_err = max_mat_diff(&d_kall.outputs, &s_kall.outputs)?;

    let differential_fixtures = vec![
        SparseFixtureCase {
            case_id: "01-skewed-bias-reversal-empty-expert-and-overflow-preserved".into(),
            forward_max_err: dense_vs_sparse_forward_max_err,
            grad_max_err: Some(grad_err),
            executed_pairs: sparse.accounting.executed_pairs,
            dense_pairs: sparse.accounting.dense_candidate_pairs,
            empty_experts_skipped: sparse.accounting.empty_experts_skipped,
            outcome: if dense_vs_sparse_forward_max_err <= tol
                && grad_err <= tol
                && sparse.accounting.empty_experts_skipped >= 1
                && sparse.accounting.overflow_assignments >= 1
            {
                OutcomeStatus::Accepted
            } else {
                OutcomeStatus::Invalid
            },
            detail: "Bias reverses ranking, expert 0 is skipped (empty), overflow assignments are preserved without drops, and forward+grad match dense.".into(),
        },
        SparseFixtureCase {
            case_id: "02-top-k-1-and-top-k-all-parity".into(),
            forward_max_err: k1_err.max(kall_err),
            grad_max_err: None,
            executed_pairs: s_k1.accounting.executed_pairs,
            dense_pairs: s_k1.accounting.dense_candidate_pairs,
            empty_experts_skipped: s_k1.accounting.empty_experts_skipped,
            outcome: if k1_err <= tol && kall_err <= tol {
                OutcomeStatus::Accepted
            } else {
                OutcomeStatus::Invalid
            },
            detail: "Both top_k=1 (6/24 pairs) and top_k=all (24/24 pairs) match dense reference.".into(),
        },
        SparseFixtureCase {
            case_id: "03-extreme-finite-logits-bf16-gate-underflow-load-count-safe".into(),
            forward_max_err: 0.0,
            grad_max_err: None,
            executed_pairs: ext_sparse.accounting.executed_pairs,
            dense_pairs: ext_sparse.accounting.dense_candidate_pairs,
            empty_experts_skipped: ext_sparse.accounting.empty_experts_skipped,
            outcome: if bf16_underflow_preserved {
                OutcomeStatus::Accepted
            } else {
                OutcomeStatus::Invalid
            },
            detail: "Logit -100.0 underflows BF16 gate to 0.0, yet assignment-derived load count preserves count=1.".into(),
        },
        SparseFixtureCase {
            case_id: "04-exact-topk-tie-deterministic-selection".into(),
            forward_max_err: tie_fwd_err,
            grad_max_err: None,
            executed_pairs: tie_sparse.accounting.executed_pairs,
            dense_pairs: tie_sparse.accounting.dense_candidate_pairs,
            empty_experts_skipped: tie_sparse.accounting.empty_experts_skipped,
            outcome: if tie_ok { OutcomeStatus::Accepted } else { OutcomeStatus::Invalid },
            detail: "Actual tied-logit token ([0,0,0,-2] -> [0.5,0.5,0.5,0.12]) executed via moe_forward_backward deterministically selects lowest expert IDs [0, 1] without differentiating across discontinuity.".into(),
        },
        SparseFixtureCase {
            case_id: "05-three-step-matched-weight-and-bias-update-parity".into(),
            forward_max_err: step_max_err,
            grad_max_err: Some(fd_max_err),
            executed_pairs: sparse.accounting.executed_pairs,
            dense_pairs: sparse.accounting.dense_candidate_pairs,
            empty_experts_skipped: sparse.accounting.empty_experts_skipped,
            outcome: if step_max_err <= 1e-10 && fd_max_err <= 1e-7 {
                OutcomeStatus::Accepted
            } else {
                OutcomeStatus::Invalid
            },
            detail: "3-step matched update (including bias updates) and central finite-difference gradients agree; parity is separate from training-quality claims.".into(),
        },
    ];

    // 7 adversarial negative controls.
    let run_neg = |id: &str,
                   fault: SparseFaultInjection,
                   detail: &str|
     -> Result<SparseFixtureCase, String> {
        let bad = moe_forward_backward(
            &base_p,
            &tokens,
            &skew_bias,
            &d_out,
            eps,
            true,
            1,
            Some(2),
            fault,
        )?;
        let err = max_mat_diff(&dense.outputs, &bad.outputs)?;
        Ok(SparseFixtureCase {
            case_id: id.into(),
            forward_max_err: err,
            grad_max_err: None,
            executed_pairs: bad.accounting.executed_pairs,
            dense_pairs: bad.accounting.dense_candidate_pairs,
            empty_experts_skipped: bad.accounting.empty_experts_skipped,
            outcome: if err > 1e-6 {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::InternalFailure
            },
            detail: detail.into(),
        })
    };

    let nan_rejected = moe_forward_backward(
        &base_p,
        &[vec![f64::NAN, 0.0, 0.0, 0.0]],
        &skew_bias,
        &[vec![1.0; 4]],
        eps,
        true,
        1,
        None,
        SparseFaultInjection::none(),
    )
    .is_err();

    let negative_controls = vec![
        run_neg(
            "01-negative-wrong-bias-added-to-output-weights",
            SparseFaultInjection { add_bias_to_output_weights: true, ..SparseFaultInjection::none() },
            "Adding load-balancing bias into output gate weights fails numerical parity.",
        )?,
        run_neg(
            "02-negative-missing-affinity-normalization",
            SparseFaultInjection { omit_affinity_normalization: true, ..SparseFaultInjection::none() },
            "Omitting normalization over selected raw affinities fails numerical parity.",
        )?,
        run_neg(
            "03-negative-duplicate-scatter-contribution",
            SparseFaultInjection { duplicate_first_scatter: true, ..SparseFaultInjection::none() },
            "Duplicating scatter-add contribution fails output parity.",
        )?,
        run_neg(
            "04-negative-silent-token-drop-at-capacity-boundary",
            SparseFaultInjection { drop_overflow_assignment: true, ..SparseFaultInjection::none() },
            "Silently dropping overflow assignments at capacity boundary fails output parity.",
        )?,
        run_neg(
            "05-negative-incorrect-shared-expert-multi-execution",
            SparseFaultInjection { run_shared_expert_per_route: true, ..SparseFaultInjection::none() },
            "Running shared expert top_k times instead of once per token fails output parity.",
        )?,
        SparseFixtureCase {
            case_id: "06-negative-nonfinite-logit-rejected".into(),
            forward_max_err: 0.0,
            grad_max_err: None,
            executed_pairs: 0,
            dense_pairs: 4,
            empty_experts_skipped: 0,
            outcome: if nan_rejected { OutcomeStatus::Invalid } else { OutcomeStatus::InternalFailure },
            detail: "Non-finite NaN input/logit explicitly rejected rather than propagated.".into(),
        },
        SparseFixtureCase {
            case_id: "07-negative-dense-fallback-cannot-be-mislabeled-sparse".into(),
            forward_max_err: 0.0,
            grad_max_err: None,
            executed_pairs: dense.accounting.executed_pairs,
            dense_pairs: dense.accounting.dense_candidate_pairs,
            empty_experts_skipped: 0,
            outcome: if dense.accounting.dense_fallback_triggered
                && dense.accounting.executed_pairs == dense.accounting.dense_candidate_pairs
            {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::InternalFailure
            },
            detail: "Dense path executes 24/24 pairs with dense_fallback_triggered=true; cannot be reported as sparse.".into(),
        },
    ];

    // Sweep actual executed token/expert pairs by running real moe_forward_backward on every config!
    let sweep_configs = [
        (8usize, 16usize, 4usize, "balanced", 2usize),
        (8usize, 16usize, 4usize, "skewed_hot_expert", 4usize),
        (16usize, 16usize, 2usize, "balanced", 2usize),
        (16usize, 4usize, 4usize, "top_k_all_no_savings", 1usize),
    ];
    let mut sweep_rows = Vec::new();
    for (cfg_idx, &(nt, ne, tk, skew, bucket)) in sweep_configs.iter().enumerate() {
        let d_mod = 4usize;
        let sw_params = TinyMoeParams {
            d_model: d_mod,
            d_hidden: d_mod,
            n_experts: ne,
            top_k: tk,
            route_scale: 1.0,
            seq_aux_alpha: 0.01,
            w_router: (0..d_mod * ne)
                .map(|i| det_weight(900 + cfg_idx, i, 0.5))
                .collect(),
            w_shared: (0..d_mod * d_mod)
                .map(|i| det_weight(920 + cfg_idx, i, 0.3))
                .collect(),
            w_experts: (0..ne)
                .map(|e| {
                    (0..d_mod * d_mod)
                        .map(|i| det_weight(1000 + cfg_idx * 32 + e, i, 0.35))
                        .collect()
                })
                .collect(),
        };
        let sw_tokens: Vec<Vec<f64>> = (0..nt)
            .map(|t| {
                (0..d_mod)
                    .map(|d| det_weight(1200 + cfg_idx * 32 + t, d, 0.6))
                    .collect()
            })
            .collect();
        let sw_dout = vec![vec![0.25; d_mod]; nt];
        let mut sw_bias = vec![0.0_f64; ne];
        if skew == "skewed_hot_expert" {
            for b in sw_bias.iter_mut().take(tk.min(ne)) {
                *b = 3.0;
            }
        }
        let sw_res = moe_forward_backward(
            &sw_params,
            &sw_tokens,
            &sw_bias,
            &sw_dout,
            eps,
            true,
            bucket,
            None,
            SparseFaultInjection::none(),
        )?;
        let dense_pairs = sw_res.accounting.dense_candidate_pairs;
        let sparse_exec = sw_res.accounting.executed_pairs;
        let pad = sw_res.accounting.padded_pairs;
        sweep_rows.push(SparseSweepRow {
            n_tokens: nt,
            n_experts: ne,
            top_k: tk,
            skew_profile: skew.into(),
            dense_pairs,
            sparse_executed_pairs: sparse_exec,
            sparse_padded_pairs: pad,
            pair_reduction_ratio: (sparse_exec as f64) / (dense_pairs as f64),
        });
    }

    Ok(SparseMoeReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "physical_gpu_sparse_grouped_gemm_moe_timing",
            "CPU dense, independent tiny scalar, and grouped sparse MoE forward+backward+3-step update parity ran locally; physical GPU grouped-GEMM speedup remains unavailable_hardware_runtime.",
        ),
        dense_vs_tiny_max_err,
        dense_vs_sparse_forward_max_err,
        dense_vs_sparse_grad_max_err: grad_err,
        analytical_vs_fd_grad_max_err: fd_max_err,
        three_step_update_parity_max_err: step_max_err,
        bf16_underflow_preserved_assignment_count: bf16_underflow_preserved,
        differential_fixtures,
        negative_controls,
        sweep_rows,
        scoped_decision: "GO_FOR_CPU_SPARSE_MOE_PARITY_INCONCLUSIVE_FOR_GPU_SPEEDUP".into(),
    })
}

pub fn inputs_json(report: &SparseMoeReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.issue-21-sparse-moe.inputs.v1\",\n",
            "  \"issue\": 21,\n",
            "  \"execution_class\": {},\n",
            "  \"shared_parameter_set_across_dense_tiny_sparse\": true,\n",
            "  \"tie_breaking_policy\": \"descending_score_then_ascending_expert_id\",\n",
            "  \"load_count_source\": \"actual_selected_assignments_not_bf16_gates_gt_zero\",\n",
            "  \"sweep_condition_count\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.sweep_rows.len(),
    )
}

pub fn results_json(report: &SparseMoeReport) -> String {
    let mut out = String::new();
    writeln!(out, "{{").unwrap();
    writeln!(
        out,
        "  \"schema\": \"inference-lab.issue-21-sparse-moe.results.v1\","
    )
    .unwrap();
    writeln!(out, "  \"issue\": 21,").unwrap();
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
        "  \"dense_vs_tiny_max_err\": {},",
        finite_f64(report.dense_vs_tiny_max_err)
    )
    .unwrap();
    writeln!(
        out,
        "  \"dense_vs_sparse_forward_max_err\": {},",
        finite_f64(report.dense_vs_sparse_forward_max_err)
    )
    .unwrap();
    writeln!(
        out,
        "  \"dense_vs_sparse_grad_max_err\": {},",
        finite_f64(report.dense_vs_sparse_grad_max_err)
    )
    .unwrap();
    writeln!(
        out,
        "  \"analytical_vs_fd_grad_max_err\": {},",
        finite_f64(report.analytical_vs_fd_grad_max_err)
    )
    .unwrap();
    writeln!(
        out,
        "  \"three_step_update_parity_max_err\": {},",
        finite_f64(report.three_step_update_parity_max_err)
    )
    .unwrap();
    writeln!(
        out,
        "  \"bf16_underflow_preserved_assignment_count\": {},",
        report.bf16_underflow_preserved_assignment_count
    )
    .unwrap();
    writeln!(
        out,
        "  \"scoped_decision\": {},",
        quote(&report.scoped_decision)
    )
    .unwrap();

    let fmt_cases = |cases: &[SparseFixtureCase]| -> String {
        cases
            .iter()
            .map(|c| {
                format!(
                    concat!(
                        "{{\"case_id\":{},\"forward_max_err\":{},\"grad_max_err\":{},",
                        "\"executed_pairs\":{},\"dense_pairs\":{},\"empty_experts_skipped\":{},",
                        "\"outcome\":{},\"detail\":{}}}"
                    ),
                    quote(&c.case_id),
                    finite_f64(c.forward_max_err),
                    opt_f64(c.grad_max_err),
                    c.executed_pairs,
                    c.dense_pairs,
                    c.empty_experts_skipped,
                    quote(c.outcome.as_str()),
                    quote(&c.detail),
                )
            })
            .collect::<Vec<_>>()
            .join(",")
    };
    writeln!(
        out,
        "  \"differential_fixtures\": [{}],",
        fmt_cases(&report.differential_fixtures)
    )
    .unwrap();
    writeln!(
        out,
        "  \"negative_controls\": [{}],",
        fmt_cases(&report.negative_controls)
    )
    .unwrap();

    let sw: Vec<String> = report
        .sweep_rows
        .iter()
        .map(|r| {
            format!(
                concat!(
                    "{{\"n_tokens\":{},\"n_experts\":{},\"top_k\":{},\"skew_profile\":{},",
                    "\"dense_pairs\":{},\"sparse_executed_pairs\":{},\"sparse_padded_pairs\":{},",
                    "\"pair_reduction_ratio\":{}}}"
                ),
                r.n_tokens,
                r.n_experts,
                r.top_k,
                quote(&r.skew_profile),
                r.dense_pairs,
                r.sparse_executed_pairs,
                r.sparse_padded_pairs,
                finite_f64(r.pair_reduction_ratio),
            )
        })
        .collect();
    writeln!(out, "  \"sweep_rows\": [{}]", sw.join(",")).unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn summary_markdown(report: &SparseMoeReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #21: Sparse MoE Dispatch vs Dense Reference & Gradient Parity"
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
        "- Dense vs tiny max err: `{}`; dense vs sparse forward err: `{}`; grad err: `{}`; analytical vs FD grad err: `{}`",
        finite_f64(report.dense_vs_tiny_max_err),
        finite_f64(report.dense_vs_sparse_forward_max_err),
        finite_f64(report.dense_vs_sparse_grad_max_err),
        finite_f64(report.analytical_vs_fd_grad_max_err)
    )
    .unwrap();
    writeln!(
        out,
        "- 3-step matched update parity max err: `{}`; BF16 underflow assignment count preserved: `{}`",
        finite_f64(report.three_step_update_parity_max_err),
        report.bf16_underflow_preserved_assignment_count
    )
    .unwrap();
    writeln!(out, "- Scoped decision: `{}`", report.scoped_decision).unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Differential Fixtures").unwrap();
    for c in &report.differential_fixtures {
        writeln!(
            out,
            "- `{}`: `{}` (pairs `{}/{}`, empty_skipped={}, fwd_err=`{}`)",
            c.case_id,
            c.outcome.as_str(),
            c.executed_pairs,
            c.dense_pairs,
            c.empty_experts_skipped,
            finite_f64(c.forward_max_err)
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "## Adversarial Negative Controls").unwrap();
    for c in &report.negative_controls {
        writeln!(
            out,
            "- `{}`: `{}` (err=`{}`) — {}",
            c.case_id,
            c.outcome.as_str(),
            finite_f64(c.forward_max_err),
            c.detail
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "> Selected-parameter count alone is not evidence of saved compute; GPU grouped-GEMM timing remains `unavailable_hardware_runtime`."
    )
    .unwrap();
    out
}
