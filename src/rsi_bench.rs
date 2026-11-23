//! Issue #12: Consolidate standalone RSI inference benchmarks with parity and provenance gates.
//!
//! Pure deterministic core implementing:
//! - Explicit donor-to-destination ownership and provenance manifest pinned to
//!   `OnixResearch/rsi-lab@01e47e268d71bc1c53cc36e11c179f2b8f22f5c5` (5 selected donor surfaces)
//!   while keeping RSI stateful/decision owners (`packing_flip_replay.py`, `gpu_memory_graph_probe.py`,
//!   `gpu_packed_read_probe.py`, trainer/simulator, closed `#18` receipts) in RSI.
//! - Real pure-Rust FP64 RWKV-7 state-update recurrence (`rwkv7_step_f64`, `rwkv7_forward_f64`)
//!   verifying token-by-token vs chunked output and final-state parity plus negative controls
//!   (altered transform/input, perturbed final state).
//! - RSI `#82` kernel-sweep qualification gate (requiring precomputed inputs, output + final-state
//!   parity, allocation/RNG excluded from timed regions, warmups, and synchronized timing; keeping
//!   broken historical sweep as `superseded_invalid`).
//! - Encoder cold/warm/compile phase separation, fake-quantization INT8-speedup rejection,
//!   private-to-public publication allowlist, and immutable closed `#18` receipt preservation.

use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_f64, opt_str, quote, str_array};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const RSI_DONOR_COMMIT: &str = "01e47e268d71bc1c53cc36e11c179f2b8f22f5c5";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DonorSurfaceEntry {
    pub donor_path: String,
    pub logical_group: String,
    pub destination_responsibility: String,
    pub preserved_limit: String,
    pub publishable_public: bool,
    pub dependency_env: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Rwkv7TokenInput {
    /// Decay vector per head: `[n_heads][head_dim]`, values in `(0, 1]`.
    pub decay_w: Vec<Vec<f64>>,
    /// In-context learning rate vector `a`: `[n_heads][head_dim]`.
    pub iclr_a: Vec<Vec<f64>>,
    /// Replacement key `b`: `[n_heads][head_dim]`.
    pub key_b: Vec<Vec<f64>>,
    /// Key vector `k`: `[n_heads][head_dim]`.
    pub key_k: Vec<Vec<f64>>,
    /// Value vector `v`: `[n_heads][head_dim]`.
    pub val_v: Vec<Vec<f64>>,
    /// Receptance vector `r`: `[n_heads][head_dim]`.
    pub receptance_r: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Rwkv7StateF64 {
    pub n_heads: usize,
    pub head_dim: usize,
    /// Flattened `[n_heads, head_dim, head_dim]` row-major state matrix (`S[h][i][j]`).
    pub data: Vec<f64>,
}

impl Rwkv7StateF64 {
    pub fn zeros(n_heads: usize, head_dim: usize) -> Result<Self, String> {
        if n_heads == 0 || head_dim == 0 {
            return Err("rwkv7 dimensions must be positive".into());
        }
        let total = n_heads
            .checked_mul(head_dim)
            .and_then(|v| v.checked_mul(head_dim))
            .ok_or_else(|| "rwkv7 state size overflow".to_string())?;
        Ok(Self {
            n_heads,
            head_dim,
            data: vec![0.0; total],
        })
    }

    pub fn byte_len(&self) -> usize {
        self.data.len() * std::mem::size_of::<f64>()
    }

    pub fn max_abs_diff(&self, other: &Self) -> Result<f64, String> {
        if self.n_heads != other.n_heads
            || self.head_dim != other.head_dim
            || self.data.len() != other.data.len()
        {
            return Err("rwkv7 state shape mismatch".into());
        }
        let mut max_err = 0.0_f64;
        for (&a, &b) in self.data.iter().zip(other.data.iter()) {
            if !a.is_finite() || !b.is_finite() {
                return Err("non-finite value in rwkv7 state comparison".into());
            }
            let diff = (a - b).abs();
            if diff > max_err {
                max_err = diff;
            }
        }
        Ok(max_err)
    }
}

/// Pure-Rust FP64 single-step RWKV-7 generalized delta-rule recurrence:
/// `S_t[h] = S_{t-1}[h] * diag(w_t) + (S_{t-1}[h] @ a_t) * b_t^T + v_t * k_t^T`
/// `y_t[h] = S_t[h] @ r_t`
pub fn rwkv7_step_f64(
    state: &mut Rwkv7StateF64,
    input: &Rwkv7TokenInput,
) -> Result<Vec<f64>, String> {
    let h_count = state.n_heads;
    let d = state.head_dim;
    if input.decay_w.len() != h_count
        || input.iclr_a.len() != h_count
        || input.key_b.len() != h_count
        || input.key_k.len() != h_count
        || input.val_v.len() != h_count
        || input.receptance_r.len() != h_count
    {
        return Err("rwkv7 head count mismatch".into());
    }

    let mut out = vec![0.0_f64; h_count * d];
    for h in 0..h_count {
        let w = &input.decay_w[h];
        let a = &input.iclr_a[h];
        let b = &input.key_b[h];
        let k = &input.key_k[h];
        let v = &input.val_v[h];
        let r = &input.receptance_r[h];
        if w.len() != d
            || a.len() != d
            || b.len() != d
            || k.len() != d
            || v.len() != d
            || r.len() != d
        {
            return Err("rwkv7 head_dim mismatch".into());
        }
        for slice in [w, a, b, k, v, r] {
            if slice.iter().any(|x| !x.is_finite()) {
                return Err("non-finite value in rwkv7 step input".into());
            }
        }

        let head_offset = h * d * d;
        for row in 0..d {
            let row_offset = head_offset + row * d;
            // sa = sum_j S[row, j] * a[j]
            let mut sa = 0.0_f64;
            for (j, &aj) in a.iter().enumerate() {
                sa += state.data[row_offset + j] * aj;
            }
            let mut y_row = 0.0_f64;
            for j in 0..d {
                let updated = state.data[row_offset + j] * w[j] + sa * b[j] + v[row] * k[j];
                if !updated.is_finite() {
                    return Err("non-finite overflow in rwkv7 state update".into());
                }
                state.data[row_offset + j] = updated;
                y_row += updated * r[j];
            }
            if !y_row.is_finite() {
                return Err("non-finite overflow in rwkv7 output".into());
            }
            out[h * d + row] = y_row;
        }
    }
    Ok(out)
}

/// Run an entire sequence of RWKV-7 token steps in FP64, returning outputs `[T][n_heads * head_dim]`.
pub fn rwkv7_forward_f64(
    initial_state: &Rwkv7StateF64,
    inputs: &[Rwkv7TokenInput],
) -> Result<(Vec<Vec<f64>>, Rwkv7StateF64), String> {
    let mut state = initial_state.clone();
    let mut outputs = Vec::with_capacity(inputs.len());
    for tok in inputs {
        outputs.push(rwkv7_step_f64(&mut state, tok)?);
    }
    Ok((outputs, state))
}

/// Deterministic synthetic RWKV-7 sequence fixture reused across `#12` and `#16`.
pub fn synthetic_rwkv7_sequence(
    n_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<Rwkv7TokenInput> {
    let mut seq = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let mut decay_w = Vec::with_capacity(n_heads);
        let mut iclr_a = Vec::with_capacity(n_heads);
        let mut key_b = Vec::with_capacity(n_heads);
        let mut key_k = Vec::with_capacity(n_heads);
        let mut val_v = Vec::with_capacity(n_heads);
        let mut receptance_r = Vec::with_capacity(n_heads);

        for h in 0..n_heads {
            let mut w_h = Vec::with_capacity(head_dim);
            let mut a_h = Vec::with_capacity(head_dim);
            let mut b_h = Vec::with_capacity(head_dim);
            let mut k_h = Vec::with_capacity(head_dim);
            let mut v_h = Vec::with_capacity(head_dim);
            let mut r_h = Vec::with_capacity(head_dim);

            for d in 0..head_dim {
                let phase = ((t * 17 + h * 11 + d * 7 + 3) as f64) * 0.13;
                // Strictly contractive decay in (0.55, 0.92) for stable long-horizon recurrence.
                w_h.push(0.74 + 0.16 * phase.sin());
                a_h.push(0.08 * (phase * 1.1).cos());
                b_h.push(-0.06 * (phase * 0.9).sin());
                k_h.push(0.25 * (phase * 1.3).sin());
                v_h.push(0.30 * (phase * 0.7).cos());
                r_h.push(0.35 * (phase * 0.5 + 0.4).sin());
            }
            decay_w.push(w_h);
            iclr_a.push(a_h);
            key_b.push(b_h);
            key_k.push(k_h);
            val_v.push(v_h);
            receptance_r.push(r_h);
        }

        seq.push(Rwkv7TokenInput {
            decay_w,
            iclr_a,
            key_b,
            key_k,
            val_v,
            receptance_r,
        });
    }
    seq
}

/// Pure-Rust symmetric int8 fake-quantization (dequantized back to FP64 — numeric error probe only!).
pub fn fake_quant_int8_slice(values: &[f64]) -> Result<(Vec<f64>, f64), String> {
    if values.is_empty() || values.iter().any(|x| !x.is_finite()) {
        return Err("fake_quant_int8_slice requires non-empty finite values".into());
    }
    let max_abs = values.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
    if max_abs == 0.0 {
        return Ok((vec![0.0; values.len()], 0.0));
    }
    let scale = max_abs / 127.0;
    let mut out = Vec::with_capacity(values.len());
    let mut max_err = 0.0_f64;
    for &x in values {
        let q = (x / scale).round().clamp(-127.0, 127.0);
        let deq = q * scale;
        max_err = max_err.max((x - deq).abs());
        out.push(deq);
    }
    Ok((out, max_err))
}

#[derive(Debug, Clone, PartialEq)]
pub struct Rsi82SweepQualificationInput {
    pub case_id: String,
    pub precomputed_equivalent_inputs: bool,
    pub output_max_abs_err: f64,
    pub final_state_max_abs_err: f64,
    pub tolerance: f64,
    pub rng_or_alloc_inside_timed_region: bool,
    pub warmups_completed: usize,
    pub synchronized_wall_and_device_timing: bool,
    pub retains_prior_sweep_as_superseded: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Rsi82SweepVerdict {
    pub case_id: String,
    pub qualified_baseline: bool,
    pub outcome: OutcomeStatus,
    pub rejection_reason: Option<String>,
}

/// Evaluate RSI `#82` kernel-sweep qualification before any crossover baseline can be consumed.
pub fn evaluate_rsi82_sweep(input: &Rsi82SweepQualificationInput) -> Rsi82SweepVerdict {
    let reject = |reason: &str| Rsi82SweepVerdict {
        case_id: input.case_id.clone(),
        qualified_baseline: false,
        outcome: OutcomeStatus::Invalid,
        rejection_reason: Some(reason.into()),
    };

    if !input.precomputed_equivalent_inputs {
        return reject("inputs_and_gates_not_precomputed_equivalently");
    }
    if input.rng_or_alloc_inside_timed_region {
        return reject("rng_or_allocation_inside_timed_region_violates_rsi82");
    }
    if input.warmups_completed == 0 {
        return reject("missing_warmup_iterations");
    }
    if !input.synchronized_wall_and_device_timing {
        return reject("missing_synchronized_wall_and_device_timing");
    }
    if !input.retains_prior_sweep_as_superseded {
        return reject("historical_broken_sweep_overwritten_instead_of_superseded");
    }
    if !input.output_max_abs_err.is_finite()
        || input.output_max_abs_err < 0.0
        || !input.final_state_max_abs_err.is_finite()
        || input.final_state_max_abs_err < 0.0
        || !input.tolerance.is_finite()
        || input.tolerance < 0.0
    {
        return reject("nonfinite_or_negative_parity_metric_or_tolerance");
    }
    if input.output_max_abs_err > input.tolerance {
        return reject("output_parity_exceeds_tolerance");
    }
    if input.final_state_max_abs_err > input.tolerance {
        return reject("final_recurrent_state_parity_exceeds_tolerance");
    }

    Rsi82SweepVerdict {
        case_id: input.case_id.clone(),
        qualified_baseline: true,
        outcome: OutcomeStatus::Accepted,
        rejection_reason: None,
    }
}

/// Validate that encoder benchmark reports separate compile/capture/warmup from steady-state.
pub fn validate_encoder_phase_separation(
    compile_and_capture_mixed_into_steady_state: bool,
    steady_state_ms: f64,
) -> OutcomeStatus {
    if !steady_state_ms.is_finite()
        || steady_state_ms < 0.0
        || compile_and_capture_mixed_into_steady_state
    {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    }
}

/// Reject inferring INT8 runtime speedup from fake-quantization numeric-error probes.
pub fn validate_fake_quant_claim(
    is_fake_quant_probe: bool,
    claims_int8_runtime_speedup: bool,
) -> OutcomeStatus {
    if is_fake_quant_probe && claims_int8_runtime_speedup {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    }
}

/// Enforce immutability of historical closed RSI `#18` receipts (new runs require fresh IDs).
pub fn validate_rsi18_receipt_immutability(
    prior_receipt_id: &str,
    new_run_id: &str,
    overwrites_prior_receipt: bool,
) -> OutcomeStatus {
    if prior_receipt_id.is_empty()
        || new_run_id.is_empty()
        || prior_receipt_id == new_run_id
        || overwrites_prior_receipt
    {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    }
}

/// Enforce private-to-public publication allowlist (zero private logs, traces, or model assets).
pub fn validate_public_publication_allowlist(
    contains_private_logs_traces_or_weights: bool,
) -> OutcomeStatus {
    if contains_private_logs_traces_or_weights {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    }
}

/// Missing optional GPU/FLA/LinQuant runtime stays `UnavailableHardwareRuntime`, never `Accepted`.
pub fn validate_optional_runtime_availability(
    requires_gpu_or_optional_dep: bool,
    runtime_available: bool,
) -> OutcomeStatus {
    if requires_gpu_or_optional_dep && !runtime_available {
        OutcomeStatus::UnavailableHardwareRuntime
    } else {
        OutcomeStatus::Accepted
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConsolidationControl {
    pub control_id: String,
    pub category: String,
    pub outcome: OutcomeStatus,
    pub max_abs_err: Option<f64>,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct RsiBenchReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub donor_commit: String,
    pub donor_surfaces: Vec<DonorSurfaceEntry>,
    pub rsi_retained_boundaries: Vec<String>,
    pub rsi82_verdicts: Vec<Rsi82SweepVerdict>,
    pub controls: Vec<ConsolidationControl>,
    pub fake_quant_max_abs_err: f64,
    pub fake_quant_implies_int8_speedup: bool,
    pub historical_rsi18_preserved_immutable: bool,
    pub nvfp4_mxfp8_consumer_note: String,
    pub fixture_digest: DigestRef,
}

pub fn run_default_harness() -> Result<RsiBenchReport, String> {
    let donor_surfaces = vec![
        DonorSurfaceEntry {
            donor_path: "python/scripts/gpu_rwkv7_kernel_sweep.py".into(),
            logical_group: "benchmarks/recurrent".into(),
            destination_responsibility: "Recurrent kernel crossover benchmark (gated on RSI #82 repair)".into(),
            preserved_limit: "Current timed fused baseline unfixed until RSI #82 lands; old receipt retained as superseded_invalid".into(),
            publishable_public: true,
            dependency_env: "opt-in torch+fla CUDA env (unavailable locally)".into(),
        },
        DonorSurfaceEntry {
            donor_path: "python/scripts/gpu_encode_bench.py (+ compile/graph probes)".into(),
            logical_group: "benchmarks/encoders".into(),
            destination_responsibility: "Encoder prefill, compilation, CUDA-graph capture, and steady-state latency".into(),
            preserved_limit: "Cold, warm, compile, and graph-capture phases strictly separated from steady state".into(),
            publishable_public: true,
            dependency_env: "opt-in torch+transformers CUDA env (unavailable locally)".into(),
        },
        DonorSurfaceEntry {
            donor_path: "python/scripts/rwkv7_invariances.py".into(),
            logical_group: "benchmarks/recurrent".into(),
            destination_responsibility: "Model-free RWKV-7 recurrence reference, transform & state-continuation controls".into(),
            preserved_limit: "Positive and negative controls run CPU-only with zero weights, network, or GPU".into(),
            publishable_public: true,
            dependency_env: "pure-rust-cpu-core (executed locally)".into(),
        },
        DonorSurfaceEntry {
            donor_path: "python/scripts/gpu_rwkv7_quant_arm.py".into(),
            logical_group: "experiments/quantization".into(),
            destination_responsibility: "Held-out numerical precision probe (fake-quantization error only)".into(),
            preserved_limit: "Fake-quantizes in higher precision; zero INT8 runtime speedup may be inferred".into(),
            publishable_public: true,
            dependency_env: "opt-in torch CUDA env (unavailable locally)".into(),
        },
        DonorSurfaceEntry {
            donor_path: "python/src/rwkv_clm_adapter/linquant_gdn.py (+ gdn_linquant.py fixture)".into(),
            logical_group: "experiments/quantization".into(),
            destination_responsibility: "Pinned GDN reference adapter and synthetic known-answer fixture".into(),
            preserved_limit: "Preserve pinned upstream revision, notices, and explicit optional-dep skip semantics".into(),
            publishable_public: true,
            dependency_env: "opt-in linquant env (unavailable locally)".into(),
        },
    ];

    let rsi_retained_boundaries = vec![
        "packing_flip_replay.py and RSI #11 remain the decision-stability acceptance owner in RSI Lab.".into(),
        "gpu_memory_graph_probe.py and gpu_packed_read_probe.py remain in RSI Lab (depend on FlaRwkv7Memory state/cache).".into(),
        "Stateful backend wrappers, checkpoints/snapshots, policy calibration, simulator labels, trainer, and promotion remain in RSI Lab.".into(),
        "Closed RSI #18 receipts keep their original protocol/commit/receipt identities immutably; open RSI #82 stays open until met.".into(),
        "Private donor histories, models, traces, host inventories, and operational logs are excluded from public publication.".into(),
    ];

    // Real pure-Rust RWKV-7 recurrence invariance suite (n_heads=2, head_dim=4, seq_len=8).
    let seq = synthetic_rwkv7_sequence(2, 4, 8);
    let init = Rwkv7StateF64::zeros(2, 4)?;
    let (full_out, full_state) = rwkv7_forward_f64(&init, &seq)?;

    // Chunked split (3 + 5 tokens) must match token-by-token full run exactly.
    let (chunk1_out, mid_state) = rwkv7_forward_f64(&init, &seq[..3])?;
    let (chunk2_out, chunk_state) = rwkv7_forward_f64(&mid_state, &seq[3..])?;
    let mut combined_out = chunk1_out;
    combined_out.extend(chunk2_out);

    let mut max_out_err = 0.0_f64;
    for (a, b) in full_out.iter().zip(combined_out.iter()) {
        for (&va, &vb) in a.iter().zip(b.iter()) {
            max_out_err = max_out_err.max((va - vb).abs());
        }
    }
    let state_err = full_state.max_abs_diff(&chunk_state)?;

    // Negative control 1: deliberately altered transform (flipped key sign at t=4).
    let mut altered_seq = seq.clone();
    altered_seq[4].key_k[0][0] = -altered_seq[4].key_k[0][0];
    let (altered_out, altered_state) = rwkv7_forward_f64(&init, &altered_seq)?;
    let mut altered_out_err = 0.0_f64;
    for (a, b) in full_out.iter().zip(altered_out.iter()) {
        for (&va, &vb) in a.iter().zip(b.iter()) {
            altered_out_err = altered_out_err.max((va - vb).abs());
        }
    }
    let altered_state_err = full_state.max_abs_diff(&altered_state)?;

    // Negative control 2: deliberately perturbed final state (same outputs, corrupted state).
    let mut perturbed_state = full_state.clone();
    perturbed_state.data[0] += 0.25;
    let perturbed_state_err = full_state.max_abs_diff(&perturbed_state)?;

    // Fake-quant numeric error probe (explicitly non-speedup).
    let (_, fq_err) = fake_quant_int8_slice(&full_state.data)?;

    let rsi82_verdicts = vec![
        evaluate_rsi82_sweep(&Rsi82SweepQualificationInput {
            case_id: "01-positive-repaired-rsi82-contract".into(),
            precomputed_equivalent_inputs: true,
            output_max_abs_err: max_out_err,
            final_state_max_abs_err: state_err,
            tolerance: 1e-12,
            rng_or_alloc_inside_timed_region: false,
            warmups_completed: 3,
            synchronized_wall_and_device_timing: true,
            retains_prior_sweep_as_superseded: true,
        }),
        evaluate_rsi82_sweep(&Rsi82SweepQualificationInput {
            case_id: "02-negative-legacy-unfixed-rng-in-timed-region".into(),
            precomputed_equivalent_inputs: false,
            output_max_abs_err: 0.0,
            final_state_max_abs_err: 0.0,
            tolerance: 1e-12,
            rng_or_alloc_inside_timed_region: true,
            warmups_completed: 0,
            synchronized_wall_and_device_timing: false,
            retains_prior_sweep_as_superseded: true,
        }),
        evaluate_rsi82_sweep(&Rsi82SweepQualificationInput {
            case_id: "03-negative-altered-final-state-fails-rsi82".into(),
            precomputed_equivalent_inputs: true,
            output_max_abs_err: 0.0,
            final_state_max_abs_err: perturbed_state_err,
            tolerance: 1e-12,
            rng_or_alloc_inside_timed_region: false,
            warmups_completed: 3,
            synchronized_wall_and_device_timing: true,
            retains_prior_sweep_as_superseded: true,
        }),
        evaluate_rsi82_sweep(&Rsi82SweepQualificationInput {
            case_id: "04-negative-overwriting-superseded-evidence-rejected".into(),
            precomputed_equivalent_inputs: true,
            output_max_abs_err: 0.0,
            final_state_max_abs_err: 0.0,
            tolerance: 1e-12,
            rng_or_alloc_inside_timed_region: false,
            warmups_completed: 3,
            synchronized_wall_and_device_timing: true,
            retains_prior_sweep_as_superseded: false,
        }),
    ];

    let controls = vec![
        ConsolidationControl {
            control_id: "01-positive-rwkv7-chunked-output-and-state-parity".into(),
            category: "rwkv7_invariances".into(),
            outcome: if max_out_err <= 1e-12 && state_err <= 1e-12 {
                OutcomeStatus::Accepted
            } else {
                OutcomeStatus::Invalid
            },
            max_abs_err: Some(max_out_err.max(state_err)),
            detail: "3+5 chunked split matches uninterrupted 8-token FP64 recurrence on both outputs and final state.".into(),
        },
        ConsolidationControl {
            control_id: "02-negative-altered-transform-fails-parity".into(),
            category: "rwkv7_invariances".into(),
            outcome: if altered_out_err > 1e-6 || altered_state_err > 1e-6 {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::InternalFailure
            },
            max_abs_err: Some(altered_out_err.max(altered_state_err)),
            detail: "Sign-flipped key input at t=4 produces detectable output and final-state divergence.".into(),
        },
        ConsolidationControl {
            control_id: "03-negative-perturbed-final-state-fails-parity".into(),
            category: "rwkv7_invariances".into(),
            outcome: if perturbed_state_err > 1e-12 {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::InternalFailure
            },
            max_abs_err: Some(perturbed_state_err),
            detail: "Corrupted final recurrent state is rejected even if intermediate outputs matched.".into(),
        },
        ConsolidationControl {
            control_id: "04-negative-encoder-compile-capture-conflation-rejected".into(),
            category: "encoder_bench".into(),
            outcome: validate_encoder_phase_separation(true, 12.5),
            max_abs_err: None,
            detail: "Mixing cold compile/CUDA-graph capture into steady-state prefill latency fails phase separation.".into(),
        },
        ConsolidationControl {
            control_id: "05-negative-fake-quant-claimed-as-int8-speedup-rejected".into(),
            category: "quantization_probe".into(),
            outcome: validate_fake_quant_claim(true, true),
            max_abs_err: Some(fq_err),
            detail: "Fake-quantization measures numeric error only; inferring INT8 runtime speedup is explicitly rejected.".into(),
        },
        ConsolidationControl {
            control_id: "06-negative-overwrite-closed-rsi18-receipt-rejected".into(),
            category: "provenance_immutability".into(),
            outcome: validate_rsi18_receipt_immutability("rsi-18-orig", "rsi-18-orig", true),
            max_abs_err: None,
            detail: "Closed RSI #18 receipts retain original protocol/commit/receipt identities; new runs require fresh IDs.".into(),
        },
        ConsolidationControl {
            control_id: "07-negative-private-trace-or-model-asset-leak-rejected".into(),
            category: "public_publication_boundary".into(),
            outcome: validate_public_publication_allowlist(true),
            max_abs_err: None,
            detail: "Private donor logs, traces, host inventories, and model weights are excluded from public manifest.".into(),
        },
        ConsolidationControl {
            control_id: "08-control-missing-gpu-or-fla-stays-unavailable".into(),
            category: "optional_runtime_boundary".into(),
            outcome: validate_optional_runtime_availability(true, false),
            max_abs_err: None,
            detail: "Missing optional Torch/FLA/LinQuant/CUDA runtime remains explicitly unavailable, never a successful migration benchmark.".into(),
        },
    ];

    let mut digest_bytes = Vec::new();
    for val in &full_state.data {
        digest_bytes.extend_from_slice(&val.to_bits().to_le_bytes());
    }
    let fixture_digest = DigestRef::sha256(&digest_bytes);

    Ok(RsiBenchReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "cuda_gpu_torch_fla_linquant_runtime",
            "CPU-only contract & RWKV-7 recurrence fixtures ran locally; optional Torch/FLA/Transformers/LinQuant GPU sweeps remain unavailable_hardware_runtime until executed on a qualified GPU host.",
        ),
        donor_commit: RSI_DONOR_COMMIT.into(),
        donor_surfaces,
        rsi_retained_boundaries,
        rsi82_verdicts,
        controls,
        fake_quant_max_abs_err: fq_err,
        fake_quant_implies_int8_speedup: false,
        historical_rsi18_preserved_immutable: true,
        nvfp4_mxfp8_consumer_note: "Issue #26 (NVFP4-forward/MXFP8-backward qualification) is a separately scoped consumer of #12's provenance and numerical-vs-native conventions; it does not block #12.".into(),
        fixture_digest,
    })
}

pub fn inputs_json(report: &RsiBenchReport) -> String {
    let surfaces: Vec<String> = report
        .donor_surfaces
        .iter()
        .map(|s| {
            format!(
                concat!(
                    "{{\"donor_path\":{},\"logical_group\":{},\"destination_responsibility\":{},",
                    "\"preserved_limit\":{},\"publishable_public\":{},\"dependency_env\":{}}}"
                ),
                quote(&s.donor_path),
                quote(&s.logical_group),
                quote(&s.destination_responsibility),
                quote(&s.preserved_limit),
                s.publishable_public,
                quote(&s.dependency_env),
            )
        })
        .collect();

    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.issue-12-rsi-bench.inputs.v1\",\n",
            "  \"issue\": 12,\n",
            "  \"execution_class\": {},\n",
            "  \"donor_repo\": \"OnixResearch/rsi-lab\",\n",
            "  \"donor_commit\": {},\n",
            "  \"donor_surfaces\": [{}],\n",
            "  \"rsi_retained_boundaries\": {},\n",
            "  \"nvfp4_mxfp8_consumer_note\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(&report.donor_commit),
        surfaces.join(","),
        str_array(&report.rsi_retained_boundaries),
        quote(&report.nvfp4_mxfp8_consumer_note),
    )
}

pub fn results_json(report: &RsiBenchReport) -> String {
    let mut out = String::new();
    writeln!(out, "{{").unwrap();
    writeln!(
        out,
        "  \"schema\": \"inference-lab.issue-12-rsi-bench.results.v1\","
    )
    .unwrap();
    writeln!(out, "  \"issue\": 12,").unwrap();
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
        "  \"fixture_final_state_digest\": {},",
        report.fixture_digest.to_json()
    )
    .unwrap();
    writeln!(
        out,
        "  \"fake_quant_max_abs_err\": {},",
        finite_f64(report.fake_quant_max_abs_err)
    )
    .unwrap();
    writeln!(
        out,
        "  \"fake_quant_implies_int8_speedup\": {},",
        report.fake_quant_implies_int8_speedup
    )
    .unwrap();
    writeln!(
        out,
        "  \"historical_rsi18_preserved_immutable\": {},",
        report.historical_rsi18_preserved_immutable
    )
    .unwrap();

    let v82: Vec<String> = report
        .rsi82_verdicts
        .iter()
        .map(|v| {
            format!(
                "{{\"case_id\":{},\"qualified_baseline\":{},\"outcome\":{},\"rejection_reason\":{}}}",
                quote(&v.case_id),
                v.qualified_baseline,
                quote(v.outcome.as_str()),
                opt_str(v.rejection_reason.as_deref()),
            )
        })
        .collect();
    writeln!(out, "  \"rsi82_gate_verdicts\": [{}],", v82.join(",")).unwrap();

    let ctrls: Vec<String> = report
        .controls
        .iter()
        .map(|c| {
            format!(
                "{{\"control_id\":{},\"category\":{},\"outcome\":{},\"max_abs_err\":{},\"detail\":{}}}",
                quote(&c.control_id),
                quote(&c.category),
                quote(c.outcome.as_str()),
                opt_f64(c.max_abs_err),
                quote(&c.detail),
            )
        })
        .collect();
    writeln!(out, "  \"controls\": [{}]", ctrls.join(",")).unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn summary_markdown(report: &RsiBenchReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #12: Standalone RSI Inference Benchmark Consolidation"
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
        "- Donor pin: `OnixResearch/rsi-lab@{}`",
        report.donor_commit
    )
    .unwrap();
    writeln!(
        out,
        "- Fake-quantization max abs error: `{}` (`implies_int8_speedup={}`)",
        finite_f64(report.fake_quant_max_abs_err),
        report.fake_quant_implies_int8_speedup
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## RSI `#82` Sweep Qualification Verdicts").unwrap();
    for v in &report.rsi82_verdicts {
        writeln!(
            out,
            "- `{}`: qualified=`{}` (`{}` — {})",
            v.case_id,
            v.qualified_baseline,
            v.outcome.as_str(),
            v.rejection_reason.as_deref().unwrap_or("passed")
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "## Consolidation & Recurrence Controls").unwrap();
    writeln!(out, "| Control | Category | Outcome | Max Abs Err |").unwrap();
    writeln!(out, "| --- | --- | --- | ---: |").unwrap();
    for c in &report.controls {
        writeln!(
            out,
            "| `{}` | `{}` | `{}` | {} |",
            c.control_id,
            c.category,
            c.outcome.as_str(),
            opt_f64(c.max_abs_err)
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "> Pure-Rust CPU recurrence and contract checks executed locally. Optional GPU Torch/FLA/Transformers/LinQuant sweeps remain `unavailable_hardware_runtime`; RSI `#11` and stateful/decision owners stay in RSI Lab."
    )
    .unwrap();
    out
}
