//! Issue #16: Qualify `ggml` RWKV-7 recurrence and state-continuation parity against native
//! RSI paths.
//!
//! Pure deterministic core implementing:
//! - Pinned `ggml-org/ggml@417a024b4fb6bc3116d41af7c067166a039bdf12` (`src/ggml.c#L5894-L5939`,
//!   `src/ggml-cpu/ops.cpp`, `src/ggml-cuda/wkv.cu`) and `rsi-lab@01e47e26...` provenance.
//! - Independent pure-Rust FP32 `ggml`-style `wkv7` operator (`Rwkv7GgmlStateF32`) compared
//!   directly against `rsi_bench::rwkv7_forward_f64` on frozen `[n_heads, head_dim, head_dim]`
//!   layouts across token-at-a-time vs chunked execution, 64-step long-horizon drift checkpoints,
//!   and exact little-endian binary state export/import round-trips with SHA-256 receipts.
//! - Negative controls: altered exported state, sequence boundary shift, BF16-style precision
//!   drift, explicit rejection of unsupported backward/training claims, and zero silent CPU
//!   fallback when `ggml-cuda` (`wkv.cu`) is requested.
//! - Explicit boundary: `asserts_agent_persistence_correctness = false` (kernel numerical state
//!   never establishes durable RSI agent epochs, snapshots, or policy commits).

use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};
use crate::rsi_bench::{
    rwkv7_forward_f64, synthetic_rwkv7_sequence, Rwkv7StateF64, Rwkv7TokenInput, RSI_DONOR_COMMIT,
};

pub const GGML_RWKV7_COMMIT: &str = "417a024b4fb6bc3116d41af7c067166a039bdf12";

/// Independent FP32 `ggml`-style RWKV-7 operator state (`[n_heads, head_dim, head_dim]`,
/// little-endian IEEE-754 f32 serialization).
#[derive(Debug, Clone, PartialEq)]
pub struct Rwkv7GgmlStateF32 {
    pub n_heads: usize,
    pub head_dim: usize,
    pub data: Vec<f32>,
}

impl Rwkv7GgmlStateF32 {
    pub fn zeros(n_heads: usize, head_dim: usize) -> Result<Self, String> {
        if n_heads == 0 || head_dim == 0 {
            return Err("ggml wkv7 dimensions must be positive".into());
        }
        let total = n_heads
            .checked_mul(head_dim)
            .and_then(|v| v.checked_mul(head_dim))
            .ok_or_else(|| "ggml wkv7 state overflow".to_string())?;
        Ok(Self {
            n_heads,
            head_dim,
            data: vec![0.0_f32; total],
        })
    }

    pub fn byte_len(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    pub fn export_le_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.byte_len());
        for &v in &self.data {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    pub fn import_le_bytes(n_heads: usize, head_dim: usize, bytes: &[u8]) -> Result<Self, String> {
        let expected_elems = n_heads
            .checked_mul(head_dim)
            .and_then(|v| v.checked_mul(head_dim))
            .ok_or_else(|| "ggml wkv7 import overflow".to_string())?;
        let expected_bytes = expected_elems
            .checked_mul(4)
            .ok_or_else(|| "ggml wkv7 import byte-length overflow".to_string())?;
        if bytes.len() != expected_bytes {
            return Err("ggml wkv7 imported state byte length mismatch".into());
        }
        let mut data = Vec::with_capacity(expected_elems);
        for chunk in bytes.chunks_exact(4) {
            let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if !v.is_finite() {
                return Err("non-finite f32 in imported ggml wkv7 state".into());
            }
            data.push(v);
        }
        Ok(Self {
            n_heads,
            head_dim,
            data,
        })
    }

    pub fn max_abs_diff_vs_f64(&self, reference: &Rwkv7StateF64) -> Result<f64, String> {
        if self.n_heads != reference.n_heads
            || self.head_dim != reference.head_dim
            || self.data.len() != reference.data.len()
        {
            return Err("ggml vs reference state shape mismatch".into());
        }
        let mut err = 0.0_f64;
        for (&a, &b) in self.data.iter().zip(reference.data.iter()) {
            let af = f64::from(a);
            if !af.is_finite() || !b.is_finite() {
                return Err("non-finite value in state comparison".into());
            }
            err = err.max((af - b).abs());
        }
        Ok(err)
    }
}

/// Optional precision simulation: `bf16_trunc = true` truncates state updates to BF16 mantissa
/// (upper 16 bits of IEEE-754 f32) for precision-drift negative control.
fn truncate_bf16(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0xffff_0000)
}

/// Independent FP32 `ggml`-style `wkv7` step.
pub fn ggml_wkv7_step_f32(
    state: &mut Rwkv7GgmlStateF32,
    input: &Rwkv7TokenInput,
    simulate_bf16_drift: bool,
) -> Result<Vec<f32>, String> {
    let n_heads = state.n_heads;
    let d = state.head_dim;
    if input.decay_w.len() != n_heads
        || input.iclr_a.len() != n_heads
        || input.key_b.len() != n_heads
        || input.key_k.len() != n_heads
        || input.val_v.len() != n_heads
        || input.receptance_r.len() != n_heads
    {
        return Err("ggml wkv7 head count mismatch".into());
    }
    let mut out = vec![0.0_f32; n_heads * d];
    for h in 0..n_heads {
        if input.decay_w[h].len() != d
            || input.iclr_a[h].len() != d
            || input.key_b[h].len() != d
            || input.key_k[h].len() != d
            || input.val_v[h].len() != d
            || input.receptance_r[h].len() != d
        {
            return Err("ggml wkv7 head_dim mismatch".into());
        }
        for slice in [
            &input.decay_w[h],
            &input.iclr_a[h],
            &input.key_b[h],
            &input.key_k[h],
            &input.val_v[h],
            &input.receptance_r[h],
        ] {
            if slice.iter().any(|x| !x.is_finite()) {
                return Err("non-finite value in ggml wkv7 step input".into());
            }
        }
        let base = h * d * d;
        for i in 0..d {
            let row = base + i * d;
            let mut sa = 0.0_f32;
            for j in 0..d {
                sa += state.data[row + j] * (input.iclr_a[h][j] as f32);
            }
            let vi = input.val_v[h][i] as f32;
            let mut acc = 0.0_f32;
            for j in 0..d {
                let wj = input.decay_w[h][j] as f32;
                let bj = input.key_b[h][j] as f32;
                let kj = input.key_k[h][j] as f32;
                let rj = input.receptance_r[h][j] as f32;
                let mut next = state.data[row + j] * wj + sa * bj + vi * kj;
                if simulate_bf16_drift {
                    next = truncate_bf16(next);
                }
                if !next.is_finite() {
                    return Err("non-finite value in ggml wkv7 f32 step".into());
                }
                state.data[row + j] = next;
                acc += next * rj;
            }
            if !acc.is_finite() {
                return Err("non-finite output in ggml wkv7 f32 step".into());
            }
            out[h * d + i] = acc;
        }
    }
    Ok(out)
}

pub fn ggml_wkv7_forward_f32(
    init: &Rwkv7GgmlStateF32,
    seq: &[Rwkv7TokenInput],
    simulate_bf16_drift: bool,
) -> Result<(Vec<Vec<f32>>, Rwkv7GgmlStateF32), String> {
    let mut state = init.clone();
    let mut outs = Vec::with_capacity(seq.len());
    for tok in seq {
        outs.push(ggml_wkv7_step_f32(&mut state, tok, simulate_bf16_drift)?);
    }
    Ok((outs, state))
}

fn max_out_diff_f32_vs_f64(a: &[Vec<f32>], b: &[Vec<f64>]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err("sequence length mismatch".into());
    }
    let mut max_err = 0.0_f64;
    for (va, vb) in a.iter().zip(b.iter()) {
        if va.len() != vb.len() {
            return Err("output vector length mismatch".into());
        }
        for (&xa, &xb) in va.iter().zip(vb.iter()) {
            let xaf = f64::from(xa);
            if !xaf.is_finite() || !xb.is_finite() {
                return Err("non-finite value in output parity comparison".into());
            }
            let diff = (xaf - xb).abs();
            if diff > max_err {
                max_err = diff;
            }
        }
    }
    Ok(max_err)
}

/// Capability and backend-availability validator for `ggml` RWKV-7 requests.
pub fn validate_ggml_rwkv7_request(
    mode: &str,
    requested_backend: &str,
    cuda_available: bool,
) -> OutcomeStatus {
    if mode != "forward_inference" {
        return OutcomeStatus::Unsupported;
    }
    if requested_backend == "ggml-cuda" && !cuda_available {
        return OutcomeStatus::UnavailableHardwareRuntime;
    }
    if requested_backend != "ggml-cpu" && requested_backend != "ggml-cuda" {
        return OutcomeStatus::Unsupported;
    }
    OutcomeStatus::Accepted
}

#[derive(Debug, Clone, PartialEq)]
pub struct DriftCheckpoint {
    pub step: usize,
    pub max_output_abs_err: f64,
    pub max_state_abs_err: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RwkvGgmlControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub max_abs_err: Option<f64>,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct RwkvGgmlReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub ggml_commit: String,
    pub rsi_commit: String,
    pub state_layout: String,
    pub n_heads: usize,
    pub head_dim: usize,
    pub state_bytes_f32: usize,
    pub state_bytes_f64: usize,
    pub parity_tolerance: f64,
    pub exported_mid_state_digest: DigestRef,
    pub drift_checkpoints: Vec<DriftCheckpoint>,
    pub controls: Vec<RwkvGgmlControl>,
    pub asserts_agent_persistence_correctness: bool,
}

pub fn run_default_harness() -> Result<RwkvGgmlReport, String> {
    let n_heads = 2usize;
    let head_dim = 4usize;
    let seq_len = 64usize;
    let tol = 1e-5_f64;

    let seq = synthetic_rwkv7_sequence(n_heads, head_dim, seq_len);
    let init_f64 = Rwkv7StateF64::zeros(n_heads, head_dim)?;
    let init_f32 = Rwkv7GgmlStateF32::zeros(n_heads, head_dim)?;

    let (ref_outs, ref_final) = rwkv7_forward_f64(&init_f64, &seq)?;
    let (ggml_outs, ggml_final) = ggml_wkv7_forward_f32(&init_f32, &seq, false)?;

    let full_out_err = max_out_diff_f32_vs_f64(&ggml_outs, &ref_outs)?;
    let full_state_err = ggml_final.max_abs_diff_vs_f64(&ref_final)?;

    // Long-horizon drift checkpoints at steps 1, 8, 16, 32, 64.
    let mut drift_checkpoints = Vec::new();
    for &step in &[1usize, 8, 16, 32, 64] {
        let (r_o, r_s) = rwkv7_forward_f64(&init_f64, &seq[..step])?;
        let (g_o, g_s) = ggml_wkv7_forward_f32(&init_f32, &seq[..step], false)?;
        drift_checkpoints.push(DriftCheckpoint {
            step,
            max_output_abs_err: max_out_diff_f32_vs_f64(&g_o, &r_o)?,
            max_state_abs_err: g_s.max_abs_diff_vs_f64(&r_s)?,
        });
    }

    // Export/import state continuation across split point (24 + 40 tokens).
    let split = 24usize;
    let (chunk1_o, mid_state) = ggml_wkv7_forward_f32(&init_f32, &seq[..split], false)?;
    let exported_bytes = mid_state.export_le_bytes();
    let exported_mid_state_digest = DigestRef::sha256(&exported_bytes);
    let imported_state = Rwkv7GgmlStateF32::import_le_bytes(n_heads, head_dim, &exported_bytes)?;
    let (chunk2_o, cont_final) = ggml_wkv7_forward_f32(&imported_state, &seq[split..], false)?;
    let mut cont_outs = chunk1_o;
    cont_outs.extend(chunk2_o);
    let cont_out_err = max_out_diff_f32_vs_f64(&cont_outs, &ref_outs)?;
    let cont_state_err = cont_final.max_abs_diff_vs_f64(&ref_final)?;

    // Negative control 1: altered exported state (corrupt one element before import) — check
    // both continuation outputs AND final state because contractive decay attenuates final state!
    let mut tampered = imported_state.clone();
    tampered.data[0] += 0.5_f32;
    let (tampered_cont_outs, tampered_final) =
        ggml_wkv7_forward_f32(&tampered, &seq[split..], false)?;
    let tampered_out_err = max_out_diff_f32_vs_f64(&tampered_cont_outs, &ref_outs[split..])?;
    let tampered_err = tampered_out_err.max(tampered_final.max_abs_diff_vs_f64(&ref_final)?);

    // Negative control 2: sequence boundary error (re-execute token `split - 1` across boundary).
    let (boundary_cont_outs, boundary_final) =
        ggml_wkv7_forward_f32(&imported_state, &seq[(split - 1)..], false)?;
    let boundary_out_err = max_out_diff_f32_vs_f64(
        &boundary_cont_outs[..ref_outs[split..].len()],
        &ref_outs[split..],
    )?;
    let boundary_err = boundary_out_err.max(boundary_final.max_abs_diff_vs_f64(&ref_final)?);

    // Negative control 3: coarse BF16 precision drift over 64 steps.
    let (bf16_outs, bf16_final) = ggml_wkv7_forward_f32(&init_f32, &seq, true)?;
    let bf16_err = max_out_diff_f32_vs_f64(&bf16_outs, &ref_outs)?
        .max(bf16_final.max_abs_diff_vs_f64(&ref_final)?);

    let controls = vec![
        RwkvGgmlControl {
            case_id: "01-positive-token-and-full-parity-f32-vs-f64-ref".into(),
            outcome: if full_out_err <= tol && full_state_err <= tol {
                OutcomeStatus::Accepted
            } else {
                OutcomeStatus::Invalid
            },
            max_abs_err: Some(full_out_err.max(full_state_err)),
            detail: "Independent FP32 ggml-style wkv7 matches FP64 native reference across 64 steps.".into(),
        },
        RwkvGgmlControl {
            case_id: "02-positive-export-import-state-continuation-24-plus-40".into(),
            outcome: if cont_out_err <= tol && cont_state_err <= tol {
                OutcomeStatus::Accepted
            } else {
                OutcomeStatus::Invalid
            },
            max_abs_err: Some(cont_out_err.max(cont_state_err)),
            detail: "Binary LE export/import at t=24 reproduces uninterrupted 64-step outputs and final state.".into(),
        },
        RwkvGgmlControl {
            case_id: "03-negative-altered-exported-state-detected".into(),
            outcome: if tampered_err > tol {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::InternalFailure
            },
            max_abs_err: Some(tampered_err),
            detail: "Perturbing exported state[0] before continuation breaks final-state parity.".into(),
        },
        RwkvGgmlControl {
            case_id: "04-negative-sequence-boundary-duplicate-token-detected".into(),
            outcome: if boundary_err > tol {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::InternalFailure
            },
            max_abs_err: Some(boundary_err),
            detail: "Off-by-one chunk boundary duplication at t=23..64 fails state parity.".into(),
        },
        RwkvGgmlControl {
            case_id: "05-negative-bf16-precision-drift-exceeds-f32-tolerance".into(),
            outcome: if bf16_err > tol {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::InternalFailure
            },
            max_abs_err: Some(bf16_err),
            detail: "BF16-truncated recurrence drifts beyond 1e-5 F32 tolerance over 64 steps.".into(),
        },
        RwkvGgmlControl {
            case_id: "06-negative-backward-training-claim-unsupported".into(),
            outcome: validate_ggml_rwkv7_request("backward_training", "ggml-cpu", false),
            max_abs_err: None,
            detail: "ggml wkv7 inference qualification explicitly rejects unsupported backward/training claims.".into(),
        },
        RwkvGgmlControl {
            case_id: "07-control-ggml-cuda-wkv-unavailable-no-cpu-fallback".into(),
            outcome: validate_ggml_rwkv7_request("forward_inference", "ggml-cuda", false),
            max_abs_err: None,
            detail: "Physical ggml-cuda wkv.cu is unavailable locally and never falls back silently to CPU.".into(),
        },
    ];

    Ok(RwkvGgmlReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "compiled_ggml_cpu_and_cuda_wkv7_kernels",
            "Pure-Rust FP64 reference vs independent FP32 ggml-style wkv7 parity and export/import round-trips ran locally; compiled C/CUDA ggml wkv.cu execution remains unavailable_hardware_runtime.",
        ),
        ggml_commit: GGML_RWKV7_COMMIT.into(),
        rsi_commit: RSI_DONOR_COMMIT.into(),
        state_layout: "row_major_[n_heads,head_dim,head_dim]".into(),
        n_heads,
        head_dim,
        state_bytes_f32: init_f32.byte_len(),
        state_bytes_f64: init_f64.byte_len(),
        parity_tolerance: tol,
        exported_mid_state_digest,
        drift_checkpoints,
        controls,
        asserts_agent_persistence_correctness: false,
    })
}

pub fn inputs_json(report: &RwkvGgmlReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.issue-16-rwkv-ggml.inputs.v1\",\n",
            "  \"issue\": 16,\n",
            "  \"execution_class\": {},\n",
            "  \"ggml_commit\": {},\n",
            "  \"rsi_commit\": {},\n",
            "  \"pinned_ggml_sources\": [\"src/ggml.c#L5894-L5939\",\"src/ggml-cpu/ops.cpp\",\"src/ggml-cuda/wkv.cu\"],\n",
            "  \"state_layout\": {},\n",
            "  \"n_heads\": {},\n",
            "  \"head_dim\": {},\n",
            "  \"state_bytes_f32\": {},\n",
            "  \"state_bytes_f64\": {},\n",
            "  \"parity_tolerance\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(&report.ggml_commit),
        quote(&report.rsi_commit),
        quote(&report.state_layout),
        report.n_heads,
        report.head_dim,
        report.state_bytes_f32,
        report.state_bytes_f64,
        finite_f64(report.parity_tolerance),
    )
}

pub fn results_json(report: &RwkvGgmlReport) -> String {
    let mut out = String::new();
    writeln!(out, "{{").unwrap();
    writeln!(
        out,
        "  \"schema\": \"inference-lab.issue-16-rwkv-ggml.results.v1\","
    )
    .unwrap();
    writeln!(out, "  \"issue\": 16,").unwrap();
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
        "  \"exported_mid_state_digest\": {},",
        report.exported_mid_state_digest.to_json()
    )
    .unwrap();
    writeln!(
        out,
        "  \"asserts_agent_persistence_correctness\": {},",
        report.asserts_agent_persistence_correctness
    )
    .unwrap();

    let drifts: Vec<String> = report
        .drift_checkpoints
        .iter()
        .map(|d| {
            format!(
                "{{\"step\":{},\"max_output_abs_err\":{},\"max_state_abs_err\":{}}}",
                d.step,
                finite_f64(d.max_output_abs_err),
                finite_f64(d.max_state_abs_err),
            )
        })
        .collect();
    writeln!(out, "  \"drift_checkpoints\": [{}],", drifts.join(",")).unwrap();

    let ctrls: Vec<String> = report
        .controls
        .iter()
        .map(|c| {
            format!(
                "{{\"case_id\":{},\"outcome\":{},\"max_abs_err\":{},\"detail\":{}}}",
                quote(&c.case_id),
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

pub fn summary_markdown(report: &RwkvGgmlReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #16: `ggml` RWKV-7 Recurrence & State-Continuation Parity"
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
        "- State layout: `{}` (`f32={} B`, `f64={} B`)",
        report.state_layout, report.state_bytes_f32, report.state_bytes_f64
    )
    .unwrap();
    writeln!(
        out,
        "- Exported t=24 state SHA-256: `{}`",
        report.exported_mid_state_digest.hex
    )
    .unwrap();
    writeln!(
        out,
        "- Asserts RSI agent-level persistence correctness: `{}`",
        report.asserts_agent_persistence_correctness
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Long-Horizon Drift Checkpoints").unwrap();
    writeln!(out, "| Step | Max Output Abs Err | Max State Abs Err |").unwrap();
    writeln!(out, "| ---: | ---: | ---: |").unwrap();
    for d in &report.drift_checkpoints {
        writeln!(
            out,
            "| {} | `{}` | `{}` |",
            d.step,
            finite_f64(d.max_output_abs_err),
            finite_f64(d.max_state_abs_err)
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "## Parity & Negative Controls").unwrap();
    for c in &report.controls {
        writeln!(
            out,
            "- `{}`: `{}` (err=`{}`) — {}",
            c.case_id,
            c.outcome.as_str(),
            opt_f64(c.max_abs_err),
            c.detail
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "> Kernel numerical state parity never asserts durable RSI agent epochs, snapshots, or policy commits."
    )
    .unwrap();
    out
}
