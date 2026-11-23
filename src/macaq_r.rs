//! Issue #24: Evaluate MACaQ-R native MoE GGUF refinement with matched artifacts and held-out
//! task checks.
//!
//! Pinned upstream: `huawei-csl/MACaQ@3810efca0b9e819c8eb51addd93c3c87225b7476`.
//! Isolated from `#18` (Qwen3.8-27B dense model quantization/serving comparison): `#24` targets
//! matched original vs refined native K-quant (`Q4_K`/`Q5_K`/`Q6_K`) MoE projections executed
//! through standard `llama.cpp` without creation-time HF weights, teacher, or calibration data.
//!
//! Provides:
//! - Synthetic MoE GGUF container with `Q4_K`, `Q5_K`, `Q6_K` MoE expert projections (explicit
//!   block scales, integer codes, activation basis) plus out-of-scope `F16`/`Q8_0` tensors and
//!   explicitly unsupported `IQ4_XS` / split-GGUF cases.
//! - Real bounded coordinate refinement pass optimizing supported MoE weight codes against a
//!   frozen FP64 teacher on a calibration split while strictly preserving block scales, tensor
//!   formats, activation basis, and byte-for-byte unchanged out-of-scope tensors.
//! - Independent artifact inspector, creation-vs-serving preflight & digest-bound resume receipt
//!   validator, calibration/validation/held-out disjointness verifier, offline paired-trial
//!   rescorer (`zero accepted -> cost_per_accepted_task = None`), and 8 executable negative controls.

use std::collections::BTreeSet;
use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, softmax_stable};
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const UPSTREAM_REPO: &str = "huawei-csl/MACaQ";
pub const UPSTREAM_COMMIT: &str = "3810efca0b9e819c8eb51addd93c3c87225b7476";
pub const UPSTREAM_UNVERIFIED_CLAIM: &str =
    "Qwen3.6-35B-A3B KL 0.015731 -> 0.013367, 22.36 GB GGUF, 122B full-MACaQ W4A4 (non-gating)";
pub const TARGET_RECIPE: &str =
    "Qwen3.6-35B-A3B / Unsloth UD-Q4_K_XL (separate from #18 Qwen3.8-27B)";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTensorFormat {
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
    F16,
    Iq4Xs,
}

impl GgufTensorFormat {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8_0 => "Q8_0",
            Self::F16 => "F16",
            Self::Iq4Xs => "IQ4_XS",
        }
    }

    pub const fn code_bounds(self) -> Option<(i16, i16)> {
        match self {
            Self::Q4K => Some((0, 15)),
            Self::Q5K => Some((0, 31)),
            Self::Q6K => Some((-32, 31)),
            Self::Q8_0 | Self::F16 | Self::Iq4Xs => None,
        }
    }
}

/// One tensor inside our synthetic native MoE GGUF container.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufTensor {
    pub name: String,
    pub is_moe_projection: bool,
    pub format: GgufTensorFormat,
    pub rows: usize,
    pub cols: usize,
    pub block_scale_d: f64,
    pub block_min_dmin: f64,
    pub activation_basis_id: String,
    /// Discrete weight codes for K-quant MoE projections.
    pub codes: Vec<i16>,
    /// Raw payload bytes for out-of-scope tensors (must stay byte-for-byte identical).
    pub raw_bytes: Vec<u8>,
}

impl GgufTensor {
    pub fn is_macaq_r_eligible(&self) -> bool {
        self.is_moe_projection && self.format.code_bounds().is_some()
    }

    pub fn dequantize_matrix(&self) -> Vec<f64> {
        self.codes
            .iter()
            .map(|&c| self.block_scale_d * (c as f64) - self.block_min_dmin)
            .collect()
    }

    pub fn payload_bytes(&self) -> usize {
        if self.raw_bytes.is_empty() {
            // Fixed quantized block representation size (unchanged by code-only refinement).
            let bits_per_code = match self.format {
                GgufTensorFormat::Q4K => 4,
                GgufTensorFormat::Q5K => 5,
                GgufTensorFormat::Q6K => 6,
                _ => 16,
            };
            16 + (self.codes.len() * bits_per_code).div_ceil(8)
        } else {
            self.raw_bytes.len()
        }
    }

    pub fn digest(&self) -> DigestRef {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(self.name.as_bytes());
        bytes.extend_from_slice(self.format.as_str().as_bytes());
        bytes.extend_from_slice(&self.block_scale_d.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.block_min_dmin.to_bits().to_le_bytes());
        bytes.extend_from_slice(self.activation_basis_id.as_bytes());
        for &c in &self.codes {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
        bytes.extend_from_slice(&self.raw_bytes);
        DigestRef::sha256(&bytes)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticGgufArtifact {
    pub artifact_id: String,
    pub is_split_gguf: bool,
    pub metadata_bytes: usize,
    pub tensors: Vec<GgufTensor>,
    pub requires_creation_time_deps_for_serve: bool,
}

impl SyntheticGgufArtifact {
    pub fn tensor_payload_bytes(&self) -> usize {
        self.tensors.iter().map(GgufTensor::payload_bytes).sum()
    }

    pub fn total_gguf_bytes(&self) -> usize {
        self.metadata_bytes + self.tensor_payload_bytes()
    }

    pub fn digest(&self) -> DigestRef {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(self.metadata_bytes as u64).to_le_bytes());
        for t in &self.tensors {
            bytes.extend_from_slice(t.digest().hex.as_bytes());
        }
        DigestRef::sha256(&bytes)
    }
}

/// Evaluate synthetic MoE projection logits `x @ W^T` across refined tensors for KL/MSE comparison.
fn eval_moe_logits(tensors: &[GgufTensor], x: &[f64]) -> Vec<f64> {
    let mut logits = vec![0.0_f64; 4];
    for t in tensors.iter().filter(|t| t.is_macaq_r_eligible()) {
        let w = t.dequantize_matrix();
        for r in 0..t.rows.min(logits.len()) {
            let mut acc = 0.0_f64;
            for c in 0..t.cols.min(x.len()) {
                acc += w[r * t.cols + c] * x[c];
            }
            logits[r] += acc;
        }
    }
    logits
}

fn eval_teacher_logits(
    teacher_weights: &[Vec<f64>],
    rows: usize,
    cols: usize,
    x: &[f64],
) -> Vec<f64> {
    let mut logits = vec![0.0_f64; 4];
    for w in teacher_weights {
        for r in 0..rows.min(logits.len()) {
            let mut acc = 0.0_f64;
            for c in 0..cols.min(x.len()) {
                acc += w[r * cols + c] * x[c];
            }
            logits[r] += acc;
        }
    }
    logits
}

fn kl_divergence(teacher_logits: &[f64], student_logits: &[f64]) -> Result<f64, String> {
    if teacher_logits.len() != student_logits.len() || teacher_logits.is_empty() {
        return Err("mismatched or empty logits in kl_divergence".into());
    }
    let p = softmax_stable(teacher_logits)?;
    let q = softmax_stable(student_logits)?;
    let mut kl = 0.0_f64;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > 0.0 {
            if qi <= 0.0 || !qi.is_finite() {
                return Err("zero or non-finite student probability in KL".into());
            }
            kl += pi * (pi / qi).ln();
        }
    }
    if !kl.is_finite() {
        return Err("non-finite KL divergence".into());
    }
    Ok(if kl < 0.0 { 0.0 } else { kl })
}

/// Real bounded MACaQ-R coordinate refinement: modifies ONLY integer weight codes of eligible
/// `Q4_K`/`Q5_K`/`Q6_K` MoE tensors to reduce teacher reconstruction MSE on calibration inputs,
/// strictly preserving block scales, tensor formats, activation basis, and out-of-scope tensors!
pub fn refine_macaq_r(
    original: &SyntheticGgufArtifact,
    teacher_weights: &[Vec<f64>],
    calib_inputs: &[Vec<f64>],
    max_iters: usize,
) -> Result<(SyntheticGgufArtifact, usize), String> {
    if original.is_split_gguf {
        return Err("split GGUF is unsupported in MACaQ-R initial scope (upstream #2)".into());
    }
    if calib_inputs.is_empty() {
        return Err("calib_inputs cannot be empty in refine_macaq_r".into());
    }
    let mut refined = original.clone();
    refined.artifact_id = format!("{}-macaq-r-refined", original.artifact_id);
    // Add modest metadata provenance header bytes (proving total GGUF bytes != tensor payload bytes).
    refined.metadata_bytes += 128;

    let mut refined_tensor_count = 0usize;
    let mut eligible_idx = 0usize;
    for tensor in &mut refined.tensors {
        if !tensor.is_macaq_r_eligible() {
            continue;
        }
        if tensor.codes.len() != tensor.rows * tensor.cols {
            return Err(format!(
                "tensor `{}` codes length != rows*cols",
                tensor.name
            ));
        }
        let tw = teacher_weights
            .get(eligible_idx)
            .ok_or_else(|| "missing teacher weight matrix for eligible MoE tensor".to_string())?;
        if tw.len() != tensor.rows * tensor.cols || tw.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "invalid teacher weight matrix for `{}`",
                tensor.name
            ));
        }
        for x in calib_inputs {
            if x.len() < tensor.cols || x.iter().any(|v| !v.is_finite()) {
                return Err("invalid calibration vector in refine_macaq_r".into());
            }
        }
        let (cmin, cmax) = tensor.format.code_bounds().unwrap();
        eligible_idx += 1;
        refined_tensor_count += 1;

        let loss_fn = |codes: &[i16], d: f64, dmin: f64| -> f64 {
            let mut mse = 0.0_f64;
            for x in calib_inputs {
                for r in 0..tensor.rows {
                    let mut ty = 0.0_f64;
                    let mut sy = 0.0_f64;
                    for (c, &xc) in x.iter().enumerate().take(tensor.cols) {
                        let idx = r * tensor.cols + c;
                        ty += tw[idx] * xc;
                        sy += (d * (codes[idx] as f64) - dmin) * xc;
                    }
                    let diff = ty - sy;
                    mse += diff * diff;
                }
            }
            mse / (calib_inputs.len() as f64)
        };

        let mut cur_loss = loss_fn(&tensor.codes, tensor.block_scale_d, tensor.block_min_dmin);
        for _iter in 0..max_iters {
            let mut improved = false;
            for pos in 0..tensor.codes.len() {
                let orig_code = tensor.codes[pos];
                for delta in [-1i16, 1i16] {
                    let cand = orig_code + delta;
                    if cand < cmin || cand > cmax {
                        continue;
                    }
                    tensor.codes[pos] = cand;
                    let cand_loss =
                        loss_fn(&tensor.codes, tensor.block_scale_d, tensor.block_min_dmin);
                    if cand_loss + 1e-12 < cur_loss {
                        cur_loss = cand_loss;
                        improved = true;
                        break;
                    } else {
                        tensor.codes[pos] = orig_code;
                    }
                }
            }
            if !improved {
                break;
            }
        }
    }
    if refined_tensor_count == 0 {
        return Err("no supported Q4_K/Q5_K/Q6_K MoE projections found".into());
    }
    Ok((refined, refined_tensor_count))
}

/// Independent artifact inspector checking before/after invariants.
pub fn inspect_refinement_invariants(
    original: &SyntheticGgufArtifact,
    refined: &SyntheticGgufArtifact,
) -> Result<(), String> {
    if original.tensors.len() != refined.tensors.len() {
        return Err("tensor inventory count changed".into());
    }
    if refined.requires_creation_time_deps_for_serve {
        return Err("refined artifact cannot require creation-time HF/teacher/calibration deps at serve time".into());
    }
    let mut any_code_changed = false;
    for (a, b) in original.tensors.iter().zip(refined.tensors.iter()) {
        if a.name != b.name || a.rows != b.rows || a.cols != b.cols || a.format != b.format {
            return Err(format!("tensor metadata/format changed on `{}`", a.name));
        }
        if a.block_scale_d.to_bits() != b.block_scale_d.to_bits()
            || a.block_min_dmin.to_bits() != b.block_min_dmin.to_bits()
            || a.activation_basis_id != b.activation_basis_id
        {
            return Err(format!(
                "block scale or activation basis mutated on `{}` (MACaQ-R must preserve native scales/basis)",
                a.name
            ));
        }
        if !a.is_macaq_r_eligible() {
            if a.digest() != b.digest() {
                return Err(format!("out-of-scope tensor `{}` was modified", a.name));
            }
        } else if a.codes != b.codes {
            any_code_changed = true;
        }
    }
    if !any_code_changed {
        return Err("refinement changed zero supported MoE weight codes".into());
    }
    Ok(())
}

/// Resume receipt binding creation work to exact source, teacher, calibration, and settings digests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreationResumeReceipt {
    pub source_gguf_digest: String,
    pub teacher_gguf_digest: String,
    pub calibration_digest: String,
    pub settings_digest: String,
    pub completed_iters: usize,
    pub interrupted_wall_ms: u64,
    pub resumed_wall_ms: u64,
    pub peak_host_ram_gib: u64,
    pub peak_device_vram_gib: u64,
    pub peak_working_disk_gib: u64,
}

impl CreationResumeReceipt {
    pub fn total_creation_wall_ms(&self) -> u64 {
        self.interrupted_wall_ms + self.resumed_wall_ms
    }

    pub fn validate_against(&self, expected: &CreationResumeReceipt) -> Result<(), String> {
        if self.source_gguf_digest != expected.source_gguf_digest
            || self.teacher_gguf_digest != expected.teacher_gguf_digest
            || self.calibration_digest != expected.calibration_digest
            || self.settings_digest != expected.settings_digest
        {
            return Err("stale or mismatched resume state identity rejected".into());
        }
        Ok(())
    }
}

/// Verify calibration, validation, and held-out prompt/task IDs are pairwise disjoint.
pub fn verify_partition_disjointness(
    calib_ids: &[&str],
    val_ids: &[&str],
    held_out_ids: &[&str],
) -> Result<(), String> {
    let c: BTreeSet<&str> = calib_ids.iter().copied().collect();
    let v: BTreeSet<&str> = val_ids.iter().copied().collect();
    let h: BTreeSet<&str> = held_out_ids.iter().copied().collect();
    if !c.is_disjoint(&h) {
        return Err("held-out contamination detected in calibration split".into());
    }
    if !v.is_disjoint(&h) {
        return Err("held-out contamination detected in validation split".into());
    }
    if !c.is_disjoint(&v) {
        return Err("validation overlap detected in calibration split".into());
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct PairedTaskOutcome {
    pub task_id: String,
    pub status: OutcomeStatus,
    pub latency_ms: f64,
    pub mtp_enabled: bool,
    pub runtime_build: String,
}

#[derive(Debug, Clone)]
pub struct ArmRescoreSummary {
    pub arm_name: String,
    pub planned: usize,
    pub accepted: usize,
    pub non_accepted: usize,
    pub total_cost_ms: f64,
    pub cost_per_accepted_task_ms: Option<f64>,
    pub amortized_cost_per_accepted_100_uses_ms: Option<f64>,
}

pub fn rescore_arm(
    arm_name: &str,
    expected_runtime: &str,
    outcomes: &[PairedTaskOutcome],
    creation_cost_ms: f64,
    amortized_uses: usize,
) -> Result<ArmRescoreSummary, String> {
    if outcomes.is_empty()
        || amortized_uses == 0
        || !creation_cost_ms.is_finite()
        || creation_cost_ms < 0.0
    {
        return Err("empty outcomes, zero amortized use count, or invalid creation_cost_ms".into());
    }
    let mut accepted = 0usize;
    let mut total_cost = 0.0_f64;
    for o in outcomes {
        if o.mtp_enabled {
            return Err(format!(
                "MTP must stay off for matched serving arms (violation on {})",
                o.task_id
            ));
        }
        if o.runtime_build != expected_runtime {
            return Err(format!(
                "mismatched runtime build `{}` vs expected `{expected_runtime}`",
                o.runtime_build
            ));
        }
        if !o.latency_ms.is_finite() || o.latency_ms < 0.0 {
            return Err("invalid latency in rescore_arm".into());
        }
        total_cost += o.latency_ms;
        if o.status == OutcomeStatus::Accepted {
            accepted += 1;
        }
    }
    let cost_per_accepted = if accepted == 0 {
        None
    } else {
        Some(total_cost / (accepted as f64))
    };
    let amortized = if accepted == 0 {
        None
    } else {
        Some((total_cost + creation_cost_ms / (amortized_uses as f64)) / (accepted as f64))
    };
    Ok(ArmRescoreSummary {
        arm_name: arm_name.into(),
        planned: outcomes.len(),
        accepted,
        non_accepted: outcomes.len() - accepted,
        total_cost_ms: total_cost,
        cost_per_accepted_task_ms: cost_per_accepted,
        amortized_cost_per_accepted_100_uses_ms: amortized,
    })
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct MacaqRReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub original_digest: DigestRef,
    pub refined_digest: DigestRef,
    pub original_total_gguf_bytes: usize,
    pub refined_total_gguf_bytes: usize,
    pub original_tensor_payload_bytes: usize,
    pub refined_tensor_payload_bytes: usize,
    pub refined_tensor_count: usize,
    pub calib_kl_before: f64,
    pub calib_kl_after: f64,
    pub heldout_kl_before: f64,
    pub heldout_kl_after: f64,
    pub receipt: CreationResumeReceipt,
    pub original_arm: ArmRescoreSummary,
    pub refined_arm: ArmRescoreSummary,
    pub zero_accepted_arm: ArmRescoreSummary,
    pub negative_controls: Vec<NegativeControl>,
    pub decision: String,
}

pub fn run_default_harness() -> Result<MacaqRReport, String> {
    let rows = 4usize;
    let cols = 4usize;
    // Build synthetic teacher weights for 3 eligible MoE projections (Q4_K, Q5_K, Q6_K).
    let teacher_weights: Vec<Vec<f64>> = (0..3)
        .map(|idx| {
            (0..rows * cols)
                .map(|i| det_weight(2400 + idx, i, 1.2))
                .collect()
        })
        .collect();

    let quantize_init =
        |name: &str, fmt: GgufTensorFormat, tw: &[f64], d: f64, dmin: f64| -> GgufTensor {
            let (cmin, cmax) = fmt.code_bounds().unwrap();
            // Deliberately perturb initial nearest-round codes by +/-1 on a few entries to simulate
            // unrefined RTN quantization error that MACaQ-R coordinate search improves!
            let codes: Vec<i16> = tw
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let raw = ((v + dmin) / d).round() as i16;
                    let nudged = if i % 3 == 0 {
                        raw + 1
                    } else if i % 5 == 0 {
                        raw - 1
                    } else {
                        raw
                    };
                    nudged.clamp(cmin, cmax)
                })
                .collect();
            GgufTensor {
                name: name.into(),
                is_moe_projection: true,
                format: fmt,
                rows,
                cols,
                block_scale_d: d,
                block_min_dmin: dmin,
                activation_basis_id: "native_llamacpp_kquant_identity_v1".into(),
                codes,
                raw_bytes: Vec::new(),
            }
        };

    let original = SyntheticGgufArtifact {
        artifact_id: "synthetic-moe-ud-q4-k-xl-orig".into(),
        is_split_gguf: false,
        metadata_bytes: 512,
        requires_creation_time_deps_for_serve: false,
        tensors: vec![
            quantize_init(
                "blk.0.ffn_gate_exps.weight",
                GgufTensorFormat::Q4K,
                &teacher_weights[0],
                0.16,
                1.20,
            ),
            quantize_init(
                "blk.0.ffn_up_exps.weight",
                GgufTensorFormat::Q5K,
                &teacher_weights[1],
                0.08,
                1.24,
            ),
            quantize_init(
                "blk.0.ffn_down_exps.weight",
                GgufTensorFormat::Q6K,
                &teacher_weights[2],
                0.04,
                0.0,
            ),
            GgufTensor {
                name: "token_embd.weight".into(),
                is_moe_projection: false,
                format: GgufTensorFormat::F16,
                rows: 8,
                cols: 4,
                block_scale_d: 1.0,
                block_min_dmin: 0.0,
                activation_basis_id: "f16_unchanged".into(),
                codes: Vec::new(),
                raw_bytes: vec![0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0],
            },
            GgufTensor {
                name: "blk.0.ffn_gate_inp.weight".into(),
                is_moe_projection: false,
                format: GgufTensorFormat::Q8_0,
                rows: 4,
                cols: 4,
                block_scale_d: 1.0,
                block_min_dmin: 0.0,
                activation_basis_id: "q8_0_router_unchanged".into(),
                codes: Vec::new(),
                raw_bytes: vec![0xaa, 0xbb, 0xcc, 0xdd, 0x11, 0x22, 0x33, 0x44],
            },
        ],
    };

    let calib_inputs: Vec<Vec<f64>> = (0..16)
        .map(|s| (0..cols).map(|c| det_weight(2450 + s, c, 0.9)).collect())
        .collect();
    let heldout_inputs: Vec<Vec<f64>> = (0..12)
        .map(|s| (0..cols).map(|c| det_weight(2480 + s, c, 0.9)).collect())
        .collect();

    verify_partition_disjointness(
        &["cal-01", "cal-02", "cal-03"],
        &["val-01", "val-02"],
        &["held-01", "held-02", "held-03", "held-04"],
    )?;

    let (refined, refined_tensor_count) =
        refine_macaq_r(&original, &teacher_weights, &calib_inputs, 8)?;
    inspect_refinement_invariants(&original, &refined)?;

    let mean_kl = |artifact: &SyntheticGgufArtifact, xs: &[Vec<f64>]| -> Result<f64, String> {
        let mut s = 0.0_f64;
        for x in xs {
            let tl = eval_teacher_logits(&teacher_weights, rows, cols, x);
            let sl = eval_moe_logits(&artifact.tensors, x);
            s += kl_divergence(&tl, &sl)?;
        }
        Ok(s / (xs.len() as f64))
    };
    let calib_kl_before = mean_kl(&original, &calib_inputs)?;
    let calib_kl_after = mean_kl(&refined, &calib_inputs)?;
    let heldout_kl_before = mean_kl(&original, &heldout_inputs)?;
    let heldout_kl_after = mean_kl(&refined, &heldout_inputs)?;

    let receipt = CreationResumeReceipt {
        source_gguf_digest: original.digest().hex,
        teacher_gguf_digest: DigestRef::sha256(b"synthetic-fp64-teacher-v1").hex,
        calibration_digest: DigestRef::sha256(b"synthetic-calib-split-v1").hex,
        settings_digest: DigestRef::sha256(b"macaq-r-iters-8-mtp-off-v1").hex,
        completed_iters: 8,
        interrupted_wall_ms: 1_200,
        resumed_wall_ms: 3_800,
        peak_host_ram_gib: 64,
        peak_device_vram_gib: 24,
        peak_working_disk_gib: 110,
    };

    // Synthetic offline paired-trial rescoring (retaining every non-accepted attempt!).
    let runtime = "llama.cpp-pinned-std-v1";
    let orig_outcomes = vec![
        PairedTaskOutcome {
            task_id: "t1".into(),
            status: OutcomeStatus::Accepted,
            latency_ms: 420.0,
            mtp_enabled: false,
            runtime_build: runtime.into(),
        },
        PairedTaskOutcome {
            task_id: "t2".into(),
            status: OutcomeStatus::Invalid,
            latency_ms: 410.0,
            mtp_enabled: false,
            runtime_build: runtime.into(),
        },
        PairedTaskOutcome {
            task_id: "t3".into(),
            status: OutcomeStatus::Timeout,
            latency_ms: 1000.0,
            mtp_enabled: false,
            runtime_build: runtime.into(),
        },
        PairedTaskOutcome {
            task_id: "t4".into(),
            status: OutcomeStatus::Accepted,
            latency_ms: 390.0,
            mtp_enabled: false,
            runtime_build: runtime.into(),
        },
    ];
    let ref_outcomes = vec![
        PairedTaskOutcome {
            task_id: "t1".into(),
            status: OutcomeStatus::Accepted,
            latency_ms: 425.0,
            mtp_enabled: false,
            runtime_build: runtime.into(),
        },
        PairedTaskOutcome {
            task_id: "t2".into(),
            status: OutcomeStatus::Accepted,
            latency_ms: 405.0,
            mtp_enabled: false,
            runtime_build: runtime.into(),
        },
        PairedTaskOutcome {
            task_id: "t3".into(),
            status: OutcomeStatus::Timeout,
            latency_ms: 1000.0,
            mtp_enabled: false,
            runtime_build: runtime.into(),
        },
        PairedTaskOutcome {
            task_id: "t4".into(),
            status: OutcomeStatus::Accepted,
            latency_ms: 395.0,
            mtp_enabled: false,
            runtime_build: runtime.into(),
        },
    ];
    let zero_outcomes = vec![
        PairedTaskOutcome {
            task_id: "z1".into(),
            status: OutcomeStatus::Invalid,
            latency_ms: 500.0,
            mtp_enabled: false,
            runtime_build: runtime.into(),
        },
        PairedTaskOutcome {
            task_id: "z2".into(),
            status: OutcomeStatus::Oom,
            latency_ms: 120.0,
            mtp_enabled: false,
            runtime_build: runtime.into(),
        },
    ];

    let original_arm = rescore_arm("original_gguf", runtime, &orig_outcomes, 0.0, 100)?;
    let refined_arm = rescore_arm(
        "macaq_r_refined_gguf",
        runtime,
        &ref_outcomes,
        receipt.total_creation_wall_ms() as f64,
        100,
    )?;
    let zero_accepted_arm =
        rescore_arm("zero_accepted_control", runtime, &zero_outcomes, 0.0, 100)?;

    // Eight executable negative controls!
    let mut negative_controls = Vec::new();

    // 1. Mutated block scale during refinement
    let mut bad_scale = refined.clone();
    bad_scale.tensors[0].block_scale_d += 0.01;
    let r1 = inspect_refinement_invariants(&original, &bad_scale);
    negative_controls.push(NegativeControl {
        case_id: "mutated_block_scale_rejected".into(),
        outcome: if r1.is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: r1
            .err()
            .unwrap_or_else(|| "missed mutated block scale".into()),
    });

    // 2. Modified out-of-scope tensor (token_embd.weight)
    let mut bad_oos = refined.clone();
    bad_oos.tensors[3].raw_bytes[0] ^= 0xff;
    let r2 = inspect_refinement_invariants(&original, &bad_oos);
    negative_controls.push(NegativeControl {
        case_id: "modified_out_of_scope_tensor_rejected".into(),
        outcome: if r2.is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: r2
            .err()
            .unwrap_or_else(|| "missed out-of-scope edit".into()),
    });

    // 3. Unsupported IQ4_XS / split-GGUF rejected
    let mut split_art = original.clone();
    split_art.is_split_gguf = true;
    let r3 = refine_macaq_r(&split_art, &teacher_weights, &calib_inputs, 2);
    let iq4_only = SyntheticGgufArtifact {
        artifact_id: "iq4-only".into(),
        is_split_gguf: false,
        metadata_bytes: 64,
        requires_creation_time_deps_for_serve: false,
        tensors: vec![GgufTensor {
            name: "blk.0.ffn_gate_exps.weight".into(),
            is_moe_projection: true,
            format: GgufTensorFormat::Iq4Xs,
            rows: 2,
            cols: 2,
            block_scale_d: 1.0,
            block_min_dmin: 0.0,
            activation_basis_id: "iq4_xs".into(),
            codes: Vec::new(),
            raw_bytes: vec![1, 2, 3, 4],
        }],
    };
    let r3b = refine_macaq_r(&iq4_only, &teacher_weights, &calib_inputs, 2);
    negative_controls.push(NegativeControl {
        case_id: "unsupported_split_gguf_and_iq4xs_rejected".into(),
        outcome: if r3.is_err() && r3b.is_err() {
            OutcomeStatus::Unsupported
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("split_err={}, iq4xs_err={}", r3.is_err(), r3b.is_err()),
    });

    // 4. Calibration-to-held-out contamination rejected
    let r4 =
        verify_partition_disjointness(&["cal-01", "held-02"], &["val-01"], &["held-01", "held-02"]);
    negative_controls.push(NegativeControl {
        case_id: "calibration_heldout_contamination_rejected".into(),
        outcome: if r4.is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: r4.err().unwrap_or_else(|| "missed contamination".into()),
    });

    // 5. Stale resume receipt rejected
    let mut stale = receipt.clone();
    stale.teacher_gguf_digest = "deadbeef".into();
    let r5 = stale.validate_against(&receipt);
    negative_controls.push(NegativeControl {
        case_id: "stale_resume_receipt_rejected".into(),
        outcome: if r5.is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: r5.err().unwrap_or_else(|| "missed stale resume".into()),
    });

    // 6. Mismatched runtime setting (MTP turned on in one serving arm) rejected
    let mut bad_mtp = ref_outcomes.clone();
    bad_mtp[0].mtp_enabled = true;
    let r6 = rescore_arm("mismatched_mtp_arm", runtime, &bad_mtp, 0.0, 100);
    negative_controls.push(NegativeControl {
        case_id: "mismatched_serving_mtp_setting_rejected".into(),
        outcome: if r6.is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: r6.err().unwrap_or_else(|| "missed MTP mismatch".into()),
    });

    // 7. Creation-time dependency required at serve time rejected
    let mut bad_dep = refined.clone();
    bad_dep.requires_creation_time_deps_for_serve = true;
    let r7 = inspect_refinement_invariants(&original, &bad_dep);
    negative_controls.push(NegativeControl {
        case_id: "creation_time_dependency_at_serve_rejected".into(),
        outcome: if r7.is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: r7
            .err()
            .unwrap_or_else(|| "missed creation dep leak".into()),
    });

    // 8. Zero accepted tasks keeps cost_per_accepted_task undefined (None, never 0 or Inf)
    negative_controls.push(NegativeControl {
        case_id: "zero_accepted_tasks_cost_undefined".into(),
        outcome: if zero_accepted_arm.cost_per_accepted_task_ms.is_none()
            && zero_accepted_arm
                .amortized_cost_per_accepted_100_uses_ms
                .is_none()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "cost_per_accepted_task=null when accepted=0".into(),
    });

    Ok(MacaqRReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "physical_node_qwen36_35b_macaq_r_llamacpp",
            "Synthetic MoE K-quant refinement and artifact/rescoring contracts verified on CPU; 22.36 GB Qwen3.6-35B-A3B creation and standard llama.cpp serving remain unexecuted locally",
        ),
        original_digest: original.digest(),
        refined_digest: refined.digest(),
        original_total_gguf_bytes: original.total_gguf_bytes(),
        refined_total_gguf_bytes: refined.total_gguf_bytes(),
        original_tensor_payload_bytes: original.tensor_payload_bytes(),
        refined_tensor_payload_bytes: refined.tensor_payload_bytes(),
        refined_tensor_count,
        calib_kl_before,
        calib_kl_after,
        heldout_kl_before,
        heldout_kl_after,
        receipt,
        original_arm,
        refined_arm,
        zero_accepted_arm,
        negative_controls,
        decision: "INCONCLUSIVE_PHYSICAL_NODE_CREATION_AND_SERVE_REQUIRED".into(),
    })
}

pub fn inputs_json(report: &MacaqRReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.macaq-r.inputs.v1\",\n",
            "  \"issue\": 24,\n",
            "  \"upstream\": {{\"repo\":{},\"commit\":{},\"unverified_related_work_claim\":{}}},\n",
            "  \"target_recipe\": {},\n",
            "  \"isolated_from_issue_18_qwen38_27b_dense\": true,\n",
            "  \"execution_class\": {},\n",
            "  \"original_synthetic_digest\": {},\n",
            "  \"refined_synthetic_digest\": {}\n",
            "}}\n"
        ),
        quote(UPSTREAM_REPO),
        quote(UPSTREAM_COMMIT),
        quote(UPSTREAM_UNVERIFIED_CLAIM),
        quote(TARGET_RECIPE),
        quote(report.execution_class.as_str()),
        report.original_digest.to_json(),
        report.refined_digest.to_json(),
    )
}

fn arm_json(a: &ArmRescoreSummary) -> String {
    format!(
        concat!(
            "{{\"synthetic_contract_fixture\":true,\"arm_name\":{},\"planned\":{},\"accepted\":{},\"non_accepted\":{},",
            "\"total_cost_ms\":{},\"cost_per_accepted_task_ms\":{},\"amortized_cost_per_accepted_100_uses_ms\":{}}}"
        ),
        quote(&a.arm_name),
        a.planned,
        a.accepted,
        a.non_accepted,
        finite_f64(a.total_cost_ms),
        opt_f64(a.cost_per_accepted_task_ms),
        opt_f64(a.amortized_cost_per_accepted_100_uses_ms),
    )
}

pub fn results_json(report: &MacaqRReport) -> String {
    let neg: Vec<String> = report
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
            "  \"schema\": \"inference-lab.macaq-r.results.v1\",\n",
            "  \"issue\": 24,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"decision\": {},\n",
            "  \"unchanged_format_implies_identical_process_memory\": false,\n",
            "  \"refined_tensor_count\": {},\n",
            "  \"original_total_gguf_bytes\": {},\n",
            "  \"refined_total_gguf_bytes\": {},\n",
            "  \"original_tensor_payload_bytes\": {},\n",
            "  \"refined_tensor_payload_bytes\": {},\n",
            "  \"synthetic_calib_kl_before\": {},\n",
            "  \"synthetic_calib_kl_after\": {},\n",
            "  \"synthetic_heldout_kl_before\": {},\n",
            "  \"synthetic_heldout_kl_after\": {},\n",
            "  \"creation_receipt\": {{\"synthetic_contract_fixture\":true,\"total_creation_wall_ms\":{},\"interrupted_wall_ms\":{},\"resumed_wall_ms\":{},\"peak_host_ram_gib\":{},\"peak_device_vram_gib\":{},\"peak_working_disk_gib\":{}}},\n",
            "  \"paired_rescore\": [{},{},{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        quote(&report.decision),
        report.refined_tensor_count,
        report.original_total_gguf_bytes,
        report.refined_total_gguf_bytes,
        report.original_tensor_payload_bytes,
        report.refined_tensor_payload_bytes,
        finite_f64(report.calib_kl_before),
        finite_f64(report.calib_kl_after),
        finite_f64(report.heldout_kl_before),
        finite_f64(report.heldout_kl_after),
        report.receipt.total_creation_wall_ms(),
        report.receipt.interrupted_wall_ms,
        report.receipt.resumed_wall_ms,
        report.receipt.peak_host_ram_gib,
        report.receipt.peak_device_vram_gib,
        report.receipt.peak_working_disk_gib,
        arm_json(&report.original_arm),
        arm_json(&report.refined_arm),
        arm_json(&report.zero_accepted_arm),
        neg.join(","),
    )
}

pub fn summary_markdown(report: &MacaqRReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #24: MACaQ-R Native MoE GGUF Refinement Contract"
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
    writeln!(out, "- Decision: `{}`", report.decision).unwrap();
    writeln!(
        out,
        "- Synthetic K-quant MoE refinement (`Q4_K`/`Q5_K`/`Q6_K`, {} tensors): calib KL `{:.6}` -> `{:.6}`; held-out KL `{:.6}` -> `{:.6}`",
        report.refined_tensor_count,
        report.calib_kl_before,
        report.calib_kl_after,
        report.heldout_kl_before,
        report.heldout_kl_after
    )
    .unwrap();
    writeln!(
        out,
        "- Byte separation: tensor payload `{}` vs `{}` bytes (unchanged); total GGUF `{}` vs `{}` bytes (metadata receipt added)",
        report.original_tensor_payload_bytes,
        report.refined_tensor_payload_bytes,
        report.original_total_gguf_bytes,
        report.refined_total_gguf_bytes,
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Negative controls").unwrap();
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
