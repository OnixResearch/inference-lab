//! Issue #15: Adopt `ggml` `test-backend-ops` as a pinned numerical capability and performance
//! qualification harness.
//!
//! Pure deterministic core implementing:
//! - Pinned upstream source reference `ggml-org/ggml@417a024b4fb6bc3116d41af7c067166a039bdf12`
//!   (`tests/test-backend-ops.cpp`, `include/ggml-backend.h`).
//! - Four strictly independent qualification axes: `advertised_support`, `executed_success`,
//!   `numerical_pass`, and `performance_measured` (never collapsed into one boolean).
//! - Real pure-Rust bounded quantized dot/matmul reference math for `F32`, `Q8_0` (32-element
//!   block, `f32` scale + `[i8; 32]`), and `Q4_0` (32-element block, `f32` scale + 16 packed
//!   nibble bytes representing 32 values in `[-8, 7]`), plus explicit `MXFP4`/`NVFP4` unsupported
//!   status when a backend lacks those formats.
//! - Qualification verdict engine disqualifying incorrect outputs, non-finite values, and silent
//!   CPU fallback when an accelerator backend was requested, while preserving variance across
//!   repeated trials and `grants_placement_authority = false`.

use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_f64, opt_str, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const GGML_PIN_COMMIT: &str = "417a024b4fb6bc3116d41af7c067166a039bdf12";
pub const GGML_BLOCK_SIZE: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlDtype {
    F32,
    Q8_0,
    Q4_0,
    Mxfp4,
    Nvfp4,
}

impl GgmlDtype {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::Q8_0 => "Q8_0",
            Self::Q4_0 => "Q4_0",
            Self::Mxfp4 => "MXFP4",
            Self::Nvfp4 => "NVFP4",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Q8_0Block {
    pub scale: f32,
    pub quants: [i8; GGML_BLOCK_SIZE],
}

impl Q8_0Block {
    pub fn quantize(values: &[f32; GGML_BLOCK_SIZE]) -> Result<Self, String> {
        if values.iter().any(|x| !x.is_finite()) {
            return Err("non-finite value in Q8_0 quantize".into());
        }
        let amax = values.iter().fold(0.0_f32, |acc, &x| acc.max(x.abs()));
        let scale = amax / 127.0;
        let inv = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        let mut quants = [0i8; GGML_BLOCK_SIZE];
        for (i, &x) in values.iter().enumerate() {
            quants[i] = (x * inv).round().clamp(-127.0, 127.0) as i8;
        }
        Ok(Self { scale, quants })
    }

    pub fn dot_f32(&self, rhs: &[f32; GGML_BLOCK_SIZE]) -> Result<f32, String> {
        if !self.scale.is_finite() || rhs.iter().any(|x| !x.is_finite()) {
            return Err("non-finite value in Q8_0 dot".into());
        }
        let mut sum = 0.0_f64;
        for (&q, &r) in self.quants.iter().zip(rhs.iter()) {
            sum += f64::from(self.scale) * f64::from(q) * f64::from(r);
        }
        Ok(sum as f32)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Q4_0Block {
    pub scale: f32,
    /// 16 bytes packing 32 unsigned 4-bit offsets `q + 8` (`q in [-8, 7]`).
    pub packed: [u8; GGML_BLOCK_SIZE / 2],
}

impl Q4_0Block {
    pub fn quantize(values: &[f32; GGML_BLOCK_SIZE]) -> Result<Self, String> {
        if values.iter().any(|x| !x.is_finite()) {
            return Err("non-finite value in Q4_0 quantize".into());
        }
        let mut max_abs = 0.0_f32;
        let mut signed_max = 0.0_f32;
        for &x in values {
            if x.abs() > max_abs {
                max_abs = x.abs();
                signed_max = x;
            }
        }
        let scale = signed_max / -8.0;
        let inv = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        let mut packed = [0u8; GGML_BLOCK_SIZE / 2];
        for i in 0..(GGML_BLOCK_SIZE / 2) {
            let v0 = values[i];
            let v1 = values[i + GGML_BLOCK_SIZE / 2];
            let q0 = ((v0 * inv + 8.5).floor() as i32).clamp(0, 15) as u8;
            let q1 = ((v1 * inv + 8.5).floor() as i32).clamp(0, 15) as u8;
            packed[i] = q0 | (q1 << 4);
        }
        Ok(Self { scale, packed })
    }

    pub fn dot_f32(&self, rhs: &[f32; GGML_BLOCK_SIZE]) -> Result<f32, String> {
        if !self.scale.is_finite() || rhs.iter().any(|x| !x.is_finite()) {
            return Err("non-finite value in Q4_0 dot".into());
        }
        let mut sum = 0.0_f64;
        for i in 0..(GGML_BLOCK_SIZE / 2) {
            let byte = self.packed[i];
            let q0 = i32::from(byte & 0x0f) - 8;
            let q1 = i32::from(byte >> 4) - 8;
            sum += f64::from(self.scale)
                * (f64::from(q0) * f64::from(rhs[i])
                    + f64::from(q1) * f64::from(rhs[i + GGML_BLOCK_SIZE / 2]));
        }
        Ok(sum as f32)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BackendOpQualificationCase {
    pub case_id: String,
    pub mode: String,
    pub op_name: String,
    pub dtype: GgmlDtype,
    pub shape: [usize; 3],
    pub requested_backend: String,
    pub actual_backend: String,
    pub advertised_support: bool,
    pub executed_success: bool,
    pub skipped: bool,
    pub max_abs_err: Option<f64>,
    pub tolerance: f64,
    pub has_nonfinite_output: bool,
    pub repeated_latencies_us: Vec<f64>,
    pub warmup_runs: usize,
    pub graph_reallocations: usize,
    pub memory_growth_bytes: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BackendOpQualificationVerdict {
    pub case_id: String,
    pub mode: String,
    pub op_name: String,
    pub dtype: GgmlDtype,
    pub shape: [usize; 3],
    pub requested_backend: String,
    pub actual_backend: String,
    pub advertised_support: bool,
    pub executed_success: bool,
    pub skipped: bool,
    pub numerical_pass: bool,
    pub performance_measured: bool,
    pub silent_cpu_fallback_detected: bool,
    pub backend_mismatch_detected: bool,
    /// Structurally distinguishes `"passed"`, `"failed"`, `"unsupported"`, and `"skipped"`.
    pub qualification_state: &'static str,
    pub outcome: OutcomeStatus,
    pub warmup_runs: usize,
    pub graph_reallocations: usize,
    pub memory_growth_bytes: u64,
    pub mean_latency_us: Option<f64>,
    pub stddev_latency_us: Option<f64>,
    pub rejection_reason: Option<String>,
}

fn mean_and_stddev(samples: &[f64]) -> Result<(Option<f64>, Option<f64>), String> {
    if samples.is_empty() {
        return Ok((None, None));
    }
    if samples.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return Err("non-finite or negative latency sample".into());
    }
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    if samples.len() < 2 {
        return Ok((Some(mean), Some(0.0)));
    }
    let var = samples
        .iter()
        .map(|&x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / (n - 1.0);
    Ok((Some(mean), Some(var.sqrt())))
}

/// Evaluate one `test-backend-ops` qualification record with strictly separated axes.
pub fn qualify_backend_op(case: &BackendOpQualificationCase) -> BackendOpQualificationVerdict {
    let backend_mismatch = case.executed_success && case.requested_backend != case.actual_backend;
    let silent_fallback =
        backend_mismatch && case.requested_backend != "CPU" && case.actual_backend == "CPU";

    let (mean_us, std_us) = match mean_and_stddev(&case.repeated_latencies_us) {
        Ok(pair) => pair,
        Err(_) => {
            return BackendOpQualificationVerdict {
                case_id: case.case_id.clone(),
                mode: case.mode.clone(),
                op_name: case.op_name.clone(),
                dtype: case.dtype,
                shape: case.shape,
                requested_backend: case.requested_backend.clone(),
                actual_backend: case.actual_backend.clone(),
                advertised_support: case.advertised_support,
                executed_success: case.executed_success,
                skipped: case.skipped,
                numerical_pass: false,
                performance_measured: false,
                silent_cpu_fallback_detected: silent_fallback,
                backend_mismatch_detected: backend_mismatch,
                qualification_state: "failed",
                outcome: OutcomeStatus::Invalid,
                warmup_runs: case.warmup_runs,
                graph_reallocations: case.graph_reallocations,
                memory_growth_bytes: case.memory_growth_bytes,
                mean_latency_us: None,
                stddev_latency_us: None,
                rejection_reason: Some("invalid_nonfinite_or_negative_latency_samples".into()),
            };
        }
    };
    let perf_measured = mean_us.is_some() && case.executed_success && !backend_mismatch;

    let valid_tol = case.tolerance.is_finite() && case.tolerance >= 0.0;
    let invalid_err_or_tol =
        !valid_tol || case.max_abs_err.is_some_and(|e| !e.is_finite() || e < 0.0);

    let numerical_pass =
        if !case.executed_success || case.has_nonfinite_output || invalid_err_or_tol {
            false
        } else if let Some(err) = case.max_abs_err {
            err <= case.tolerance
        } else {
            false
        };

    let (qualification_state, outcome, reason) = if !case.advertised_support {
        (
            "unsupported",
            OutcomeStatus::Unsupported,
            Some("backend_explicitly_does_not_support_op_or_dtype".into()),
        )
    } else if case.skipped {
        (
            "skipped",
            OutcomeStatus::Unsupported,
            Some("explicitly_skipped_distinct_from_pass_or_fail".into()),
        )
    } else if silent_fallback {
        (
            "failed",
            OutcomeStatus::Invalid,
            Some("silent_cpu_fallback_disqualifies_accelerator_qualification".into()),
        )
    } else if backend_mismatch {
        (
            "failed",
            OutcomeStatus::Invalid,
            Some("actual_backend_mismatch_disqualifies_qualification".into()),
        )
    } else if !case.executed_success {
        (
            "failed",
            OutcomeStatus::InternalFailure,
            Some("advertised_supported_op_failed_execution".into()),
        )
    } else if case.has_nonfinite_output {
        (
            "failed",
            OutcomeStatus::Invalid,
            Some("nonfinite_output_detected_disqualified".into()),
        )
    } else if invalid_err_or_tol {
        (
            "failed",
            OutcomeStatus::Invalid,
            Some("negative_or_nonfinite_error_or_tolerance".into()),
        )
    } else if !numerical_pass {
        (
            "failed",
            OutcomeStatus::Invalid,
            Some("numerical_error_exceeds_frozen_tolerance".into()),
        )
    } else {
        ("passed", OutcomeStatus::Accepted, None)
    };

    BackendOpQualificationVerdict {
        case_id: case.case_id.clone(),
        mode: case.mode.clone(),
        op_name: case.op_name.clone(),
        dtype: case.dtype,
        shape: case.shape,
        requested_backend: case.requested_backend.clone(),
        actual_backend: case.actual_backend.clone(),
        advertised_support: case.advertised_support,
        executed_success: case.executed_success,
        skipped: case.skipped,
        numerical_pass,
        performance_measured: perf_measured,
        silent_cpu_fallback_detected: silent_fallback,
        backend_mismatch_detected: backend_mismatch,
        qualification_state,
        outcome,
        warmup_runs: case.warmup_runs,
        graph_reallocations: case.graph_reallocations,
        memory_growth_bytes: case.memory_growth_bytes,
        mean_latency_us: mean_us,
        stddev_latency_us: std_us,
        rejection_reason: reason,
    }
}

#[derive(Debug, Clone)]
pub struct GgmlOpsReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub ggml_pin_commit: String,
    pub f32_ref_dot: f64,
    pub q8_0_abs_err: f64,
    pub q4_0_abs_err: f64,
    pub verdicts: Vec<BackendOpQualificationVerdict>,
    pub grants_placement_authority: bool,
    pub input_digest: DigestRef,
}

pub fn run_default_harness() -> Result<GgmlOpsReport, String> {
    let mut lhs = [0.0_f32; GGML_BLOCK_SIZE];
    let mut rhs = [0.0_f32; GGML_BLOCK_SIZE];
    for i in 0..GGML_BLOCK_SIZE {
        let f = i as f32;
        lhs[i] = ((f * 0.37).sin()) * 1.5;
        rhs[i] = ((f * 0.23 + 0.5).cos()) * 1.2;
    }

    let f32_ref_dot: f64 = lhs
        .iter()
        .zip(rhs.iter())
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum();
    let q8 = Q8_0Block::quantize(&lhs)?;
    let q8_dot = f64::from(q8.dot_f32(&rhs)?);
    let q8_err = (q8_dot - f32_ref_dot).abs();

    let q4 = Q4_0Block::quantize(&lhs)?;
    let q4_dot = f64::from(q4.dot_f32(&rhs)?);
    let q4_err = (q4_dot - f32_ref_dot).abs();

    let cases = vec![
        BackendOpQualificationCase {
            case_id: "01-positive-cpu-ref-mul-mat-f32".into(),
            mode: "test".into(),
            op_name: "MUL_MAT".into(),
            dtype: GgmlDtype::F32,
            shape: [32, 16, 1],
            requested_backend: "CPU".into(),
            actual_backend: "CPU".into(),
            advertised_support: true,
            executed_success: true,
            skipped: false,
            max_abs_err: Some(0.0),
            tolerance: 1e-5,
            has_nonfinite_output: false,
            repeated_latencies_us: vec![],
            warmup_runs: 2,
            graph_reallocations: 0,
            memory_growth_bytes: 0,
        },
        BackendOpQualificationCase {
            case_id: "02-positive-cpu-ref-mul-mat-q8_0".into(),
            mode: "test".into(),
            op_name: "MUL_MAT".into(),
            dtype: GgmlDtype::Q8_0,
            shape: [32, 16, 1],
            requested_backend: "CPU".into(),
            actual_backend: "CPU".into(),
            advertised_support: true,
            executed_success: true,
            skipped: false,
            max_abs_err: Some(q8_err),
            tolerance: 0.15,
            has_nonfinite_output: false,
            repeated_latencies_us: vec![],
            warmup_runs: 2,
            graph_reallocations: 0,
            memory_growth_bytes: 0,
        },
        BackendOpQualificationCase {
            case_id: "03-positive-cpu-ref-mul-mat-q4_0".into(),
            mode: "test".into(),
            op_name: "MUL_MAT".into(),
            dtype: GgmlDtype::Q4_0,
            shape: [32, 16, 1],
            requested_backend: "CPU".into(),
            actual_backend: "CPU".into(),
            advertised_support: true,
            executed_success: true,
            skipped: false,
            max_abs_err: Some(q4_err),
            tolerance: 1.5,
            has_nonfinite_output: false,
            repeated_latencies_us: vec![],
            warmup_runs: 2,
            graph_reallocations: 0,
            memory_growth_bytes: 0,
        },
        BackendOpQualificationCase {
            case_id: "04-control-unsupported-mxfp4-explicit".into(),
            mode: "supported".into(),
            op_name: "MUL_MAT".into(),
            dtype: GgmlDtype::Mxfp4,
            shape: [32, 16, 1],
            requested_backend: "CPU".into(),
            actual_backend: "CPU".into(),
            advertised_support: false,
            executed_success: false,
            skipped: false,
            max_abs_err: None,
            tolerance: 0.25,
            has_nonfinite_output: false,
            repeated_latencies_us: vec![],
            warmup_runs: 0,
            graph_reallocations: 0,
            memory_growth_bytes: 0,
        },
        BackendOpQualificationCase {
            case_id: "05-control-unsupported-nvfp4-explicit".into(),
            mode: "supported".into(),
            op_name: "MUL_MAT".into(),
            dtype: GgmlDtype::Nvfp4,
            shape: [32, 16, 1],
            requested_backend: "CUDA0".into(),
            actual_backend: "none".into(),
            advertised_support: false,
            executed_success: false,
            skipped: false,
            max_abs_err: None,
            tolerance: 0.25,
            has_nonfinite_output: false,
            repeated_latencies_us: vec![],
            warmup_runs: 0,
            graph_reallocations: 0,
            memory_growth_bytes: 0,
        },
        BackendOpQualificationCase {
            case_id: "06-negative-silent-cpu-fallback-disqualified".into(),
            mode: "perf".into(),
            op_name: "MUL_MAT".into(),
            dtype: GgmlDtype::Q8_0,
            shape: [32, 16, 1],
            requested_backend: "CUDA0".into(),
            actual_backend: "CPU".into(),
            advertised_support: true,
            executed_success: true,
            skipped: false,
            max_abs_err: Some(0.01),
            tolerance: 0.15,
            has_nonfinite_output: false,
            repeated_latencies_us: vec![120.0, 122.0, 121.0],
            warmup_runs: 3,
            graph_reallocations: 0,
            memory_growth_bytes: 0,
        },
        BackendOpQualificationCase {
            case_id: "07-negative-numerical-mismatch-disqualified".into(),
            mode: "test".into(),
            op_name: "MUL_MAT".into(),
            dtype: GgmlDtype::F32,
            shape: [32, 16, 1],
            requested_backend: "CUDA0".into(),
            actual_backend: "CUDA0".into(),
            advertised_support: true,
            executed_success: true,
            skipped: false,
            max_abs_err: Some(0.42),
            tolerance: 1e-4,
            has_nonfinite_output: false,
            repeated_latencies_us: vec![],
            warmup_runs: 2,
            graph_reallocations: 0,
            memory_growth_bytes: 0,
        },
        BackendOpQualificationCase {
            case_id: "08-negative-nonfinite-nan-output-disqualified".into(),
            mode: "test".into(),
            op_name: "MUL_MAT".into(),
            dtype: GgmlDtype::Q4_0,
            shape: [32, 16, 1],
            requested_backend: "CUDA0".into(),
            actual_backend: "CUDA0".into(),
            advertised_support: true,
            executed_success: true,
            skipped: false,
            max_abs_err: None,
            tolerance: 1.5,
            has_nonfinite_output: true,
            repeated_latencies_us: vec![],
            warmup_runs: 2,
            graph_reallocations: 0,
            memory_growth_bytes: 0,
        },
        BackendOpQualificationCase {
            case_id: "09-negative-advertised-supported-but-exec-failed".into(),
            mode: "test".into(),
            op_name: "MUL_MAT".into(),
            dtype: GgmlDtype::Q8_0,
            shape: [32, 16, 1],
            requested_backend: "CUDA0".into(),
            actual_backend: "CUDA0".into(),
            advertised_support: true,
            executed_success: false,
            skipped: false,
            max_abs_err: None,
            tolerance: 0.15,
            has_nonfinite_output: false,
            repeated_latencies_us: vec![],
            warmup_runs: 0,
            graph_reallocations: 0,
            memory_growth_bytes: 0,
        },
        BackendOpQualificationCase {
            case_id: "10-control-explicit-skip-distinct-from-pass-or-fail".into(),
            mode: "test".into(),
            op_name: "MUL_MAT".into(),
            dtype: GgmlDtype::Q4_0,
            shape: [32, 16, 1],
            requested_backend: "CPU".into(),
            actual_backend: "CPU".into(),
            advertised_support: true,
            executed_success: false,
            skipped: true,
            max_abs_err: None,
            tolerance: 1.5,
            has_nonfinite_output: false,
            repeated_latencies_us: vec![],
            warmup_runs: 0,
            graph_reallocations: 0,
            memory_growth_bytes: 0,
        },
    ];

    let verdicts = cases.iter().map(qualify_backend_op).collect();
    let mut raw_in = Vec::new();
    for x in lhs.iter().chain(rhs.iter()) {
        raw_in.extend_from_slice(&x.to_bits().to_le_bytes());
    }

    Ok(GgmlOpsReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "pinned_ggml_test_backend_ops_accelerator_binary",
            "Pure-Rust F32/Q8_0/Q4_0 reference dot kernels and 4-axis qualification controls ran locally; compiled ggml test-backend-ops binary and accelerator execution remain unavailable_hardware_runtime.",
        ),
        ggml_pin_commit: GGML_PIN_COMMIT.into(),
        f32_ref_dot,
        q8_0_abs_err: q8_err,
        q4_0_abs_err: q4_err,
        verdicts,
        grants_placement_authority: false,
        input_digest: DigestRef::sha256(&raw_in),
    })
}

pub fn inputs_json(report: &GgmlOpsReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.issue-15-ggml-ops.inputs.v1\",\n",
            "  \"issue\": 15,\n",
            "  \"execution_class\": {},\n",
            "  \"ggml_repo\": \"ggml-org/ggml\",\n",
            "  \"ggml_pin_commit\": {},\n",
            "  \"pinned_files\": [\"tests/test-backend-ops.cpp\",\"include/ggml-backend.h\"],\n",
            "  \"bounded_dtypes\": [\"F32\",\"Q8_0\",\"Q4_0\",\"MXFP4\",\"NVFP4\"],\n",
            "  \"input_digest\": {},\n",
            "  \"grants_placement_authority\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(&report.ggml_pin_commit),
        report.input_digest.to_json(),
        report.grants_placement_authority,
    )
}

pub fn results_json(report: &GgmlOpsReport) -> String {
    let mut out = String::new();
    writeln!(out, "{{").unwrap();
    writeln!(
        out,
        "  \"schema\": \"inference-lab.issue-15-ggml-ops.results.v1\","
    )
    .unwrap();
    writeln!(out, "  \"issue\": 15,").unwrap();
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
        "  \"f32_ref_dot\": {},",
        finite_f64(report.f32_ref_dot)
    )
    .unwrap();
    writeln!(
        out,
        "  \"q8_0_abs_err\": {},",
        finite_f64(report.q8_0_abs_err)
    )
    .unwrap();
    writeln!(
        out,
        "  \"q4_0_abs_err\": {},",
        finite_f64(report.q4_0_abs_err)
    )
    .unwrap();
    writeln!(
        out,
        "  \"grants_placement_authority\": {},",
        report.grants_placement_authority
    )
    .unwrap();

    let rows: Vec<String> = report
        .verdicts
        .iter()
        .map(|v| {
            format!(
                concat!(
                    "{{\"case_id\":{},\"mode\":{},\"op_name\":{},\"dtype\":{},\"shape\":[{},{},{}],",
                    "\"requested_backend\":{},\"actual_backend\":{},",
                    "\"advertised_support\":{},\"executed_success\":{},\"skipped\":{},\"numerical_pass\":{},",
                    "\"performance_measured\":{},\"silent_cpu_fallback_detected\":{},\"backend_mismatch_detected\":{},",
                    "\"qualification_state\":{},\"outcome\":{},\"warmup_runs\":{},\"graph_reallocations\":{},",
                    "\"memory_growth_bytes\":{},\"mean_latency_us\":{},\"stddev_latency_us\":{},",
                    "\"rejection_reason\":{}}}"
                ),
                quote(&v.case_id),
                quote(&v.mode),
                quote(&v.op_name),
                quote(v.dtype.as_str()),
                v.shape[0],
                v.shape[1],
                v.shape[2],
                quote(&v.requested_backend),
                quote(&v.actual_backend),
                v.advertised_support,
                v.executed_success,
                v.skipped,
                v.numerical_pass,
                v.performance_measured,
                v.silent_cpu_fallback_detected,
                v.backend_mismatch_detected,
                quote(v.qualification_state),
                quote(v.outcome.as_str()),
                v.warmup_runs,
                v.graph_reallocations,
                v.memory_growth_bytes,
                opt_f64(v.mean_latency_us),
                opt_f64(v.stddev_latency_us),
                opt_str(v.rejection_reason.as_deref()),
            )
        })
        .collect();
    writeln!(out, "  \"qualification_verdicts\": [{}]", rows.join(",")).unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn summary_markdown(report: &GgmlOpsReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #15: Pinned `ggml` Backend-Ops Qualification Harness"
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
    writeln!(out, "- Pinned `ggml` commit: `{}`", report.ggml_pin_commit).unwrap();
    writeln!(
        out,
        "- Reference block-32 dot (`F32`): `{}` (`Q8_0` abs err: `{}`, `Q4_0` abs err: `{}`)",
        finite_f64(report.f32_ref_dot),
        finite_f64(report.q8_0_abs_err),
        finite_f64(report.q4_0_abs_err)
    )
    .unwrap();
    writeln!(
        out,
        "- Grants placement authority: `{}` (Kiln `#60` retains scheduling)",
        report.grants_placement_authority
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## 4-Axis Qualification Verdicts").unwrap();
    writeln!(
        out,
        "| Case | Dtype | Req->Act | Adv | Exec | Num | Perf | Outcome |"
    )
    .unwrap();
    writeln!(out, "| --- | --- | --- | --- | --- | --- | --- | --- |").unwrap();
    for v in &report.verdicts {
        writeln!(
            out,
            "| `{}` | `{}` | `{}->{}` | `{}` | `{}` | `{}` | `{}` | `{}` |",
            v.case_id,
            v.dtype.as_str(),
            v.requested_backend,
            v.actual_backend,
            v.advertised_support,
            v.executed_success,
            v.numerical_pass,
            v.performance_measured,
            v.outcome.as_str()
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "> Pure-Rust quantized reference kernels and qualification controls executed locally; no fabricated accelerator performance or compiled `test-backend-ops` GPU run is claimed."
    )
    .unwrap();
    out
}
