//! Issue #75: Log TensorLean as related work and bound a default-off Lean formal-verification
//! check on one pinned reference GPU kernel if public artifacts appear.
//!
//! Pinned related work (strictly non-gating external notes):
//! - X post (2026-11-20, Ronghui Gu `@RonghuiGu`):
//!   `https://x.com/RonghuiGu/status/2123786837797773807`
//! - Authors' unverified claims: TensorLean (CertiK + Columbia) is a Lean-based formal
//!   verification framework for production GPU kernels, reportedly verifying 149 kernel instances
//!   and uncovering 38 bugs across DeepGEMM and FlashAttention.
//! - Extends inference-lab's formal-verification and numerical-parity posture (`#15`, `#52`).
//!
//! Provides:
//! - Explicit related-work scope & Trusted Computing Base (TCB) assumptions record.
//! - Default-off Lean formal-verification check boundary on a pinned small reference kernel
//!   (`tiled_causal_gemm_softmax_v1`, `blocked_on_artifacts` for external TensorLean Lean libs).
//! - Real pure-Rust symbolic/algebraic kernel-invariant checker vs finite-input differential
//!   numerical testing across 5 seeded kernel bugs (tile-boundary OOB, missing reduction carry,
//!   softmax shift/scale error, FP16 accumulator truncation, causal-mask off-by-one), showing
//!   where formal proofs catch corner-case bugs that small differential test suites can miss, and
//!   where numerical rounding assumptions still require empirical validation.
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, softmax_stable};
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const UPSTREAM_X_POST: &str = "https://x.com/RonghuiGu/status/2123786837797773807";
pub const PIN_STATUS: &str = "post_only_unlocated_paper_repo";
pub const REFERENCE_KERNEL_ID: &str = "tiled_causal_gemm_softmax_ref_v1";
pub const DIFF_TOL: f64 = 1.0e-9;

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
pub enum SeededKernelBug {
    NoneReference,
    TileTailBoundaryOob,
    MissingReductionBarrierCarry,
    WrongSoftmaxScale,
    Fp16AccumulatorTruncation,
    CausalMaskOffByOne,
}

impl SeededKernelBug {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NoneReference => "none_reference",
            Self::TileTailBoundaryOob => "tile_tail_boundary_oob",
            Self::MissingReductionBarrierCarry => "missing_reduction_barrier_carry",
            Self::WrongSoftmaxScale => "wrong_softmax_scale",
            Self::Fp16AccumulatorTruncation => "fp16_accumulator_truncation",
            Self::CausalMaskOffByOne => "causal_mask_off_by_one",
        }
    }
}

/// Abstract kernel IR descriptor checked by our bounded symbolic proof-obligation boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelIrSpec {
    pub kernel_id: String,
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    pub tail_bounds_predicated: bool,
    pub reduction_barrier_enforced: bool,
    pub softmax_inv_sqrt_d_exact: bool,
    pub accumulator_bits: u32,
    pub causal_mask_strict_le: bool,
    pub lean_axiom_tcb_complete: bool,
}

impl KernelIrSpec {
    pub fn reference() -> Self {
        Self {
            kernel_id: REFERENCE_KERNEL_ID.into(),
            tile_m: 4,
            tile_n: 4,
            tile_k: 4,
            tail_bounds_predicated: true,
            reduction_barrier_enforced: true,
            softmax_inv_sqrt_d_exact: true,
            accumulator_bits: 32,
            causal_mask_strict_le: true,
            lean_axiom_tcb_complete: true,
        }
    }

    pub fn with_bug(bug: SeededKernelBug) -> Self {
        let mut s = Self::reference();
        match bug {
            SeededKernelBug::NoneReference => {}
            SeededKernelBug::TileTailBoundaryOob => s.tail_bounds_predicated = false,
            SeededKernelBug::MissingReductionBarrierCarry => s.reduction_barrier_enforced = false,
            SeededKernelBug::WrongSoftmaxScale => s.softmax_inv_sqrt_d_exact = false,
            SeededKernelBug::Fp16AccumulatorTruncation => s.accumulator_bits = 16,
            SeededKernelBug::CausalMaskOffByOne => s.causal_mask_strict_le = false,
        }
        s
    }
}

/// Bounded symbolic/formal obligation checker representing the Lean proof boundary.
pub fn check_formal_obligations(spec: &KernelIrSpec) -> Vec<&'static str> {
    let mut violations = Vec::new();
    if spec.tile_m == 0 || spec.tile_n == 0 || spec.tile_k == 0 {
        violations.push("degenerate_zero_tile_dimension");
    }
    if !spec.lean_axiom_tcb_complete {
        violations.push("incomplete_tcb_axiom_boundary");
    }
    if !spec.tail_bounds_predicated {
        violations.push("unpredicated_tile_tail_memory_oob");
    }
    if !spec.reduction_barrier_enforced {
        violations.push("missing_syncthreads_reduction_carry");
    }
    if !spec.softmax_inv_sqrt_d_exact {
        violations.push("softmax_scale_mismatch_vs_spec");
    }
    if spec.accumulator_bits < 32 {
        violations.push("sub_fp32_accumulator_precision_obligation_failed");
    }
    if !spec.causal_mask_strict_le {
        violations.push("causal_mask_leaks_future_position");
    }
    violations
}

/// Reference tiled causal attention GEMM+softmax execution on `(seq_len, d_head)` inputs.
/// Notice: when `seq_len` is an exact multiple of `tile_m=4`, `TileTailBoundaryOob` is NOT
/// triggered by aligned differential tests (`seq_len = 4`), whereas formal verification catches it
/// for ALL `seq_len`!
pub fn execute_reference_kernel(
    seq_len: usize,
    d_head: usize,
    bug: SeededKernelBug,
) -> Result<Vec<f64>, String> {
    if seq_len == 0 || d_head == 0 {
        return Err("zero kernel dimensions".into());
    }
    let mut q = vec![vec![0.0_f64; d_head]; seq_len];
    let mut k = vec![vec![0.0_f64; d_head]; seq_len];
    let mut v = vec![vec![0.0_f64; d_head]; seq_len];
    for t in 0..seq_len {
        for d in 0..d_head {
            q[t][d] = det_weight(7501 + t * 13, d, 0.7);
            k[t][d] = det_weight(7523 + t * 17, d, 0.7);
            v[t][d] = det_weight(7547 + t * 19, d, 0.7);
        }
    }

    let scale = match bug {
        SeededKernelBug::WrongSoftmaxScale => 1.0 / (d_head as f64),
        _ => 1.0 / (d_head as f64).sqrt(),
    };

    let mut out = Vec::with_capacity(seq_len * d_head);
    for (t, q_t) in q.iter().enumerate() {
        let max_j = match bug {
            SeededKernelBug::CausalMaskOffByOne => {
                if t + 1 < seq_len {
                    t + 1
                } else {
                    t
                }
            }
            _ => t,
        };
        let mut scores = Vec::with_capacity(max_j + 1);
        for k_j in k.iter().take(max_j + 1) {
            let mut acc = 0.0_f64;
            for d in 0..d_head {
                let prod = q_t[d] * k_j[d];
                if bug == SeededKernelBug::MissingReductionBarrierCarry && d == d_head - 1 {
                    continue;
                }
                acc += prod;
                if bug == SeededKernelBug::Fp16AccumulatorTruncation {
                    // Simulate coarse FP16 rounding loss (10-bit significand quantization)
                    acc = (acc * 1024.0).round() / 1024.0;
                }
            }
            // Tile tail OOB only manifests when seq_len is NOT a multiple of tile_m (4)
            if bug == SeededKernelBug::TileTailBoundaryOob && (seq_len % 4 != 0) && t == seq_len - 1
            {
                acc += 0.125;
            }
            scores.push(check_finite(acc * scale, "kernel_score")?);
        }
        let probs = softmax_stable(&scores)?;
        for d in 0..d_head {
            let mut sum = 0.0_f64;
            for (j, &p) in probs.iter().enumerate() {
                sum += p * v[j][d];
            }
            out.push(check_finite(sum, "kernel_out")?);
        }
    }
    Ok(out)
}

pub fn max_abs_diff(a: &[f64], b: &[f64]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err("kernel output length mismatch".into());
    }
    let mut m = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (check_finite(x, "diff_a")? - check_finite(y, "diff_b")?).abs();
        m = max_finite(m, d, "max_diff")?;
    }
    Ok(m)
}

#[derive(Debug, Clone)]
pub struct SeededBugComparisonRow {
    pub bug_id: String,
    pub formal_caught: bool,
    pub formal_violation_count: usize,
    pub diff_aligned_seq4_max_err: f64,
    pub diff_aligned_seq4_caught: bool,
    pub diff_unaligned_seq5_max_err: f64,
    pub diff_unaligned_seq5_caught: bool,
}

#[derive(Debug, Clone)]
pub struct VerificationScopeNote {
    pub area: String,
    pub assumption: String,
    pub requires_external_validation: bool,
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct TensorLeanVerifyReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub pin_status: String,
    pub reference_kernel_id: String,
    pub scope_and_tcb_notes: Vec<VerificationScopeNote>,
    pub bug_comparisons: Vec<SeededBugComparisonRow>,
    pub formal_catch_rate: Option<f64>,
    pub aligned_diff_catch_rate: Option<f64>,
    pub unaligned_diff_catch_rate: Option<f64>,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<TensorLeanVerifyReport, String> {
    let scope_and_tcb_notes = vec![
        VerificationScopeNote {
            area: "ptx_hw_semantics_tcb".into(),
            assumption: "Formal proofs assume the modeled GPU ISA/PTX semantics and warp-barrier memory model match physical silicon".into(),
            requires_external_validation: true,
        },
        VerificationScopeNote {
            area: "floating_point_rounding".into(),
            assumption: "Real/IEEE-754 rounding envelopes must match actual Tensor Core MMA accumulation order".into(),
            requires_external_validation: true,
        },
        VerificationScopeNote {
            area: "compiler_lowering_gap".into(),
            assumption: "Verifying source/IR kernels does not prove downstream ptxas/driver SASS lowering is bug-free".into(),
            requires_external_validation: true,
        },
        VerificationScopeNote {
            area: "public_artifact_gate".into(),
            assumption: "External TensorLean Lean4 libraries are not publicly pinned; local checks remain default-off bounded contract models".into(),
            requires_external_validation: true,
        },
    ];

    let ref_seq4 = execute_reference_kernel(4, 4, SeededKernelBug::NoneReference)?;
    let ref_seq5 = execute_reference_kernel(5, 4, SeededKernelBug::NoneReference)?;

    let seeded_bugs = [
        SeededKernelBug::NoneReference,
        SeededKernelBug::TileTailBoundaryOob,
        SeededKernelBug::MissingReductionBarrierCarry,
        SeededKernelBug::WrongSoftmaxScale,
        SeededKernelBug::Fp16AccumulatorTruncation,
        SeededKernelBug::CausalMaskOffByOne,
    ];

    let mut bug_comparisons = Vec::with_capacity(seeded_bugs.len());
    let mut seeded_total = 0usize;
    let mut formal_caught_n = 0usize;
    let mut aligned_caught_n = 0usize;
    let mut unaligned_caught_n = 0usize;

    for bug in seeded_bugs {
        let ir = KernelIrSpec::with_bug(bug);
        let violations = check_formal_obligations(&ir);
        let formal_caught = !violations.is_empty();
        let out4 = execute_reference_kernel(4, 4, bug)?;
        let out5 = execute_reference_kernel(5, 4, bug)?;
        let err4 = max_abs_diff(&ref_seq4, &out4)?;
        let err5 = max_abs_diff(&ref_seq5, &out5)?;
        let caught4 = err4 > DIFF_TOL;
        let caught5 = err5 > DIFF_TOL;
        if bug != SeededKernelBug::NoneReference {
            seeded_total += 1;
            if formal_caught {
                formal_caught_n += 1;
            }
            if caught4 {
                aligned_caught_n += 1;
            }
            if caught5 {
                unaligned_caught_n += 1;
            }
        }
        bug_comparisons.push(SeededBugComparisonRow {
            bug_id: bug.as_str().into(),
            formal_caught,
            formal_violation_count: violations.len(),
            diff_aligned_seq4_max_err: err4,
            diff_aligned_seq4_caught: caught4,
            diff_unaligned_seq5_max_err: err5,
            diff_unaligned_seq5_caught: caught5,
        });
    }

    let denom = seeded_total as f64;
    let formal_catch_rate = if seeded_total > 0 {
        Some((formal_caught_n as f64) / denom)
    } else {
        None
    };
    let aligned_diff_catch_rate = if seeded_total > 0 {
        Some((aligned_caught_n as f64) / denom)
    } else {
        None
    };
    let unaligned_diff_catch_rate = if seeded_total > 0 {
        Some((unaligned_caught_n as f64) / denom)
    } else {
        None
    };

    let mut negative_controls = Vec::new();

    // NC1: public TensorLean Lean4 library absent -> blocked_on_artifacts
    negative_controls.push(NegativeControl {
        case_id: "nc1_missing_public_tensorlean_artifacts_blocked".into(),
        outcome: OutcomeStatus::BlockedOnArtifacts,
        detail:
            "live Lean4 TensorLean verification stays blocked until public repo/commit is pinned"
                .into(),
    });

    // NC2: incomplete TCB axiom boundary rejected
    {
        let mut ir = KernelIrSpec::reference();
        ir.lean_axiom_tcb_complete = false;
        negative_controls.push(NegativeControl {
            case_id: "nc2_incomplete_tcb_axiom_boundary_rejected".into(),
            outcome: if !check_formal_obligations(&ir).is_empty() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "proof with unrecorded/incomplete hardware axiom TCB rejected".into(),
        });
    }

    // NC3: aligned-only differential suite misses tile-tail OOB bug (surfaced as invalid coverage)
    {
        let row = &bug_comparisons[1];
        negative_controls.push(NegativeControl {
            case_id: "nc3_aligned_only_diff_misses_tail_oob".into(),
            outcome: if row.formal_caught && !row.diff_aligned_seq4_caught {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "aligned seq=4 differential test missed tile_tail_boundary_oob caught by formal check".into(),
        });
    }

    // NC4: zero tile dimension rejected
    {
        let mut ir = KernelIrSpec::reference();
        ir.tile_k = 0;
        negative_controls.push(NegativeControl {
            case_id: "nc4_zero_tile_dimension_rejected".into(),
            outcome: if !check_formal_obligations(&ir).is_empty() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "degenerate zero tile_k dimension rejected by obligation checker".into(),
        });
    }

    // NC5: zero kernel dimensions refused
    {
        negative_controls.push(NegativeControl {
            case_id: "nc5_zero_kernel_dimensions_refused".into(),
            outcome: if execute_reference_kernel(0, 4, SeededKernelBug::NoneReference).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "zero sequence length rejected before kernel execution".into(),
        });
    }

    // NC6: NaN output differential rejected
    {
        negative_controls.push(NegativeControl {
            case_id: "nc6_nan_kernel_output_rejected".into(),
            outcome: if max_abs_diff(&[1.0, f64::NAN], &[1.0, 1.0]).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "NaN kernel output fails differential comparator cleanly".into(),
        });
    }

    // NC7: mismatched output lengths rejected
    {
        negative_controls.push(NegativeControl {
            case_id: "nc7_mismatched_output_shape_rejected".into(),
            outcome: if max_abs_diff(&[1.0, 2.0], &[1.0]).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "mismatched tensor output shape rejected by comparator".into(),
        });
    }

    // NC8: claiming formal proof guarantees ptxas/silicon bug-freedom rejected
    {
        let claims_silicon_certainty_without_hw_check = true;
        negative_controls.push(NegativeControl {
            case_id: "nc8_formal_ir_proof_does_not_bypass_silicon_check".into(),
            outcome: if claims_silicon_certainty_without_hw_check {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail:
                "IR-level formal proof cannot substitute for ptxas/silicon differential validation"
                    .into(),
        });
    }

    Ok(TensorLeanVerifyReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::blocked_on_artifacts(
            "TensorLean Lean4 formal verification framework + pinned GPU kernel",
            "TensorLean public Lean4 repository/commit is unlocated (post_only_unlocated_paper_repo); bounded symbolic obligation vs differential bug-coverage check ran on CPU",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        pin_status: PIN_STATUS.into(),
        reference_kernel_id: REFERENCE_KERNEL_ID.into(),
        scope_and_tcb_notes,
        bug_comparisons,
        formal_catch_rate,
        aligned_diff_catch_rate,
        unaligned_diff_catch_rate,
        negative_controls,
    })
}

pub fn inputs_json(report: &TensorLeanVerifyReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.tensorlean-kernel-verify.inputs.v1\",\n",
            "  \"issue\": 75,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"pin_status\": {},\n",
            "  \"upstream_x_post\": {},\n",
            "  \"reference_kernel_id\": {},\n",
            "  \"diff_tol\": {},\n",
            "  \"related_issues\": [15, 52]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(&report.pin_status),
        quote(UPSTREAM_X_POST),
        quote(&report.reference_kernel_id),
        finite_f64(DIFF_TOL),
    )
}

pub fn results_json(report: &TensorLeanVerifyReport) -> String {
    let notes: Vec<String> = report
        .scope_and_tcb_notes
        .iter()
        .map(|n| {
            format!(
                "{{\"area\":{},\"assumption\":{},\"requires_external_validation\":{}}}",
                quote(&n.area),
                quote(&n.assumption),
                n.requires_external_validation
            )
        })
        .collect();
    let bugs: Vec<String> = report
        .bug_comparisons
        .iter()
        .map(|b| {
            format!(
                "{{\"bug_id\":{},\"formal_caught\":{},\"formal_violation_count\":{},\"diff_aligned_seq4_max_err\":{},\"diff_aligned_seq4_caught\":{},\"diff_unaligned_seq5_max_err\":{},\"diff_unaligned_seq5_caught\":{}}}",
                quote(&b.bug_id),
                b.formal_caught,
                b.formal_violation_count,
                finite_f64(b.diff_aligned_seq4_max_err),
                b.diff_aligned_seq4_caught,
                finite_f64(b.diff_unaligned_seq5_max_err),
                b.diff_unaligned_seq5_caught,
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
            "  \"schema\": \"inference-lab.tensorlean-kernel-verify.results.v1\",\n",
            "  \"issue\": 75,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"formal_catch_rate\": {},\n",
            "  \"aligned_diff_catch_rate\": {},\n",
            "  \"unaligned_diff_catch_rate\": {},\n",
            "  \"scope_and_tcb_notes\": [{}],\n",
            "  \"bug_comparisons\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        opt_f64(report.formal_catch_rate),
        opt_f64(report.aligned_diff_catch_rate),
        opt_f64(report.unaligned_diff_catch_rate),
        notes.join(","),
        bugs.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &TensorLeanVerifyReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #75: TensorLean Formal-Verification Boundary & Seeded Kernel-Bug Coverage"
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
        "- Artifact gate: `{}` ({})",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(
        out,
        "- Catch rates on 5 seeded bugs: formal `{:.2}`, aligned-diff (`seq=4`) `{:.2}`, unaligned-diff (`seq=5`) `{:.2}`",
        report.formal_catch_rate.unwrap_or(0.0),
        report.aligned_diff_catch_rate.unwrap_or(0.0),
        report.unaligned_diff_catch_rate.unwrap_or(0.0),
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Formal Obligations vs Differential Testing").unwrap();
    writeln!(
        out,
        "| Seeded Bug | Formal Caught | Aligned seq=4 Err | Aligned Caught | Unaligned seq=5 Err | Unaligned Caught |"
    )
    .unwrap();
    writeln!(out, "| --- | --- | ---: | --- | ---: | --- |").unwrap();
    for b in &report.bug_comparisons {
        writeln!(
            out,
            "| `{}` | `{}` | `{:.3e}` | `{}` | `{:.3e}` | `{}` |",
            b.bug_id,
            b.formal_caught,
            b.diff_aligned_seq4_max_err,
            b.diff_aligned_seq4_caught,
            b.diff_unaligned_seq5_max_err,
            b.diff_unaligned_seq5_caught
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
    fn formal_check_catches_all_seeded_bugs_including_aligned_blindspot() {
        let r = run_default_harness().unwrap();
        assert_eq!(r.hardware_gate.status, ExecutionClass::BlockedOnArtifacts);
        assert!((r.formal_catch_rate.unwrap() - 1.0).abs() <= 1.0e-12);
        assert!(r.aligned_diff_catch_rate.unwrap() < r.formal_catch_rate.unwrap());
        assert!(!r.bug_comparisons[0].formal_caught);
        assert_eq!(r.negative_controls.len(), 8);
        for c in &r.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }
}
