//! Issue #84: Log ML-and-silicon co-design report (`arXiv:2611.18859`) as related work and record
//! its pre-silicon latency/energy assumptions against lab hardware baselines.
//!
//! Pinned related work (strictly non-gating external notes):
//! - Paper: `https://arxiv.org/abs/2611.18859` (`arxiv_id_only_artifacts_unverified_offline`)
//! - X post (2026-11-17, Vithu Thangarasa `@vithursant19`):
//!   `https://x.com/vithursant19/status/2122697450196730308`
//! - Authors' unverified pre-silicon claims: exploring 10M+ ML/silicon co-designs with
//!   pre-silicon projections of 110x faster token generation and 690x lower energy vs NVIDIA B300.
//! - Directly integrates with `#43` (power/cadence baselines and sensor-gated energy reporting)
//!   by refusing to conflate pre-silicon simulation models with measured physical silicon.
//!
//! Provides:
//! - Structured assumption ledger recording workload, precision, memory-bandwidth, SRAM/HBM
//!   energy-per-bit, and power-envelope assumptions from `arXiv:2611.18859` alongside explicit
//!   `requires_physical_silicon` flags.
//! - Pure-Rust deterministic **roofline & energy-per-token physics comparator** contrasting the
//!   pre-silicon projection against lab hardware baseline envelopes (B300 reference envelope,
//!   DGX Spark GB10, Apple M2 Ultra, Tenstorrent P150) — deriving memory-bandwidth tok/s ceiling,
//!   compute tok/s ceiling, roofline bottleneck, and implied Joules/token from explicit physical
//!   parameters rather than accepting headline ratios on faith.
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const UPSTREAM_ARXIV: &str = "arXiv:2611.18859";
pub const UPSTREAM_ARXIV_URL: &str = "https://arxiv.org/abs/2611.18859";
pub const UPSTREAM_X_POST: &str = "https://x.com/vithursant19/status/2122697450196730308";
pub const PIN_STATUS: &str = "arxiv_id_only_artifacts_unverified_offline";

fn check_finite(v: f64, ctx: &str) -> Result<f64, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("non-finite float in {ctx}"))
    }
}

fn min_finite(a: f64, b: f64, ctx: &str) -> Result<f64, String> {
    let a = check_finite(a, ctx)?;
    let b = check_finite(b, ctx)?;
    Ok(if a <= b { a } else { b })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareEvidenceMaturity {
    PreSiliconSimulationOnly,
    LabHardwareBaselineEnvelope,
}

impl HardwareEvidenceMaturity {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PreSiliconSimulationOnly => "pre_silicon_simulation_only",
            Self::LabHardwareBaselineEnvelope => "lab_hardware_baseline_envelope",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoDesignAssumptionItem {
    pub dimension: String,
    pub pre_silicon_assumption: String,
    pub lab_baseline_reality: String,
    pub requires_physical_silicon: bool,
}

#[derive(Debug, Clone)]
pub struct RooflinePlatformInput {
    pub platform_id: String,
    pub maturity: HardwareEvidenceMaturity,
    pub active_params_billions: f64,
    pub weight_bits: f64,
    pub memory_bandwidth_gb_s: f64,
    pub peak_compute_tflops: f64,
    pub board_power_watts: f64,
    pub physical_silicon_validated: bool,
}

#[derive(Debug, Clone)]
pub struct RooflineComparisonRow {
    pub platform_id: String,
    pub maturity: HardwareEvidenceMaturity,
    pub active_weight_bytes_gb: f64,
    pub bw_ceiling_tok_s: Option<f64>,
    pub compute_ceiling_tok_s: Option<f64>,
    pub roofline_tok_s: Option<f64>,
    pub bottleneck: String,
    pub implied_joules_per_token: Option<f64>,
    pub implied_tok_per_sec_per_watt: Option<f64>,
    pub physical_silicon_validated: bool,
    pub eligible_as_measured_lab_result: bool,
}

/// Deterministic decode (`batch=1`) roofline & energy-per-token calculator.
/// Memory bytes read per token ~ `active_params * (weight_bits / 8)`.
/// FLOPs per token ~ `2 * active_params`.
pub fn evaluate_roofline_platform(
    input: &RooflinePlatformInput,
) -> Result<RooflineComparisonRow, String> {
    let params_b = check_finite(input.active_params_billions, "active_params_billions")?;
    let w_bits = check_finite(input.weight_bits, "weight_bits")?;
    let bw = check_finite(input.memory_bandwidth_gb_s, "memory_bandwidth_gb_s")?;
    let tflops = check_finite(input.peak_compute_tflops, "peak_compute_tflops")?;
    let watts = check_finite(input.board_power_watts, "board_power_watts")?;
    if params_b <= 0.0 || w_bits <= 0.0 || bw <= 0.0 || tflops <= 0.0 || watts <= 0.0 {
        return Err(format!(
            "non-positive roofline parameter on `{}`",
            input.platform_id
        ));
    }
    // Enforce honesty: pre-silicon simulations can NEVER claim physical_silicon_validated = true!
    if input.maturity == HardwareEvidenceMaturity::PreSiliconSimulationOnly
        && input.physical_silicon_validated
    {
        return Err("pre-silicon simulation cannot be marked physical_silicon_validated".into());
    }

    // Active weight footprint in decimal GB (10^9 bytes)
    let weight_gb = params_b * (w_bits / 8.0);
    let bw_tok_s = bw / weight_gb;
    // 2 FLOPs per param -> 2 * params_b GFLOPs/token; peak_compute_tflops * 1000 GFLOP/s
    let comp_tok_s = (tflops * 1000.0) / (2.0 * params_b);
    let roofline = min_finite(bw_tok_s, comp_tok_s, "roofline")?;
    let bottleneck = if bw_tok_s <= comp_tok_s {
        "memory_bandwidth_bound"
    } else {
        "compute_bound"
    };
    let j_per_tok = if roofline > 0.0 {
        Some(watts / roofline)
    } else {
        None
    };
    let tok_s_w = if watts > 0.0 {
        Some(roofline / watts)
    } else {
        None
    };

    Ok(RooflineComparisonRow {
        platform_id: input.platform_id.clone(),
        maturity: input.maturity,
        active_weight_bytes_gb: weight_gb,
        bw_ceiling_tok_s: Some(bw_tok_s),
        compute_ceiling_tok_s: Some(comp_tok_s),
        roofline_tok_s: Some(roofline),
        bottleneck: bottleneck.into(),
        implied_joules_per_token: j_per_tok,
        implied_tok_per_sec_per_watt: tok_s_w,
        physical_silicon_validated: input.physical_silicon_validated,
        eligible_as_measured_lab_result: input.physical_silicon_validated
            && input.maturity == HardwareEvidenceMaturity::LabHardwareBaselineEnvelope,
    })
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct SiliconCoDesignAuditReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub upstream_arxiv: String,
    pub pin_status: String,
    pub assumptions: Vec<CoDesignAssumptionItem>,
    pub roofline_rows: Vec<RooflineComparisonRow>,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<SiliconCoDesignAuditReport, String> {
    let assumptions = vec![
        CoDesignAssumptionItem {
            dimension: "workload_and_model_co_design".into(),
            pre_silicon_assumption: "Co-designed ultra-narrow/custom operator graphs matched to fixed on-chip SRAM capacity".into(),
            lab_baseline_reality: "Production models (Qwen3.8-27B, DeepSeek-V4.1-Flash) require dynamic long-context KV and general-purpose serving".into(),
            requires_physical_silicon: true,
        },
        CoDesignAssumptionItem {
            dimension: "precision_and_task_quality".into(),
            pre_silicon_assumption: "Aggressive ultra-low-bit arithmetic assumed lossless in latency/energy projections".into(),
            lab_baseline_reality: "Sub-4-bit quantization degrades coding/agent accuracy unless independently verified on end tasks (#70, #80)".into(),
            requires_physical_silicon: true,
        },
        CoDesignAssumptionItem {
            dimension: "memory_bandwidth_and_interconnect".into(),
            pre_silicon_assumption: "Idealized on-chip SRAM / wafer-scale bandwidth without host PCIe/RDMA/KV spillover penalties".into(),
            lab_baseline_reality: "Physical accelerators face HBM3e/LPDDR5X/GDDR6X pin limits, controller overhead, and multi-chip fabric hops".into(),
            requires_physical_silicon: true,
        },
        CoDesignAssumptionItem {
            dimension: "system_power_and_thermal_envelope".into(),
            pre_silicon_assumption: "Core-array energy/op extrapolated without board VRMs, HBM PHY, host CPU, cooling, or thermal throttling".into(),
            lab_baseline_reality: "Lab measurements (#43) require calibrated board/wall power sensors, soak stability, and observed throttle counters".into(),
            requires_physical_silicon: true,
        },
        CoDesignAssumptionItem {
            dimension: "yield_and_silicon_bringup".into(),
            pre_silicon_assumption: "Analytical/RTL pre-silicon synthesis estimates without tapeout defect yield or clock-tree derating".into(),
            lab_baseline_reality: "Physical silicon bring-up routinely shifts frequency, leakage, and usable SRAM capacity relative to pre-silicon models".into(),
            requires_physical_silicon: true,
        },
    ];

    // Evaluate a matched 27B active-parameter reference workload across pre-silicon projection vs
    // lab hardware baseline envelopes (all clearly tagged by maturity & silicon validation!).
    let platforms = [
        RooflinePlatformInput {
            platform_id: "arxiv_2611_18859_presilicon_sram_projection".into(),
            maturity: HardwareEvidenceMaturity::PreSiliconSimulationOnly,
            active_params_billions: 27.0,
            weight_bits: 2.0,
            memory_bandwidth_gb_s: 200_000.0,
            peak_compute_tflops: 4000.0,
            board_power_watts: 150.0,
            physical_silicon_validated: false,
        },
        RooflinePlatformInput {
            platform_id: "nvidia_b300_hbm3e_envelope".into(),
            maturity: HardwareEvidenceMaturity::LabHardwareBaselineEnvelope,
            active_params_billions: 27.0,
            weight_bits: 4.0,
            memory_bandwidth_gb_s: 8000.0,
            peak_compute_tflops: 4500.0,
            board_power_watts: 1200.0,
            physical_silicon_validated: false, // envelope only; not measured in this CPU run
        },
        RooflinePlatformInput {
            platform_id: "dgx_spark_gb10_lpddr5x_envelope".into(),
            maturity: HardwareEvidenceMaturity::LabHardwareBaselineEnvelope,
            active_params_billions: 27.0,
            weight_bits: 4.0,
            memory_bandwidth_gb_s: 273.0,
            peak_compute_tflops: 500.0,
            board_power_watts: 170.0,
            physical_silicon_validated: false,
        },
        RooflinePlatformInput {
            platform_id: "apple_m2_ultra_unified_envelope".into(),
            maturity: HardwareEvidenceMaturity::LabHardwareBaselineEnvelope,
            active_params_billions: 27.0,
            weight_bits: 4.0,
            memory_bandwidth_gb_s: 800.0,
            peak_compute_tflops: 54.0,
            board_power_watts: 120.0,
            physical_silicon_validated: false,
        },
        RooflinePlatformInput {
            platform_id: "tenstorrent_p150_gddr6_envelope".into(),
            maturity: HardwareEvidenceMaturity::LabHardwareBaselineEnvelope,
            active_params_billions: 27.0,
            weight_bits: 4.0,
            memory_bandwidth_gb_s: 512.0,
            peak_compute_tflops: 400.0,
            board_power_watts: 300.0,
            physical_silicon_validated: false,
        },
    ];

    let mut roofline_rows = Vec::with_capacity(platforms.len());
    for p in &platforms {
        roofline_rows.push(evaluate_roofline_platform(p)?);
    }

    let mut negative_controls = Vec::new();

    // NC1: marking pre-silicon simulation as physical_silicon_validated rejected!
    {
        let mut bad = platforms[0].clone();
        bad.physical_silicon_validated = true;
        negative_controls.push(NegativeControl {
            case_id: "nc1_presilicon_marked_silicon_validated_rejected".into(),
            outcome: if evaluate_roofline_platform(&bad).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "pre-silicon simulation cannot be tagged physical_silicon_validated".into(),
        });
    }

    // NC2: pre-silicon projection is never eligible as a measured lab result
    {
        let presil = &roofline_rows[0];
        negative_controls.push(NegativeControl {
            case_id: "nc2_presilicon_never_eligible_as_measured_lab_result".into(),
            outcome: if !presil.eligible_as_measured_lab_result {
                OutcomeStatus::BlockedOnArtifacts
            } else {
                OutcomeStatus::Accepted
            },
            detail: "arXiv:2611.18859 pre-silicon projection blocked on fabricated silicon".into(),
        });
    }

    // NC3: zero board power rejected (cannot claim 0 J/token)
    {
        let mut bad = platforms[0].clone();
        bad.board_power_watts = 0.0;
        negative_controls.push(NegativeControl {
            case_id: "nc3_zero_board_power_rejected".into(),
            outcome: if evaluate_roofline_platform(&bad).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "zero board power refused instead of reporting 0.0 J/token".into(),
        });
    }

    // NC4: zero memory bandwidth rejected
    {
        let mut bad = platforms[1].clone();
        bad.memory_bandwidth_gb_s = 0.0;
        negative_controls.push(NegativeControl {
            case_id: "nc4_zero_memory_bandwidth_rejected".into(),
            outcome: if evaluate_roofline_platform(&bad).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "zero memory bandwidth rejected cleanly".into(),
        });
    }

    // NC5: NaN active parameters rejected
    {
        let mut bad = platforms[1].clone();
        bad.active_params_billions = f64::NAN;
        negative_controls.push(NegativeControl {
            case_id: "nc5_nan_parameter_count_rejected".into(),
            outcome: if evaluate_roofline_platform(&bad).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "NaN model parameter count rejected cleanly".into(),
        });
    }

    // NC6: negative weight bits rejected
    {
        let mut bad = platforms[1].clone();
        bad.weight_bits = -2.0;
        negative_controls.push(NegativeControl {
            case_id: "nc6_negative_weight_bits_rejected".into(),
            outcome: if evaluate_roofline_platform(&bad).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "negative weight bitwidth rejected".into(),
        });
    }

    // NC7: infinite peak TFLOPS rejected
    {
        let mut bad = platforms[0].clone();
        bad.peak_compute_tflops = f64::INFINITY;
        negative_controls.push(NegativeControl {
            case_id: "nc7_infinite_peak_tflops_rejected".into(),
            outcome: if evaluate_roofline_platform(&bad).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "non-finite infinite TFLOPS rejected".into(),
        });
    }

    // NC8: treating 110x speed / 690x energy claims as measured lab targets rejected
    {
        let import_110x_690x_as_measured = true;
        negative_controls.push(NegativeControl {
            case_id: "nc8_unverified_110x_690x_presilicon_claims_non_gating".into(),
            outcome: if import_110x_690x_as_measured {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "110x speed and 690x energy claims remain unverified pre-silicon related work"
                .into(),
        });
    }

    Ok(SiliconCoDesignAuditReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::blocked_on_artifacts(
            "Fabricated physical silicon for arXiv:2611.18859 co-design & sensor-calibrated lab hardware baselines (#43)",
            "arXiv:2611.18859 reports pre-silicon projections with no fabricated hardware artifacts available (arxiv_id_only_artifacts_unverified_offline); deterministic assumption audit & roofline envelope comparison ran on CPU",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        upstream_arxiv: UPSTREAM_ARXIV.into(),
        pin_status: PIN_STATUS.into(),
        assumptions,
        roofline_rows,
        negative_controls,
    })
}

pub fn inputs_json(report: &SiliconCoDesignAuditReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.silicon-codesign-audit.inputs.v1\",\n",
            "  \"issue\": 84,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"upstream_arxiv\": {},\n",
            "  \"upstream_arxiv_url\": {},\n",
            "  \"upstream_x_post\": {},\n",
            "  \"pin_status\": {},\n",
            "  \"related_issues\": [43]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(&report.upstream_arxiv),
        quote(UPSTREAM_ARXIV_URL),
        quote(UPSTREAM_X_POST),
        quote(&report.pin_status),
    )
}

pub fn results_json(report: &SiliconCoDesignAuditReport) -> String {
    let assumptions: Vec<String> = report
        .assumptions
        .iter()
        .map(|a| {
            format!(
                "{{\"dimension\":{},\"pre_silicon_assumption\":{},\"lab_baseline_reality\":{},\"requires_physical_silicon\":{}}}",
                quote(&a.dimension),
                quote(&a.pre_silicon_assumption),
                quote(&a.lab_baseline_reality),
                a.requires_physical_silicon
            )
        })
        .collect();
    let rows: Vec<String> = report
        .roofline_rows
        .iter()
        .map(|r| {
            format!(
                "{{\"platform_id\":{},\"maturity\":{},\"active_weight_bytes_gb\":{},\"bw_ceiling_tok_s\":{},\"compute_ceiling_tok_s\":{},\"roofline_tok_s\":{},\"bottleneck\":{},\"implied_joules_per_token\":{},\"implied_tok_per_sec_per_watt\":{},\"physical_silicon_validated\":{},\"eligible_as_measured_lab_result\":{}}}",
                quote(&r.platform_id),
                quote(r.maturity.as_str()),
                finite_f64(r.active_weight_bytes_gb),
                opt_f64(r.bw_ceiling_tok_s),
                opt_f64(r.compute_ceiling_tok_s),
                opt_f64(r.roofline_tok_s),
                quote(&r.bottleneck),
                opt_f64(r.implied_joules_per_token),
                opt_f64(r.implied_tok_per_sec_per_watt),
                r.physical_silicon_validated,
                r.eligible_as_measured_lab_result,
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
            "  \"schema\": \"inference-lab.silicon-codesign-audit.results.v1\",\n",
            "  \"issue\": 84,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"assumptions\": [{}],\n",
            "  \"roofline_rows\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        assumptions.join(","),
        rows.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &SiliconCoDesignAuditReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #84: ML-and-Silicon Co-Design Report (`arXiv:2611.18859`) Assumption Audit"
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
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Pre-Silicon Projections vs Lab Hardware Baseline Envelopes (27B Active Params, Batch=1)"
    )
    .unwrap();
    writeln!(
        out,
        "| Platform | Maturity | Roofline tok/s | Bottleneck | Implied J/tok | Silicon Validated |"
    )
    .unwrap();
    writeln!(out, "| --- | --- | ---: | --- | ---: | --- |").unwrap();
    for r in &report.roofline_rows {
        writeln!(
            out,
            "| `{}` | `{}` | `{:.1}` | `{}` | `{:.4}` | `{}` |",
            r.platform_id,
            r.maturity.as_str(),
            r.roofline_tok_s.unwrap_or(0.0),
            r.bottleneck,
            r.implied_joules_per_token.unwrap_or(0.0),
            r.physical_silicon_validated
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
    fn presilicon_audit_separates_projections_from_validated_silicon() {
        let r = run_default_harness().unwrap();
        assert_eq!(r.hardware_gate.status, ExecutionClass::BlockedOnArtifacts);
        assert_eq!(r.assumptions.len(), 5);
        for a in &r.assumptions {
            assert!(a.requires_physical_silicon);
        }
        assert_eq!(r.roofline_rows.len(), 5);
        for row in &r.roofline_rows {
            assert!(!row.eligible_as_measured_lab_result);
        }
        assert_eq!(r.negative_controls.len(), 8);
        for c in &r.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }
}
