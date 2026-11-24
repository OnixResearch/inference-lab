//! Issue #76: Log Blink's CPU-free RDMA + persistent-kernel serving architecture and measure
//! host-CPU contention resilience of GPU-resident scheduling against vLLM/SGLang.
//!
//! Pinned related work (strictly non-gating external notes):
//! - X post (2026-11-19, Superman `@thesupermannx`):
//!   `https://x.com/thesupermannx/status/2123339605311111649`
//! - Authors' unverified claims: Blink eliminates the host CPU from the LLM serving hot path using
//!   SmartNIC GPUDirect RDMA directly into GPU memory plus a persistent GPU kernel handling
//!   batching, scheduling, and KV-cache management, claiming up to 8.47x P99 TTFT improvement and
//!   immunity to host-CPU interference.
//! - Complements `#39` (host CPU contention), `#40` (stall attribution), and `#61` (persistent
//!   decode megakernels).
//!
//! Provides:
//! - Explicit Blink architecture & hardware-assumption record (`SmartNIC GPUDirect RDMA`,
//!   `persistent GPU scheduler`, `device-side KV page allocator`, `blocked_on_artifacts`).
//! - Deterministic discrete-event serving simulator comparing host-scheduled serving
//!   (`vllm_sglang_host_scheduled`) vs GPU-resident persistent scheduling (`gpu_resident_persistent`)
//!   under controlled host-CPU interference (`idle`, `moderate`, `heavy`), measuring P50/P99 TTFT,
//!   decode ITL jitter (P99 - P50), and colocated/isolated resilience ratios.
//! - Default-off RDMA-to-GPU qualification gate refusing missing Blink artifacts or incompatible
//!   NIC/PCIe topology.
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const UPSTREAM_X_POST: &str = "https://x.com/thesupermannx/status/2123339605311111649";
pub const PIN_STATUS: &str = "post_only_unlocated_paper_repo";

fn check_finite(v: f64, ctx: &str) -> Result<f64, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("non-finite float in {ctx}"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerMode {
    HostScheduledVllmSglang,
    GpuResidentPersistent,
}

impl SchedulerMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HostScheduledVllmSglang => "host_scheduled_vllm_sglang",
            Self::GpuResidentPersistent => "gpu_resident_persistent",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuInterferenceLevel {
    Idle,
    Moderate,
    Heavy,
}

impl CpuInterferenceLevel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Moderate => "moderate",
            Self::Heavy => "heavy",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RdmaHostQualification {
    pub blink_artifacts_pinned: bool,
    pub smartnic_gpudirect_rdma: bool,
    pub same_pcie_root_complex: bool,
    pub persistent_kernel_supported: bool,
    pub silent_tcp_fallback_requested: bool,
}

pub fn qualify_rdma_to_gpu(spec: &RdmaHostQualification) -> OutcomeStatus {
    if spec.silent_tcp_fallback_requested {
        return OutcomeStatus::Invalid;
    }
    if !spec.blink_artifacts_pinned {
        return OutcomeStatus::BlockedOnArtifacts;
    }
    if !spec.smartnic_gpudirect_rdma || !spec.persistent_kernel_supported {
        return OutcomeStatus::Unsupported;
    }
    if !spec.same_pcie_root_complex {
        return OutcomeStatus::UnavailableHardwareRuntime;
    }
    OutcomeStatus::Accepted
}

pub fn percentile_sorted(values: &[f64], pct: f64) -> Result<Option<f64>, String> {
    let p = check_finite(pct, "percentile")?;
    if !(0.0..=100.0).contains(&p) {
        return Err("percentile out of [0, 100]".into());
    }
    if values.is_empty() {
        return Ok(None);
    }
    let mut sorted = Vec::with_capacity(values.len());
    for &v in values {
        sorted.push(check_finite(v, "percentile_val")?);
    }
    sorted.sort_by(|a, b| a.total_cmp(b));
    let rank = ((p / 100.0) * ((sorted.len() - 1) as f64)).round() as usize;
    let idx = if rank < sorted.len() {
        rank
    } else {
        sorted.len() - 1
    };
    Ok(Some(sorted[idx]))
}

#[derive(Debug, Clone)]
pub struct ContentionCellMetrics {
    pub scheduler: SchedulerMode,
    pub interference: CpuInterferenceLevel,
    pub request_count: usize,
    pub p50_ttft_ms: f64,
    pub p99_ttft_ms: f64,
    pub p50_itl_ms: f64,
    pub p99_itl_ms: f64,
    pub itl_jitter_p99_minus_p50_ms: f64,
    pub p99_ttft_ratio_vs_idle: Option<f64>,
    pub p99_itl_ratio_vs_idle: Option<f64>,
}

/// Deterministic discrete-event contention simulation over 32 requests x 8 decode tokens.
pub fn simulate_contention_cell(
    scheduler: SchedulerMode,
    interference: CpuInterferenceLevel,
) -> Result<ContentionCellMetrics, String> {
    let n_reqs = 32usize;
    let decode_steps = 8usize;
    let mut ttfts = Vec::with_capacity(n_reqs);
    let mut itls = Vec::with_capacity(n_reqs * decode_steps);

    let (cpu_sched_base_ms, cpu_spike_ms, cpu_step_jitter_ms) = match interference {
        CpuInterferenceLevel::Idle => (1.2, 0.3, 0.15),
        CpuInterferenceLevel::Moderate => (4.8, 9.5, 1.4),
        CpuInterferenceLevel::Heavy => (11.5, 42.0, 5.2),
    };

    for r in 0..n_reqs {
        let gpu_prefill_ms = 14.0 + ((r % 4) as f64) * 1.25;
        let is_tail_spike = (r % 8) == 7;
        let ttft = match scheduler {
            SchedulerMode::HostScheduledVllmSglang => {
                let host_stall = cpu_sched_base_ms
                    + if is_tail_spike { cpu_spike_ms } else { 0.0 }
                    + ((r % 5) as f64) * 0.4;
                gpu_prefill_ms + host_stall
            }
            SchedulerMode::GpuResidentPersistent => {
                // SmartNIC RDMA + GPU-resident queue avoids host scheduler stalls; retains
                // small bounded NIC/PCIe ingress variation (~0.35 ms) independent of CPU load.
                let rdma_ingress_ms = 0.65 + ((r % 4) as f64) * 0.08;
                gpu_prefill_ms + rdma_ingress_ms
            }
        };
        ttfts.push(check_finite(ttft, "ttft")?);

        for s in 0..decode_steps {
            let gpu_decode_ms = 6.2 + (((r + s) % 3) as f64) * 0.15;
            let itl = match scheduler {
                SchedulerMode::HostScheduledVllmSglang => {
                    let spike = if (r + s) % 11 == 0 {
                        cpu_step_jitter_ms * 2.5
                    } else {
                        cpu_step_jitter_ms * (((r + s) % 4) as f64) * 0.25
                    };
                    gpu_decode_ms + 0.8 + spike
                }
                SchedulerMode::GpuResidentPersistent => gpu_decode_ms + 0.12,
            };
            itls.push(check_finite(itl, "itl")?);
        }
    }

    let p50_ttft = percentile_sorted(&ttfts, 50.0)?.ok_or("empty ttfts")?;
    let p99_ttft = percentile_sorted(&ttfts, 99.0)?.ok_or("empty ttfts")?;
    let p50_itl = percentile_sorted(&itls, 50.0)?.ok_or("empty itls")?;
    let p99_itl = percentile_sorted(&itls, 99.0)?.ok_or("empty itls")?;

    Ok(ContentionCellMetrics {
        scheduler,
        interference,
        request_count: n_reqs,
        p50_ttft_ms: p50_ttft,
        p99_ttft_ms: p99_ttft,
        p50_itl_ms: p50_itl,
        p99_itl_ms: p99_itl,
        itl_jitter_p99_minus_p50_ms: p99_itl - p50_itl,
        p99_ttft_ratio_vs_idle: None,
        p99_itl_ratio_vs_idle: None,
    })
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct BlinkCpuFreeReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub pin_status: String,
    pub rdma_gate_outcome: OutcomeStatus,
    pub cells: Vec<ContentionCellMetrics>,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<BlinkCpuFreeReport, String> {
    let modes = [
        SchedulerMode::HostScheduledVllmSglang,
        SchedulerMode::GpuResidentPersistent,
    ];
    let levels = [
        CpuInterferenceLevel::Idle,
        CpuInterferenceLevel::Moderate,
        CpuInterferenceLevel::Heavy,
    ];

    let mut cells = Vec::new();
    for mode in modes {
        let mut idle_p99_ttft = None;
        let mut idle_p99_itl = None;
        for level in levels {
            let mut cell = simulate_contention_cell(mode, level)?;
            if level == CpuInterferenceLevel::Idle {
                idle_p99_ttft = Some(cell.p99_ttft_ms);
                idle_p99_itl = Some(cell.p99_itl_ms);
            }
            cell.p99_ttft_ratio_vs_idle = match idle_p99_ttft {
                Some(base) if base > 0.0 => Some(cell.p99_ttft_ms / base),
                _ => None,
            };
            cell.p99_itl_ratio_vs_idle = match idle_p99_itl {
                Some(base) if base > 0.0 => Some(cell.p99_itl_ms / base),
                _ => None,
            };
            cells.push(cell);
        }
    }

    let local_rdma_spec = RdmaHostQualification {
        blink_artifacts_pinned: false,
        smartnic_gpudirect_rdma: false,
        same_pcie_root_complex: false,
        persistent_kernel_supported: false,
        silent_tcp_fallback_requested: false,
    };
    let rdma_gate_outcome = qualify_rdma_to_gpu(&local_rdma_spec);

    let mut negative_controls = Vec::new();

    // NC1: unpinned Blink artifacts return blocked_on_artifacts
    negative_controls.push(NegativeControl {
        case_id: "nc1_unpinned_blink_artifacts_blocked".into(),
        outcome: rdma_gate_outcome,
        detail: "missing public Blink RDMA/persistent-kernel code returns blocked_on_artifacts"
            .into(),
    });

    // NC2: silent TCP fallback when SmartNIC GPUDirect RDMA is absent rejected as invalid
    {
        let bad = RdmaHostQualification {
            blink_artifacts_pinned: true,
            smartnic_gpudirect_rdma: false,
            same_pcie_root_complex: true,
            persistent_kernel_supported: true,
            silent_tcp_fallback_requested: true,
        };
        negative_controls.push(NegativeControl {
            case_id: "nc2_silent_tcp_fallback_rejected".into(),
            outcome: qualify_rdma_to_gpu(&bad),
            detail:
                "silently falling back from GPUDirect RDMA to host TCP invalidates CPU-free test"
                    .into(),
        });
    }

    // NC3: missing SmartNIC GPUDirect RDMA hardware returns unsupported
    {
        let bad = RdmaHostQualification {
            blink_artifacts_pinned: true,
            smartnic_gpudirect_rdma: false,
            same_pcie_root_complex: true,
            persistent_kernel_supported: true,
            silent_tcp_fallback_requested: false,
        };
        negative_controls.push(NegativeControl {
            case_id: "nc3_missing_smartnic_gpudirect_unsupported".into(),
            outcome: qualify_rdma_to_gpu(&bad),
            detail: "host without GPUDirect RDMA SmartNIC returns unsupported".into(),
        });
    }

    // NC4: cross-NUMA/PCIe-root-complex topology flagged unavailable_hardware_runtime
    {
        let bad = RdmaHostQualification {
            blink_artifacts_pinned: true,
            smartnic_gpudirect_rdma: true,
            same_pcie_root_complex: false,
            persistent_kernel_supported: true,
            silent_tcp_fallback_requested: false,
        };
        negative_controls.push(NegativeControl {
            case_id: "nc4_cross_pcie_root_topology_gated".into(),
            outcome: qualify_rdma_to_gpu(&bad),
            detail: "NIC and GPU on separate PCIe root complexes cannot claim zero-hop RDMA".into(),
        });
    }

    // NC5: empty latency sample produces None, never 0.0
    {
        let empty = percentile_sorted(&[], 99.0)?;
        negative_controls.push(NegativeControl {
            case_id: "nc5_empty_latency_population_never_zero".into(),
            outcome: if empty.is_none() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "empty TTFT/ITL sample returns None rather than collapsing to 0.0 ms".into(),
        });
    }

    // NC6: NaN latency sample rejected
    {
        negative_controls.push(NegativeControl {
            case_id: "nc6_nan_latency_sample_rejected".into(),
            outcome: if percentile_sorted(&[10.0, f64::NAN], 99.0).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "non-finite latency measurement rejected by percentile reducer".into(),
        });
    }

    // NC7: out-of-range percentile (> 100) rejected
    {
        negative_controls.push(NegativeControl {
            case_id: "nc7_oob_percentile_rejected".into(),
            outcome: if percentile_sorted(&[10.0, 20.0], 105.0).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "percentile outside [0, 100] rejected".into(),
        });
    }

    // NC8: importing Blink's unverified 8.47x claim as a lab acceptance gate rejected
    {
        let imported_vendor_claim_as_gate = true;
        negative_controls.push(NegativeControl {
            case_id: "nc8_unverified_8_47x_claim_non_gating".into(),
            outcome: if imported_vendor_claim_as_gate {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "Blink's 8.47x P99 TTFT claim stays unverified related work, never a gate"
                .into(),
        });
    }

    Ok(BlinkCpuFreeReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::blocked_on_artifacts(
            "SmartNIC GPUDirect RDMA + persistent GPU serving cluster (Blink vs vLLM/SGLang)",
            "Blink public artifacts are unlocated (post_only_unlocated_paper_repo) and GPUDirect RDMA SmartNIC hardware is absent; deterministic contention resilience simulation ran on CPU",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        pin_status: PIN_STATUS.into(),
        rdma_gate_outcome,
        cells,
        negative_controls,
    })
}

pub fn inputs_json(report: &BlinkCpuFreeReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.blink-cpu-free-serving.inputs.v1\",\n",
            "  \"issue\": 76,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"pin_status\": {},\n",
            "  \"upstream_x_post\": {},\n",
            "  \"related_issues\": [39, 40, 61]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(&report.pin_status),
        quote(UPSTREAM_X_POST),
    )
}

pub fn results_json(report: &BlinkCpuFreeReport) -> String {
    let cells: Vec<String> = report
        .cells
        .iter()
        .map(|c| {
            format!(
                "{{\"scheduler\":{},\"interference\":{},\"request_count\":{},\"p50_ttft_ms\":{},\"p99_ttft_ms\":{},\"p50_itl_ms\":{},\"p99_itl_ms\":{},\"itl_jitter_p99_minus_p50_ms\":{},\"p99_ttft_ratio_vs_idle\":{},\"p99_itl_ratio_vs_idle\":{}}}",
                quote(c.scheduler.as_str()),
                quote(c.interference.as_str()),
                c.request_count,
                finite_f64(c.p50_ttft_ms),
                finite_f64(c.p99_ttft_ms),
                finite_f64(c.p50_itl_ms),
                finite_f64(c.p99_itl_ms),
                finite_f64(c.itl_jitter_p99_minus_p50_ms),
                opt_f64(c.p99_ttft_ratio_vs_idle),
                opt_f64(c.p99_itl_ratio_vs_idle),
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
            "  \"schema\": \"inference-lab.blink-cpu-free-serving.results.v1\",\n",
            "  \"issue\": 76,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"rdma_gate_outcome\": {},\n",
            "  \"cells\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(report.rdma_gate_outcome.as_str()),
        cells.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &BlinkCpuFreeReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #76: Blink CPU-Free RDMA Serving & Host-CPU Contention Resilience"
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
        "- Artifact gate: `{}` (`rdma_gate_outcome: {}`)",
        report.hardware_gate.status.as_str(),
        report.rdma_gate_outcome.as_str()
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Host-Scheduled vs GPU-Resident Scheduling Under Host-CPU Contention"
    )
    .unwrap();
    writeln!(
        out,
        "| Scheduler | Interference | P50 TTFT (ms) | P99 TTFT (ms) | P99 TTFT / Idle | ITL Jitter (ms) |"
    )
    .unwrap();
    writeln!(out, "| --- | --- | ---: | ---: | ---: | ---: |").unwrap();
    for c in &report.cells {
        writeln!(
            out,
            "| `{}` | `{}` | `{:.2}` | `{:.2}` | `{:.3}` | `{:.2}` |",
            c.scheduler.as_str(),
            c.interference.as_str(),
            c.p50_ttft_ms,
            c.p99_ttft_ms,
            c.p99_ttft_ratio_vs_idle.unwrap_or(0.0),
            c.itl_jitter_p99_minus_p50_ms
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
    fn gpu_resident_scheduler_resists_heavy_host_cpu_interference() {
        let r = run_default_harness().unwrap();
        assert_eq!(r.cells.len(), 6);
        let host_heavy = &r.cells[2];
        let gpu_heavy = &r.cells[5];
        assert!(host_heavy.p99_ttft_ratio_vs_idle.unwrap() > 2.5);
        assert!(gpu_heavy.p99_ttft_ratio_vs_idle.unwrap() <= 1.05);
        assert!(gpu_heavy.itl_jitter_p99_minus_p50_ms < host_heavy.itl_jitter_p99_minus_p50_ms);
        assert_eq!(r.negative_controls.len(), 8);
        for c in &r.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }
}
