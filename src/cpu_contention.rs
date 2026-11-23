//! Issue #39: Measure inference sensitivity to host CPU contention and topology-aware placement.
//!
//! Pinned research inputs (strictly unverified non-gating external claims):
//! - Eunomia CPU-pressure / eBPF CPU-noise blog posts (2026-10-17, 2026-05-24): adopt the controlled
//!   question, NEVER transplant hardware-specific CPU masks or speedup claims.
//! - Linux `sched_ext` kernel docs & SchedCP (prior art only; agent scheduler synthesis out of scope;
//!   BPF verifier load != inference SLO).
//! - Blink (`arXiv:2604.07609`, KTH & RISE): unverified claims (up to 8.47x P99 TTFT / 3.40x P99
//!   TPOT); SmartNIC/RDMA intake is a non-goal, while colocated-to-isolated throughput ratio and
//!   P99 TTFT/TPOT per background-load level are reported in comparable form alongside an explicit
//!   workload-reproducibility gap note.
//! - Reuses `#7`/`#8` serving measurements, `#11` disposable lifecycle, and `#40` trace attribution.
//!
//! Provides:
//! - Host topology discovery descriptor (physical/logical cores, asymmetric core classes, NUMA,
//!   NIC/GPU locality) binding requested vs effective CPU masks and affected threads.
//! - Single-factor arm matrix across bounded background CPU loads (`idle`, `moderate`, `heavy`):
//!   `default_placement`, `static_topology_affinity` (affinity only; priority/IRQ strictly separate),
//!   and `host_light_decode` (CUDA graphs / persistent decode without new hardware).
//! - Comparable background-work progress accounting (flagging starvation-based recovery),
//!   Blink-comparable colocated/isolated ratios, strict ITL-naming guard, `#40` causal-attribution
//!   guard, no-benefit/regression retention, conditional `sched_ext` gate, and 8 negative controls.

use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const BLINK_PAPER: &str = "arXiv:2604.07609";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementArm {
    DefaultPlacement,
    StaticTopologyAffinity,
    HostLightDecode,
}

impl PlacementArm {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DefaultPlacement => "default_placement",
            Self::StaticTopologyAffinity => "static_topology_affinity",
            Self::HostLightDecode => "host_light_decode_cuda_graphs",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackgroundLoad {
    Idle,
    Moderate,
    Heavy,
}

impl BackgroundLoad {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Moderate => "moderate_cpu_load",
            Self::Heavy => "heavy_cpu_load",
        }
    }
}

#[derive(Debug, Clone)]
pub struct HostTopologySpec {
    pub host_class: String,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub asymmetric_core_classes: bool,
    pub numa_nodes: usize,
    pub gpu_numa_node: usize,
    pub requested_mask: String,
    pub effective_mask: String,
    pub transplanted_from_blog: bool,
    pub combined_priority_or_irq_unlabeled: bool,
}

pub fn validate_topology_placement(spec: &HostTopologySpec) -> Result<(), String> {
    if spec.transplanted_from_blog {
        return Err(
            "refusing blindly transplanted blog CPU mask; derive from selected host topology"
                .into(),
        );
    }
    if spec.combined_priority_or_irq_unlabeled {
        return Err("affinity, priority, and IRQ interventions must be varied independently in labeled arms".into());
    }
    if spec.requested_mask.is_empty() || spec.effective_mask.is_empty() {
        return Err("requested and effective CPU masks must both be recorded".into());
    }
    if spec.physical_cores == 0 || spec.logical_cores < spec.physical_cores {
        return Err("invalid physical/logical core counts".into());
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct ContentionTrialRow {
    pub arm: PlacementArm,
    pub load: BackgroundLoad,
    pub planned_requests: usize,
    pub valid_requests: usize,
    pub throughput_req_s: f64,
    pub colocated_to_isolated_ratio: Option<f64>,
    pub p99_ttft_ms: f64,
    pub p99_tpot_ms: f64,
    pub host_runnable_delay_ms: f64,
    pub background_work_completed_units: f64,
    pub background_starved: bool,
    pub outcome_note: String,
}

#[derive(Debug, Clone)]
pub struct AttributionAudit {
    pub claims_gpu_idle_from_cuda_api_or_irq_alone: bool,
    pub relabeled_mean_itl_as_token_gap_percentile: bool,
    pub sched_ext_bpf_verifier_claimed_as_slo: bool,
    pub traced_vs_untraced_compared: bool,
}

pub fn validate_attribution_audit(a: &AttributionAudit) -> (OutcomeStatus, String) {
    if a.claims_gpu_idle_from_cuda_api_or_irq_alone {
        return (
            OutcomeStatus::Invalid,
            "CUDA API duration / off-CPU / IRQ overlap alone cannot establish GPU idle or causal stall"
                .into(),
        );
    }
    if a.relabeled_mean_itl_as_token_gap_percentile {
        return (
            OutcomeStatus::Invalid,
            "per-call mean ITL must never be relabeled as a within-stream token-gap percentile"
                .into(),
        );
    }
    if a.sched_ext_bpf_verifier_claimed_as_slo {
        return (
            OutcomeStatus::Invalid,
            "successful BPF verifier load for sched_ext is not an inference SLO result".into(),
        );
    }
    if !a.traced_vs_untraced_compared {
        return (
            OutcomeStatus::Invalid,
            "traced and untraced runs must be compared to bound observer overhead".into(),
        );
    }
    (OutcomeStatus::Accepted, "attribution audit passed".into())
}

pub fn compute_colocated_ratio(colocated: f64, isolated: f64) -> Result<Option<f64>, String> {
    if !colocated.is_finite() || !isolated.is_finite() || colocated < 0.0 || isolated < 0.0 {
        return Err("non-finite or negative throughput".into());
    }
    if isolated == 0.0 {
        return Ok(None);
    }
    Ok(Some(colocated / isolated))
}

pub fn validate_background_fairness(
    expected_bg_units: f64,
    completed_bg_units: f64,
    min_progress_ratio: f64,
) -> Result<OutcomeStatus, String> {
    if !expected_bg_units.is_finite()
        || !completed_bg_units.is_finite()
        || !min_progress_ratio.is_finite()
        || expected_bg_units < 0.0
        || completed_bg_units < 0.0
    {
        return Err("non-finite or negative background work units".into());
    }
    if expected_bg_units > 0.0 && completed_bg_units < expected_bg_units * min_progress_ratio {
        Ok(OutcomeStatus::Invalid)
    } else {
        Ok(OutcomeStatus::Accepted)
    }
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct CpuContentionReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub synthetic_contract_fixture: bool,
    pub topology: HostTopologySpec,
    pub blink_interference_workload_reproducible: bool,
    pub blink_gap_note: String,
    pub sched_ext_followon_triggered: bool,
    pub trials: Vec<ContentionTrialRow>,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<CpuContentionReport, String> {
    let topology = HostTopologySpec {
        host_class: "dgx-spark-gb10-arm64-template".into(),
        physical_cores: 20,
        logical_cores: 20,
        asymmetric_core_classes: true, // Cortex-X925 + Cortex-A725 asymmetric classes
        numa_nodes: 1,
        gpu_numa_node: 0,
        requested_mask: "0-7".into(),
        effective_mask: "0-7".into(),
        transplanted_from_blog: false,
        combined_priority_or_irq_unlabeled: false,
    };
    validate_topology_placement(&topology)?;

    // Synthetic contract matrix showing both recovery under heavy load AND slight regression
    // when over-restricting cores in idle regime (no cherry-picking only best stressed run!).
    type RawRow<'a> = (
        PlacementArm,
        BackgroundLoad,
        f64,
        f64,
        f64,
        f64,
        f64,
        &'a str,
    );
    let raw_rows: Vec<RawRow<'_>> = vec![
        (
            PlacementArm::DefaultPlacement,
            BackgroundLoad::Idle,
            4.00,
            48.0,
            14.0,
            0.2,
            0.0,
            "idle baseline",
        ),
        (
            PlacementArm::DefaultPlacement,
            BackgroundLoad::Moderate,
            3.10,
            96.0,
            21.0,
            3.8,
            100.0,
            "moderate contention degrades tail",
        ),
        (
            PlacementArm::DefaultPlacement,
            BackgroundLoad::Heavy,
            1.45,
            310.0,
            49.0,
            18.5,
            192.0,
            "heavy contention triggers host launch stall",
        ),
        (
            PlacementArm::StaticTopologyAffinity,
            BackgroundLoad::Idle,
            3.88,
            50.0,
            14.5,
            0.3,
            0.0,
            "retained idle regression from reduced core pool",
        ),
        (
            PlacementArm::StaticTopologyAffinity,
            BackgroundLoad::Moderate,
            3.65,
            62.0,
            16.0,
            1.1,
            96.0,
            "affinity recovers tail with comparable background work",
        ),
        (
            PlacementArm::StaticTopologyAffinity,
            BackgroundLoad::Heavy,
            3.20,
            88.0,
            19.5,
            2.4,
            184.0,
            "affinity limits heavy-load degradation",
        ),
        (
            PlacementArm::HostLightDecode,
            BackgroundLoad::Idle,
            4.05,
            47.0,
            13.5,
            0.1,
            0.0,
            "CUDA-graph host-light idle",
        ),
        (
            PlacementArm::HostLightDecode,
            BackgroundLoad::Moderate,
            3.92,
            54.0,
            14.2,
            0.4,
            98.0,
            "host-light separates per-token host path",
        ),
        (
            PlacementArm::HostLightDecode,
            BackgroundLoad::Heavy,
            3.74,
            68.0,
            15.6,
            0.9,
            190.0,
            "host-light minimizes heavy-load tail impact",
        ),
    ];

    let mut trials = Vec::new();
    for &(arm, load, tput, p99_ttft, p99_tpot, run_delay, bg_work, note) in &raw_rows {
        let iso = raw_rows
            .iter()
            .find(|r| r.0 == arm && r.1 == BackgroundLoad::Idle)
            .map(|r| r.2)
            .unwrap_or(0.0);
        let ratio = compute_colocated_ratio(tput, iso)?;
        let expected_bg = match load {
            BackgroundLoad::Idle => 0.0,
            BackgroundLoad::Moderate => 100.0,
            BackgroundLoad::Heavy => 190.0,
        };
        let background_starved = expected_bg > 0.0 && bg_work < expected_bg * 0.70;
        trials.push(ContentionTrialRow {
            arm,
            load,
            planned_requests: 64,
            valid_requests: 64,
            throughput_req_s: tput,
            colocated_to_isolated_ratio: ratio,
            p99_ttft_ms: p99_ttft,
            p99_tpot_ms: p99_tpot,
            host_runnable_delay_ms: run_delay,
            background_work_completed_units: bg_work,
            background_starved,
            outcome_note: note.into(),
        });
    }

    let mut negative_controls = Vec::new();

    // NC1: transplanted blog CPU mask rejected
    {
        let mut bad = topology.clone();
        bad.transplanted_from_blog = true;
        negative_controls.push(NegativeControl {
            case_id: "nc1_transplanted_blog_cpu_mask_rejected".into(),
            outcome: if validate_topology_placement(&bad).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "blog-specific CPU masks cannot be transplanted across hosts".into(),
        });
    }

    // NC2: unlabeled combined affinity + priority/IRQ rejected
    {
        let mut bad = topology.clone();
        bad.combined_priority_or_irq_unlabeled = true;
        negative_controls.push(NegativeControl {
            case_id: "nc2_unlabeled_combined_affinity_priority_irq".into(),
            outcome: if validate_topology_placement(&bad).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "affinity, priority, and IRQ must be separated into distinct labeled arms"
                .into(),
        });
    }

    // NC3: CUDA API / IRQ overlap claimed as GPU idle stall
    {
        let (st, detail) = validate_attribution_audit(&AttributionAudit {
            claims_gpu_idle_from_cuda_api_or_irq_alone: true,
            relabeled_mean_itl_as_token_gap_percentile: false,
            sched_ext_bpf_verifier_claimed_as_slo: false,
            traced_vs_untraced_compared: true,
        });
        negative_controls.push(NegativeControl {
            case_id: "nc3_cuda_api_irq_not_causal_gpu_idle".into(),
            outcome: st,
            detail,
        });
    }

    // NC4: per-call mean ITL relabeled as token-gap percentile
    {
        let (st, detail) = validate_attribution_audit(&AttributionAudit {
            claims_gpu_idle_from_cuda_api_or_irq_alone: false,
            relabeled_mean_itl_as_token_gap_percentile: true,
            sched_ext_bpf_verifier_claimed_as_slo: false,
            traced_vs_untraced_compared: true,
        });
        negative_controls.push(NegativeControl {
            case_id: "nc4_mean_itl_relabeled_as_token_gap_percentile".into(),
            outcome: st,
            detail,
        });
    }

    // NC5: sched_ext BPF verifier load claimed as SLO pass
    {
        let (st, detail) = validate_attribution_audit(&AttributionAudit {
            claims_gpu_idle_from_cuda_api_or_irq_alone: false,
            relabeled_mean_itl_as_token_gap_percentile: false,
            sched_ext_bpf_verifier_claimed_as_slo: true,
            traced_vs_untraced_compared: true,
        });
        negative_controls.push(NegativeControl {
            case_id: "nc5_bpf_verifier_load_not_inference_slo".into(),
            outcome: st,
            detail,
        });
    }

    // NC6: missing traced-vs-untraced observer overhead comparison
    {
        let (st, detail) = validate_attribution_audit(&AttributionAudit {
            claims_gpu_idle_from_cuda_api_or_irq_alone: false,
            relabeled_mean_itl_as_token_gap_percentile: false,
            sched_ext_bpf_verifier_claimed_as_slo: false,
            traced_vs_untraced_compared: false,
        });
        negative_controls.push(NegativeControl {
            case_id: "nc6_missing_traced_vs_untraced_overhead_check".into(),
            outcome: st,
            detail,
        });
    }

    // NC7: recovery obtained by starving competing background workload flagged
    {
        let st = validate_background_fairness(190.0, 15.0, 0.70)?;
        negative_controls.push(NegativeControl {
            case_id: "nc7_background_starvation_recovery_flagged".into(),
            outcome: st,
            detail: "recovery achieved by starving background work (<70% progress) is flagged"
                .into(),
        });
    }

    // NC8: NaN throughput rejected cleanly
    {
        let res = compute_colocated_ratio(f64::NAN, 4.0);
        negative_controls.push(NegativeControl {
            case_id: "nc8_nonfinite_throughput_rejected".into(),
            outcome: if res.is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "NaN throughput fails colocated/isolated ratio computation".into(),
        });
    }

    Ok(CpuContentionReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "Physical Linux / DGX Spark host CPU contention & GPU serving trials",
            "Topology validation, single-factor arm matrix, Blink ratio schema, and attribution guards verified on CPU; physical contention runs did not execute locally",
        ),
        synthetic_contract_fixture: true,
        topology,
        blink_interference_workload_reproducible: false,
        blink_gap_note: "Blink (arXiv:2604.07609) Fig. 1 interference workload & CPU budget are not fully specified for exact reproduction; gap recorded rather than invented".into(),
        sched_ext_followon_triggered: false,
        trials,
        negative_controls,
    })
}

pub fn inputs_json(report: &CpuContentionReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.cpu-contention.inputs.v1\",\n",
            "  \"issue\": 39,\n",
            "  \"execution_class\": {},\n",
            "  \"host_class\": {},\n",
            "  \"physical_cores\": {},\n",
            "  \"asymmetric_core_classes\": {},\n",
            "  \"requested_mask\": {},\n",
            "  \"effective_mask\": {},\n",
            "  \"blink_reference\": {},\n",
            "  \"blink_interference_workload_reproducible\": {},\n",
            "  \"synthetic_contract_fixture\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(&report.topology.host_class),
        report.topology.physical_cores,
        report.topology.asymmetric_core_classes,
        quote(&report.topology.requested_mask),
        quote(&report.topology.effective_mask),
        quote(BLINK_PAPER),
        report.blink_interference_workload_reproducible,
        report.synthetic_contract_fixture,
    )
}

pub fn results_json(report: &CpuContentionReport) -> String {
    let trials: Vec<String> = report
        .trials
        .iter()
        .map(|t| {
            format!(
                "{{\"arm\":{},\"load\":{},\"planned_requests\":{},\"valid_requests\":{},\"throughput_req_s\":{},\"colocated_to_isolated_ratio\":{},\"p99_ttft_ms\":{},\"p99_tpot_ms\":{},\"host_runnable_delay_ms\":{},\"background_work_completed\":{},\"background_starved\":{},\"note\":{}}}",
                quote(t.arm.as_str()),
                quote(t.load.as_str()),
                t.planned_requests,
                t.valid_requests,
                finite_f64(t.throughput_req_s),
                opt_f64(t.colocated_to_isolated_ratio),
                finite_f64(t.p99_ttft_ms),
                finite_f64(t.p99_tpot_ms),
                finite_f64(t.host_runnable_delay_ms),
                finite_f64(t.background_work_completed_units),
                t.background_starved,
                quote(&t.outcome_note),
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
            "  \"schema\": \"inference-lab.cpu-contention.results.v1\",\n",
            "  \"issue\": 39,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"blink_interference_workload_reproducible\": {},\n",
            "  \"blink_gap_note\": {},\n",
            "  \"sched_ext_followon_triggered\": {},\n",
            "  \"trials\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.synthetic_contract_fixture,
        report.blink_interference_workload_reproducible,
        quote(&report.blink_gap_note),
        report.sched_ext_followon_triggered,
        trials.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &CpuContentionReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #39: Host CPU Contention & Topology-Aware Placement"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Execution class: `{}` (`synthetic_contract_fixture: {}`)",
        report.execution_class.as_str(),
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
    writeln!(
        out,
        "- Blink workload reproducible from paper: `{}` ({})",
        report.blink_interference_workload_reproducible, report.blink_gap_note
    )
    .unwrap();
    writeln!(
        out,
        "- Conditional `sched_ext` follow-on triggered: `{}`",
        report.sched_ext_followon_triggered
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Synthetic Contract Contention Matrix (Blink-Comparable Form)"
    )
    .unwrap();
    for t in &report.trials {
        writeln!(
            out,
            "- `{}` @ `{}`: `ratio={:.3}`, `p99_ttft={:.1}ms`, `p99_tpot={:.1}ms`, `bg_work={:.0}` (`starved={}`) — {}",
            t.arm.as_str(),
            t.load.as_str(),
            t.colocated_to_isolated_ratio.unwrap_or(0.0),
            t.p99_ttft_ms,
            t.p99_tpot_ms,
            t.background_work_completed_units,
            t.background_starved,
            t.outcome_note
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
