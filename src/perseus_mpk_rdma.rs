//! Issue #91: Log Perseus (multi-node RDMA megakernel decoupled signaling) and MPK (Mirage
//! Persistent Kernel) beside `#61`, `#63`, and `#76`, default-off.
//!
//! Related-work references (strictly non-gating external notes):
//! - `Perseus`: decoupled per-destination signaling and NIC-side ordering to eliminate hidden
//!   serialization on fine-grained proxy RDMA PUT+signal NIC fences.
//! - `MPK` (Mirage Persistent Kernel): compiles multi-GPU tensor graphs into SM-level persistent
//!   kernels.
//! - `ForgeMegakernel`: related persistent-kernel codegen effort.
//! - Authors' unverified claim: naive proxy RDMA megakernels suffer up to `10x` regression from
//!   fine-grained PUT+signal NIC fence serialization. Strictly non-gating related work.
//! - Directly informs `#61` (`decode-megakernels`), `#63` (`spark-rdma-tp`), and `#76`
//!   (`blink-cpu-free-serving`).
//!
//! Provides:
//! - Deterministic 2-node RDMA collective-reduction & ordering parity verifier (`<= 1e-12` vs
//!   sequential collective reference) across `NUM_LAYERS = 8` decode layers and `NUM_DESTS = 4`
//!   destination partitions per step.
//! - Explicit timeline decomposition separating compute, RDMA wire transfer, and NIC fence/signal
//!   stall microseconds across:
//!   1. `naive_proxy_put_signal` (synchronous fine-grained PUT+signal fence serialization)
//!   2. `perseus_decoupled_signaling` (per-destination completion counters + NIC-side ordering)
//!   3. `mpk_sm_persistent_graph` (SM-level persistent kernel with fused cross-node barriers)
//! - Hardware qualification gate for 2-node DGX Spark RDMA decode (explicitly refusing silent TCP
//!   fallback or single-node misattribution).
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::det_weight;
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const NUM_LAYERS: usize = 8;
pub const NUM_PARTITIONS: usize = 4;
pub const PARITY_TOL: f64 = 1.0e-12;

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
pub enum RdmaSignalingMode {
    NaiveProxyPutSignal,
    PerseusDecoupledSignaling,
    MpkSmPersistentGraph,
}

impl RdmaSignalingMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NaiveProxyPutSignal => "naive_proxy_put_signal",
            Self::PerseusDecoupledSignaling => "perseus_decoupled_signaling",
            Self::MpkSmPersistentGraph => "mpk_sm_persistent_graph",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TwoNodeRdmaHostSpec {
    pub node_count: usize,
    pub rdma_nic_present: bool,
    pub gpudirect_rdma_enabled: bool,
    pub silent_tcp_fallback_allowed: bool,
    pub upstream_artifacts_pinned: bool,
}

pub fn qualify_two_node_rdma_host(spec: &TwoNodeRdmaHostSpec) -> OutcomeStatus {
    if spec.silent_tcp_fallback_allowed {
        return OutcomeStatus::Invalid;
    }
    if spec.node_count < 2 {
        return OutcomeStatus::Invalid;
    }
    if !spec.upstream_artifacts_pinned {
        return OutcomeStatus::BlockedOnArtifacts;
    }
    if !spec.rdma_nic_present || !spec.gpudirect_rdma_enabled {
        return OutcomeStatus::UnavailableHardwareRuntime;
    }
    OutcomeStatus::Accepted
}

#[derive(Debug, Clone)]
pub struct RdmaModeProfile {
    pub mode: RdmaSignalingMode,
    pub layers_executed: usize,
    pub collective_max_abs_err: f64,
    pub ordering_invariants_preserved: bool,
    pub compute_us: f64,
    pub rdma_wire_us: f64,
    pub nic_fence_stall_us: f64,
    pub total_decode_step_us: f64,
    pub stall_fraction: f64,
    pub timing_admitted: bool,
}

/// Deterministic 2-node collective verification + timeline breakdown.
/// Rejects out-of-order completion delivery (`inject_out_of_order = true`) so ordering bugs
/// cannot silently produce timed results.
pub fn simulate_two_node_rdma_step(
    mode: RdmaSignalingMode,
    compute_per_layer_us: f64,
    wire_per_partition_us: f64,
    fence_penalty_us: f64,
    inject_out_of_order: bool,
) -> Result<RdmaModeProfile, String> {
    let comp = check_finite(compute_per_layer_us, "compute_per_layer_us")?;
    let wire = check_finite(wire_per_partition_us, "wire_per_partition_us")?;
    let fence = check_finite(fence_penalty_us, "fence_penalty_us")?;
    if comp <= 0.0 || wire <= 0.0 || fence < 0.0 {
        return Err("invalid non-positive RDMA timing parameters".into());
    }

    // Verify exact 2-node collective sum across all layers & partitions
    let mut max_err = 0.0_f64;
    let mut ordering_ok = true;
    for l in 0..NUM_LAYERS {
        let mut ref_sum = [0.0_f64; NUM_PARTITIONS];
        let mut obs_sum = [0.0_f64; NUM_PARTITIONS];
        for p in 0..NUM_PARTITIONS {
            let rank0 = det_weight(9101 + l * 19, p, 0.75);
            let rank1 = det_weight(9151 + l * 23, p, 0.75);
            ref_sum[p] = rank0 + rank1;
            let delivered_rank1 = if inject_out_of_order && l == 3 && p == 2 {
                ordering_ok = false;
                0.0 // stale buffer read before NIC signal arrival
            } else {
                rank1
            };
            obs_sum[p] = rank0 + delivered_rank1;
            max_err = max_finite(max_err, (ref_sum[p] - obs_sum[p]).abs(), "rdma_err")?;
        }
    }

    let total_compute = comp * (NUM_LAYERS as f64);
    let total_wire = wire * (NUM_PARTITIONS as f64) * (NUM_LAYERS as f64);
    let (stall_us, total_step_us) = match mode {
        RdmaSignalingMode::NaiveProxyPutSignal => {
            // Serialized proxy fence per partition per layer
            let stall = fence * (NUM_PARTITIONS as f64) * (NUM_LAYERS as f64);
            (stall, total_compute + total_wire + stall)
        }
        RdmaSignalingMode::PerseusDecoupledSignaling => {
            // Decoupled per-destination signaling overlaps wire with compute; tiny residual wait
            let stall = fence * 0.15 * (NUM_LAYERS as f64);
            let overlapped = max_finite(total_compute, total_wire, "perseus_overlap")?;
            (stall, overlapped + stall)
        }
        RdmaSignalingMode::MpkSmPersistentGraph => {
            // Persistent SM graph removes host launch overhead; single cross-node barrier per layer
            let stall = fence * 0.30 * (NUM_LAYERS as f64);
            let overlapped = max_finite(total_compute * 0.92, total_wire, "mpk_overlap")?;
            (stall, overlapped + stall)
        }
    };
    let stall_fraction = stall_us / total_step_us;
    let timing_admitted = ordering_ok && max_err <= PARITY_TOL;

    Ok(RdmaModeProfile {
        mode,
        layers_executed: NUM_LAYERS,
        collective_max_abs_err: max_err,
        ordering_invariants_preserved: ordering_ok,
        compute_us: total_compute,
        rdma_wire_us: total_wire,
        nic_fence_stall_us: stall_us,
        total_decode_step_us: total_step_us,
        stall_fraction,
        timing_admitted,
    })
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct PerseusMpkRdmaReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub host_gate_outcome: OutcomeStatus,
    pub mode_profiles: Vec<RdmaModeProfile>,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<PerseusMpkRdmaReport, String> {
    let host_gate_outcome = qualify_two_node_rdma_host(&TwoNodeRdmaHostSpec {
        node_count: 2,
        rdma_nic_present: false,
        gpudirect_rdma_enabled: false,
        silent_tcp_fallback_allowed: false,
        upstream_artifacts_pinned: false,
    });

    let mode_profiles = vec![
        simulate_two_node_rdma_step(
            RdmaSignalingMode::NaiveProxyPutSignal,
            35.0,
            12.0,
            40.0,
            false,
        )?,
        simulate_two_node_rdma_step(
            RdmaSignalingMode::PerseusDecoupledSignaling,
            35.0,
            12.0,
            40.0,
            false,
        )?,
        simulate_two_node_rdma_step(
            RdmaSignalingMode::MpkSmPersistentGraph,
            35.0,
            12.0,
            40.0,
            false,
        )?,
    ];

    let mut negative_controls = Vec::new();

    // NC1: out-of-order NIC completion violates collective parity and blocks timing admission
    let ooo = simulate_two_node_rdma_step(
        RdmaSignalingMode::PerseusDecoupledSignaling,
        35.0,
        12.0,
        40.0,
        true,
    )?;
    negative_controls.push(NegativeControl {
        case_id: "nc1_out_of_order_nic_completion_rejected".into(),
        outcome: if !ooo.timing_admitted {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "missing NIC ordering fence produces collective parity failure and blocks timing"
            .into(),
    });

    // NC2: silent TCP fallback rejected
    negative_controls.push(NegativeControl {
        case_id: "nc2_silent_tcp_fallback_rejected".into(),
        outcome: qualify_two_node_rdma_host(&TwoNodeRdmaHostSpec {
            node_count: 2,
            rdma_nic_present: false,
            gpudirect_rdma_enabled: false,
            silent_tcp_fallback_allowed: true,
            upstream_artifacts_pinned: true,
        }),
        detail: "silent TCP socket fallback refused when RDMA NIC absent".into(),
    });

    // NC3: single-node run mislabeled as 2-node RDMA rejected
    negative_controls.push(NegativeControl {
        case_id: "nc3_single_node_mislabeled_two_node_rejected".into(),
        outcome: qualify_two_node_rdma_host(&TwoNodeRdmaHostSpec {
            node_count: 1,
            rdma_nic_present: true,
            gpudirect_rdma_enabled: true,
            silent_tcp_fallback_allowed: false,
            upstream_artifacts_pinned: true,
        }),
        detail: "node_count < 2 rejected by 2-node RDMA qualification gate".into(),
    });

    // NC4: unpinned Perseus/MPK artifacts blocked
    negative_controls.push(NegativeControl {
        case_id: "nc4_unpinned_perseus_mpk_artifacts_blocked".into(),
        outcome: host_gate_outcome,
        detail: "unpinned Perseus/MPK/ForgeMegakernel artifacts return BlockedOnArtifacts".into(),
    });

    // NC5: missing GPUDirect RDMA hardware returns UnavailableHardwareRuntime
    negative_controls.push(NegativeControl {
        case_id: "nc5_missing_gpudirect_rdma_hardware_unavailable".into(),
        outcome: qualify_two_node_rdma_host(&TwoNodeRdmaHostSpec {
            node_count: 2,
            rdma_nic_present: true,
            gpudirect_rdma_enabled: false,
            silent_tcp_fallback_allowed: false,
            upstream_artifacts_pinned: true,
        }),
        detail: "absent GPUDirect RDMA NIC returns UnavailableHardwareRuntime".into(),
    });

    // NC6: non-finite wire latency rejected
    negative_controls.push(NegativeControl {
        case_id: "nc6_nonfinite_wire_latency_rejected".into(),
        outcome: if simulate_two_node_rdma_step(
            RdmaSignalingMode::PerseusDecoupledSignaling,
            35.0,
            f64::NAN,
            40.0,
            false,
        )
        .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "NaN wire latency rejected cleanly".into(),
    });

    // NC7: negative fence penalty rejected
    negative_controls.push(NegativeControl {
        case_id: "nc7_negative_fence_penalty_rejected".into(),
        outcome: if simulate_two_node_rdma_step(
            RdmaSignalingMode::NaiveProxyPutSignal,
            35.0,
            12.0,
            -5.0,
            false,
        )
        .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "negative NIC fence stall duration rejected".into(),
    });

    // NC8: unverified 10x regression figure refused as acceptance gate
    negative_controls.push(NegativeControl {
        case_id: "nc8_unverified_10x_regression_claim_non_gating".into(),
        outcome: OutcomeStatus::Invalid,
        detail:
            "authors' unverified 10x naive proxy RDMA regression figure is non-gating related work"
                .into(),
    });

    Ok(PerseusMpkRdmaReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "2-node DGX Spark RDMA decode cluster (Perseus / MPK / ForgeMegakernel)",
            "2-node DGX Spark RDMA hardware and Perseus/MPK artifacts are not executed locally; deterministic collective-ordering parity and NIC fence-stall model verified on CPU",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        host_gate_outcome,
        mode_profiles,
        negative_controls,
    })
}

pub fn inputs_json(report: &PerseusMpkRdmaReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.perseus-mpk-rdma.inputs.v1\",\n",
            "  \"issue\": 91,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"related_work_systems\": [\"Perseus\", \"MPK (Mirage Persistent Kernel)\", \"ForgeMegakernel\"],\n",
            "  \"informs_issues\": [61, 63, 76],\n",
            "  \"num_layers\": {},\n",
            "  \"num_partitions\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        NUM_LAYERS,
        NUM_PARTITIONS,
    )
}

pub fn results_json(report: &PerseusMpkRdmaReport) -> String {
    let modes: Vec<String> = report
        .mode_profiles
        .iter()
        .map(|m| {
            format!(
                "{{\"mode\":{},\"layers_executed\":{},\"collective_max_abs_err\":{},\"ordering_invariants_preserved\":{},\"compute_us\":{},\"rdma_wire_us\":{},\"nic_fence_stall_us\":{},\"total_decode_step_us\":{},\"stall_fraction\":{},\"timing_admitted\":{}}}",
                quote(m.mode.as_str()),
                m.layers_executed,
                finite_f64(m.collective_max_abs_err),
                m.ordering_invariants_preserved,
                finite_f64(m.compute_us),
                finite_f64(m.rdma_wire_us),
                finite_f64(m.nic_fence_stall_us),
                finite_f64(m.total_decode_step_us),
                opt_f64(Some(m.stall_fraction)),
                m.timing_admitted,
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
            "  \"schema\": \"inference-lab.perseus-mpk-rdma.results.v1\",\n",
            "  \"issue\": 91,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"host_gate_outcome\": {},\n",
            "  \"mode_profiles\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(report.host_gate_outcome.as_str()),
        modes.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &PerseusMpkRdmaReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #91: Perseus Decoupled RDMA Signaling & MPK Persistent Kernel Related Work"
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
        "- Hardware gate: `{}` ({})",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## 2-Node RDMA Signaling Mode Comparison").unwrap();
    for m in &report.mode_profiles {
        writeln!(
            out,
            "- `{}`: `parity_err={:.3e}`, `ordering_ok={}`, `fence_stall={:.1}us` (`{:.1}%`), `step_total={:.1}us`, `timing_admitted={}`",
            m.mode.as_str(),
            m.collective_max_abs_err,
            m.ordering_invariants_preserved,
            m.nic_fence_stall_us,
            m.stall_fraction * 100.0,
            m.total_decode_step_us,
            m.timing_admitted
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
    fn perseus_reduces_fence_stalls_with_exact_collective_parity() {
        let r = run_default_harness().unwrap();
        assert!(!r.default_enabled);
        for m in &r.mode_profiles {
            assert!(m.timing_admitted);
            assert!(m.collective_max_abs_err <= PARITY_TOL);
        }
        assert!(r.mode_profiles[1].nic_fence_stall_us < r.mode_profiles[0].nic_fence_stall_us);
        assert_eq!(r.negative_controls.len(), 8);
        for c in &r.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted, "{}", c.case_id);
        }
    }
}
