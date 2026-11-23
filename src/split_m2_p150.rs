//! Issue #42: Qualify M2 Ultra + P150 split-model inference with matched numerical and performance controls.
//!
//! Pinned source basis & ownership map (strictly non-gating upstream notes):
//! - Reuses Inference Lab `#41` (`crate::stage`) versioned transformer stage contract & 4-layer
//!   causal dense reference (`unsplit`, `2-stage`, `3-stage`, full & chunked prefill, forced-token
//!   decode, reset/reuse, zero silent CPU fallback for Apple GPU / Tenstorrent P150).
//! - Hardware & Tenstorrent execution remain with TT-Onix (`#48` Asahi/ARM Linux PCIe gate,
//!   `#49` Darwin DriverKit/Metal gate, `#50` contiguous-block TT partition, `#51` static MoE).
//! - `joesinvestments/apple-nvidia-sidecar@c59a276a1ce5fc3c2e12e05f070a0180550cb619` supplies
//!   concrete MoE exchange reference only (M5 Max/RTX 5090/TB5, zero performance receipts;
//!   inherited Strata files are NOT results for our Mac Pro M2 Ultra + internal PCIe P150 cohort).
//!
//! Provides:
//! - Frozen split profile separating Linux+Asahi vs macOS+Metal, internal-PCIe Mac Pro vs
//!   conventional Linux/network prototypes, distributed Apple unified memory vs P150 GDDR
//!   inventories, and independent `P150x1`, `P150x2`, and `MoE` acceptance verdicts.
//! - Dense split numerical qualification reusing `#41` (`stage::run_default_harness()`) plus an
//!   independent pure-Rust disjoint-expert MoE split-vs-unsplit reference (verifying exact weighted
//!   combination and zero borrowed dense-model pass).
//! - Exact derived logical payload math (`hidden=8192 * 2 bytes = 16 KiB/token`, `4096 rows = 64 MiB`,
//!   explicitly marked `derived_logical_payload_not_measured_throughput: true`), serial latency vs
//!   capacity vs concurrent pipeline distinction, and 8 executable negative controls.

use std::collections::BTreeSet;
use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, softmax_stable};
use crate::json::{finite_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};
use crate::stage;

pub const SIDECAR_UPSTREAM_COMMIT: &str = "c59a276a1ce5fc3c2e12e05f070a0180550cb619";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitHostCohort {
    MacProInternalPcieLinuxAsahi,
    MacProInternalPcieMacOsMetal,
    ConventionalLinuxPrototype,
    NetworkConnectedPrototype,
}

impl SplitHostCohort {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MacProInternalPcieLinuxAsahi => "mac_pro_internal_pcie_linux_asahi",
            Self::MacProInternalPcieMacOsMetal => "mac_pro_internal_pcie_macos_metal",
            Self::ConventionalLinuxPrototype => "conventional_linux_prototype",
            Self::NetworkConnectedPrototype => "network_connected_prototype",
        }
    }

    pub const fn is_internal_pcie_mac_pro(self) -> bool {
        matches!(
            self,
            Self::MacProInternalPcieLinuxAsahi | Self::MacProInternalPcieMacOsMetal
        )
    }
}

#[derive(Debug, Clone)]
pub struct DeviceMemoryInventory {
    pub device_name: String,
    pub weights_bytes: u64,
    pub kv_state_bytes: u64,
    pub allocator_reserve_bytes: u64,
    pub workspace_bytes: u64,
    pub transfer_buffer_bytes: u64,
}

impl DeviceMemoryInventory {
    pub fn total_bytes(&self) -> u64 {
        self.weights_bytes
            + self.kv_state_bytes
            + self.allocator_reserve_bytes
            + self.workspace_bytes
            + self.transfer_buffer_bytes
    }
}

#[derive(Debug, Clone)]
pub struct SplitQualificationClaim {
    pub cohort: SplitHostCohort,
    pub claims_internal_pcie_mac_pro_result: bool,
    pub claims_unified_plus_gddr_coherent_pool: bool,
    pub numerical_and_continuation_passed: bool,
    pub claims_performance_qualification: bool,
    pub quantization_changed_at_boundary: bool,
    pub claims_exact_parity_across_quant_change: bool,
    pub borrows_dense_pass_for_moe: bool,
    pub borrows_p150x1_pass_for_p150x2: bool,
}

pub fn validate_split_claim(c: &SplitQualificationClaim) -> (OutcomeStatus, String) {
    if c.claims_internal_pcie_mac_pro_result && !c.cohort.is_internal_pcie_mac_pro() {
        return (
            OutcomeStatus::Invalid,
            "reference/simulator/network/conventional-Linux prototype cannot substitute for internal-PCIe Mac Pro".into(),
        );
    }
    if c.claims_unified_plus_gddr_coherent_pool {
        return (
            OutcomeStatus::Invalid,
            "Apple unified memory and P150 GDDR are distributed capacity, not one coherent pool"
                .into(),
        );
    }
    if c.claims_performance_qualification && !c.numerical_and_continuation_passed {
        return (
            OutcomeStatus::Invalid,
            "numerical and continuation acceptance must precede any performance claim".into(),
        );
    }
    if c.quantization_changed_at_boundary && c.claims_exact_parity_across_quant_change {
        return (
            OutcomeStatus::Invalid,
            "changed quantization across stage boundary is numerical translation, not exact parity"
                .into(),
        );
    }
    if c.borrows_dense_pass_for_moe {
        return (
            OutcomeStatus::Invalid,
            "MoE split requires independent checkpoint and routing qualification; cannot borrow dense pass".into(),
        );
    }
    if c.borrows_p150x1_pass_for_p150x2 {
        return (
            OutcomeStatus::Invalid,
            "P150x2 requires independent link/parallelism qualification; cannot inherit P150x1 pass".into(),
        );
    }
    (OutcomeStatus::Accepted, "split claim valid".into())
}

/// Exact logical dense boundary payload calculation (derived logical payload, NOT throughput!).
pub fn derived_dense_boundary_payload_bytes(
    hidden_dim: usize,
    bytes_per_elem: usize,
    tokens: usize,
) -> usize {
    hidden_dim * bytes_per_elem * tokens
}

/// Independent pure-Rust disjoint-expert MoE split verification (Apple M2 experts vs P150 experts).
pub fn verify_disjoint_moe_split(
    input: &[f64; 4],
    apple_experts: &[usize],
    tt_experts: &[usize],
    num_experts: usize,
) -> Result<f64, String> {
    if input.iter().any(|x| !x.is_finite()) {
        return Err("non-finite MoE input".into());
    }
    let mut seen = BTreeSet::new();
    for &e in apple_experts.iter().chain(tt_experts.iter()) {
        if e >= num_experts {
            return Err(format!("expert index {e} out of bounds {num_experts}"));
        }
        if !seen.insert(e) {
            return Err(format!(
                "overlapping expert assignment detected for expert {e}"
            ));
        }
    }
    if seen.len() != num_experts {
        return Err(format!(
            "incomplete expert partition: assigned {} of {}",
            seen.len(),
            num_experts
        ));
    }

    // Router logits over all experts
    let mut router_logits = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let mut s = 0.0_f64;
        for (i, &xv) in input.iter().enumerate() {
            s += xv * det_weight(4201 + e * 7, i, 0.65);
        }
        router_logits.push(s);
    }
    let weights = softmax_stable(&router_logits)?;

    let run_expert = |e: usize| -> [f64; 4] {
        let mut out = [0.0_f64; 4];
        for (o, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0_f64;
            for (i, &xv) in input.iter().enumerate() {
                acc += xv * det_weight(4250 + e * 19 + o * 5, i, 0.5);
            }
            *slot = acc.tanh();
        }
        out
    };

    // Unsplit reference
    let mut unsplit = [0.0_f64; 4];
    for (e, &w) in weights.iter().enumerate() {
        let y = run_expert(e);
        for d in 0..4 {
            unsplit[d] += w * y[d];
        }
    }

    // Disjoint split across Apple M2 + P150
    let mut apple_part = [0.0_f64; 4];
    for &e in apple_experts {
        let y = run_expert(e);
        for d in 0..4 {
            apple_part[d] += weights[e] * y[d];
        }
    }
    let mut tt_part = [0.0_f64; 4];
    for &e in tt_experts {
        let y = run_expert(e);
        for d in 0..4 {
            tt_part[d] += weights[e] * y[d];
        }
    }

    let mut max_err = 0.0_f64;
    for d in 0..4 {
        let combined = apple_part[d] + tt_part[d];
        if !combined.is_finite() || !unsplit[d].is_finite() {
            return Err("non-finite MoE split output".into());
        }
        let err = (combined - unsplit[d]).abs();
        if err > max_err {
            max_err = err;
        }
    }
    Ok(max_err)
}

#[derive(Debug, Clone)]
pub struct IndependentProfileVerdict {
    pub profile_id: String,
    pub cohort: SplitHostCohort,
    pub p150_count: usize,
    pub is_moe: bool,
    pub verdict: String,
    pub blocker: String,
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct SplitM2P150Report {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub synthetic_contract_fixture: bool,
    pub stage_contract_reused_checks: usize,
    pub dense_split_max_err: f64,
    pub moe_disjoint_split_max_err: f64,
    pub logical_payload_1tok_bytes: usize,
    pub logical_payload_4096tok_bytes: usize,
    pub derived_logical_payload_not_measured_throughput: bool,
    pub apple_memory_inventory: DeviceMemoryInventory,
    pub p150_memory_inventory: DeviceMemoryInventory,
    pub profile_verdicts: Vec<IndependentProfileVerdict>,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<SplitM2P150Report, String> {
    // Reuse #41 (`crate::stage`) full 4-layer dense transformer split verification!
    let stage_rep = stage::run_default_harness()?;
    let mut dense_split_max_err = 0.0_f64;
    for chk in &stage_rep.parity_checks {
        if !chk.max_logit_abs_error.is_finite() || !chk.max_boundary_abs_error.is_finite() {
            return Err("non-finite stage parity diff".into());
        }
        if chk.max_logit_abs_error > dense_split_max_err {
            dense_split_max_err = chk.max_logit_abs_error;
        }
        if chk.max_boundary_abs_error > dense_split_max_err {
            dense_split_max_err = chk.max_boundary_abs_error;
        }
    }

    // Independent disjoint-expert MoE split verification (6 experts: 3 on Apple M2, 3 on P150)
    let moe_input = [0.45, -0.82, 0.31, 0.67];
    let moe_disjoint_split_max_err =
        verify_disjoint_moe_split(&moe_input, &[0, 2, 4], &[1, 3, 5], 6)?;

    // Exact logical dense boundary payload checks from issue spec
    let logical_payload_1tok_bytes = derived_dense_boundary_payload_bytes(8192, 2, 1); // 16 KiB
    let logical_payload_4096tok_bytes = derived_dense_boundary_payload_bytes(8192, 2, 4096); // 64 MiB

    let apple_memory_inventory = DeviceMemoryInventory {
        device_name: "Apple M2 Ultra Unified Memory (distributed, non-coherent with GDDR)".into(),
        weights_bytes: 24_000_000_000,
        kv_state_bytes: 4_000_000_000,
        allocator_reserve_bytes: 2_000_000_000,
        workspace_bytes: 1_500_000_000,
        transfer_buffer_bytes: 256_000_000,
    };
    let p150_memory_inventory = DeviceMemoryInventory {
        device_name: "Tenstorrent P150 GDDR (card-local)".into(),
        weights_bytes: 20_000_000_000,
        kv_state_bytes: 3_500_000_000,
        allocator_reserve_bytes: 1_500_000_000,
        workspace_bytes: 1_000_000_000,
        transfer_buffer_bytes: 256_000_000,
    };

    let profile_verdicts = vec![
        IndependentProfileVerdict {
            profile_id: "m2-ultra-p150x1-dense-linux-asahi".into(),
            cohort: SplitHostCohort::MacProInternalPcieLinuxAsahi,
            p150_count: 1,
            is_moe: false,
            verdict: "UNAVAILABLE_HARDWARE_RUNTIME".into(),
            blocker: "blocked on TT-Onix #48 (Asahi/ARM Linux PCIe host gate) and #50 (contiguous TT partition)".into(),
        },
        IndependentProfileVerdict {
            profile_id: "m2-ultra-p150x1-dense-macos-metal".into(),
            cohort: SplitHostCohort::MacProInternalPcieMacOsMetal,
            p150_count: 1,
            is_moe: false,
            verdict: "UNAVAILABLE_HARDWARE_RUNTIME".into(),
            blocker: "blocked on TT-Onix #49 (Darwin DriverKit/native-kernel gate)".into(),
        },
        IndependentProfileVerdict {
            profile_id: "m2-ultra-p150x2-dense".into(),
            cohort: SplitHostCohort::MacProInternalPcieLinuxAsahi,
            p150_count: 2,
            is_moe: false,
            verdict: "UNAVAILABLE_HARDWARE_RUNTIME".into(),
            blocker: "independent second-card qualification gated on P150x1 correctness + observed TT links".into(),
        },
        IndependentProfileVerdict {
            profile_id: "m2-ultra-p150x1-moe-static-placement".into(),
            cohort: SplitHostCohort::MacProInternalPcieLinuxAsahi,
            p150_count: 1,
            is_moe: true,
            verdict: "UNAVAILABLE_HARDWARE_RUNTIME".into(),
            blocker: "independent MoE checkpoint qualification gated on TT-Onix #51".into(),
        },
    ];

    let base_claim = SplitQualificationClaim {
        cohort: SplitHostCohort::MacProInternalPcieLinuxAsahi,
        claims_internal_pcie_mac_pro_result: true,
        claims_unified_plus_gddr_coherent_pool: false,
        numerical_and_continuation_passed: true,
        claims_performance_qualification: false,
        quantization_changed_at_boundary: false,
        claims_exact_parity_across_quant_change: false,
        borrows_dense_pass_for_moe: false,
        borrows_p150x1_pass_for_p150x2: false,
    };

    let mut negative_controls = Vec::new();

    // NC1: network/prototype result claiming internal-PCIe Mac Pro qualification
    {
        let mut bad = base_claim.clone();
        bad.cohort = SplitHostCohort::NetworkConnectedPrototype;
        let (st, detail) = validate_split_claim(&bad);
        negative_controls.push(NegativeControl {
            case_id: "nc1_network_prototype_not_internal_pcie".into(),
            outcome: st,
            detail,
        });
    }

    // NC2: summing Apple unified memory + P150 GDDR as one coherent allocation
    {
        let mut bad = base_claim.clone();
        bad.claims_unified_plus_gddr_coherent_pool = true;
        let (st, detail) = validate_split_claim(&bad);
        negative_controls.push(NegativeControl {
            case_id: "nc2_unified_plus_gddr_not_coherent_pool".into(),
            outcome: st,
            detail,
        });
    }

    // NC3: performance claim without numerical/continuation acceptance
    {
        let mut bad = base_claim.clone();
        bad.numerical_and_continuation_passed = false;
        bad.claims_performance_qualification = true;
        let (st, detail) = validate_split_claim(&bad);
        negative_controls.push(NegativeControl {
            case_id: "nc3_performance_before_numerical_acceptance".into(),
            outcome: st,
            detail,
        });
    }

    // NC4: quantization change claimed as exact parity
    {
        let mut bad = base_claim.clone();
        bad.quantization_changed_at_boundary = true;
        bad.claims_exact_parity_across_quant_change = true;
        let (st, detail) = validate_split_claim(&bad);
        negative_controls.push(NegativeControl {
            case_id: "nc4_quant_translation_not_exact_parity".into(),
            outcome: st,
            detail,
        });
    }

    // NC5: dense-model pass borrowed for MoE
    {
        let mut bad = base_claim.clone();
        bad.borrows_dense_pass_for_moe = true;
        let (st, detail) = validate_split_claim(&bad);
        negative_controls.push(NegativeControl {
            case_id: "nc5_dense_pass_not_borrowed_for_moe".into(),
            outcome: st,
            detail,
        });
    }

    // NC6: P150x1 pass inherited by P150x2
    {
        let mut bad = base_claim.clone();
        bad.borrows_p150x1_pass_for_p150x2 = true;
        let (st, detail) = validate_split_claim(&bad);
        negative_controls.push(NegativeControl {
            case_id: "nc6_p150x1_not_inherited_by_p150x2".into(),
            outcome: st,
            detail,
        });
    }

    // NC7: overlapping expert assignment in MoE split rejected
    {
        let res = verify_disjoint_moe_split(&moe_input, &[0, 2, 4], &[2, 3, 5], 6);
        negative_controls.push(NegativeControl {
            case_id: "nc7_overlapping_moe_expert_assignment".into(),
            outcome: if res.is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "duplicate expert 2 across Apple and P150 rejected".into(),
        });
    }

    // NC8: incomplete expert partition (missing expert 5) rejected
    {
        let res = verify_disjoint_moe_split(&moe_input, &[0, 2, 4], &[1, 3], 6);
        negative_controls.push(NegativeControl {
            case_id: "nc8_incomplete_moe_expert_partition".into(),
            outcome: if res.is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "missing expert 5 in split partition rejected".into(),
        });
    }

    Ok(SplitM2P150Report {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "Mac Pro Apple M2 Ultra + internal PCIe Tenstorrent P150 / P150x2",
            "Reuses #41 dense stage parity and verifies disjoint MoE split & cohort/memory rules on CPU; physical M2 Ultra + P150 execution blocked on TT-Onix #48/#49/#50/#51",
        ),
        synthetic_contract_fixture: true,
        stage_contract_reused_checks: stage_rep.parity_checks.len(),
        dense_split_max_err,
        moe_disjoint_split_max_err,
        logical_payload_1tok_bytes,
        logical_payload_4096tok_bytes,
        derived_logical_payload_not_measured_throughput: true,
        apple_memory_inventory,
        p150_memory_inventory,
        profile_verdicts,
        negative_controls,
    })
}

pub fn inputs_json(report: &SplitM2P150Report) -> String {
    let digest = DigestRef::sha256(b"split-m2-p150-contract-v1");
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.split-m2-p150.inputs.v1\",\n",
            "  \"issue\": 42,\n",
            "  \"execution_class\": {},\n",
            "  \"reuses_stage_issue\": 41,\n",
            "  \"pinned_sidecar_commit\": {},\n",
            "  \"contract_digest\": {},\n",
            "  \"synthetic_contract_fixture\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(SIDECAR_UPSTREAM_COMMIT),
        digest.to_json(),
        report.synthetic_contract_fixture,
    )
}

pub fn results_json(report: &SplitM2P150Report) -> String {
    let verdicts: Vec<String> = report
        .profile_verdicts
        .iter()
        .map(|v| {
            format!(
                "{{\"profile_id\":{},\"cohort\":{},\"p150_count\":{},\"is_moe\":{},\"verdict\":{},\"blocker\":{}}}",
                quote(&v.profile_id),
                quote(v.cohort.as_str()),
                v.p150_count,
                v.is_moe,
                quote(&v.verdict),
                quote(&v.blocker),
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
            "  \"schema\": \"inference-lab.split-m2-p150.results.v1\",\n",
            "  \"issue\": 42,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"stage_contract_reused_checks\": {},\n",
            "  \"dense_split_max_err\": {},\n",
            "  \"moe_disjoint_split_max_err\": {},\n",
            "  \"logical_payload_1tok_bytes\": {},\n",
            "  \"logical_payload_4096tok_bytes\": {},\n",
            "  \"derived_logical_payload_not_measured_throughput\": {},\n",
            "  \"apple_memory_total_bytes_planned\": {},\n",
            "  \"p150_memory_total_bytes_planned\": {},\n",
            "  \"profile_verdicts\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.synthetic_contract_fixture,
        report.stage_contract_reused_checks,
        finite_f64(report.dense_split_max_err),
        finite_f64(report.moe_disjoint_split_max_err),
        report.logical_payload_1tok_bytes,
        report.logical_payload_4096tok_bytes,
        report.derived_logical_payload_not_measured_throughput,
        report.apple_memory_inventory.total_bytes(),
        report.p150_memory_inventory.total_bytes(),
        verdicts.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &SplitM2P150Report) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #42: M2 Ultra + Tenstorrent P150 Split-Model Qualification"
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
        "- Dense split parity (reusing `#41` over {} checks): `max_err = {:.3e}`; Disjoint MoE split: `max_err = {:.3e}`",
        report.stage_contract_reused_checks,
        report.dense_split_max_err,
        report.moe_disjoint_split_max_err
    )
    .unwrap();
    writeln!(
        out,
        "- Derived logical payload (`hidden=8192`, `2B/elem`, NOT measured throughput): `{} B/tok` (`16 KiB`), `{} B / 4096 rows` (`64 MiB`)",
        report.logical_payload_1tok_bytes, report.logical_payload_4096tok_bytes
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Independent Profile Verdicts").unwrap();
    for v in &report.profile_verdicts {
        writeln!(
            out,
            "- `{}` (`{}`, `P150x{}`, `moe={}`): `{}` — {}",
            v.profile_id,
            v.cohort.as_str(),
            v.p150_count,
            v.is_moe,
            v.verdict,
            v.blocker
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
