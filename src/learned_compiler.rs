//! Issue #9: Recover external learned-compiler experiments (`compiler-lab#30` Qwen causal
//! baseline and `compiler-lab#31` DiffusionGemma result) with explicit `blocked_on_artifacts`
//! asset recovery inventory, preserved historical baselines, Compiler Lab checker ownership,
//! Darwin/AppleContainer isolation gate, and model-free integration controls.

use std::fmt::Write as FmtWrite;

use crate::json::{opt_str, quote, str_array};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetInventoryEntry {
    pub asset_id: String,
    pub expected_origin: String,
    pub recovery_status: String,
    pub publishability_class: String,
    pub original_digest: Option<DigestRef>,
    pub notes: String,
}

/// Preserves historical claim, current replay outcome, and newly measured result as distinct fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreservedBaselineRecord {
    pub issue_ref: String,
    pub condition_name: String,
    pub declared_split: String,
    pub historical_claim: String,
    pub current_replay_outcome: OutcomeStatus,
    pub newly_measured_result: Option<String>,
    pub unmet_criteria_preserved: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeIsolationHost {
    Darwin,
    AppleContainer,
    LinuxUnavailable,
}

impl NativeIsolationHost {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Darwin => "darwin_native_isolation",
            Self::AppleContainer => "apple_container_isolation",
            Self::LinuxUnavailable => "linux_unavailable_no_darwin_or_apple_container",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeReplayCandidate {
    pub candidate_id: String,
    pub expected_abi_profile: String,
    pub candidate_abi_profile: String,
    pub expected_checker_digest: DigestRef,
    pub observed_checker_digest: DigestRef,
    pub partition_feedback_contaminated: bool,
    pub attempt_index: usize,
    pub max_attempts: usize,
    pub uses_hidden_teacher_compiler_repair: bool,
    pub model_supplies_trusted_expected_output: bool,
    pub host: NativeIsolationHost,
    pub assembly_valid: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeReplayControlVerdict {
    pub candidate_id: String,
    pub outcome: OutcomeStatus,
    pub isolation_weakened: bool,
    pub rejection_reason: Option<String>,
}

/// Evaluate Compiler Lab native candidate boundary controls without duplicating Compiler Lab's checker.
pub fn check_native_integration_control(
    cand: &NativeReplayCandidate,
) -> NativeReplayControlVerdict {
    let reject = |outcome: OutcomeStatus, reason: &str| NativeReplayControlVerdict {
        candidate_id: cand.candidate_id.clone(),
        outcome,
        isolation_weakened: false,
        rejection_reason: Some(reason.into()),
    };

    if cand.candidate_abi_profile != cand.expected_abi_profile {
        return reject(OutcomeStatus::Invalid, "frozen_abi_profile_mismatch");
    }
    if cand.observed_checker_digest != cand.expected_checker_digest {
        return reject(OutcomeStatus::Invalid, "frozen_checker_identity_mismatch");
    }
    if cand.partition_feedback_contaminated {
        return reject(
            OutcomeStatus::Invalid,
            "feedback_final_partition_contamination",
        );
    }
    if cand.attempt_index >= cand.max_attempts {
        return reject(OutcomeStatus::Exhausted, "attempt_budget_exhausted");
    }
    if cand.uses_hidden_teacher_compiler_repair || cand.model_supplies_trusted_expected_output {
        return reject(
            OutcomeStatus::Invalid,
            "hidden_teacher_replacement_or_model_oracle_forbidden",
        );
    }
    if cand.host == NativeIsolationHost::LinuxUnavailable {
        return reject(
            OutcomeStatus::UnavailableHardwareRuntime,
            "linux_host_lacks_darwin_or_apple_container_isolation",
        );
    }
    if !cand.assembly_valid {
        return reject(OutcomeStatus::Invalid, "failed_assembly_gate");
    }
    NativeReplayControlVerdict {
        candidate_id: cand.candidate_id.clone(),
        outcome: OutcomeStatus::Accepted,
        isolation_weakened: false,
        rejection_reason: None,
    }
}

#[derive(Debug, Clone)]
pub struct LearnedCompilerReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub compiler_lab_checked_commit: String,
    pub asset_inventory: Vec<AssetInventoryEntry>,
    pub preserved_baselines: Vec<PreservedBaselineRecord>,
    pub integration_controls: Vec<NativeReplayControlVerdict>,
    pub retained_compiler_lab_boundaries: Vec<String>,
}

pub fn run_default_harness() -> LearnedCompilerReport {
    let compiler_lab_checked_commit = "50dcbdced71df3a4f4cfad2d99638ad6e7bd69ae".to_string();
    let checker_digest = DigestRef::sha256(b"compiler-lab@50dcbdce:src/native_candidate.rs");

    let asset_inventory = vec![
        AssetInventoryEntry {
            asset_id: "research/learned-compiler".into(),
            expected_origin: "external checkout referenced by compiler-lab#30 and #31".into(),
            recovery_status: "missing_unlocated_not_in_compiler_lab_50dcbdce".into(),
            publishability_class: "review_required_before_public_import".into(),
            original_digest: None,
            notes: "Not tracked in brittonr/compiler-lab@50dcbdce; cannot claim completed migration.".into(),
        },
        AssetInventoryEntry {
            asset_id: "acceptance.json".into(),
            expected_origin: "research/learned-compiler/acceptance.json".into(),
            recovery_status: "missing_unlocated_original_receipt".into(),
            publishability_class: "restricted_private_donor_hash_only".into(),
            original_digest: None,
            notes: "Original receipt must be located and hash-bound before claiming reproduced historical run.".into(),
        },
        AssetInventoryEntry {
            asset_id: "qwen30-causal-adapter-and-weights".into(),
            expected_origin: "compiler-lab#30".into(),
            recovery_status: "missing_unlocated_weights_never_committed".into(),
            publishability_class: "weights_excluded_adapter_review_required".into(),
            original_digest: None,
            notes: "Causal baseline recovered alongside diffusion so continuations have a real baseline.".into(),
        },
        AssetInventoryEntry {
            asset_id: "diffusiongemma31-adapter-and-failed-outputs".into(),
            expected_origin: "compiler-lab#31".into(),
            recovery_status: "missing_unlocated_failed_assembly_outputs".into(),
            publishability_class: "restricted_private_donor_review_required".into(),
            original_digest: None,
            notes: "Failed/truncated outputs must be retained unchanged once located.".into(),
        },
    ];

    let preserved_baselines = vec![
        PreservedBaselineRecord {
            issue_ref: "brittonr/compiler-lab#30".into(),
            condition_name: "qwen-causal-baseline (released vs adapted; reasoning-mode separated)".into(),
            declared_split: "14 training / 2 held-out tasks".into(),
            historical_claim: "Unverified historical report: 27 evaluated outputs across released/adapted & reasoning-mode separated conditions, with one 12-byte proved artifact recorded in external acceptance.json (not rechecked locally until receipt recovered)".into(),
            current_replay_outcome: OutcomeStatus::BlockedOnArtifacts,
            newly_measured_result: None,
            unmet_criteria_preserved: vec![
                "Original acceptance.json receipt and research/learned-compiler checkout not yet located".into(),
                "Released vs adapted and reasoning-mode separation must survive recovery unchanged".into(),
            ],
        },
        PreservedBaselineRecord {
            issue_ref: "brittonr/compiler-lab#31".into(),
            condition_name: "diffusiongemma-baseline (BF16 / unmapped_float32 / mapped_float32)".into(),
            declared_split: "Same frozen corpus split as #30".into(),
            historical_claim: "Unverified historical report: every generated output failed assembly (0 accepted), 900s timeout on the 16-canvas branch/register case, separate BF16 / unmapped_float32 / mapped_float32 conditions, and official JAX/Orbax trainer unmet (bounded PyTorch non-equivalent)".into(),
            current_replay_outcome: OutcomeStatus::BlockedOnArtifacts,
            newly_measured_result: None,
            unmet_criteria_preserved: vec![
                "Zero outputs passed assembly across BF16, unmapped_float32, and mapped_float32 conditions".into(),
                "900s timeout on the 16-canvas branch/register case preserved".into(),
                "Official JAX/Orbax trainer unmet; bounded PyTorch run is non-equivalent".into(),
                "Any continuation requires a separately named hypothesis (e.g. continuous/flow or mnemonic completion)".into(),
            ],
        },
    ];

    let base_cand = |id: &str| NativeReplayCandidate {
        candidate_id: id.into(),
        expected_abi_profile: "arm64-darwin-aapcs64".into(),
        candidate_abi_profile: "arm64-darwin-aapcs64".into(),
        expected_checker_digest: checker_digest.clone(),
        observed_checker_digest: checker_digest.clone(),
        partition_feedback_contaminated: false,
        attempt_index: 0,
        max_attempts: 4,
        uses_hidden_teacher_compiler_repair: false,
        model_supplies_trusted_expected_output: false,
        host: NativeIsolationHost::AppleContainer,
        assembly_valid: true,
    };

    let mut c_neg_asm = base_cand("synthetic-negative-diffusion-failed-assembly");
    c_neg_asm.assembly_valid = false;

    let mut c_linux_unavail = base_cand("negative-linux-host-no-isolation-weakening");
    c_linux_unavail.host = NativeIsolationHost::LinuxUnavailable;

    let mut c_abi = base_cand("negative-abi-profile-mismatch");
    c_abi.candidate_abi_profile = "x86_64-linux-sysv".into();

    let mut c_checker = base_cand("negative-frozen-checker-identity-mismatch");
    c_checker.observed_checker_digest = DigestRef::sha256(b"tampered-native-checker");

    let mut c_part = base_cand("negative-feedback-final-contamination");
    c_part.partition_feedback_contaminated = true;

    let mut c_budget = base_cand("negative-attempt-budget-exhausted");
    c_budget.attempt_index = 4;

    let mut c_teacher = base_cand("negative-hidden-teacher-compiler-repair");
    c_teacher.uses_hidden_teacher_compiler_repair = true;

    let mut c_model_oracle = base_cand("negative-model-supplies-trusted-expected-output");
    c_model_oracle.model_supplies_trusted_expected_output = true;

    let integration_controls = vec![
        check_native_integration_control(&base_cand("synthetic-positive-qwen-apple-container")),
        check_native_integration_control(&c_neg_asm),
        check_native_integration_control(&c_linux_unavail),
        check_native_integration_control(&c_abi),
        check_native_integration_control(&c_checker),
        check_native_integration_control(&c_part),
        check_native_integration_control(&c_budget),
        check_native_integration_control(&c_teacher),
        check_native_integration_control(&c_model_oracle),
    ];

    LearnedCompilerReport {
        execution_class: ExecutionClass::BlockedOnArtifacts,
        hardware_gate: HardwareGate::blocked_on_artifacts(
            "external_learned_compiler_checkout_and_darwin_apple_container_host",
            "External `research/learned-compiler` checkout and `acceptance.json` are absent from `compiler-lab@50dcbdce`, and this Linux host lacks Darwin/AppleContainer native isolation.",
        ),
        compiler_lab_checked_commit,
        asset_inventory,
        preserved_baselines,
        integration_controls,
        retained_compiler_lab_boundaries: vec![
            "compiler-lab/src/dataset.rs & dataset/partition.rs: canonical task identity and feedback/final splits".into(),
            "compiler-lab/src/native_candidate.rs: assembly/ABI checks, semantic evaluation, proof admission, isolation & replay".into(),
            "compiler-lab/src/native_candidate/comparison.rs: attempt/feedback accounting and scored-input protection".into(),
            "Distinct from Inference Lab #2 (source proof completion) and compiler-lab#216 (compiler-input generation)".into(),
        ],
    }
}

pub fn inputs_json(report: &LearnedCompilerReport) -> String {
    let inv: Vec<String> = report
        .asset_inventory
        .iter()
        .map(|a| {
            format!(
                concat!(
                    "{{\"asset_id\":{},\"expected_origin\":{},\"recovery_status\":{},",
                    "\"publishability_class\":{},\"original_digest\":{},\"notes\":{}}}"
                ),
                quote(&a.asset_id),
                quote(&a.expected_origin),
                quote(&a.recovery_status),
                quote(&a.publishability_class),
                a.original_digest
                    .as_ref()
                    .map_or_else(|| "null".into(), DigestRef::to_json),
                quote(&a.notes),
            )
        })
        .collect();
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.issue-09-learned-compiler.inputs.v1\",\n",
            "  \"issue\": 9,\n",
            "  \"execution_class\": {},\n",
            "  \"compiler_lab_checked_commit\": {},\n",
            "  \"asset_inventory\": [{}],\n",
            "  \"retained_compiler_lab_boundaries\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(&report.compiler_lab_checked_commit),
        inv.join(","),
        str_array(&report.retained_compiler_lab_boundaries),
    )
}

pub fn results_json(report: &LearnedCompilerReport) -> String {
    let mut out = String::new();
    writeln!(out, "{{").unwrap();
    writeln!(
        out,
        "  \"schema\": \"inference-lab.issue-09-learned-compiler.results.v1\","
    )
    .unwrap();
    writeln!(out, "  \"issue\": 9,").unwrap();
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

    let base: Vec<String> = report
        .preserved_baselines
        .iter()
        .map(|b| {
            format!(
                concat!(
                    "{{\"issue_ref\":{},\"condition_name\":{},\"declared_split\":{},",
                    "\"historical_claim\":{},\"current_replay_outcome\":{},",
                    "\"newly_measured_result\":{},\"unmet_criteria_preserved\":{}}}"
                ),
                quote(&b.issue_ref),
                quote(&b.condition_name),
                quote(&b.declared_split),
                quote(&b.historical_claim),
                quote(b.current_replay_outcome.as_str()),
                opt_str(b.newly_measured_result.as_deref()),
                str_array(&b.unmet_criteria_preserved),
            )
        })
        .collect();
    writeln!(out, "  \"preserved_baselines\": [{}],", base.join(",")).unwrap();

    let ctrls: Vec<String> = report
        .integration_controls
        .iter()
        .map(|c| {
            format!(
                concat!(
                    "{{\"candidate_id\":{},\"outcome\":{},\"isolation_weakened\":{},",
                    "\"rejection_reason\":{}}}"
                ),
                quote(&c.candidate_id),
                quote(c.outcome.as_str()),
                c.isolation_weakened,
                opt_str(c.rejection_reason.as_deref()),
            )
        })
        .collect();
    writeln!(out, "  \"integration_controls\": [{}]", ctrls.join(",")).unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn summary_markdown(report: &LearnedCompilerReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #9: Learned-Compiler Asset Recovery & Preserved Baselines"
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
        "- Artifact/host gate: `{}` (`{}`)",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(
        out,
        "- Checked Compiler Lab tree: `{}` (`research/learned-compiler` absent; migration explicitly blocked on missing assets)",
        report.compiler_lab_checked_commit
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Preserved Historical Baselines").unwrap();
    for b in &report.preserved_baselines {
        writeln!(
            out,
            "- `{}` (`{}`): historical claim preserved; current replay=`{}`; new measurement=`none`",
            b.issue_ref,
            b.condition_name,
            b.current_replay_outcome.as_str()
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "## Integration Boundary Controls").unwrap();
    writeln!(out, "| Control | Outcome | Isolation Weakened | Reason |").unwrap();
    writeln!(out, "| --- | --- | --- | --- |").unwrap();
    for c in &report.integration_controls {
        writeln!(
            out,
            "| `{}` | `{}` | `{}` | {} |",
            c.candidate_id,
            c.outcome.as_str(),
            c.isolation_weakened,
            c.rejection_reason.as_deref().unwrap_or("none")
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "> Explicitly `blocked_on_artifacts`. No private Compiler Lab asset or unlocated checkpoint was fabricated or copied."
    )
    .unwrap();
    out
}
